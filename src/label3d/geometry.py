"""Epipolar geometry, triangulation, and reprojection.

Uses calibration data (K, R, t) from aniposelib-format calibration.toml.
Coordinates are raw pixels; undistortion is applied on the fly when needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import toml

from label3d.calibration_registry import CalibrationEntry, save_calibration

if TYPE_CHECKING:
    import pandas as pd

    from label3d.points import PointsStore
    from label3d.skeleton import ProjectSkeleton

logger = logging.getLogger(__name__)


class CameraParams:
    """Calibration parameters for one camera."""

    def __init__(
        self,
        name: str,
        matrix: np.ndarray,
        dist: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        size: tuple[int, int],
    ):
        self.name = name
        self.matrix = matrix          # 3x3 intrinsic K
        self.dist = dist              # distortion coefficients
        self.rvec = rvec              # Rodrigues rotation (world→camera)
        self.tvec = tvec              # translation (world→camera)
        self.size = size              # (width, height)
        # Precompute rotation matrix
        self.R, _ = cv2.Rodrigues(self.rvec)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort pixel coordinates.

        Args:
            points: (N, 2) array of pixel coordinates.

        Returns:
            (N, 2) array of undistorted pixel coordinates.
        """
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        undistorted = cv2.undistortPoints(pts, self.matrix, self.dist, P=self.matrix)
        return undistorted.reshape(-1, 2)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D world points to 2D pixel coordinates.

        Args:
            points_3d: (N, 3) array of world coordinates.

        Returns:
            (N, 2) array of pixel coordinates.
        """
        pts, _ = cv2.projectPoints(
            points_3d.astype(np.float64),
            self.rvec,
            self.tvec,
            self.matrix,
            self.dist,
        )
        return pts.reshape(-1, 2)


def load_cameras(calibration_path: Path) -> dict[str, CameraParams]:
    """Load all cameras from a calibration.toml file.

    Returns:
        dict mapping camera name to CameraParams.
    """
    cal = toml.load(calibration_path)
    cameras = {}
    for key in sorted(cal.keys()):
        if key == "metadata":
            continue
        section = cal[key]
        name = section.get("name", key)
        cameras[name] = CameraParams(
            name=name,
            matrix=np.array(section["matrix"], dtype=np.float64),
            dist=np.array(section["distortions"], dtype=np.float64),
            rvec=np.array(section["rotation"], dtype=np.float64),
            tvec=np.array(section["translation"], dtype=np.float64),
            size=tuple(section.get("size", [0, 0])),
        )
    return cameras


def compute_fundamental_matrix(
    cam_a: CameraParams, cam_b: CameraParams
) -> np.ndarray:
    """Compute the 3x3 fundamental matrix F from camera A to camera B.

    A point x in camera A satisfies: x_b^T F x_a = 0 for corresponding x_b.
    The epipolar line in camera B for point x_a is: l = F @ [x_a, 1].

    Uses: E = [t_rel]_x @ R_rel; F = K_b^{-T} @ E @ K_a^{-1}
    where R_rel = R_b @ R_a^T, t_rel = t_b - R_rel @ t_a.
    """
    R_rel = cam_b.R @ cam_a.R.T
    t_rel = cam_b.tvec.ravel() - R_rel @ cam_a.tvec.ravel()

    # Skew-symmetric matrix of t_rel
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0],
    ], dtype=np.float64)

    E = tx @ R_rel
    K_a_inv = np.linalg.inv(cam_a.matrix)
    K_b_inv_T = np.linalg.inv(cam_b.matrix).T
    F = K_b_inv_T @ E @ K_a_inv

    # Normalize so that ||F||_fro = 1
    F = F / np.linalg.norm(F)
    return F


def compute_epipolar_line(
    F: np.ndarray,
    point_2d: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Compute epipolar line endpoints clipped to the target image.

    Args:
        F: 3x3 fundamental matrix (from source camera to target camera).
        point_2d: (x, y) pixel coordinate in the source camera.
        image_size: (width, height) of the target camera image.

    Returns:
        ((x1, y1), (x2, y2)) endpoints of the line segment, or None if
        the line does not intersect the image.
    """
    x, y = point_2d
    l = F @ np.array([x, y, 1.0], dtype=np.float64)
    a, b, c = l

    w, h = image_size

    # Avoid division by zero
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return None

    # Find intersections with image borders
    candidates = []

    if abs(b) > 1e-12:
        # y at x=0
        y0 = -c / b
        if 0 <= y0 <= h:
            candidates.append((0.0, y0))
        # y at x=w
        yw = -(a * w + c) / b
        if 0 <= yw <= h:
            candidates.append((float(w), yw))

    if abs(a) > 1e-12:
        # x at y=0
        x0 = -c / a
        if 0 <= x0 <= w:
            candidates.append((x0, 0.0))
        # x at y=h
        xh = -(b * h + c) / a
        if 0 <= xh <= w:
            candidates.append((xh, float(h)))

    # Deduplicate near-identical points
    unique = []
    for pt in candidates:
        if not any(abs(pt[0] - u[0]) < 1 and abs(pt[1] - u[1]) < 1 for u in unique):
            unique.append(pt)

    if len(unique) < 2:
        return None

    return (unique[0], unique[1])


def cameras_from_cgroup(cgroup) -> dict[str, "CameraParams"]:
    """Build a CameraParams dict from an aniposelib CameraGroup.

    Enables creating an EpipolarCache directly from a calibrated CameraGroup
    without writing/reading a TOML file.
    """
    result = {}
    for cam in cgroup.cameras:
        size = cam.get_size()
        result[cam.get_name()] = CameraParams(
            name=cam.get_name(),
            matrix=np.array(cam.get_camera_matrix(), dtype=np.float64),
            dist=np.array(cam.get_distortions(), dtype=np.float64),
            rvec=np.array(cam.get_rotation(), dtype=np.float64),
            tvec=np.array(cam.get_translation(), dtype=np.float64),
            size=tuple(size) if size is not None else (0, 0),
        )
    return result


class EpipolarCache:
    """Caches fundamental matrices between camera pairs."""

    def __init__(self, cameras: dict[str, CameraParams]):
        self._cameras = cameras
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def get_fundamental(self, cam_a_name: str, cam_b_name: str) -> np.ndarray:
        """Get (or compute and cache) F from cam_a to cam_b."""
        key = (cam_a_name, cam_b_name)
        if key not in self._cache:
            self._cache[key] = compute_fundamental_matrix(
                self._cameras[cam_a_name], self._cameras[cam_b_name]
            )
        return self._cache[key]

    def get_epipolar_line(
        self,
        source_cam: str,
        target_cam: str,
        point_2d: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Compute epipolar line in target camera for a point in source camera.

        Uses undistorted coordinates for accurate geometry.
        """
        src = self._cameras[source_cam]
        tgt = self._cameras[target_cam]

        # Undistort the source point
        undistorted = src.undistort_points(np.array([[point_2d[0], point_2d[1]]]))
        pt = (float(undistorted[0, 0]), float(undistorted[0, 1]))

        F = self.get_fundamental(source_cam, target_cam)
        return compute_epipolar_line(F, pt, tgt.size)


# -- Ad-hoc triangulation (for axis alignment) ---------------------------------

def triangulate_point(
    cameras: dict[str, "CameraParams"],
    picks_2d: dict[str, tuple[float, float]],
) -> np.ndarray | None:
    """Triangulate a single 3D point from 2D observations in ≥2 cameras.

    Args:
        cameras: calibration params keyed by camera name.
        picks_2d: {cam_name: (x, y)} raw pixel coordinates.

    Returns:
        (3,) world-space point, or None if fewer than 2 valid observations
        or if triangulation fails.
    """
    valid = {c: xy for c, xy in picks_2d.items() if c in cameras}
    if len(valid) < 2:
        return None

    cgroup = build_camera_group({c: cameras[c] for c in valid})
    cam_order = sorted(valid.keys())

    pts = np.array([[valid[c][0], valid[c][1]] for c in cam_order],
                   dtype=np.float64)[:, np.newaxis, :]  # (n, 1, 2)

    try:
        xyz = cgroup.triangulate(pts, undistort=True)  # (1, 3)
        if np.any(np.isnan(xyz)):
            return None
        return xyz[0].astype(float)
    except Exception:
        return None


# -- Axis alignment ------------------------------------------------------------

def _arb_perp(v: np.ndarray) -> np.ndarray:
    """Return an arbitrary unit vector perpendicular to v."""
    v = v / np.linalg.norm(v)
    # Pick the component of v smallest in magnitude and zero it, then cross
    i = int(np.argmin(np.abs(v)))
    basis = np.zeros(3)
    basis[i] = 1.0
    w = np.cross(v, basis)
    return w / np.linalg.norm(w)


def compute_axis_alignment(
    origin: np.ndarray,
    x_vec: "np.ndarray | None",
    y_vec: "np.ndarray | None",
    z_vec: "np.ndarray | None",
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a rotation matrix aligning NEW world axes to the calibrated world.

    At least two of x_vec, y_vec, z_vec must be non-None.

    Gram-Schmidt ordering: the longest axis vector is taken as primary (its
    direction is preserved exactly after normalisation). The next longest is
    orthogonalised against the first. The third axis is the cross product of
    the first two, giving a right-handed system; flip individual axes if the
    resulting handedness is wrong.

    Flips are applied to the RAW vectors before Gram-Schmidt.

    Returns:
        R_align: (3, 3) rotation matrix whose columns are [x_hat, y_hat, z_hat]
                 expressed in the OLD (calibrated) world coordinate system.
                 Transform: p_old = R_align @ p_new + origin
                            p_new = R_align.T @ (p_old − origin)
        origin:  (3,) origin in OLD world coords.

    New camera params after alignment:
        R_cam_new = R_cam_old @ R_align   (as rotation matrix)
        t_cam_new = R_cam_old @ origin + t_cam_old
    """
    raw: dict[str, np.ndarray | None] = {
        "x": (-x_vec if flip_x else x_vec.copy()) if x_vec is not None else None,
        "y": (-y_vec if flip_y else y_vec.copy()) if y_vec is not None else None,
        "z": (-z_vec if flip_z else z_vec.copy()) if z_vec is not None else None,
    }
    available = {k: v for k, v in raw.items() if v is not None}
    if len(available) < 2:
        raise ValueError("At least two axis vectors are required.")

    lengths = {k: float(np.linalg.norm(v)) for k, v in available.items()}

    # GS order: longest first, then next longest
    gs_order = sorted(available.keys(), key=lambda k: -lengths[k])

    basis: dict[str, np.ndarray] = {}

    # Primary axis — just normalise
    p = gs_order[0]
    lp = lengths[p]
    if lp < 1e-10:
        raise ValueError(f"Axis '{p}' vector has zero length.")
    basis[p] = available[p] / lp

    # Secondary — orthogonalise against primary
    s = gs_order[1]
    v = available[s].copy()
    v -= np.dot(v, basis[p]) * basis[p]
    ls = float(np.linalg.norm(v))
    if ls < 1e-10:
        raise ValueError(f"Axis '{s}' is parallel to '{p}'.")
    basis[s] = v / ls

    # Tertiary — cross product for right-hand rule, or GS of provided vector
    remaining = [k for k in ("x", "y", "z") if k not in basis]
    third = remaining[0]

    if third in available:
        # Orthogonalise the user-provided third vector
        w = available[third].copy()
        w -= np.dot(w, basis[p]) * basis[p]
        w -= np.dot(w, basis[s]) * basis[s]
        lw = float(np.linalg.norm(w))
        if lw < 1e-10:
            # Degenerate — fall through to cross product
            w = None
        else:
            basis[third] = w / lw
            w = basis[third]
    else:
        w = None

    if w is None:
        # Derive from right-hand cross-product rule: x×y=z, y×z=x, z×x=y
        _rh = {"z": ("x", "y"), "x": ("y", "z"), "y": ("z", "x")}
        a_lbl, b_lbl = _rh[third]
        basis[third] = np.cross(basis[a_lbl], basis[b_lbl])

    R_align = np.column_stack([basis["x"], basis["y"], basis["z"]])
    return R_align, np.asarray(origin, dtype=float)


def apply_alignment_to_cameras(
    cameras: dict[str, "CameraParams"],
    R_align: np.ndarray,
    origin: np.ndarray,
) -> dict[str, "CameraParams"]:
    """Return new CameraParams with extrinsics rotated to the aligned world.

    For each camera:
        R_new = R_cam @ R_align   (as matrix; stored back as Rodrigues)
        t_new = R_cam @ origin + t_cam
    """
    import cv2 as _cv2

    updated: dict[str, CameraParams] = {}
    for name, cp in cameras.items():
        R_new = cp.R @ R_align
        rvec_new, _ = _cv2.Rodrigues(R_new)
        tvec_new = cp.R @ origin + cp.tvec.ravel()

        updated[name] = CameraParams(
            name=cp.name,
            matrix=cp.matrix.copy(),
            dist=cp.dist.copy(),
            rvec=rvec_new.ravel(),
            tvec=tvec_new,
            size=cp.size,
        )
    return updated


def save_aligned_calibration(
    cameras: dict[str, "CameraParams"],
    exp_dir: "Path",
    R_align: np.ndarray,
    origin: np.ndarray,
    parent: str,
    label: str = "align",
) -> CalibrationEntry:
    """Save aligned camera extrinsics as an "align" registry entry.

    parent is the id of the calibration that was aligned. The rotation matrix
    and origin go under [metadata.alignment]: CameraGroup.load() would read a
    top-level [alignment] table as a camera.
    """
    return save_calibration(
        exp_dir, build_camera_group(cameras), label, "align", parent=parent,
        alignment={"origin": np.asarray(origin).tolist(), "R_align": np.asarray(R_align).tolist()},
    )


# -- Triangulation (frame-level, for labeling) ---------------------------------

def build_camera_group(cameras: dict[str, "CameraParams"]):
    """Build an aniposelib CameraGroup from a CameraParams dict.

    Returns the CameraGroup with cameras ordered by sorted camera name.
    """
    from aniposelib.cameras import Camera as AniCamera, CameraGroup

    ani_cams = []
    for name in sorted(cameras.keys()):
        cp = cameras[name]
        cam = AniCamera(
            matrix=cp.matrix.copy(),
            dist=cp.dist.copy(),
            size=list(cp.size),
            rvec=cp.rvec.copy(),
            tvec=cp.tvec.copy(),
            name=cp.name,
        )
        ani_cams.append(cam)
    return CameraGroup(ani_cams)


# A camera's point is used for triangulation in these states (not not_visible).
TRIANGULATED_STATES = ("set", "predicted")


def triangulation_mask(
    df: "pd.DataFrame", cam: str, skeletons: "ProjectSkeleton | None" = None
) -> np.ndarray:
    """Rows where cam's point counts for triangulation: set/predicted, x/y present.

    With skeletons, the node must also be in cam's skeleton for that row's trial.
    """
    mask = (
        df[(cam, "state")].isin(TRIANGULATED_STATES).to_numpy()
        & df[(cam, "x")].notna().to_numpy()
        & df[(cam, "y")].notna().to_numpy()
    )
    if skeletons is not None and mask.any():
        mask &= skeletons.available_mask(df.index, cam)
    return mask


def triangulate_rows(
    cameras: dict[str, "CameraParams"],
    df: "pd.DataFrame",
    skeletons: "ProjectSkeleton | None" = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Triangulate every row of a points DataFrame in one batch.

    Each row is one (frame, instance, bodypart). Cameras whose point is set or
    predicted are undistorted to normalised coordinates and stacked into one
    DLT system per row; all rows are solved with a single batched SVD. This is
    the same DLT as aniposelib's CameraGroup.triangulate, without its per-point
    Python loop. With skeletons, a camera's point counts only where its
    skeleton has the node (see triangulation_mask).

    Returns:
        p3d: (N, 3) world points, NaN for rows seen by fewer than 2 cameras.
        reproj: {camera: (N, 2)} reprojected pixels, NaN where p3d is NaN.
    """
    cams = [c for c in sorted(cameras) if (c, "x") in df.columns]
    n = len(df)
    p3d = np.full((n, 3), np.nan)
    reproj = {cam: np.full((n, 2), np.nan) for cam in cams}
    if n == 0:
        return p3d, reproj

    A = np.zeros((n, 2 * len(cams), 4))
    n_views = np.zeros(n, dtype=int)
    for i, cam in enumerate(cams):
        good = triangulation_mask(df, cam, skeletons)
        if not good.any():
            continue
        cp = cameras[cam]
        xy = df.loc[good, [(cam, "x"), (cam, "y")]].to_numpy(dtype=np.float64)
        norm = cv2.undistortPoints(xy.reshape(-1, 1, 2), cp.matrix, cp.dist).reshape(-1, 2)
        M = np.hstack([cp.R, cp.tvec.reshape(3, 1)])
        A[good, 2 * i] = norm[:, :1] * M[2] - M[0]
        A[good, 2 * i + 1] = norm[:, 1:] * M[2] - M[1]
        n_views += good

    ok = n_views >= 2
    if ok.any():
        _, _, vh = np.linalg.svd(A[ok])
        X = vh[:, -1, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            p3d[ok] = X[:, :3] / X[:, 3:]
        p3d[~np.isfinite(p3d).all(axis=1)] = np.nan
        ok = ~np.isnan(p3d[:, 0])
        for cam in cams:
            reproj[cam][ok] = cameras[cam].project(p3d[ok])
    return p3d, reproj


def mean_reprojection_errors(
    df: "pd.DataFrame",
    reproj: dict[str, np.ndarray],
    skeletons: "ProjectSkeleton | None" = None,
) -> np.ndarray:
    """(N,) mean distance (px) between each row's used points and their reprojections.

    Only cameras whose point is set or predicted count; NaN where none do or
    there is no reprojection.
    """
    total = np.zeros(len(df))
    count = np.zeros(len(df))
    for cam, xy_r in reproj.items():
        err = np.hypot(
            df[(cam, "x")].to_numpy(dtype=np.float64) - xy_r[:, 0],
            df[(cam, "y")].to_numpy(dtype=np.float64) - xy_r[:, 1],
        )
        use = triangulation_mask(df, cam, skeletons) & ~np.isnan(err)
        total[use] += err[use]
        count[use] += 1
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def triangulate_trial(
    cameras: dict[str, "CameraParams"],
    points_store: "PointsStore",
    trial: str,
    frames: list[int] | None = None,
    skeletons: "ProjectSkeleton | None" = None,
) -> int:
    """Triangulate every row of a trial (or of some of its frames) and store the results.

    3D points and every camera's xr/yr are written in one batch. Rows seen by
    fewer than 2 cameras are cleared, even if they held an earlier result.
    Returns the number of rows with a 3D point.
    """
    rows = points_store.trial_rows(trial, frames)
    if rows.empty:
        return 0
    p3d, reproj = triangulate_rows(cameras, rows, skeletons)
    points_store.set_triangulation(rows.index, p3d, reproj)
    return int((~np.isnan(p3d[:, 0])).sum())


def triangulate_frame(
    cameras: dict[str, "CameraParams"],
    points_store: "PointsStore",
    trial: str,
    frame: int,
    instance: int,
    skeletons: "ProjectSkeleton | None" = None,
) -> dict[str, float]:
    """Triangulate all bodyparts for one instance in one frame.

    Same rules as triangulate_trial: points with state 'set' or 'predicted'
    are used, and bodyparts seen by fewer than 2 cameras have their 3D point
    and reprojections cleared.

    Returns dict mapping bodypart name → mean reprojection error (px).
    Bodyparts with fewer than 2 views return NaN error.
    """
    rows = points_store.instance_rows(trial, frame, instance)
    if rows.empty:
        return {}
    p3d, reproj = triangulate_rows(cameras, rows, skeletons)
    points_store.set_triangulation(rows.index, p3d, reproj)
    errors = mean_reprojection_errors(rows, reproj, skeletons)
    bodyparts = rows.index.get_level_values("bodypart")
    return {str(bp): float(e) for bp, e in zip(bodyparts, errors)}


def triangulate_all_instances(
    cameras: dict[str, "CameraParams"],
    points_store: "PointsStore",
    trial: str,
    frame: int,
    skeletons: "ProjectSkeleton | None" = None,
) -> dict[str, float]:
    """Triangulate all instances in a frame.

    Returns flat dict mapping bodypart → mean error across all instances.
    """
    instances = points_store.get_instances(trial, frame)
    all_errors: dict[str, list[float]] = {}
    for inst in instances:
        errs = triangulate_frame(cameras, points_store, trial, frame, inst, skeletons)
        for bp, e in errs.items():
            all_errors.setdefault(bp, []).append(e)

    return {
        bp: float(np.nanmean(vs)) if vs else np.nan
        for bp, vs in all_errors.items()
    }

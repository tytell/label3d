"""Instance initialization strategies for pre-populating new instances.

Three strategies:

random     — scatter bodyparts around a center point using Gaussian noise;
             if skeleton.node_offsets_cm is defined, those cm offsets are
             projected through the camera at a nominal depth instead.

average    — compute mean relative positions from existing labeled instances
             in this trial+camera, place the new instance at center_px.
             Falls back to random when fewer than 2 qualifying instances exist.

reprojected — triangulate the selected instance from its existing cameras and
              project the 3D positions into the target camera.  Only fills
              bodyparts that are currently not_set; never overwrites set/predicted.
              Returns None if triangulation is impossible.

availability — available_strategies() checks which strategies are usable given
               the current state; used to enable/disable menu items.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import pandas as pd

from label3d.geometry import triangulate_point

if TYPE_CHECKING:
    from label3d.geometry import CameraParams
    from label3d.points import PointsStore
    from label3d.skeleton import Skeleton


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_center(frame_size: tuple[int, int]) -> tuple[float, float]:
    w, h = frame_size
    return (w / 2.0, h / 2.0)


def _pixel_to_world_at_depth(
    cam: CameraParams,
    px: float,
    py: float,
    depth: float,
) -> np.ndarray:
    """Back-project a pixel to a world point at the given camera-space depth.

    Returns a (3,) world-space array.
    """
    pt = np.array([[[px, py]]], dtype=np.float64)
    # Normalised camera coords (removes K and distortion)
    pt_norm = cv2.undistortPoints(pt, cam.matrix, cam.dist)  # (1, 1, 2)
    ray_cam = np.array([pt_norm[0, 0, 0], pt_norm[0, 0, 1], 1.0]) * depth
    # Camera → world: R.T @ (p_cam − t)
    return cam.R.T @ (ray_cam - cam.tvec.ravel())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_random(
    skeleton: Skeleton,
    camera_name: str,
    cameras: dict[str, CameraParams] | None,
    center_px: tuple[float, float] | None,
    frame_size: tuple[int, int],
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float]]:
    """Return {bodypart: (x, y)} with random placement around center_px.

    If skeleton.node_offsets_cm covers all nodes and calibration is provided,
    each bodypart's cm offset is projected through the camera at a nominal depth
    (50 cm) with 10% jitter.  Otherwise uses an isotropic Gaussian with
    sigma = min(w, h) / 8.
    """
    if rng is None:
        rng = np.random.default_rng()

    w, h = frame_size
    cx, cy = center_px if center_px is not None else _frame_center(frame_size)

    result: dict[str, tuple[float, float]] = {}

    cam = cameras.get(camera_name) if cameras else None
    node_offsets: dict[str, list[float]] = getattr(skeleton, "node_offsets_cm", {})

    if (
        cam is not None
        and node_offsets
        and all(bp in node_offsets for bp in skeleton.nodes)
    ):
        try:
            nominal_depth = 50.0  # cm
            world_center = _pixel_to_world_at_depth(cam, cx, cy, nominal_depth)

            projected: list[np.ndarray] = []
            for bp in skeleton.nodes:
                offset = np.array(node_offsets[bp], dtype=float)
                world_pt = world_center + offset
                pix = cam.project(world_pt.reshape(1, 3))[0]
                projected.append(pix)

            pts = np.array(projected)  # (n, 2)
            span = max(
                float(pts[:, 0].max() - pts[:, 0].min()),
                float(pts[:, 1].max() - pts[:, 1].min()),
                1.0,
            )
            sigma = span * 0.1

            for bp, pix in zip(skeleton.nodes, projected):
                jitter = rng.normal(0.0, sigma, size=2)
                x = float(np.clip(pix[0] + jitter[0], 0, w))
                y = float(np.clip(pix[1] + jitter[1], 0, h))
                result[bp] = (x, y)
            return result
        except Exception:
            pass  # Fall through to Gaussian

    # Isotropic Gaussian fallback
    sigma = min(w, h) / 8.0
    for bp in skeleton.nodes:
        x = float(np.clip(rng.normal(cx, sigma), 0, w))
        y = float(np.clip(rng.normal(cy, sigma), 0, h))
        result[bp] = (x, y)

    return result


def init_average(
    points_store: PointsStore,
    experiment: str,
    trial: str,
    camera_name: str,
    skeleton: Skeleton,
    center_px: tuple[float, float] | None,
    frame_size: tuple[int, int],
) -> tuple[dict[str, tuple[float, float]], bool]:
    """Return (coords, used_fallback) for average-relative placement.

    Collects all instances in experiment+trial that have state set/predicted in
    ≥ half of the skeleton nodes for camera_name.  Computes per-bodypart offsets
    relative to each instance's centroid, then averages them.  The new instance
    is placed centred at center_px (or frame centre).

    used_fallback is True when fewer than 2 qualifying instances were found
    (random placement is used instead).
    """
    w, h = frame_size
    cx, cy = center_px if center_px is not None else _frame_center(frame_size)

    min_fraction = 0.5
    n_nodes = len(skeleton.nodes)
    offsets_per_bp: dict[str, list[tuple[float, float]]] = {
        bp: [] for bp in skeleton.nodes
    }
    qualifying_count = 0

    try:
        trial_df = points_store.df.loc[(experiment, trial)]
    except KeyError:
        trial_df = None

    if trial_df is not None and (camera_name, "state") in trial_df.columns:
        for (_frame, _inst), group in trial_df.groupby(level=["frame", "instance"]):
            cam_states = group[(camera_name, "state")]
            n_labeled = int(cam_states.isin(["set", "predicted"]).sum())
            if n_labeled < max(1, int(n_nodes * min_fraction)):
                continue

            # Centroid from labeled bodyparts only
            cam_x = pd.to_numeric(group[(camera_name, "x")], errors="coerce")
            cam_y = pd.to_numeric(group[(camera_name, "y")], errors="coerce")
            valid_mask = cam_states.isin(["set", "predicted"])
            xs = cam_x[valid_mask].dropna().values.astype(float)
            ys = cam_y[valid_mask].dropna().values.astype(float)
            if len(xs) == 0:
                continue

            qualifying_count += 1
            cent_x = float(xs.mean())
            cent_y = float(ys.mean())

            for bp in skeleton.nodes:
                if bp not in group.index:
                    continue
                row = group.loc[bp]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                state = row[(camera_name, "state")]
                bx = row[(camera_name, "x")]
                by = row[(camera_name, "y")]
                if state not in ("set", "predicted"):
                    continue
                try:
                    bx_f, by_f = float(bx), float(by)
                except (TypeError, ValueError):
                    continue
                if np.isnan(bx_f) or np.isnan(by_f):
                    continue
                offsets_per_bp[bp].append((bx_f - cent_x, by_f - cent_y))

    if qualifying_count < 2:
        # Random fallback
        fallback = init_random(skeleton, camera_name, None, center_px, frame_size)
        return fallback, True

    result: dict[str, tuple[float, float]] = {}
    for bp in skeleton.nodes:
        if offsets_per_bp[bp]:
            avg_dx = float(np.mean([o[0] for o in offsets_per_bp[bp]]))
            avg_dy = float(np.mean([o[1] for o in offsets_per_bp[bp]]))
        else:
            avg_dx, avg_dy = 0.0, 0.0
        x = float(np.clip(cx + avg_dx, 0, w))
        y = float(np.clip(cy + avg_dy, 0, h))
        result[bp] = (x, y)

    return result, False


def init_reprojected(
    points_store: PointsStore,
    experiment: str,
    trial: str,
    frame: int,
    instance: int,
    target_camera: str,
    cameras: dict[str, CameraParams],
) -> dict[str, tuple[float, float]] | None:
    """Triangulate instance from other cameras and project into target_camera.

    Returns {bodypart: (x, y)} for bodyparts that could be reprojected, or None
    if no bodypart had ≥2 source-camera observations.  Coordinates are clipped
    to the target camera frame bounds.
    """
    target_cam = cameras.get(target_camera)
    if target_cam is None:
        return None

    w, h = target_cam.size

    try:
        inst_df = points_store.df.loc[(experiment, trial, frame, instance)]
        bodyparts = inst_df.index.tolist()
    except KeyError:
        return None

    result: dict[str, tuple[float, float]] = {}

    for bp in bodyparts:
        picks_2d: dict[str, tuple[float, float]] = {}
        for cam_name, cam in cameras.items():
            if cam_name == target_camera:
                continue
            pt = points_store.get_point(trial, frame, instance, bp, cam_name)
            if pt.state in ("set", "predicted") and not np.isnan(pt.x):
                picks_2d[cam_name] = (pt.x, pt.y)

        if len(picks_2d) < 2:
            continue

        xyz = triangulate_point(cameras, picks_2d)
        if xyz is None:
            continue

        # Reject points behind the camera
        cam_pt = target_cam.R @ xyz + target_cam.tvec.ravel()
        if cam_pt[2] <= 0:
            continue

        pix = target_cam.project(xyz.reshape(1, 3))[0]
        px = float(np.clip(pix[0], 0, w))
        py = float(np.clip(pix[1], 0, h))
        result[bp] = (px, py)

    return result if result else None


def available_strategies(
    points_store: PointsStore,
    experiment: str,
    trial: str,
    frame: int,
    instance: int | None,
    camera_name: str,
) -> set[Literal["random", "average", "reprojected"]]:
    """Return the set of strategies currently usable.

    "random"      — always available.
    "average"     — available if any instance in this trial+camera has at least
                    one set/predicted point (init_average will fall back to random
                    with a status message if < 2 qualifying instances are found).
    "reprojected" — available if instance is not None and that instance has
                    set/predicted points in ≥2 cameras other than camera_name
                    in this frame.
    """
    result: set[Literal["random", "average", "reprojected"]] = {"random"}

    # Average: quick check — any labeled point in trial+camera
    try:
        trial_df = points_store.df.loc[(experiment, trial)]
        if (camera_name, "state") in trial_df.columns:
            if trial_df[(camera_name, "state")].isin(["set", "predicted"]).any():
                result.add("average")
    except KeyError:
        pass

    # Reprojected: selected instance has ≥2 other cameras with valid data
    if instance is not None and instance >= 0:
        try:
            inst_df = points_store.df.loc[(experiment, trial, frame, instance)]
            other_cam_count = 0
            for cam in points_store.cameras:
                if cam == camera_name:
                    continue
                if (cam, "state") not in inst_df.columns:
                    continue
                if inst_df[(cam, "state")].isin(["set", "predicted"]).any():
                    other_cam_count += 1
            if other_cam_count >= 2:
                result.add("reprojected")
        except KeyError:
            pass

    return result

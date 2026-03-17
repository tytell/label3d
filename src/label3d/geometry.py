"""Epipolar geometry: fundamental matrix and epipolar line computation.

Uses calibration data (K, R, t) from aniposelib-format calibration.toml.
Coordinates are raw pixels; undistortion is applied on the fly when needed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import toml


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

"""Projection: 3D world points to 2D pixel coordinates via camera model."""

import cv2
import numpy as np


def project_points(
    points_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    """Project Nx3 world points to Nx2 pixel coordinates."""
    rvec, _ = cv2.Rodrigues(R)
    pts = points_3d.reshape(-1, 1, 3).astype(np.float64)
    projected, _ = cv2.projectPoints(pts, rvec, t, K, dist)
    return projected.reshape(-1, 2)


def project_all_frames(
    all_pts_3d: list[np.ndarray], cameras: dict
) -> dict[str, list[np.ndarray]]:
    """Project every frame into every camera.

    Returns dict[cam_name → list of (n_nodes, 2) arrays].
    """
    projected = {}
    for cam_name, cam in cameras.items():
        cam_proj = []
        for pts_3d in all_pts_3d:
            px = project_points(pts_3d, cam["K"], cam["R"], cam["t"], cam["dist"])
            cam_proj.append(px)
        projected[cam_name] = cam_proj
    return projected

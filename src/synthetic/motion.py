"""Rigid-body motion: rotation matrices, body poses, world point computation."""

import numpy as np


def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def compute_body_poses(
    motion_cfg: dict, num_frames: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (position, rotation_matrix) for each frame.

    Linear interpolation of translation and rotation from frame 0 to N-1.
    Rotation order: Rx then Ry then Rz (body-fixed pitch, yaw, roll).
    """
    start = np.array(motion_cfg["start_position"], dtype=np.float64)
    total_t = np.array(motion_cfg["total_translation"], dtype=np.float64)
    total_r = np.array(motion_cfg["total_rotation_deg"], dtype=np.float64)

    poses = []
    for i in range(num_frames):
        frac = i / max(num_frames - 1, 1)
        pos = start + frac * total_t
        angles = np.radians(frac * total_r)
        R = _Rz(angles[2]) @ _Ry(angles[1]) @ _Rx(angles[0])
        poses.append((pos, R))
    return poses


def compute_world_points(
    poses: list[tuple[np.ndarray, np.ndarray]],
    node_names: list[str],
    node_offsets: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """Return per-frame 3D world positions: list of (n_nodes, 3) arrays."""
    all_pts = []
    for pos, R_body in poses:
        pts = np.array([pos + R_body @ node_offsets[n] for n in node_names])
        all_pts.append(pts)
    return all_pts

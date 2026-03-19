"""Camera geometry: look-at, intrinsics, and camera construction from config."""

import cv2
import numpy as np
import toml


def load_config(path: str) -> dict:
    """Load TOML configuration file."""
    with open(path) as f:
        return toml.load(f)


def look_at(
    position: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-to-camera rotation matrix R and translation t.

    OpenCV camera convention: X right, Y down, Z forward.
    Returns R (3×3) and t (3,) such that p_cam = R @ p_world + t.
    """
    pos = np.asarray(position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)

    # Camera axes in world coordinates
    z_cam = tgt - pos  # forward (into scene)
    z_cam /= np.linalg.norm(z_cam)

    x_cam = np.cross(z_cam, up)  # right
    norm = np.linalg.norm(x_cam)
    if norm < 1e-8:
        # z_cam is parallel to world_up — use world +x as fallback up
        fallback_up = np.array([1.0, 0.0, 0.0])
        x_cam = np.cross(z_cam, fallback_up)
        norm = np.linalg.norm(x_cam)
    x_cam /= norm

    y_cam = np.cross(z_cam, x_cam)  # down (Z × X = Y in right-handed)

    R = np.array([x_cam, y_cam, z_cam])  # rows are camera axes
    t = -R @ pos

    return R, t


def compute_intrinsics(
    position: list,
    target: list,
    resolution: list[int],
    fov_at_target: float,
) -> np.ndarray:
    """Compute 3×3 intrinsic matrix K from field-of-view specification.

    fov_at_target is the horizontal extent (cm) visible at the target point.
    Square pixels: fx = fy.
    """
    distance = np.linalg.norm(np.array(target) - np.array(position))
    half_width = fov_at_target / 2.0
    tan_half = half_width / distance
    f = (resolution[0] / 2.0) / tan_half  # focal length in pixels

    cx = resolution[0] / 2.0
    cy = resolution[1] / 2.0

    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def build_cameras(config: dict) -> dict:
    """Build camera parameters from config.  Returns dict[cam_name, cam_dict]."""
    world_up = np.array(config.get("world", {}).get("up", [0, 0, 1]), dtype=np.float64)

    cameras = {}
    for cam_name, cam_cfg in config["cameras"].items():
        cam_up = np.array(cam_cfg.get("up", world_up), dtype=np.float64)
        R, t = look_at(cam_cfg["position"], cam_cfg["target"], cam_up)
        K = compute_intrinsics(
            cam_cfg["position"],
            cam_cfg["target"],
            cam_cfg["resolution"],
            cam_cfg["fov_at_target"],
        )
        rvec, _ = cv2.Rodrigues(R)
        dist = np.array(cam_cfg.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)

        cameras[cam_name] = {
            "name": cam_name,
            "K": K,
            "R": R,
            "t": t,
            "rvec": rvec.ravel(),
            "dist": dist,
            "resolution": cam_cfg["resolution"],
            "position": np.array(cam_cfg["position"], dtype=np.float64),
        }
    return cameras

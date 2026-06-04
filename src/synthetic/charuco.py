"""Synthetic Charuco board image generation for calibration testing.

Renders a Charuco board at multiple poses as seen by each camera, including
lens distortion. Output images can be fed into aniposelib's calibration
pipeline to verify end-to-end correctness.
"""

from pathlib import Path

import cv2
import numpy as np

from synthetic.projection import project_points


# ArUco dictionary mapping: (marker_bits, dict_size) → OpenCV constant
_ARUCO_DICTS = {
    (4, 50): cv2.aruco.DICT_4X4_50,
    (4, 100): cv2.aruco.DICT_4X4_100,
    (4, 250): cv2.aruco.DICT_4X4_250,
    (4, 1000): cv2.aruco.DICT_4X4_1000,
    (5, 50): cv2.aruco.DICT_5X5_50,
    (5, 100): cv2.aruco.DICT_5X5_100,
    (5, 250): cv2.aruco.DICT_5X5_250,
    (5, 1000): cv2.aruco.DICT_5X5_1000,
    (6, 50): cv2.aruco.DICT_6X6_50,
    (6, 100): cv2.aruco.DICT_6X6_100,
    (6, 250): cv2.aruco.DICT_6X6_250,
    (6, 1000): cv2.aruco.DICT_6X6_1000,
    (7, 50): cv2.aruco.DICT_7X7_50,
    (7, 100): cv2.aruco.DICT_7X7_100,
    (7, 250): cv2.aruco.DICT_7X7_250,
    (7, 1000): cv2.aruco.DICT_7X7_1000,
}


def _make_charuco_board(charuco_cfg: dict) -> tuple[cv2.aruco.CharucoBoard, np.ndarray]:
    """Create a CharucoBoard and render its flat image.

    Returns (board, board_image_bgr).
    """
    squares_x = charuco_cfg["squares_x"]
    squares_y = charuco_cfg["squares_y"]
    square_length = charuco_cfg["square_length"]
    marker_length = charuco_cfg["marker_length"]
    marker_bits = charuco_cfg.get("marker_bits", 4)
    dict_size = charuco_cfg.get("dict_size", 50)
    img_res = charuco_cfg.get("board_image_resolution", 2000)

    key = (marker_bits, dict_size)
    if key not in _ARUCO_DICTS:
        raise ValueError(
            f"Unsupported ArUco dict: marker_bits={marker_bits}, dict_size={dict_size}. "
            f"Supported: {sorted(_ARUCO_DICTS.keys())}"
        )
    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICTS[key])

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, aruco_dict
    )

    # Render at high resolution, maintaining aspect ratio
    aspect = squares_x / squares_y
    if aspect >= 1:
        img_w, img_h = img_res, int(img_res / aspect)
    else:
        img_w, img_h = int(img_res * aspect), img_res

    board_img = board.generateImage((img_w, img_h), marginSize=10, borderBits=1)

    # Convert to BGR if grayscale
    if len(board_img.shape) == 2:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    return board, board_img


def _compute_board_poses(
    charuco_cfg: dict, cameras: dict, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute random board poses (position, R_board) in world frame.

    R_board transforms from board-local to world coordinates.
    Board local frame: X along width, Y along height, Z is outward normal.
    The board normal is oriented to roughly face the cameras, with random tilt.
    """
    placement = charuco_cfg["placement"]
    center = np.array(placement["center"], dtype=np.float64)
    extent = np.array(placement["extent"], dtype=np.float64)
    tilt_range = np.radians(placement["tilt_range_deg"])
    num_images = charuco_cfg["num_images"]

    # Mean camera position — board normal will roughly point toward this
    cam_positions = np.array([cam["position"] for cam in cameras.values()])
    mean_cam_pos = np.mean(cam_positions, axis=0)

    poses = []
    for _ in range(num_images):
        # Random position within placement volume
        pos = center + rng.uniform(-1, 1, 3) * extent

        # Board normal should face toward mean camera position
        normal = mean_cam_pos - pos
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / norm_len

        # Build rotation matrix: board Z → normal direction
        # Choose a "right" vector not parallel to normal
        if abs(normal[2]) < 0.9:
            up_hint = np.array([0.0, 0.0, 1.0])
        else:
            up_hint = np.array([0.0, 1.0, 0.0])

        x_axis = np.cross(up_hint, normal)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # Base rotation: board-local (X, Y, Z) → world (x_axis, y_axis, normal)
        R_base = np.column_stack([x_axis, y_axis, normal])

        # Apply random tilt around the board's X and Y axes
        tilt_x = rng.uniform(-tilt_range, tilt_range)
        tilt_y = rng.uniform(-tilt_range, tilt_range)
        cx, sx = np.cos(tilt_x), np.sin(tilt_x)
        cy, sy = np.cos(tilt_y), np.sin(tilt_y)
        R_tilt_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        R_tilt_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R_tilt = R_tilt_y @ R_tilt_x

        R_board = R_base @ R_tilt
        poses.append((pos, R_board))

    return poses


def _render_board_in_camera(
    board_img: np.ndarray,
    board_corners_3d: np.ndarray,
    cam: dict,
    bg_color: list[int],
) -> np.ndarray:
    """Render a Charuco board image as seen by a camera, with lens distortion.

    Uses ray-plane intersection (reverse mapping) so that distortion is
    correctly applied — the board edges appear curved as they would in a
    real distorted image.

    board_corners_3d: (4, 3) — the four board corners in world frame,
        ordered: top-left, top-right, bottom-right, bottom-left.
    """
    w, h = cam["resolution"]
    K, R, t, dist = cam["K"], cam["R"], cam["t"], cam["dist"]
    bh, bw = board_img.shape[:2]

    # Board plane geometry from the 4 corners
    origin = board_corners_3d[0]  # top-left
    x_edge = board_corners_3d[1] - board_corners_3d[0]  # top-left → top-right
    y_edge = board_corners_3d[3] - board_corners_3d[0]  # top-left → bottom-left
    board_width = np.linalg.norm(x_edge)
    board_height = np.linalg.norm(y_edge)
    x_hat = x_edge / board_width
    y_hat = y_edge / board_height
    normal = np.cross(x_hat, y_hat)

    # Camera center in world: C = -R^T @ t
    C = -R.T @ t

    # Generate all pixel coordinates
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([u_coords.ravel(), v_coords.ravel()], axis=1).astype(np.float64)

    # Undistort to get normalized camera coordinates (removes lens distortion)
    undistorted = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, dist)
    # Result is in normalized coords (x/z, y/z); add z=1 to get ray direction in camera frame
    rays_cam = np.hstack([undistorted.reshape(-1, 2), np.ones((len(pixels), 1))])

    # Transform ray directions to world frame
    rays_world = (R.T @ rays_cam.T).T  # (N, 3)

    # Ray-plane intersection: P = C + lambda * d, (P - origin) . normal = 0
    denom = rays_world @ normal  # (N,)
    numer = (origin - C) @ normal  # scalar

    valid = np.abs(denom) > 1e-10
    lambdas = np.full(len(pixels), np.inf)
    lambdas[valid] = numer / denom[valid]

    # Only keep intersections in front of camera
    valid &= lambdas > 0

    # Intersection points in world frame
    intersections = C + lambdas[:, np.newaxis] * rays_world

    # Convert to board-local coordinates
    rel = intersections - origin
    u_board = rel @ x_hat
    v_board = rel @ y_hat

    # Normalize to [0, 1] in board dimensions
    u_norm = u_board / board_width
    v_norm = v_board / board_height

    # Check if within board bounds
    valid &= (u_norm >= 0) & (u_norm < 1) & (v_norm >= 0) & (v_norm < 1)

    # Build remap arrays (float coords for bilinear interpolation)
    # Invalid pixels get -1, which cv2.remap treats as border
    # Flip U: the board image is drawn from the viewer's perspective, but the
    # board's local X axis points the opposite way when seen by the camera
    # (they face each other), so we mirror horizontally.
    map_x = np.full(h * w, -1.0, dtype=np.float32)
    map_y = np.full(h * w, -1.0, dtype=np.float32)
    map_x[valid] = ((1.0 - u_norm[valid]) * (bw - 1)).astype(np.float32)
    map_y[valid] = (v_norm[valid] * (bh - 1)).astype(np.float32)

    map_x = map_x.reshape(h, w)
    map_y = map_y.reshape(h, w)

    bg_bgr = (bg_color[2], bg_color[1], bg_color[0])
    output = cv2.remap(
        board_img, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=bg_bgr,
    )

    return output


def generate_charuco_images(
    charuco_cfg: dict,
    cameras: dict,
    output_dir: Path,
    experiment_name: str = "",
    calib_keyword: str = "calib",
):
    """Generate synthetic Charuco board calibration images for all cameras.

    Images are written to output_dir/calibration/{cam_name}/ with filenames
    {calib_keyword}_{experiment_name}_{cam_name}_board_NNN.png, matching the
    flat folder naming convention so they can be copied without renaming.

    Board dimensions in the config (square_length, marker_length) are in mm;
    they are converted to cm for 3D placement since the project uses cm throughout.
    """
    board, board_img = _make_charuco_board(charuco_cfg)

    seed = charuco_cfg.get("placement", {}).get("seed", 91571)
    rng = np.random.default_rng(seed)
    bg_color = charuco_cfg.get("background_color", [40, 40, 40])

    # Board physical size in cm (config values are in mm)
    square_length_cm = charuco_cfg["square_length"] / 10.0
    board_width_cm = charuco_cfg["squares_x"] * square_length_cm
    board_height_cm = charuco_cfg["squares_y"] * square_length_cm

    poses = _compute_board_poses(charuco_cfg, cameras, rng)

    cal_dir = output_dir / "calibration"

    prefix_parts = [p for p in (calib_keyword, experiment_name) if p]
    prefix = "_".join(prefix_parts)

    for cam_name, cam in cameras.items():
        cam_dir = cal_dir / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)

        for pi, (pos, R_board) in enumerate(poses):
            # Board corners in world frame (centered on pos)
            # Local corners: centered, X along width, Y along height
            half_w, half_h = board_width_cm / 2, board_height_cm / 2
            local_corners = np.array([
                [-half_w, -half_h, 0],  # top-left
                [ half_w, -half_h, 0],  # top-right
                [ half_w,  half_h, 0],  # bottom-right
                [-half_w,  half_h, 0],  # bottom-left
            ], dtype=np.float64)

            # Transform to world
            corners_3d = (R_board @ local_corners.T).T + pos

            # Check that at least some of the board is visible
            corners_2d = project_points(corners_3d, cam["K"], cam["R"], cam["t"], cam["dist"])
            w, h = cam["resolution"]
            visible = np.any(
                (corners_2d[:, 0] >= -w * 0.5)
                & (corners_2d[:, 0] < w * 1.5)
                & (corners_2d[:, 1] >= -h * 0.5)
                & (corners_2d[:, 1] < h * 1.5)
            )
            if not visible:
                print(f"  WARNING: board pose {pi} fully outside {cam_name}, rendering anyway")

            img = _render_board_in_camera(board_img, corners_3d, cam, bg_color)
            img_path = cam_dir / f"{prefix}_{cam_name}_board_{pi:03d}.png"
            cv2.imwrite(str(img_path), img)

        print(f"  {cam_dir}/ ({len(poses)} images)")

    # Run detection verification if aniposelib is available
    _verify_charuco_detection(charuco_cfg, cameras, cal_dir)


def _verify_charuco_detection(charuco_cfg: dict, cameras: dict, cal_dir: Path):
    """Verify that generated images can be detected by OpenCV's Charuco detector."""
    print("\nCharuco detection verification...")

    board, _ = _make_charuco_board(charuco_cfg)
    num_images = charuco_cfg["num_images"]
    detector = cv2.aruco.CharucoDetector(board)
    expected_corners = (charuco_cfg["squares_x"] - 1) * (charuco_cfg["squares_y"] - 1)

    total_detected = 0
    total_images = 0

    for cam_name in cameras:
        cam_dir = cal_dir / cam_name
        cam_detected = 0
        img_paths = sorted(cam_dir.glob("*.png")) if cam_dir.is_dir() else []
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            total_images += 1

            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

            if charuco_corners is not None and len(charuco_corners) > 0:
                cam_detected += 1
                total_detected += 1

        ratio = cam_detected / num_images if num_images > 0 else 0
        status = "✓" if ratio > 0.5 else "✗"
        print(f"  {cam_name}: {cam_detected}/{num_images} images detected ({ratio:.0%})  {status}")

    if total_images > 0:
        overall = total_detected / total_images
        print(f"  Overall: {total_detected}/{total_images} ({overall:.0%})")

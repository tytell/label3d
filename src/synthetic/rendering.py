"""Video rendering: frame composition and MP4 writing."""

from pathlib import Path

import cv2
import numpy as np

from synthetic.projection import project_points


def _rgb_to_bgr(rgb: list[int]) -> tuple[int, ...]:
    return (rgb[2], rgb[1], rgb[0])


def render_frame(
    resolution: list[int],
    projected: np.ndarray,
    node_names: list[str],
    node_colors: dict[str, list[int]],
    edges: list[tuple[str, str]],
    point_radius: int,
    bg_color: list[int],
    frame_idx: int,
    cam_name: str,
    axes_2d: np.ndarray | None = None,
) -> np.ndarray:
    """Render a single video frame with skeleton dots, edges, and labels."""
    w, h = resolution
    img = np.full((h, w, 3), _rgb_to_bgr(bg_color), dtype=np.uint8)

    # Draw world axes if provided (origin + 3 axis endpoints, projected)
    if axes_2d is not None:
        origin = tuple(axes_2d[0].astype(int))
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X=red, Y=green, Z=blue
        axis_labels = ["X", "Y", "Z"]
        for k in range(3):
            tip = tuple(axes_2d[1 + k].astype(int))
            cv2.line(img, origin, tip, axis_colors[k], 2, cv2.LINE_AA)
            cv2.putText(
                img, axis_labels[k], (tip[0] + 4, tip[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_colors[k], 1, cv2.LINE_AA,
            )

    # Skeleton edges
    for src, dst in edges:
        si, di = node_names.index(src), node_names.index(dst)
        pt1 = tuple(projected[si].astype(int))
        pt2 = tuple(projected[di].astype(int))
        cv2.line(img, pt1, pt2, (128, 128, 128), 1, cv2.LINE_AA)

    # Nodes
    for i, name in enumerate(node_names):
        pt = tuple(projected[i].astype(int))
        color_bgr = _rgb_to_bgr(node_colors[name])
        cv2.circle(img, pt, point_radius, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(img, pt, point_radius, (255, 255, 255), 1, cv2.LINE_AA)  # outline
        cv2.putText(
            img, name, (pt[0] + point_radius + 3, pt[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

    # Frame counter and camera label
    cv2.putText(
        img, f"{cam_name}  frame {frame_idx}", (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    )

    return img


def write_videos(
    cameras: dict,
    all_projected: dict[str, list[np.ndarray]],
    node_names: list[str],
    node_colors: dict[str, list[int]],
    edges: list[tuple[str, str]],
    render_cfg: dict,
    trial_name: str,
    num_frames: int,
    video_dir: Path,
):
    """Render and write MP4 video files, one per camera."""
    point_radius = render_cfg.get("point_radius", 8)
    bg_color = render_cfg.get("background_color", [40, 40, 40])
    codec = render_cfg.get("codec", "mp4v")
    fps = render_cfg.get("fps", 30)
    draw_axes = render_cfg.get("draw_axes", False)
    axes_len = render_cfg.get("axes_length", 5.0)

    # Pre-compute projected axes for each camera (origin + 3 axis tips)
    axes_3d = np.array([
        [0, 0, 0],
        [axes_len, 0, 0],
        [0, axes_len, 0],
        [0, 0, axes_len],
    ], dtype=np.float64)

    for cam_name, cam in cameras.items():
        res = cam["resolution"]
        video_path = video_dir / f"{cam_name}_{trial_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (res[0], res[1]))

        if draw_axes:
            axes_px = project_points(axes_3d, cam["K"], cam["R"], cam["t"], cam["dist"])
        else:
            axes_px = None

        for fi in range(num_frames):
            img = render_frame(
                res, all_projected[cam_name][fi],
                node_names, node_colors, edges,
                point_radius, bg_color, fi, cam_name,
                axes_2d=axes_px,
            )
            writer.write(img)

        writer.release()
        print(f"  {video_path}")

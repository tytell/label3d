#!/usr/bin/env python3
"""Generate synthetic test data for label3d.

Usage:
    python generate_synthetic.py synthetic_config.toml

Generates:
    <output_dir>/
    ├── videos/
    │   ├── trials.csv
    │   ├── cam0_trial1.mp4
    │   └── cam1_trial1.mp4
    ├── calibration.toml
    ├── points.csv
    ├── skeleton.json
    └── slp/                     (if sleap_io is installed)
        ├── cam0_trial1.slp
        └── cam1_trial1.slp

Dependencies: numpy, opencv-python, pandas, toml
Optional:     sleap-io (for .slp file generation)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import toml


# ── Config loading ───────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    """Load TOML configuration file."""
    with open(path) as f:
        return toml.load(f)


# ── Camera geometry ──────────────────────────────────────────────────────────


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
    x_cam /= np.linalg.norm(x_cam)

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
        R, t = look_at(cam_cfg["position"], cam_cfg["target"], world_up)
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


# ── Rigid-body motion ────────────────────────────────────────────────────────


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


# ── Projection ───────────────────────────────────────────────────────────────


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


# ── Video rendering ──────────────────────────────────────────────────────────


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


# ── Output writers ───────────────────────────────────────────────────────────


def write_calibration_toml(cameras: dict, output_path: Path):
    """Write aniposelib-compatible calibration.toml."""
    d = {}
    for i, (cam_name, cam) in enumerate(cameras.items()):
        d[f"cam_{i}"] = {
            "name": cam_name,
            "size": list(cam["resolution"]),
            "matrix": cam["K"].tolist(),
            "distortions": cam["dist"].tolist(),
            "rotation": cam["rvec"].tolist(),
            "translation": cam["t"].tolist(),
        }
    d["metadata"] = {"adjusted": False}

    with open(output_path, "w") as f:
        toml.dump(d, f)


def write_trials_csv(
    cameras: dict, trial_name: str, num_frames: int, video_dir: Path
):
    """Write trials.csv."""
    row = {"trial": trial_name}
    for cam_name in cameras:
        row[cam_name] = f"{cam_name}_{trial_name}.mp4"
        row[f"{cam_name}_start"] = 0
        row[f"{cam_name}_end"] = num_frames - 1

    pd.DataFrame([row]).to_csv(video_dir / "trials.csv", index=False)


def write_points_csv(
    all_pts_3d: list[np.ndarray],
    all_projected: dict[str, list[np.ndarray]],
    cameras: dict,
    node_names: list[str],
    trial_name: str,
    num_frames: int,
    output_path: Path,
):
    """Write ground-truth points.csv with zero reprojection error."""
    cam_names = list(cameras.keys())

    # Column MultiIndex
    col_tuples = []
    for cam_name in cam_names:
        for coord in ("x", "y", "xr", "yr", "state"):
            col_tuples.append((cam_name, coord))
    for coord in ("x", "y", "z"):
        col_tuples.append(("3d", coord))
    col_idx = pd.MultiIndex.from_tuples(col_tuples)

    # Build rows
    rows = []
    idx_tuples = []
    for fi in range(num_frames):
        for bi, bp in enumerate(node_names):
            idx_tuples.append((trial_name, fi, 0, bp))
            row = {}
            for cam_name in cam_names:
                px, py = all_projected[cam_name][fi][bi]
                # Synthetic: clicked == reprojected (zero error)
                row[(cam_name, "x")] = px
                row[(cam_name, "y")] = py
                row[(cam_name, "xr")] = px
                row[(cam_name, "yr")] = py
                row[(cam_name, "state")] = "set"
            p3 = all_pts_3d[fi][bi]
            row[("3d", "x")] = p3[0]
            row[("3d", "y")] = p3[1]
            row[("3d", "z")] = p3[2]
            rows.append(row)

    row_idx = pd.MultiIndex.from_tuples(
        idx_tuples, names=["trial", "frame", "instance", "bodypart"]
    )
    df = pd.DataFrame(rows, index=row_idx, columns=col_idx)
    df.to_csv(output_path)


def write_skeleton_json(
    node_names: list[str],
    edges: list[tuple[str, str]],
    node_offsets: dict[str, np.ndarray],
    output_path: Path,
):
    """Write skeleton.json (project metadata)."""
    data = {
        "nodes": node_names,
        "edges": [list(e) for e in edges],
        "symmetries": [],
        "node_offsets_cm": {n: o.tolist() for n, o in node_offsets.items()},
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {output_path}")


def write_slp_files(
    all_projected: dict[str, list[np.ndarray]],
    cameras: dict,
    node_names: list[str],
    edges: list[tuple[str, str]],
    trial_name: str,
    num_frames: int,
    video_dir: Path,
    output_dir: Path,
):
    """Write per-camera .slp files using sleap_io.  Skipped if not installed."""
    try:
        import sleap_io as sio
    except ImportError:
        print("  sleap_io not installed — skipping .slp generation")
        return

    # Build skeleton
    skeleton = sio.Skeleton(name="synthetic")
    for name in node_names:
        skeleton.add_node(name)
    for src, dst in edges:
        skeleton.add_edge(src, dst)

    output_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in cameras:
        video_path = (video_dir / f"{cam_name}_{trial_name}.mp4").resolve()
        video = sio.Video(filename=str(video_path))

        labeled_frames = []
        for fi in range(num_frames):
            # points array: (n_nodes, 2), NaN for missing
            pts_array = all_projected[cam_name][fi].astype(np.float32)

            inst = sio.PredictedInstance.from_numpy(
                pts_array, skeleton=skeleton, score=1.0
            )
            lf = sio.LabeledFrame(
                video=video, frame_idx=fi, instances=[inst]
            )
            labeled_frames.append(lf)

        labels = sio.Labels(labeled_frames=labeled_frames)
        slp_path = output_dir / f"{cam_name}_{trial_name}.slp"
        sio.save_file(labels, str(slp_path))
        print(f"  {slp_path}")


# ── Sanity checks ────────────────────────────────────────────────────────────


def _dlt_triangulate(cameras: dict, pixel_coords: np.ndarray) -> np.ndarray:
    """DLT triangulation from N cameras.  pixel_coords: (n_cams, 2)."""
    A = []
    for ci, cam in enumerate(cameras.values()):
        P = cam["K"] @ np.hstack([cam["R"], cam["t"].reshape(3, 1)])  # 3×4
        u, v = pixel_coords[ci]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def _epipolar_error(cam_a: dict, cam_b: dict, pt_a: np.ndarray, pt_b: np.ndarray) -> float:
    """Symmetric epipolar distance (pixels) between corresponding points."""
    Pa = cam_a["K"] @ np.hstack([cam_a["R"], cam_a["t"].reshape(3, 1)])
    Pb = cam_b["K"] @ np.hstack([cam_b["R"], cam_b["t"].reshape(3, 1)])

    # Essential-style: compute F from camera matrices
    # F = K_b^{-T} [t]x R K_a^{-1}  — but simpler to use the 8-pt formula:
    # Use the projection centers to compute F directly.
    # Center of camera A in world: C_a = -R_a^T t_a
    C_a = -cam_a["R"].T @ cam_a["t"]

    # Epipole e_b = project C_a into cam_b
    e_b = cam_b["K"] @ (cam_b["R"] @ C_a + cam_b["t"])
    e_b /= e_b[2]

    # F = [e_b]_x P_b P_a^+   (simplified: use cross-product matrix)
    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # Compute F via the formula F = [e_b]× Pb Pa†
    Pa_pinv = np.linalg.pinv(Pa)
    F = skew(e_b) @ Pb @ Pa_pinv

    # Epipolar line in cam_b for point pt_a: l_b = F x_a
    x_a = np.array([pt_a[0], pt_a[1], 1.0])
    x_b = np.array([pt_b[0], pt_b[1], 1.0])

    l_b = F @ x_a
    # Distance from pt_b to line l_b: |x_b . l_b| / sqrt(l[0]^2 + l[1]^2)
    dist_b = abs(x_b @ l_b) / np.sqrt(l_b[0] ** 2 + l_b[1] ** 2)

    # Symmetric: also compute l_a = F^T x_b, distance from pt_a to l_a
    l_a = F.T @ x_b
    dist_a = abs(x_a @ l_a) / np.sqrt(l_a[0] ** 2 + l_a[1] ** 2)

    return float((dist_a + dist_b) / 2)


def _run_sanity_checks(
    cameras: dict,
    all_pts_3d: list[np.ndarray],
    all_projected: dict[str, list[np.ndarray]],
    node_names: list[str],
    output_dir: Path,
):
    cam_list = list(cameras.values())
    cam_names = list(cameras.keys())
    n_cams = len(cam_list)
    n_frames = len(all_pts_3d)

    # 1. Triangulation: recover ground-truth 3D from projected pixels
    print("\nTriangulation sanity check (all frames)...")
    max_tri_err = 0.0
    for fi in range(n_frames):
        pixel_coords = np.array([all_projected[cn][fi] for cn in cam_names])
        for bi, bp in enumerate(node_names):
            X = _dlt_triangulate(cameras, pixel_coords[:, bi, :])
            gt = all_pts_3d[fi][bi]
            err = float(np.linalg.norm(X - gt))
            max_tri_err = max(max_tri_err, err)
            if err > 1e-4:
                print(f"  WARNING frame {fi} {bp}: triangulation error {err:.6f} cm")
    print(f"  Max triangulation error across all frames/bodyparts: {max_tri_err:.2e} cm  {'✓' if max_tri_err < 1e-4 else '✗'}")

    # 2. Epipolar geometry: all point pairs should lie on each other's epipolar lines
    if n_cams >= 2:
        print("\nEpipolar geometry check (frame 0, all camera pairs)...")
        max_epi_err = 0.0
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                for bi, bp in enumerate(node_names):
                    pt_i = all_projected[cam_names[i]][0][bi]
                    pt_j = all_projected[cam_names[j]][0][bi]
                    err = _epipolar_error(cam_list[i], cam_list[j], pt_i, pt_j)
                    max_epi_err = max(max_epi_err, err)
                    if err > 1e-4:
                        print(f"  WARNING {cam_names[i]}-{cam_names[j]} {bp}: epipolar dist {err:.4f} px")
        print(f"  Max symmetric epipolar distance: {max_epi_err:.2e} px  {'✓' if max_epi_err < 1e-4 else '✗'}")

    # 3. Reprojection: project GT 3D back and compare to stored 2D
    print("\nReprojection sanity check...")
    max_reproj_err = 0.0
    for cam_name, cam in cameras.items():
        for fi in range(n_frames):
            reproj = project_points(all_pts_3d[fi], cam["K"], cam["R"], cam["t"], cam["dist"])
            stored = all_projected[cam_name][fi]
            err = float(np.max(np.linalg.norm(reproj - stored, axis=1)))
            max_reproj_err = max(max_reproj_err, err)
    print(f"  Max reprojection error: {max_reproj_err:.2e} px  {'✓' if max_reproj_err < 1e-6 else '✗'}")

    # 4. Aniposelib round-trip (if installed; also handles local checkout)
    try:
        _repo = Path(__file__).parent / "aniposelib"
        if _repo.exists():
            import sys as _sys
            if str(_repo) not in _sys.path:
                _sys.path.insert(0, str(_repo))
        from aniposelib.cameras import CameraGroup  # noqa: PLC0415
        print("\nAniposelib round-trip check...")
        cgroup = CameraGroup.load(str(output_dir / "calibration.toml"))
        for i, (cam_name, cam) in enumerate(cameras.items()):
            loaded_K = cgroup.cameras[i].get_camera_matrix()
            err_K = float(np.max(np.abs(loaded_K - cam["K"])))
            loaded_rvec = cgroup.cameras[i].get_rotation()
            err_rvec = float(np.max(np.abs(loaded_rvec - cam["rvec"])))
            status = "✓" if err_K < 1e-6 and err_rvec < 1e-6 else "✗"
            print(f"  {cam_name}: K_err={err_K:.2e}, rvec_err={err_rvec:.2e}  {status}")
    except ImportError:
        pass  # aniposelib not installed, skip


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic label3d test data from a TOML config"
    )
    parser.add_argument("config", help="Path to TOML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Unpack config ────────────────────────────────────────────────────
    project_cfg = config["project"]
    output_dir = Path(project_cfg["output_dir"])
    trial_name = project_cfg["trial_name"]
    num_frames = project_cfg["num_frames"]

    skel_cfg = config["skeleton"]
    node_names = list(skel_cfg["nodes"].keys())
    node_offsets = {
        n: np.array(cfg["offset"], dtype=np.float64)
        for n, cfg in skel_cfg["nodes"].items()
    }
    node_colors = {n: cfg["color"] for n, cfg in skel_cfg["nodes"].items()}
    edges = [tuple(e) for e in skel_cfg["edges"]]

    render_cfg = config.get("rendering", {})

    # ── Build cameras ────────────────────────────────────────────────────
    cameras = build_cameras(config)

    # ── Compute motion ───────────────────────────────────────────────────
    poses = compute_body_poses(config["motion"], num_frames)
    all_pts_3d = compute_world_points(poses, node_names, node_offsets)

    # ── Project into each camera ─────────────────────────────────────────
    all_projected = project_all_frames(all_pts_3d, cameras)

    # ── Verify all points are within image bounds ────────────────────────
    for cam_name, cam in cameras.items():
        w, h = cam["resolution"]
        for fi, proj in enumerate(all_projected[cam_name]):
            for bi, (px, py) in enumerate(proj):
                if not (0 <= px < w and 0 <= py < h):
                    print(
                        f"  WARNING: {node_names[bi]} in {cam_name} frame {fi} "
                        f"is outside image at ({px:.1f}, {py:.1f})"
                    )

    # ── Create output directories ────────────────────────────────────────
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # ── Write outputs ────────────────────────────────────────────────────
    print("Writing videos...")
    write_videos(
        cameras, all_projected,
        node_names, node_colors, edges,
        render_cfg, trial_name, num_frames, video_dir,
    )

    print("Writing calibration.toml...")
    write_calibration_toml(cameras, output_dir / "calibration.toml")
    print(f"  {output_dir / 'calibration.toml'}")

    print("Writing trials.csv...")
    write_trials_csv(cameras, trial_name, num_frames, video_dir)
    print(f"  {video_dir / 'trials.csv'}")

    print("Writing points.csv...")
    write_points_csv(
        all_pts_3d, all_projected, cameras,
        node_names, trial_name, num_frames,
        output_dir / "points.csv",
    )
    print(f"  {output_dir / 'points.csv'}")

    print("Writing skeleton.json...")
    write_skeleton_json(node_names, edges, node_offsets, output_dir / "skeleton.json")

    print("Writing .slp files...")
    write_slp_files(
        all_projected, cameras, node_names, edges,
        trial_name, num_frames, video_dir, output_dir / "slp",
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nDone. Output: {output_dir}/")
    print(f"  Cameras:   {list(cameras.keys())}")
    print(f"  Frames:    {num_frames}")
    print(f"  Bodyparts: {node_names}")
    for cam_name, cam in cameras.items():
        dist_cm = np.linalg.norm(cam["position"] - np.array(config["cameras"][cam_name]["target"]))
        print(
            f"  {cam_name}: f={cam['K'][0,0]:.1f}px, "
            f"distance={dist_cm:.1f}cm, "
            f"hFOV at target={config['cameras'][cam_name]['fov_at_target']}cm"
        )

    # ── Sanity checks ────────────────────────────────────────────────────
    _run_sanity_checks(cameras, all_pts_3d, all_projected, node_names, output_dir)


if __name__ == "__main__":
    main()

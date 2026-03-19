"""Output writers: calibration.toml, trials.csv, points.csv, skeleton.json, .slp files."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import toml


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

"""Output writers: calibration registry entry, trials.csv, points.csv, skeleton.json, .slp files."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from aniposelib.cameras import Camera, CameraGroup

from label3d.calibration_registry import CalibrationEntry, save_calibration


def write_calibration(cameras: dict, experiment_dir: Path) -> CalibrationEntry:
    """Save the ground-truth rig as an "imported" calibration registry entry."""
    cams = [
        Camera(
            name=cam_name,
            size=tuple(cam["resolution"]),
            matrix=cam["K"],
            dist=cam["dist"],
            rvec=cam["rvec"],
            tvec=cam["t"],
        )
        for cam_name, cam in cameras.items()
    ]
    return save_calibration(experiment_dir, CameraGroup(cams), "ground-truth", "imported")


def write_trials_csv(trial_rows: list[dict], output_path: Path):
    """Write global trials.csv from a list of pre-built row dicts."""
    pd.DataFrame(trial_rows).to_csv(output_path, index=False)


def write_points_csv(
    all_pts_3d: list[np.ndarray],
    all_projected: dict[str, list[np.ndarray]],
    cameras: dict,
    node_names: list[str],
    trial_name: str,
    num_frames: int,
    output_path: Path,
    experiment_name: str = "default",
):
    """Write ground-truth points.csv with zero reprojection error.

    Row index is 5-level: (experiment, trial, frame, instance, bodypart).
    """
    cam_names = list(cameras.keys())

    # Column MultiIndex
    col_tuples = []
    for cam_name in cam_names:
        for coord in ("x", "y", "xr", "yr", "state", "source", "score", "sync"):
            col_tuples.append((cam_name, coord))
    for coord in ("x", "y", "z"):
        col_tuples.append(("3d", coord))
    col_idx = pd.MultiIndex.from_tuples(col_tuples)

    # Build rows
    rows = []
    idx_tuples = []
    for fi in range(num_frames):
        for bi, bp in enumerate(node_names):
            idx_tuples.append((experiment_name, trial_name, fi, 0, bp))
            row = {}
            for cam_name in cam_names:
                px, py = all_projected[cam_name][fi][bi]
                # Synthetic: clicked == reprojected (zero error)
                row[(cam_name, "x")] = px
                row[(cam_name, "y")] = py
                row[(cam_name, "xr")] = px
                row[(cam_name, "yr")] = py
                row[(cam_name, "state")] = "set"
                row[(cam_name, "source")] = "label3d"
                row[(cam_name, "score")] = np.nan
                row[(cam_name, "sync")] = np.nan
            p3 = all_pts_3d[fi][bi]
            row[("3d", "x")] = p3[0]
            row[("3d", "y")] = p3[1]
            row[("3d", "z")] = p3[2]
            rows.append(row)

    row_idx = pd.MultiIndex.from_tuples(
        idx_tuples, names=["experiment", "trial", "frame", "instance", "bodypart"]
    )
    df = pd.DataFrame(rows, index=row_idx, columns=col_idx)
    df.to_csv(output_path)


def write_project_toml(
    output_path: Path,
    project_name: str,
    charuco: dict | None = None,
    calibrations: dict[str, str] | None = None,
):
    """Write project.toml at the project root.

    calibrations (experiment -> registry id) is both the active calibration
    and the one the true 3D points in points.csv correspond to.
    """
    data: dict = {"project": {"name": project_name}}
    if charuco:
        data["charuco"] = charuco
    if calibrations:
        data["calibration"] = {
            "active": dict(calibrations),
            "triangulated_with": dict(calibrations),
        }
    with open(output_path, "w") as f:
        toml.dump(data, f)
    print(f"  {output_path}")


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
    cam_trial_data: dict[str, list[tuple[Path, list[np.ndarray]]]],
    node_names: list[str],
    edges: list[tuple[str, str]],
    output_dir: Path,
):
    """Write one .slp per camera spanning all experiments and trials.

    cam_trial_data[cam_name] = [(video_path, [frame0_pts, frame1_pts, ...]), ...]
    Each video_path is the absolute path to the MP4; frame arrays are (n_nodes, 2).
    """
    try:
        import sleap_io as sio
    except ImportError:
        print("  sleap_io not installed — skipping .slp generation")
        return

    skeleton = sio.Skeleton(name="synthetic")
    for name in node_names:
        skeleton.add_node(name)
    for src, dst in edges:
        skeleton.add_edge(src, dst)

    output_dir.mkdir(parents=True, exist_ok=True)

    for cam_name, trials in cam_trial_data.items():
        labeled_frames = []
        for video_path, frame_arrays in trials:
            video = sio.Video(filename=str(video_path))
            for fi, pts_array in enumerate(frame_arrays):
                inst = sio.PredictedInstance.from_numpy(
                    pts_array.astype(np.float32), skeleton=skeleton, score=1.0
                )
                lf = sio.LabeledFrame(video=video, frame_idx=fi, instances=[inst])
                labeled_frames.append(lf)

        labels = sio.Labels(labeled_frames=labeled_frames)
        slp_path = output_dir / f"{cam_name}.slp"
        sio.save_file(labels, str(slp_path))
        print(f"  {slp_path}")


# --- Flat folder output for testing setup_project ---

# Naming strategies: how to build filenames for each (camera, trial) pair
_NAMING_STRATEGIES = {
    "number": "{cam}_{exp}_trial{trial_num:03d}.{ext}",
    "datestamp": "{cam}Y{datestamp}H{timestamp}.{frac}.{ext}",
    "datestamp_simple": "{cam}_{datestamp}_{timestamp}.{ext}",
}

# Seconds between experiments in datestamp strategies
_DATESTAMP_EXP_OFFSET_SEC = 60.0


def _make_flat_filename(
    cam_name: str,
    exp_name: str,
    trial_index: int,
    strategy: str,
    ext: str,
    datestamp: str = "20260116",
    base_time_sec: float = 51667.0,
    time_offset_per_cam: float = 0.003,
    cam_index: int = 0,
) -> str:
    """Generate a filename for a flat folder using the given naming strategy.

    Args:
        cam_name: Camera identifier string
        exp_name: Experiment name (embedded in filename to avoid collisions)
        trial_index: 0-based global trial index across all experiments
        strategy: "number", "datestamp", or "datestamp_simple"
        ext: File extension without dot (e.g. "mp4", "slp")
        datestamp: Date string YYYYMMDD
        base_time_sec: Base time in seconds since midnight for trial index 0
        time_offset_per_cam: Sub-second offset per camera (simulates real hw)
        cam_index: 0-based camera index (used for per-camera time offset)
    """
    if strategy == "number":
        return f"{cam_name}_{exp_name}_trial{trial_index + 1:03d}.{ext}"
    elif strategy == "datestamp":
        # Y...H... format like .cine files; each trial offset by _DATESTAMP_EXP_OFFSET_SEC
        total_sec = base_time_sec + trial_index * _DATESTAMP_EXP_OFFSET_SEC + cam_index * time_offset_per_cam
        hh = int(total_sec // 3600)
        mm = int((total_sec % 3600) // 60)
        ss = int(total_sec % 60)
        frac = total_sec - int(total_sec)
        frac_str = f"{frac:.9f}"[2:]  # "000003000" style
        return f"{cam_name}Y{datestamp}H{hh:02d}{mm:02d}{ss:02d}.{frac_str}.{ext}"
    elif strategy == "datestamp_simple":
        # YYYYMMDD_HHMMSS format; each trial offset by _DATESTAMP_EXP_OFFSET_SEC
        total_sec = base_time_sec + trial_index * _DATESTAMP_EXP_OFFSET_SEC + cam_index * time_offset_per_cam
        hh = int(total_sec // 3600)
        mm = int((total_sec % 3600) // 60)
        ss = int(total_sec % 60)
        return f"{cam_name}_{datestamp}_{hh:02d}{mm:02d}{ss:02d}.{ext}"
    else:
        raise ValueError(f"Unknown naming strategy: {strategy}")


def write_flat_folder(
    project_dir: Path,
    flat_dir: Path,
    cam_names: list[str],
    experiments: list[dict],
    strategy: str,
    calib_keyword: str = "calib",
    calib_keywords: list[str] | None = None,
    charuco_config: dict | None = None,
) -> Path:
    """Reorganize a multi-experiment project into a flat folder for testing setup_project.

    Copies videos from all experiments, the single per-camera .slp files, and
    calibration videos from each experiment into a single flat directory with
    collision-free filenames. Also writes a setup_config.toml.

    Args:
        project_dir: Path to the generated structured project
        flat_dir: Path for the flat output folder
        cam_names: Ordered list of camera names
        experiments: List of dicts with keys "name", "trial_name", "num_frames"
        strategy: Naming strategy — "number", "datestamp", or "datestamp_simple"
        calib_keyword: Legacy single keyword used to tag calibration files
        calib_keywords: Take keywords (e.g. ["PreCal", "PostCal"]); each names one
            calibration take. Defaults to charuco_config["takes"], else [calib_keyword]
        charuco_config: Charuco board config dict (optional, for setup_config.toml)

    Returns:
        Path to the flat folder
    """
    flat_dir.mkdir(parents=True, exist_ok=True)

    if calib_keywords:
        takes = list(calib_keywords)
    elif charuco_config and charuco_config.get("takes"):
        takes = list(charuco_config["takes"])
    else:
        takes = [calib_keyword]

    base_time = 51667.519576
    time_offset_per_cam = 0.003

    # --- Copy videos with new names (one trial per experiment, indexed globally) ---
    for trial_idx, exp in enumerate(experiments):
        exp_name = exp["name"]
        trial_name = exp["trial_name"]
        video_dir = project_dir / "experiments" / exp_name / "videos"
        for ci, cam_name in enumerate(cam_names):
            src = video_dir / f"{cam_name}_{exp_name}_{trial_name}.mp4"
            if src.exists():
                new_name = _make_flat_filename(
                    cam_name, exp_name, trial_idx, strategy, "mp4",
                    base_time_sec=base_time,
                    time_offset_per_cam=time_offset_per_cam,
                    cam_index=ci,
                )
                shutil.copy2(str(src), str(flat_dir / new_name))

    # --- Copy single per-camera .slp files ---
    slp_dir = project_dir / "slp"
    if slp_dir.exists():
        for cam_name in cam_names:
            src = slp_dir / f"{cam_name}.slp"
            if src.exists():
                shutil.copy2(str(src), str(flat_dir / f"{cam_name}.slp"))

    # --- Copy calibration media from each experiment ---
    # Files are already named to carry the experiment, take and camera, so copy as-is.
    for exp in experiments:
        exp_name = exp["name"]
        calib_dir = project_dir / "experiments" / exp_name / "calibration"
        if calib_dir.exists():
            for cam_name in cam_names:
                cam_calib = calib_dir / cam_name
                if cam_calib.is_dir():
                    for media in sorted(cam_calib.iterdir()):
                        if media.is_file():
                            shutil.copy2(str(media), str(flat_dir / media.name))

    # --- Write setup_config.toml ---
    exp_names = [e["name"] for e in experiments]
    datestamp_window = time_offset_per_cam * len(cam_names) * 2

    if strategy == "number":
        # Experiment name is embedded; keyword matching identifies each experiment
        match_strategy = "keyword"
    else:
        match_strategy = "datestamp"

    setup_config: dict = {
        "setup": {
            "source_dir": str(flat_dir),
            "output_dir": str(flat_dir.parent / f"{flat_dir.name}_project"),
            "camera_names": cam_names,
            "calib_keyword": calib_keyword,
            "calib_keywords": takes,
            "experiment_keywords": exp_names,
            "match_strategy": match_strategy,
            "datestamp_window_sec": datestamp_window if "datestamp" in strategy else 0.0,
        },
        "project": {
            "name": f"synthetic_{strategy}",
        },
    }
    if charuco_config:
        setup_config["charuco"] = {
            k: charuco_config[k]
            for k in ("squares_x", "squares_y", "square_length",
                       "marker_length", "marker_bits", "dict_size")
            if k in charuco_config
        }

    config_path = flat_dir / "setup_config.toml"
    with open(config_path, "w") as f:
        toml.dump(setup_config, f)
    print(f"  Setup config: {config_path}")

    return flat_dir

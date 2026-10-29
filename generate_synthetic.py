#!/usr/bin/env python3
"""Generate synthetic test data for label3d.

Usage:
    python generate_synthetic.py synthetic_config.toml
    python generate_synthetic.py synthetic_config.toml --no-charuco
    python generate_synthetic.py synthetic_config.toml --flat          # all strategies
    python generate_synthetic.py synthetic_config.toml --flat number   # one strategy
    python generate_synthetic.py synthetic_config.toml --no-slp        # skeletons in project.toml

Generates:
    <output_dir>/
    ├── trials.csv
    ├── project.toml
    ├── slp/
    │   ├── cam1.slp          (all experiments & trials for cam1)
    │   ├── cam2.slp
    │   └── cam3.slp
    └── experiments/
        └── <experiment_name>/
            ├── calibrations/        (cam folders if [charuco] section present)
            │   ├── cam1/<exp>_PreCal_cam1.mp4
            │   ├── ...
            │   └── ground-truth_{timestamp}.toml
            ├── points.parquet
            └── videos/
                ├── cam1_<exp>_trial1.mp4
                └── ...

A camera's `missing_nodes` (in [cameras.<name>]) are left out of its .slp
skeleton and its points. With --no-slp, no .slp files are written and each
camera's skeleton goes to [[skeleton.assigned]] in project.toml instead.

With --flat, also generates flat folder(s) for testing setup_project:
    <output_dir>_flat_<strategy>/
    ├── cam1_exp1_trial001.mp4        (number strategy — exp name in filename)
    ├── cam1Y20260116H142107....mp4   (datestamp strategy)
    ├── calib_exp1_cam1_board_000.png (calibration — exp name prefixed)
    ├── cam1.slp                      (single .slp per camera)
    └── setup_config.toml             (config for setup_project_cli.py)

Dependencies: numpy, opencv-python, pandas, toml
Optional:     sleap-io (for .slp file generation)
"""

import argparse
import copy
import logging
from pathlib import Path

import numpy as np

from label3d.frames import shown_frame
from label3d.points import POINTS_FILE
from synthetic import (
    build_cameras,
    compute_body_poses,
    compute_world_points,
    generate_charuco_videos,
    load_config,
    project_all_frames,
    run_sanity_checks,
    camera_skeleton,
    write_calibration,
    write_flat_folder,
    write_points,
    write_project_toml,
    write_slp_files,
    write_trials_csv,
    write_videos,
)


def _apply_camera_offsets(base_config: dict, offsets: dict) -> dict:
    """Return a deep-copied config with per-camera position deltas applied."""
    cfg = copy.deepcopy(base_config)
    for cam_name, delta in offsets.items():
        if cam_name in cfg["cameras"]:
            pos = list(cfg["cameras"][cam_name]["position"])
            cfg["cameras"][cam_name]["position"] = [
                pos[i] + delta[i] for i in range(3)
            ]
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic label3d test data from a TOML config"
    )
    parser.add_argument("config", help="Path to TOML configuration file")
    parser.add_argument(
        "--no-charuco", action="store_true",
        help="Skip calibration image generation even if [charuco] section is present",
    )
    parser.add_argument(
        "--flat", nargs="?", const="all", metavar="STRATEGY",
        help=(
            "Also generate flat folder(s) for testing setup_project. "
            "STRATEGY can be: number, datestamp, datestamp_simple, or all (default). "
            "Output goes to <output_dir>_flat_<strategy>/"
        ),
    )
    parser.add_argument(
        "--no-slp", action="store_true",
        help="Write no .slp files; put each camera's skeleton in project.toml instead",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.debug:
        for pkg in ("label3d", "synthetic"):
            logging.getLogger(pkg).setLevel(logging.DEBUG)

    config = load_config(args.config)

    # ── Unpack shared config ─────────────────────────────────────────────
    project_cfg = config["project"]
    output_dir = Path(project_cfg["output_dir"])

    skel_cfg = config["skeleton"]
    node_names = list(skel_cfg["nodes"].keys())
    node_offsets = {
        n: np.array(cfg["offset"], dtype=np.float64)
        for n, cfg in skel_cfg["nodes"].items()
    }
    node_colors = {n: cfg["color"] for n, cfg in skel_cfg["nodes"].items()}
    edges = [tuple(e) for e in skel_cfg["edges"]]

    render_cfg = config.get("rendering", {})
    charuco_cfg = config.get("charuco")

    cam_names = list(config["cameras"].keys())
    missing_nodes = {
        cam: set(cfg.get("missing_nodes", [])) for cam, cfg in config["cameras"].items()
    }
    for cam, missing in missing_nodes.items():
        unknown = missing - set(node_names)
        if unknown:
            parser.error(f"{cam}: missing_nodes not in the skeleton: {sorted(unknown)}")

    # Accumulate data across experiments for global outputs
    all_trial_rows: list[dict] = []
    # cam_trial_data[cam_name] = [(video_path, [frame_pts, ...]), ...]
    cam_trial_data: dict[str, list[tuple[Path, list[np.ndarray]]]] = {
        cn: [] for cn in cam_names
    }
    experiment_infos: list[dict] = []  # for flat folder generation
    calibrations: dict[str, str] = {}  # experiment -> ground-truth registry id

    # ── Per-experiment loop ──────────────────────────────────────────────
    for exp_name, exp_cfg in config["experiments"].items():
        trial_name = exp_cfg["trial_name"]
        num_frames = exp_cfg["num_frames"]

        print(f"\n=== Experiment: {exp_name} / trial: {trial_name} ===")

        # Build cameras with per-experiment position offsets applied
        offsets = exp_cfg.get("camera_offsets", {})
        exp_config = _apply_camera_offsets(config, offsets)
        cameras = build_cameras(exp_config)

        # Compute motion and 3D points
        poses = compute_body_poses(exp_cfg["motion"], num_frames)
        all_pts_3d = compute_world_points(poses, node_names, node_offsets)
        all_projected = project_all_frames(all_pts_3d, cameras)

        # Verify all points are within image bounds
        for cam_name, cam in cameras.items():
            w, h = cam["resolution"]
            for fi, proj in enumerate(all_projected[cam_name]):
                for bi, (px, py) in enumerate(proj):
                    if not (0 <= px < w and 0 <= py < h):
                        print(
                            f"  WARNING: {node_names[bi]} in {cam_name} frame {fi} "
                            f"is outside image at ({px:.1f}, {py:.1f})"
                        )

        # Create experiment output directory
        experiment_dir = output_dir / "experiments" / exp_name
        video_dir = experiment_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Write per-experiment outputs
        print("  Writing videos...")
        write_videos(
            cameras, all_projected,
            node_names, node_colors, edges,
            render_cfg, trial_name, num_frames, video_dir,
            experiment_name=exp_name,
        )

        print("  Writing calibration...")
        calib_entry = write_calibration(cameras, experiment_dir)
        calibrations[exp_name] = calib_entry.id
        print(f"    {calib_entry.path}")

        print("  Writing points.parquet...")
        points_path = experiment_dir / POINTS_FILE
        write_points(
            all_pts_3d, all_projected, cameras,
            node_names, trial_name, num_frames,
            points_path,
            exp_name,
            missing_nodes,
        )
        print(f"    {points_path}")

        # Charuco calibration videos (per-experiment since cameras differ)
        if charuco_cfg and not args.no_charuco:
            print("  Generating Charuco calibration videos...")
            generate_charuco_videos(
                charuco_cfg, cameras, experiment_dir,
                experiment_name=exp_name,
            )

        # Sanity checks
        run_sanity_checks(cameras, all_pts_3d, all_projected, node_names, calib_entry.path)

        # Accumulate trial row for global trials.csv
        trial_row: dict = {"experiment": exp_name, "trial": trial_name}
        for cam_name in cam_names:
            trial_row[cam_name] = f"{cam_name}_{exp_name}_{trial_name}.mp4"
            trial_row[f"{cam_name}_start"] = shown_frame(0)
            trial_row[f"{cam_name}_end"] = shown_frame(num_frames - 1)
        all_trial_rows.append(trial_row)

        # Accumulate per-camera labeled frames for unified .slp files
        for cam_name in cam_names:
            video_path = (video_dir / f"{cam_name}_{exp_name}_{trial_name}.mp4").resolve()
            frame_arrays = [all_projected[cam_name][fi] for fi in range(num_frames)]
            cam_trial_data[cam_name].append((video_path, frame_arrays))

        experiment_infos.append({
            "name": exp_name,
            "trial_name": trial_name,
            "num_frames": num_frames,
        })

        # Per-experiment summary
        print(f"\n  Cameras: {list(cameras.keys())}")
        for cam_name, cam in cameras.items():
            dist_cm = np.linalg.norm(
                cam["position"] - np.array(exp_config["cameras"][cam_name]["target"])
            )
            print(
                f"    {cam_name}: f={cam['K'][0,0]:.1f}px, "
                f"distance={dist_cm:.1f}cm"
            )

    # ── Global outputs (written once after all experiments) ──────────────
    print("\nWriting trials.csv...")
    trials_csv_path = output_dir / "trials.csv"
    write_trials_csv(all_trial_rows, trials_csv_path)
    print(f"  {trials_csv_path}")

    print("Writing project.toml...")
    charuco_board = (
        {k: charuco_cfg[k] for k in
         ("squares_x", "squares_y", "square_length", "marker_length", "marker_bits", "dict_size")
         if k in charuco_cfg}
        if charuco_cfg else None
    )
    assigned = None
    if args.no_slp:
        assigned = []
        for cam in cam_names:
            skel = camera_skeleton(node_names, edges, missing_nodes[cam])
            same = next((a for a in assigned if a["nodes"] == skel["nodes"]), None)
            if same is None:
                assigned.append({**skel, "cameras": [cam]})
            else:
                same["cameras"].append(cam)
    write_project_toml(
        output_dir / "project.toml", project_cfg.get("name", output_dir.name), charuco_board,
        calibrations=calibrations,
        node_offsets=node_offsets,
        assigned_skeletons=assigned,
    )

    if args.no_slp:
        print("Skipping .slp files (--no-slp)")
    else:
        print("Writing .slp files...")
        write_slp_files(cam_trial_data, node_names, edges, output_dir / "slp", missing_nodes)

    print(f"\nDone. Output: {output_dir}/")
    print(f"  Experiments: {[e['name'] for e in experiment_infos]}")
    print(f"  Cameras:     {cam_names}")
    print(f"  Bodyparts:   {node_names}")

    # ── Flat folder output ────────────────────────────────────────────────
    if args.flat:
        strategies = (
            ["number", "datestamp", "datestamp_simple"]
            if args.flat == "all"
            else [args.flat]
        )
        for strategy in strategies:
            flat_dir = output_dir.parent / f"{output_dir.name}_flat_{strategy}"
            print(f"\nGenerating flat folder ({strategy}): {flat_dir}/")
            write_flat_folder(
                project_dir=output_dir,
                flat_dir=flat_dir,
                cam_names=cam_names,
                experiments=experiment_infos,
                strategy=strategy,
                calib_keyword="calib",
                charuco_config=charuco_cfg,
            )
            file_count = len(list(flat_dir.iterdir()))
            print(f"  {file_count} files written")


if __name__ == "__main__":
    main()

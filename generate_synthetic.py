#!/usr/bin/env python3
"""Generate synthetic test data for label3d.

Usage:
    python generate_synthetic.py synthetic_config.toml
    python generate_synthetic.py synthetic_config.toml --no-charuco
    python generate_synthetic.py synthetic_config.toml --flat          # all strategies
    python generate_synthetic.py synthetic_config.toml --flat number   # one strategy

Generates:
    <output_dir>/
    ├── trials.csv
    ├── skeleton.json
    ├── project.toml
    ├── slp/
    │   ├── cam0_trial1.slp
    │   └── cam1_trial1.slp
    └── experiments/
        └── <experiment_name>/
            ├── calibration/         (if [charuco] section present)
            │   ├── cam1/board_000.png
            │   └── ...
            ├── calibration.toml
            ├── points.csv
            └── videos/
                ├── cam0_trial1.mp4
                └── cam1_trial1.mp4

With --flat, also generates flat folder(s) for testing setup_project:
    <output_dir>_flat_<strategy>/
    ├── cam1_trial001.mp4         (number strategy)
    ├── cam1Y20260116H142107.519576000.mp4  (datestamp strategy)
    ├── calib_cam1_board_000.png  (calibration files)
    ├── cam1_trial001.slp
    └── setup_config.toml         (config for setup_project_cli.py)

Dependencies: numpy, opencv-python, pandas, toml
Optional:     sleap-io (for .slp file generation)
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from synthetic import (
    build_cameras,
    compute_body_poses,
    compute_world_points,
    generate_charuco_images,
    load_config,
    project_all_frames,
    run_sanity_checks,
    write_calibration_toml,
    write_flat_folder,
    write_points_csv,
    write_project_toml,
    write_skeleton_json,
    write_slp_files,
    write_trials_csv,
    write_videos,
)


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

    # ── Unpack config ────────────────────────────────────────────────────
    project_cfg = config["project"]
    output_dir = Path(project_cfg["output_dir"])
    experiment_name = project_cfg.get("experiment_name", "default")
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
    experiment_dir = output_dir / "experiments" / experiment_name
    video_dir = experiment_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # ── Write outputs ────────────────────────────────────────────────────
    print("Writing videos...")
    write_videos(
        cameras, all_projected,
        node_names, node_colors, edges,
        render_cfg, trial_name, num_frames, video_dir,
    )

    print("Writing calibration.toml...")
    write_calibration_toml(cameras, experiment_dir / "calibration.toml")
    print(f"  {experiment_dir / 'calibration.toml'}")

    print("Writing trials.csv...")
    trials_csv_path = output_dir / "trials.csv"
    write_trials_csv(cameras, trial_name, num_frames, trials_csv_path, experiment_name)
    print(f"  {trials_csv_path}")

    print("Writing points.csv...")
    points_csv_path = experiment_dir / "points.csv"
    write_points_csv(
        all_pts_3d, all_projected, cameras,
        node_names, trial_name, num_frames,
        points_csv_path,
        experiment_name,
    )
    print(f"  {points_csv_path}")

    print("Writing skeleton.json...")
    write_skeleton_json(node_names, edges, node_offsets, output_dir / "skeleton.json")

    print("Writing project.toml...")
    charuco_cfg = config.get("charuco")
    charuco_board = (
        {k: charuco_cfg[k] for k in
         ("squares_x", "squares_y", "square_length", "marker_length", "marker_bits", "dict_size")
         if k in charuco_cfg}
        if charuco_cfg else None
    )
    write_project_toml(output_dir / "project.toml", project_cfg.get("name", output_dir.name), charuco_board)

    print("Writing .slp files...")
    write_slp_files(
        all_projected, cameras, node_names, edges,
        trial_name, num_frames, video_dir, output_dir / "slp",
    )

    # ── Charuco calibration images ───────────────────────────────────────
    if "charuco" in config and not args.no_charuco:
        print("\nGenerating Charuco calibration images...")
        generate_charuco_images(config["charuco"], cameras, experiment_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nDone. Output: {output_dir}/")
    print(f"  Experiment: {experiment_name}")
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
    run_sanity_checks(cameras, all_pts_3d, all_projected, node_names, experiment_dir)

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
                cameras=cameras,
                trial_name=trial_name,
                strategy=strategy,
                experiment_name=experiment_name,
                calib_keyword="calib",
                charuco_config=charuco_cfg,
            )
            file_count = len(list(flat_dir.iterdir()))
            print(f"  {file_count} files written")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic test data for label3d.

Usage:
    python generate_synthetic.py synthetic_config.toml
    python generate_synthetic.py synthetic_config.toml --no-charuco

Generates:
    <output_dir>/
    ├── videos/
    │   ├── trials.csv
    │   ├── cam0_trial1.mp4
    │   └── cam1_trial1.mp4
    ├── calibration.toml
    ├── points.csv
    ├── skeleton.json
    ├── calibration/              (if [charuco] section present)
    │   ├── cam1/board_000.png
    │   └── ...
    └── slp/                     (if sleap_io is installed)
        ├── cam0_trial1.slp
        └── cam1_trial1.slp

Dependencies: numpy, opencv-python, pandas, toml
Optional:     sleap-io (for .slp file generation)
"""

import argparse
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
    write_points_csv,
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

    # ── Charuco calibration images ───────────────────────────────────────
    if "charuco" in config and not args.no_charuco:
        print("\nGenerating Charuco calibration images...")
        generate_charuco_images(config["charuco"], cameras, output_dir)

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
    run_sanity_checks(cameras, all_pts_3d, all_projected, node_names, output_dir)


if __name__ == "__main__":
    main()

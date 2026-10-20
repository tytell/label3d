"""Synthetic test data generator for label3d."""

from synthetic.cameras import build_cameras, load_config
from synthetic.charuco import generate_charuco_images, generate_charuco_videos
from synthetic.checks import run_sanity_checks
from synthetic.motion import compute_body_poses, compute_world_points
from synthetic.projection import project_all_frames, project_points
from synthetic.rendering import write_videos
from synthetic.writers import (
    write_calibration,
    write_flat_folder,
    write_points,
    write_project_toml,
    camera_skeleton,
    write_slp_files,
    write_trials_csv,
)

__all__ = [
    "build_cameras",
    "compute_body_poses",
    "compute_world_points",
    "generate_charuco_images",
    "generate_charuco_videos",
    "load_config",
    "project_all_frames",
    "project_points",
    "run_sanity_checks",
    "write_calibration",
    "write_flat_folder",
    "write_points",
    "write_project_toml",
    "camera_skeleton",
    "write_slp_files",
    "write_trials_csv",
    "write_videos",
]

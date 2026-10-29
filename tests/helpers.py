"""Shared constants and helper functions for the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import toml

from label3d import version
from label3d.frames import FRAME_BASE
from label3d.project import _extract_cameras, read_trials_csv, write_trials_csv
from label3d.skeleton import Skeleton, assign_in_config

CAMERAS = ["cam1", "cam2", "cam3"]
BODYPARTS = ["snout", "left_eye", "dorsal"]


def make_skeleton_json(tmp_path: Path, nodes, edges=None, symmetries=None) -> Path:
    """Write a minimal skeleton.json and return its path."""
    data = {
        "nodes": nodes,
        "edges": edges or [],
        "symmetries": symmetries or [],
    }
    p = tmp_path / "skeleton.json"
    p.write_text(json.dumps(data))
    return p


def make_current(root: Path) -> Path:
    """Bring a project a test built in the old style up to the current layout.

    Test projects are often written with a skeleton.json and a 0-based
    trials.csv; this moves the skeleton into project.toml, numbers trials.csv
    from FRAME_BASE and records the layout version, without the upgrade
    (tests/test_upgrade.py covers that).
    """
    root = Path(root)
    config_path = root / "project.toml"
    config = toml.load(config_path) if config_path.exists() else {}
    trials_path = root / "trials.csv"
    cameras = _extract_cameras(pd.read_csv(trials_path, nrows=0).columns.tolist())
    skeleton_json = root / "skeleton.json"
    if skeleton_json.exists():
        if "skeleton" not in config:
            skeleton = Skeleton.from_json(skeleton_json)
            config["skeleton"] = assign_in_config({}, skeleton, cameras, cameras)
        skeleton_json.unlink()
    if config.get("trials", {}).get("frame_base", 0) != FRAME_BASE:
        write_trials_csv(trials_path, read_trials_csv(trials_path, frame_base=0), backup=False)
        config.setdefault("trials", {})["frame_base"] = FRAME_BASE
    version.write_project_toml(config_path, config, new_project=True)
    return root

"""Shared constants and helper functions for the test suite."""

from __future__ import annotations

import json
from pathlib import Path

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

"""The 2D clicks of axis alignment, saved as they are made (Phase 16.5).

experiments/{name}/calibrations/alignment_picks.parquet has one row per
(take, frame, target, camera): take is an axis take, frame its 0-based
synchronized frame, and target one of TARGETS. x, y are raw pixels. Keeping
the clicks lets an alignment be recomputed with another calibration without
clicking again.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa

from label3d.calibration_registry import registry_dir
from label3d.parquet_io import write_table

logger = logging.getLogger(__name__)

PICKS_FILE = "alignment_picks.parquet"
PICK_COLUMNS = ["take", "frame", "target", "camera", "x", "y"]
PICKS_SCHEMA = pa.schema([
    pa.field("take", pa.string()),
    pa.field("frame", pa.int32()),
    pa.field("target", pa.string()),
    pa.field("camera", pa.string()),
    pa.field("x", pa.float64()),
    pa.field("y", pa.float64()),
])

# Origin, then one or two points along each axis: x1 is the only point in
# 1-point mode, or the near one in 2-point mode, where the axis runs x1 -> x2.
ORIGIN = "origin"
TARGETS = (ORIGIN, "x1", "x2", "y1", "y2", "z1", "z2")

# target -> camera -> (x, y)
Picks = dict[str, dict[str, tuple[float, float]]]


def picks_path(exp_dir: Path) -> Path:
    return registry_dir(exp_dir) / PICKS_FILE


def axis_targets(axis: str, points_per_axis: int) -> list[str]:
    """The targets clicked on one axis."""
    return [f"{axis}1", f"{axis}2"][:points_per_axis]


def targets_for(points_per_axis: int, cross_product: str = "") -> list[str]:
    """The targets to click: origin, then each axis not derived by the cross product."""
    targets = [ORIGIN]
    for axis in "xyz":
        if axis != cross_product:
            targets += axis_targets(axis, points_per_axis)
    return targets


def _read_df(exp_dir: Path) -> pd.DataFrame:
    path = picks_path(exp_dir)
    if not path.exists():
        return pd.DataFrame(columns=PICK_COLUMNS)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame(columns=PICK_COLUMNS)


def read_picks(exp_dir: Path, take: str, frame: int) -> Picks:
    """The clicks saved for one frame of an axis take."""
    df = _read_df(exp_dir)
    rows = df[(df["take"] == take) & (df["frame"] == frame)]
    picks: Picks = {}
    for row in rows.itertuples(index=False):
        picks.setdefault(str(row.target), {})[str(row.camera)] = (float(row.x), float(row.y))
    return picks


def picked_frames(exp_dir: Path, take: str) -> list[int]:
    """Frames of an axis take with saved clicks."""
    df = _read_df(exp_dir)
    return sorted(int(f) for f in df.loc[df["take"] == take, "frame"].unique())


def write_picks(exp_dir: Path, take: str, frame: int, picks: Picks) -> None:
    """Replace one frame's clicks, keeping every other frame's. Empty picks remove it."""
    df = _read_df(exp_dir)
    keep = df[~((df["take"] == take) & (df["frame"] == frame))]
    new = pd.DataFrame(
        [
            (take, frame, target, camera, float(x), float(y))
            for target in TARGETS for camera, (x, y) in sorted(picks.get(target, {}).items())
        ],
        columns=PICK_COLUMNS,
    )
    out = pd.concat([keep, new], ignore_index=True) if len(keep) else new
    path = picks_path(exp_dir)
    if out.empty and not path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out = out.astype({"take": str, "frame": "int32", "target": str, "camera": str})
    write_table(pa.Table.from_pandas(out[PICK_COLUMNS], schema=PICKS_SCHEMA, preserve_index=False), path)

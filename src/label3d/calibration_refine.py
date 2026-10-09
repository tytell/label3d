"""Refining a calibration with labelled bodypart points (Phase 12).

A refinement starts from a charuco (or imported) calibration and adjusts it
with aniposelib's bundle_adjust_iter on points labelled or predicted in
several cameras. Some frames are held out of the fit, so the reported error
is measured on points the fit never saw. No Qt here: the worker and dialog
wrap these functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from label3d.points import SOURCE_LABEL3D, SOURCE_SLEAP_PRED, SOURCE_SLEAP_USER

ALL_SOURCES = (SOURCE_LABEL3D, SOURCE_SLEAP_USER, SOURCE_SLEAP_PRED)
USED_STATES = ("set", "predicted")
POINT_INDEX = ["trial", "frame", "instance", "bodypart"]
RECORD_COLUMNS = POINT_INDEX + ["camera", "x", "y", "source", "score", "holdout"]


@dataclass
class PointSelection:
    """Which labelled points go into a refinement."""

    sources: set[str] = field(default_factory=lambda: {SOURCE_LABEL3D, SOURCE_SLEAP_USER})
    score_min: float = 0.0            # applies to sleap_pred only
    trials: list[str] | None = None   # None = all trials of the experiment
    min_cameras: int = 2


def _experiment_rows(points_df: pd.DataFrame, trials: list[str] | None) -> pd.DataFrame:
    """Rows indexed (trial, frame, instance, bodypart), optionally only some trials."""
    df = points_df
    if "experiment" in (df.index.names or []):
        df = df.droplevel("experiment")
    if trials is not None:
        df = df[df.index.get_level_values("trial").isin([str(t) for t in trials])]
    return df


def usable_mask(points_df: pd.DataFrame, cam_names: list[str], sel: PointSelection) -> pd.DataFrame:
    """Which camera observations count, as a bool frame (rows × cameras).

    An observation counts if it is set or predicted, from a chosen source,
    has finite x/y and, for SLEAP predictions, a score of at least score_min.
    label3d reprojection guesses (predicted, label3d) never count: they come
    from the calibration, so fitting to them would be circular.
    """
    df = _experiment_rows(points_df, sel.trials)
    mask = pd.DataFrame(False, index=df.index, columns=list(cam_names))
    for cam in cam_names:
        if (cam, "state") not in df.columns:
            continue
        state = df[(cam, "state")].astype(str)
        source = df[(cam, "source")].astype(str)
        x = pd.to_numeric(df[(cam, "x")], errors="coerce")
        y = pd.to_numeric(df[(cam, "y")], errors="coerce")
        score = pd.to_numeric(df[(cam, "score")], errors="coerce")
        ok = (
            state.isin(USED_STATES)
            & source.isin(sel.sources)
            & np.isfinite(x) & np.isfinite(y)
            & ~((source == SOURCE_LABEL3D) & (state == "predicted"))
            & ((source != SOURCE_SLEAP_PRED) | (score >= sel.score_min))
        )
        mask[cam] = ok.to_numpy()
    return mask


def gather_points(
    points_df: pd.DataFrame, cam_names: list[str], sel: PointSelection
) -> tuple[np.ndarray, pd.MultiIndex]:
    """The C × N × 2 raw-pixel array for bundle adjustment and its point index.

    Columns are points (one bodypart in one frame) seen by at least
    sel.min_cameras cameras that count; cameras that don't count are NaN.
    """
    df = _experiment_rows(points_df, sel.trials)
    mask = usable_mask(points_df, cam_names, sel)
    keep = (mask.sum(axis=1) >= sel.min_cameras).to_numpy()
    df, mask = df[keep], mask[keep]

    p2ds = np.full((len(cam_names), len(df), 2), np.nan)
    for c, cam in enumerate(cam_names):
        if (cam, "x") not in df.columns:
            continue
        ok = mask[cam].to_numpy()
        xy = np.column_stack([
            pd.to_numeric(df[(cam, "x")], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df[(cam, "y")], errors="coerce").to_numpy(dtype=float),
        ])
        p2ds[c, ok] = xy[ok]
    index = pd.MultiIndex.from_tuples(list(df.index), names=POINT_INDEX)
    return p2ds, index


def count_points(
    points_df: pd.DataFrame, cam_names: list[str], sel: PointSelection
) -> dict[tuple[str, str], int]:
    """Per camera pair, the number of selected points seen by both."""
    p2ds, _ = gather_points(points_df, cam_names, sel)
    seen = ~np.isnan(p2ds[:, :, 0])
    return {
        (cam_names[a], cam_names[b]): int(np.sum(seen[a] & seen[b]))
        for a, b in combinations(range(len(cam_names)), 2)
    }


def holdout_frames(
    index: pd.MultiIndex, fraction: float, seed: int
) -> set[tuple[str, int]]:
    """A reproducible random set of (trial, frame) pairs to keep out of the fit.

    The split is by frame, so a held-out frame's bodyparts are all held out.
    At least one frame is held out when fraction > 0, and at least one is
    kept for the fit.
    """
    frames = sorted({(str(t), int(f)) for t, f, *_ in index})
    if fraction <= 0 or len(frames) < 2:
        return set()
    n = min(max(1, round(fraction * len(frames))), len(frames) - 1)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(frames), size=n, replace=False)
    return {frames[i] for i in picked}


def holdout_mask(index: pd.MultiIndex, frames: set[tuple[str, int]]) -> np.ndarray:
    """Bool per point: True if its frame is held out."""
    return np.array([(str(t), int(f)) in frames for t, f, *_ in index], dtype=bool)


def frames_by_trial(index: pd.MultiIndex) -> dict[str, list[int]]:
    """trial -> sorted frames that have points, for [metadata.points]."""
    frames: dict[str, set[int]] = {}
    for t, f, *_ in index:
        frames.setdefault(str(t), set()).add(int(f))
    return {t: sorted(fs) for t, fs in sorted(frames.items())}


# -- Points record ---------------------------------------------------------------

def save_points_record(
    path: Path,
    p2ds: np.ndarray,
    index: pd.MultiIndex,
    cam_names: list[str],
    points_df: pd.DataFrame,
    holdout: np.ndarray,
) -> None:
    """Write every camera observation that went into a refinement.

    The x/y are copied rather than referenced, so the file still describes the
    fit after points.csv is edited or re-synced. Source and score are looked
    up in points_df.
    """
    df = _experiment_rows(points_df, None).reindex(index)
    records = []
    for c, cam in enumerate(cam_names):
        sources = df[(cam, "source")].astype(str).to_numpy()
        scores = pd.to_numeric(df[(cam, "score")], errors="coerce").to_numpy(dtype=float)
        for n, key in enumerate(index):
            x, y = p2ds[c, n]
            if np.isfinite(x) and np.isfinite(y):
                records.append(
                    (n, c, *key, cam, float(x), float(y), sources[n], scores[n], bool(holdout[n]))
                )
    records.sort(key=lambda r: (r[0], r[1]))   # point order, so a reload keeps it
    rows = [r[2:] for r in records]
    pd.DataFrame(rows, columns=RECORD_COLUMNS).to_csv(path, index=False)


def load_points_record(
    path: Path, cam_names: list[str]
) -> tuple[np.ndarray, pd.MultiIndex, np.ndarray]:
    """Read a points record back as (p2ds C×N×2, point index, holdout per point)."""
    df = pd.read_csv(path, dtype={"trial": str, "bodypart": str, "camera": str})
    df["frame"] = df["frame"].astype(int)
    df["instance"] = df["instance"].astype(int)
    keys = list(dict.fromkeys(zip(df["trial"], df["frame"], df["instance"], df["bodypart"])))
    position = {k: n for n, k in enumerate(keys)}
    cam_pos = {cam: c for c, cam in enumerate(cam_names)}

    p2ds = np.full((len(cam_names), len(keys), 2), np.nan)
    holdout = np.zeros(len(keys), dtype=bool)
    for rec in df.itertuples(index=False):
        n = position[(rec.trial, rec.frame, rec.instance, rec.bodypart)]
        holdout[n] = bool(rec.holdout)
        c = cam_pos.get(rec.camera)
        if c is not None:
            p2ds[c, n] = (rec.x, rec.y)
    index = pd.MultiIndex.from_tuples(keys, names=POINT_INDEX)
    return p2ds, index, holdout

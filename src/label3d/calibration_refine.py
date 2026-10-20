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
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from aniposelib.cameras import CameraGroup

from label3d.calibration_compare import PairDrift, compare_calibrations, run_spread
from label3d.calibration_io import load_corners
from label3d.calibration_registry import (
    CalibrationEntry,
    camera_dir,
    save_calibration,
    update_metadata,
)
from label3d.calibration_solve import (
    CalibrationTake,
    TakeCalibration,
    build_board,
    calibrate_repeated,
    rows_in_range,
    take_detections,
    take_ranges,
)
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
    fit after the points are edited or re-synced. Source and score are looked
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


# -- Stage 1: the charuco calibration a refinement starts from --------------------

def charuco_rows(
    exp_dir: Path, cam_names: list[str], takes: list[CalibrationTake], names: list[str]
) -> dict[str, list]:
    """Saved corners of the named takes, trimmed to each take's frame range."""
    rows = {
        cam: load_corners(camera_dir(exp_dir, cam), takes=names) or []
        for cam in cam_names
    }
    return rows_in_range(rows, takes)


def calibrate_takes(
    rows: dict[str, list],
    cam_names: list[str],
    cam_sizes: dict[str, tuple[int, int]],
    board_params: dict,
    n_repeats: int,
    progress: Callable[[str], None] | None = None,
) -> TakeCalibration:
    """Stage 1: calibrate the chosen takes jointly, keeping the best of n_repeats runs."""
    if not any(rows.values()):
        raise ValueError("no saved charuco corners for the chosen takes")
    board = build_board(board_params)
    return calibrate_repeated(
        cam_names, cam_sizes, rows, board, n_repeats, "the chosen takes", progress
    )


def matching_charuco_entry(
    entries: list[CalibrationEntry],
    names: list[str],
    takes: list[CalibrationTake],
    exp_dir: Path,
    cam_names: list[str],
) -> CalibrationEntry | None:
    """The newest charuco entry made from exactly these takes and the current corners.

    A match has the same takes, the same detection timestamps and the same
    take ranges, so recalibrating would solve the same rows again.
    """
    detected = take_detections(exp_dir, cam_names, names)
    ranges = take_ranges(takes, names)
    for entry in entries:   # newest first
        if entry.method != "charuco" or sorted(entry.charuco_takes) != sorted(names):
            continue
        charuco = entry.metadata.get("charuco", {})
        if charuco.get("detected") == detected and charuco.get("take_ranges") == ranges:
            return entry
    return None


def charuco_points(rows: dict[str, list], cam_names: list[str]) -> np.ndarray:
    """Charuco corners as a C × N × 2 array, one column per (take, frame, corner).

    Only corners seen by at least two cameras are kept, since reprojection
    error needs a triangulated point.
    """
    position: dict[tuple, int] = {}
    seen: list[dict[int, tuple[float, float]]] = []
    for c, cam in enumerate(cam_names):
        for row in rows.get(cam, []):
            corners, ids = row.get("corners"), row.get("ids")
            if corners is None or ids is None:
                continue
            xy = np.asarray(corners, dtype=float).reshape(-1, 2)
            for corner_id, (x, y) in zip(np.asarray(ids).ravel(), xy):
                key = (row.get("framenum"), int(corner_id))
                if key not in position:
                    position[key] = len(seen)
                    seen.append({})
                seen[position[key]][c] = (x, y)
    p2ds = np.full((len(cam_names), len(seen), 2), np.nan)
    for n, obs in enumerate(seen):
        for c, xy in obs.items():
            p2ds[c, n] = xy
    keep = np.sum(~np.isnan(p2ds[:, :, 0]), axis=0) >= 2
    return p2ds[:, keep]


# -- Errors ----------------------------------------------------------------------

def point_errors(cgroup: CameraGroup, p2ds: np.ndarray) -> np.ndarray:
    """Reprojection error (px) of every observation of a point seen by ≥ 2 cameras."""
    if p2ds.shape[1] == 0:
        return np.array([])
    p3ds = cgroup.triangulate(p2ds)
    errors = np.linalg.norm(cgroup.reprojection_error(p3ds, p2ds), axis=2)
    return errors[np.isfinite(errors)]


def error_stats(errors: np.ndarray) -> tuple[float, float]:
    """(median, 90th percentile), NaN when there are no errors."""
    if len(errors) == 0:
        return float("nan"), float("nan")
    return float(np.median(errors)), float(np.percentile(errors, 90))


# -- Fixing scale and world frame ------------------------------------------------

def _poses(cgroup: CameraGroup) -> tuple[np.ndarray, np.ndarray]:
    """(rotations C×3×3, translations C×3) of the cameras."""
    R = np.array([cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype=float))[0]
                  for cam in cgroup.cameras])
    t = np.array([np.asarray(cam.get_translation(), dtype=float).ravel()
                  for cam in cgroup.cameras])
    return R, t


def _nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def similarity_to(cgroup: CameraGroup, reference: CameraGroup) -> tuple[float, np.ndarray, np.ndarray]:
    """The similarity (s, R, T) with reference world ≈ s·R·(cgroup world) + T.

    With 3 or more cameras it is fitted to the camera centres (Umeyama). With
    2, the centres leave the rotation about the baseline free, so the
    rotation comes from the camera orientations and the scale from the
    baseline length.
    """
    R_a, t_a = _poses(cgroup)
    R_b, t_b = _poses(reference)
    c_a = -np.einsum("nji,nj->ni", R_a, t_a)   # centres: -Rᵀt
    c_b = -np.einsum("nji,nj->ni", R_b, t_b)
    mu_a, mu_b = c_a.mean(axis=0), c_b.mean(axis=0)
    d_a, d_b = c_a - mu_a, c_b - mu_b

    if len(c_a) >= 3:
        cov = d_b.T @ d_a / len(c_a)
        U, S, Vt = np.linalg.svd(cov)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        var_a = np.sum(d_a ** 2) / len(c_a)
        s = float(np.trace(np.diag(S) @ D) / var_a)
    else:
        # A camera keeps its orientation when R_a_i·Rᵀ = R_b_i, so R ≈ R_b_iᵀ·R_a_i.
        R = _nearest_rotation(sum(rb.T @ ra for ra, rb in zip(R_a, R_b)))
        s = float(np.sum(np.linalg.norm(d_b, axis=1)) / np.sum(np.linalg.norm(d_a, axis=1)))
    T = mu_b - s * R @ mu_a
    return s, R, T


def apply_similarity(cgroup: CameraGroup, s: float, R: np.ndarray, T: np.ndarray) -> None:
    """Move cgroup's world frame by X_new = s·R·X + T, keeping every projection."""
    R_cam, t_cam = _poses(cgroup)
    for cam, Ri, ti in zip(cgroup.cameras, R_cam, t_cam):
        R_new = Ri @ R.T
        cam.set_rotation(cv2.Rodrigues(R_new)[0].ravel())
        cam.set_translation(s * ti - R_new @ T)


def alignment_residuals(cgroup: CameraGroup, reference: CameraGroup) -> tuple[float, float, float]:
    """What an alignment of cgroup onto reference leaves unexplained.

    Returns (centre_cm, centre_pct, rotation_deg): the RMS distance between
    matching camera centres, the same as a percentage of reference's mean
    camera-pair baseline, and the RMS angle between matching camera
    orientations. With 2 cameras the centres always fit, so only the rotation
    can show a change in the rig's shape.
    """
    R_a, t_a = _poses(cgroup)
    R_b, t_b = _poses(reference)
    c_a = -np.einsum("nji,nj->ni", R_a, t_a)
    c_b = -np.einsum("nji,nj->ni", R_b, t_b)
    centre_cm = float(np.sqrt(np.mean(np.sum((c_a - c_b) ** 2, axis=1))))
    baselines = [np.linalg.norm(c_b[i] - c_b[j]) for i, j in combinations(range(len(c_b)), 2)]
    mean_baseline = float(np.mean(baselines)) if baselines else float("nan")
    centre_pct = 100.0 * centre_cm / mean_baseline if mean_baseline > 0 else float("nan")
    angles = [np.linalg.norm(cv2.Rodrigues(ra @ rb.T)[0]) for ra, rb in zip(R_a, R_b)]
    rotation_deg = float(np.degrees(np.sqrt(np.mean(np.square(angles)))))
    return centre_cm, centre_pct, rotation_deg


def max_pair_points(p2ds: np.ndarray) -> int:
    """The most points any camera pair shares."""
    seen = ~np.isnan(p2ds[:, :, 0])
    return max(
        (int(np.sum(seen[a] & seen[b])) for a, b in combinations(range(len(seen)), 2)),
        default=0,
    )


def repeats_differ(p2ds: np.ndarray, n_samp_iter: int) -> bool:
    """Whether repeated bundle adjustments of p2ds can differ.

    aniposelib subsamples at most n_samp_iter points per camera pair. When no
    pair has more, every repeat uses all the points and gives the same result.
    """
    return max_pair_points(p2ds) > n_samp_iter


def align_to(cgroup: CameraGroup, reference: CameraGroup) -> float:
    """Map cgroup onto reference's scale (cm) and world axes. Returns the scale."""
    s, R, T = similarity_to(cgroup, reference)
    apply_similarity(cgroup, s, R, T)
    return s


# -- Stage 2 ---------------------------------------------------------------------

@dataclass
class RefineOptions:
    only_extrinsics: bool = True
    holdout: float = 0.2
    n_repeats: int = 3
    seed: int | None = None       # None: drawn at random, then stored
    n_samp_full: int = 1000
    n_samp_iter: int = 200


@dataclass
class RefineResult:
    cgroup: CameraGroup                # refined and aligned onto the start
    start: CameraGroup                 # the Stage 1 group it started from
    p2ds: np.ndarray                   # every selected point, C × N × 2
    index: pd.MultiIndex
    holdout: np.ndarray                # bool per point
    seed: int
    kept_repeat: int
    n_repeats: int                     # repeats actually run (1 when they can't differ)
    scale: float                       # alignment scale correction
    spread: tuple[float, float] | None # run-to-run (rotation_deg, baseline_pct); None for one run
    errors: dict[str, float]
    drift: list[PairDrift]


def new_seed() -> int:
    return int(np.random.default_rng().integers(0, 2**31 - 1))


def run_stage2(
    start: CameraGroup, p2ds_fit: np.ndarray, options: RefineOptions, seed: int
) -> tuple[CameraGroup, float]:
    """One bundle adjustment of a copy of `start`, aligned back onto it.

    aniposelib subsamples with NumPy's global random state, so seeding it
    makes the run reproducible. Returns (refined group, scale correction).
    """
    np.random.seed(seed)
    cgroup = start.copy()
    cgroup.bundle_adjust_iter(
        p2ds_fit,
        only_extrinsics=options.only_extrinsics,
        n_samp_full=options.n_samp_full,
        n_samp_iter=options.n_samp_iter,
        verbose=False,
    )
    scale = align_to(cgroup, start)
    return cgroup, scale


def refine(
    start: CameraGroup,
    p2ds: np.ndarray,
    index: pd.MultiIndex,
    options: RefineOptions,
    charuco_p2ds: np.ndarray | None = None,
    progress: Callable[[str], None] | None = None,
) -> RefineResult:
    """Stage 2: adjust `start` with the selected points; validate on held-out frames.

    p2ds and index come from gather_points, in the order of start.get_names().
    charuco_p2ds, when given, are the charuco corners of the start's takes,
    which the refinement should not make much worse.
    """
    say = progress or (lambda _msg: None)
    seed = options.seed if options.seed is not None else new_seed()
    holdout = holdout_mask(index, holdout_frames(index, options.holdout, seed))
    fit, held = p2ds[:, ~holdout], p2ds[:, holdout]
    if fit.shape[1] == 0:
        raise ValueError("no points to refine with")
    say(f"Refining with {fit.shape[1]} points; {held.shape[1]} held out (seed {seed})")

    n_repeats = max(1, options.n_repeats)
    if n_repeats > 1 and not repeats_differ(fit, options.n_samp_iter):
        say(f"  no camera pair has more than {options.n_samp_iter} points, "
            "so repeats would be identical; running once")
        n_repeats = 1

    runs = []
    for r in range(n_repeats):
        cgroup, scale = run_stage2(start, fit, options, seed + r)
        score_points = held if held.shape[1] else fit
        err = error_stats(point_errors(cgroup, score_points))[0]
        say(f"  run {r + 1}/{n_repeats}: median error {err:.3f} px, scale {scale:.4f}")
        runs.append((err, r, cgroup, scale))
    err, kept, cgroup, scale = min(runs, key=lambda run: (np.nan_to_num(run[0], nan=np.inf), run[1]))
    spread = run_spread([run[2] for run in runs]) if n_repeats > 1 else None

    errors: dict[str, float] = {}
    errors["holdout"], errors["holdout_p90"] = error_stats(point_errors(cgroup, held))
    errors["holdout_before"], errors["holdout_p90_before"] = error_stats(point_errors(start, held))
    errors["points"] = error_stats(point_errors(cgroup, fit))[0]
    errors["points_before"] = error_stats(point_errors(start, fit))[0]
    if charuco_p2ds is not None and charuco_p2ds.shape[1]:
        errors["charuco"] = error_stats(point_errors(cgroup, charuco_p2ds))[0]
        errors["charuco_before"] = error_stats(point_errors(start, charuco_p2ds))[0]
    errors["scale_correction"] = scale
    (errors["align_centre_cm"], errors["align_centre_pct"],
     errors["align_rotation_deg"]) = alignment_residuals(cgroup, start)
    if spread is not None:
        errors["spread_rotation_deg"], errors["spread_baseline_pct"] = spread
    errors = {k: float(v) for k, v in errors.items() if np.isfinite(v)}

    return RefineResult(
        cgroup=cgroup, start=start, p2ds=p2ds, index=index, holdout=holdout,
        seed=seed, kept_repeat=kept, n_repeats=n_repeats, scale=scale, spread=spread,
        errors=errors,
        drift=compare_calibrations(start, cgroup),
    )


# -- Saving ----------------------------------------------------------------------

def save_refinement(
    exp_dir: Path,
    result: RefineResult,
    parent: CalibrationEntry,
    label: str,
    sel: PointSelection,
    options: RefineOptions,
    points_df: pd.DataFrame,
) -> CalibrationEntry:
    """Save a refine entry under refined/, with its _points.csv record."""
    cam_names = result.cgroup.get_names()
    entry = save_calibration(
        exp_dir, result.cgroup, label, "refine", parent=parent.id,
        charuco_takes=parent.charuco_takes,
        points={
            "sources": sorted(sel.sources),
            "score_min": float(sel.score_min),
            "min_cameras": int(sel.min_cameras),
            "frames": frames_by_trial(result.index),
            "n_points": int(result.p2ds.shape[1]),
        },
        options={
            "only_extrinsics": bool(options.only_extrinsics),
            "holdout": float(options.holdout),
            "n_repeats": int(result.n_repeats),
            "seed": int(result.seed),
            "kept_repeat": int(result.kept_repeat),
            "n_samp_full": int(options.n_samp_full),
            "n_samp_iter": int(options.n_samp_iter),
        },
        errors=result.errors,
    )
    points_file = f"{entry.id}_points.csv"
    save_points_record(
        entry.path.parent / points_file, result.p2ds, result.index, cam_names,
        points_df, result.holdout,
    )
    points = dict(entry.metadata["points"], points_file=points_file)
    return update_metadata(entry, points=points)

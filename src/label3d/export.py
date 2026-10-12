"""Export all points to one long-format CSV (Phase 13.2), without Qt.

One row per (experiment, trial, frame, instance, bodypart, view), where view
is a camera name or "3d". Used by File > Export points... and by the
label3d-export command line tool.

The pre-flight checks (stale_experiments, uncalibrated_experiments,
pending_slp_changes) are separate, so the dialog can ask about each one;
export_points itself only refuses stale experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from label3d.geometry import mean_reprojection_errors, triangulation_mask
from label3d.points import INDEX_NAMES, LABELED_STATES, SOURCE_SLEAP_PRED
from label3d.slp_sync import MergeResult, video_cells

if TYPE_CHECKING:
    from label3d.project import Project

# Fixed column order; scripts rely on it.
EXPORT_COLUMNS = [
    *INDEX_NAMES, "view", "video_frame",
    "x", "y", "z", "state", "source", "score",
    "xr", "yr", "error",
    "n_cameras", "mean_error", "calibration",
]
VIEW_3D = "3d"


@dataclass
class ExportOptions:
    experiments: list[str] | None = None   # None = every experiment
    include_predictions: bool = True       # False drops camera points with source sleap_pred
    only_3d: bool = False                  # keep only points that have a 3D point
    write_slp: bool = True                 # also write the mapped SLP files
    output: Path | None = None             # default: exports/points_{timestamp}.csv


@dataclass
class ExportResult:
    path: Path
    n_rows: int
    # experiment -> {(state, source): camera points}
    counts: dict[str, dict[tuple[str, str], int]] = field(default_factory=dict)
    two_d_only: list[str] = field(default_factory=list)  # experiments without a calibration
    slp_files_written: int | None = None  # None when the SLP files were not written


# -- Pre-flight checks ----------------------------------------------------------

def selected_experiments(project: Project, experiments: list[str] | None) -> list[str]:
    """The experiments to export, in project order."""
    if experiments is None:
        return list(project.experiments)
    unknown = sorted(set(experiments) - set(project.experiments))
    if unknown:
        raise KeyError(f"Unknown experiment(s): {', '.join(unknown)}")
    return [name for name in project.experiments if name in experiments]


def stale_experiments(project: Project, experiments: list[str] | None = None) -> list[str]:
    """Experiments whose 3D points don't come from their active calibration."""
    return [
        name for name in selected_experiments(project, experiments)
        if project.triangulation_is_stale(name)
    ]


def uncalibrated_experiments(project: Project, experiments: list[str] | None = None) -> list[str]:
    """Experiments with no active calibration; only their 2D points are exported."""
    return [
        name for name in selected_experiments(project, experiments)
        if project.active_calibration_id(name) is None
    ]


def slp_mapping_complete(project: Project) -> bool:
    """True when every video has an SLP file, as Sync requires."""
    return project.trials_df is not None and project.slp_mapping.is_complete(
        project.trials_df, project.cameras
    )


def pending_slp_changes(project: Project) -> dict[str, MergeResult]:
    """Experiments where reading the SLP files would change the points.

    New or changed points, new scores and conflicts all count: writing the
    SLP files over them would lose them. Only reads.
    """
    results = project.read_slp_for_sync(sync=False)
    return {
        name: result for name, result in results.items()
        if result.has_changes or result.conflicts
    }


def describe_slp_changes(results: dict[str, MergeResult]) -> str:
    """One line per experiment, for a warning or an error message."""
    lines = []
    for name, r in results.items():
        parts = []
        if r.n_new:
            parts.append(f"{r.n_new} user label(s)")
        if r.n_predictions:
            parts.append(f"{r.n_predictions} prediction(s)")
        if r.n_score_updates:
            parts.append(f"{r.n_score_updates} new score(s)")
        if r.conflicts:
            parts.append(f"{len(r.conflicts)} conflict(s)")
        lines.append(f"{name}: {', '.join(parts) or 'changes'}")
    return "\n".join(lines)


# -- Export ---------------------------------------------------------------------

def default_output_path(project: Project) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return project.root / "exports" / f"points_{stamp}.csv"


def export_points(project: Project, options: ExportOptions | None = None) -> ExportResult:
    """Write the long-format CSV and, if asked and possible, the mapped SLP files.

    Raises ValueError if a selected experiment's 3D points are stale. The SLP
    files are written only when options.write_slp is set and the mapping is
    complete; check pending_slp_changes first.
    """
    options = options or ExportOptions()
    names = selected_experiments(project, options.experiments)
    stale = stale_experiments(project, names)
    if stale:
        raise ValueError(
            f"3D points are not from the active calibration in: {', '.join(stale)}. "
            "Triangulate all experiments first."
        )

    starts = {cell: start for cell, (_video, start) in
              video_cells(project.trials_df, project.cameras).items()}
    frames = []
    result = ExportResult(path=Path(), n_rows=0)
    for name in names:
        df = project.experiment_points_df(name)
        if df.empty:
            continue
        calibration = project.triangulated_with(name)
        if project.active_calibration_id(name) is None:
            result.two_d_only.append(name)
            calibration = None
        rows = experiment_rows(df, starts, calibration, options)
        cams = rows[rows["view"] != VIEW_3D]
        result.counts[name] = {
            (str(state), str(source)): int(n)
            for (state, source), n in cams.groupby(["state", "source"]).size().items()
        }
        frames.append(rows)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EXPORT_COLUMNS)
    path = Path(options.output) if options.output else default_output_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    result.path = path
    result.n_rows = len(out)

    if options.write_slp and slp_mapping_complete(project):
        result.slp_files_written = project.write_slp_files()
    return result


def experiment_rows(
    df: pd.DataFrame,
    starts: dict[tuple[str, str, str], int],
    calibration: str | None,
    options: ExportOptions,
) -> pd.DataFrame:
    """Long-format rows for one experiment's points DataFrame.

    calibration is the id of the calibration the 3D points came from, or None
    for an experiment without one: then only camera rows are written, with no
    reprojections.
    """
    cameras = [c for c in df.columns.get_level_values(0).unique() if c != VIEW_3D]
    keys = df.index.to_frame(index=False)
    keys["frame"] = keys["frame"].astype(int)
    keys["instance"] = keys["instance"].astype(int)
    n = len(df)
    order = np.arange(n)
    with_3d = calibration is not None
    p3d = df[[(VIEW_3D, c) for c in "xyz"]].to_numpy(dtype=np.float64)
    has_3d = with_3d & ~np.isnan(p3d).any(axis=1)

    parts = []
    reproj = {}
    for v, cam in enumerate(cameras):
        x = df[(cam, "x")].to_numpy(dtype=np.float64)
        y = df[(cam, "y")].to_numpy(dtype=np.float64)
        xr = df[(cam, "xr")].to_numpy(dtype=np.float64)
        yr = df[(cam, "yr")].to_numpy(dtype=np.float64)
        if not with_3d:
            xr = np.full(n, np.nan)
            yr = np.full(n, np.nan)
        reproj[cam] = np.column_stack([xr, yr])
        state = df[(cam, "state")]
        source = df[(cam, "source")]
        keep = state.isin(LABELED_STATES).to_numpy().copy()
        if not options.include_predictions:
            keep &= (source != SOURCE_SLEAP_PRED).to_numpy()
        if options.only_3d:
            keep &= has_3d
        start = np.array(
            [starts.get((e, t, cam), np.nan) for e, t in zip(keys["experiment"], keys["trial"])],
            dtype=np.float64,
        )
        part = keys[keep].assign(
            view=cam,
            video_frame=pd.array(start[keep] + keys["frame"].to_numpy()[keep], dtype="Int64"),
            x=x[keep], y=y[keep], z=np.nan,
            state=state.to_numpy()[keep], source=source.to_numpy()[keep],
            score=df[(cam, "score")].to_numpy(dtype=np.float64)[keep],
            xr=xr[keep], yr=yr[keep], error=np.hypot(x - xr, y - yr)[keep],
            n_cameras=pd.array([pd.NA] * int(keep.sum()), dtype="Int64"),
            mean_error=np.nan, calibration=None,
            _order=order[keep], _view=v,
        )
        parts.append(part)

    if with_3d:
        n_cameras = sum(triangulation_mask(df, cam).astype(int) for cam in cameras)
        mean_error = mean_reprojection_errors(df, reproj)
        part = keys[has_3d].assign(
            view=VIEW_3D,
            video_frame=pd.array([pd.NA] * int(has_3d.sum()), dtype="Int64"),
            x=p3d[has_3d, 0], y=p3d[has_3d, 1], z=p3d[has_3d, 2],
            state=None, source=None, score=np.nan,
            xr=np.nan, yr=np.nan, error=np.nan,
            n_cameras=pd.array(np.asarray(n_cameras)[has_3d], dtype="Int64"),
            mean_error=mean_error[has_3d], calibration=calibration,
            _order=order[has_3d], _view=len(cameras),
        )
        parts.append(part)

    rows = pd.concat(parts, ignore_index=True)
    rows = rows.sort_values(["_order", "_view"], kind="stable")
    return rows[EXPORT_COLUMNS].reset_index(drop=True)


def format_summary(result: ExportResult) -> str:
    """Rows written, per-experiment counts by state and source, 2D-only and SLP notes."""
    lines = [f"Wrote {result.n_rows} rows to {result.path}"]
    for name, counts in result.counts.items():
        parts = [f"{state}/{source} {n}" for (state, source), n in sorted(counts.items())]
        lines.append(f"  {name}: {', '.join(parts) or 'no points'}")
    if result.two_d_only:
        lines.append(f"2D only (no active calibration): {', '.join(result.two_d_only)}")
    if result.slp_files_written is None:
        lines.append("SLP files were not written")
    else:
        lines.append(f"Wrote {result.slp_files_written} SLP file(s)")
    return "\n".join(lines)

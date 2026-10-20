"""Export all points to one wide-format CSV (Phase 13.2), without Qt.

One row per (experiment, trial, frame, instance, bodypart), as in the points files.
Each camera has its own block of columns, prefixed with the camera name
(LateralOrtho_x.px, ...), followed by the 3D point (x.cm, y.cm, z.cm). Units
follow a dot. Used by File > Export points... and by the label3d-export
command line tool.

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
from label3d.skeleton import ProjectSkeleton
from label3d.slp_sync import MergeResult, video_cells

if TYPE_CHECKING:
    from label3d.project import Project

VIEW_3D = "3d"
UNIT_2D = "px"
UNIT_3D = "cm"
# Per-camera columns, each written as {camera}_{field}; fixed order, scripts rely on it.
CAMERA_FIELDS = [
    "video_frame", f"x.{UNIT_2D}", f"y.{UNIT_2D}", "state", "source", "score",
    f"xr.{UNIT_2D}", f"yr.{UNIT_2D}", f"error.{UNIT_2D}",
]
POINT_3D_COLUMNS = [
    f"x.{UNIT_3D}", f"y.{UNIT_3D}", f"z.{UNIT_3D}",
    "n_cameras", f"mean_error.{UNIT_2D}", "calibration",
]


def export_columns(cameras: list[str]) -> list[str]:
    """The CSV's columns, in order, for these cameras."""
    return [
        *INDEX_NAMES,
        *(f"{cam}_{name}" for cam in cameras for name in CAMERA_FIELDS),
        *POINT_3D_COLUMNS,
    ]


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
    # Why the SLP files could not be written (skeleton conflicts, mixed files)
    slp_problems: list[str] = field(default_factory=list)
    # Points left out because their camera's skeleton lacks the node:
    # (camera, node) -> count
    outside: dict[tuple[str, str], int] = field(default_factory=dict)


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
    """Write the wide-format CSV and, if asked and possible, the mapped SLP files.

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
    cameras = project.cameras
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
        rows = experiment_rows(df, cameras, starts, calibration, options, project.skeletons)
        if project.skeletons is not None:
            for key, n in project.points_outside(name, project.skeletons).items():
                result.outside[key] = result.outside.get(key, 0) + n
        result.counts[name] = state_source_counts(rows, cameras)
        frames.append(rows)

    columns = export_columns(cameras)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    path = Path(options.output) if options.output else default_output_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    result.path = path
    result.n_rows = len(out)

    if options.write_slp and slp_mapping_complete(project):
        result.slp_problems = project.slp_write_problems()
        if not result.slp_problems:
            result.slp_files_written = project.write_slp_files()
    return result


def state_source_counts(rows: pd.DataFrame, cameras: list[str]) -> dict[tuple[str, str], int]:
    """Exported camera points by (state, source)."""
    pairs = pd.concat([
        rows[[f"{cam}_state", f"{cam}_source"]].set_axis(["state", "source"], axis=1)
        for cam in cameras
    ])
    return {
        (str(state), str(source)): int(n)
        for (state, source), n in pairs.groupby(["state", "source"]).size().items()
    }


def experiment_rows(
    df: pd.DataFrame,
    cameras: list[str],
    starts: dict[tuple[str, str, str], int],
    calibration: str | None,
    options: ExportOptions,
    skeletons: ProjectSkeleton | None = None,
) -> pd.DataFrame:
    """Wide-format rows for one experiment's points DataFrame.

    cameras gives the column order. calibration is the id of the calibration
    the 3D points came from, or None for an experiment without one: then the
    3D and reprojection columns are blank. A camera's cells are blank where it
    has no point (or a prediction, without include_predictions) or, with
    skeletons, where its skeleton lacks the node; a row is dropped when no
    camera has a point and there is no 3D point. A 3D point counts only when
    at least 2 cameras' points count for it, as in triangulation.
    """
    exp_cameras = [c for c in df.columns.get_level_values(0).unique() if c != VIEW_3D]
    keys = df.index.to_frame(index=False)
    keys["frame"] = keys["frame"].astype(int)
    keys["instance"] = keys["instance"].astype(int)
    n = len(df)
    with_3d = calibration is not None
    p3d = df[[(VIEW_3D, c) for c in "xyz"]].to_numpy(dtype=np.float64)
    has_3d = with_3d & ~np.isnan(p3d).any(axis=1)

    columns = {}
    reproj = {}
    any_camera = np.zeros(n, dtype=bool)
    for cam in exp_cameras:
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
        if skeletons is not None:
            keep &= skeletons.available_mask(df.index, cam)
        if not options.include_predictions:
            keep &= (source != SOURCE_SLEAP_PRED).to_numpy()
        any_camera |= keep
        start = np.array(
            [starts.get((e, t, cam), np.nan) for e, t in zip(keys["experiment"], keys["trial"])],
            dtype=np.float64,
        )
        values = {
            "video_frame": start + keys["frame"].to_numpy(),
            f"x.{UNIT_2D}": x, f"y.{UNIT_2D}": y,
            "state": state.to_numpy(), "source": source.to_numpy(),
            "score": df[(cam, "score")].to_numpy(dtype=np.float64),
            f"xr.{UNIT_2D}": xr, f"yr.{UNIT_2D}": yr,
            f"error.{UNIT_2D}": np.hypot(x - xr, y - yr),
        }
        for name, value in values.items():
            blank = None if value.dtype == object else np.nan
            columns[f"{cam}_{name}"] = np.where(keep, value, blank)
        columns[f"{cam}_video_frame"] = pd.array(columns[f"{cam}_video_frame"], dtype="Int64")

    if with_3d:
        n_cameras = np.asarray(sum(
            triangulation_mask(df, cam, skeletons).astype(int) for cam in exp_cameras
        ))
        mean_error = mean_reprojection_errors(df, reproj, skeletons)
        has_3d &= n_cameras >= 2
    else:
        n_cameras = np.zeros(n, dtype=int)
        mean_error = np.full(n, np.nan)
    for i, axis in enumerate("xyz"):
        columns[f"{axis}.{UNIT_3D}"] = np.where(has_3d, p3d[:, i], np.nan)
    columns["n_cameras"] = pd.array(np.where(has_3d, n_cameras, np.nan), dtype="Int64")
    columns[f"mean_error.{UNIT_2D}"] = np.where(has_3d, mean_error, np.nan)
    columns["calibration"] = np.where(has_3d, calibration, None)

    keep_row = any_camera | has_3d
    if options.only_3d:
        keep_row &= has_3d
    rows = keys.assign(**columns)[keep_row]
    return rows.reindex(columns=export_columns(cameras)).reset_index(drop=True)


def format_summary(result: ExportResult) -> str:
    """Rows written, per-experiment counts by state and source, 2D-only and SLP notes."""
    lines = [f"Wrote {result.n_rows} rows to {result.path}"]
    for name, counts in result.counts.items():
        parts = [f"{state}/{source} {n}" for (state, source), n in sorted(counts.items())]
        lines.append(f"  {name}: {', '.join(parts) or 'no points'}")
    if result.two_d_only:
        lines.append(f"2D only (no active calibration): {', '.join(result.two_d_only)}")
    if result.outside:
        parts = ", ".join(
            f"{node} in {cam} ({n})" for (cam, node), n in sorted(result.outside.items())
        )
        lines.append(
            f"Left out {sum(result.outside.values())} point(s) whose camera's skeleton "
            f"lacks the node: {parts}"
        )
    if result.slp_problems:
        lines.append("SLP files were not written: the skeletons need fixing first")
        lines.extend(f"  {problem}" for problem in result.slp_problems)
    elif result.slp_files_written is None:
        lines.append("SLP files were not written")
    else:
        lines.append(f"Wrote {result.slp_files_written} SLP file(s)")
    return "\n".join(lines)

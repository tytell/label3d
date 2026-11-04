"""Adding an axis take to a project after setup (Phase 16.4).

An axis take is a recording, often one still image per camera, of an object
with marked x, y, z axes. After setup it can be added in two ways, to the
active experiment:

- add_from_files copies one image or video per camera into
  calibrations/{camera}/ (the originals are left where they are);
- add_from_frame extracts one frame of a calibration take or trial as a still
  per camera, each camera's video read at its start offset plus the frame,
  and records where it came from under [alignment.{exp}.extracted].

Either adds a row of type axis to trials.csv (the old file is kept once as
trials.csv.bak). File > Add axis take... uses these.
"""

from __future__ import annotations

import filecmp
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from label3d import alignment as alignment_record
from label3d.alignment import AlignmentSource
from label3d.calibration_registry import camera_dir
from label3d.frames import shown_frame
from label3d.project import (
    Project,
    TrialInfo,
    _is_blank_cell,
    check_trial_videos,
    read_trials_csv,
    write_trials_csv,
)
from label3d.setup_project import VIDEO_EXTENSIONS, axis_still_name, copy_file, identify_camera
from label3d.trial_types import AXIS, CALIBRATION, TRIAL, TYPE_COLUMN, row_types
from label3d.video import IMAGE_EXTENSIONS, extract_frame, frame_count

DEFAULT_TAKE = "Align"


class AxisTakeError(ValueError):
    """The axis take can't be added; the message says why."""


@dataclass
class FileMatch:
    """Files matched to cameras by name."""

    by_camera: dict[str, Path] = field(default_factory=dict)
    unmatched: list[Path] = field(default_factory=list)   # no camera, or a second file for one
    take: str = DEFAULT_TAKE                               # suggested take name


def media_files(folder: Path) -> list[Path]:
    """The images and videos directly in a folder."""
    return sorted(
        f for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    )


def match_files(
    files: list[Path], cameras: list[str], aliases: dict[str, list[str]] | None = None
) -> FileMatch:
    """Match files to cameras by name or alias, as setup does; suggest a take name."""
    match = FileMatch()
    for f in sorted(Path(f) for f in files):
        camera = identify_camera(f.name, cameras, aliases)
        if camera is None or camera in match.by_camera:
            match.unmatched.append(f)
        else:
            match.by_camera[camera] = f
    match.take = suggest_take_name(match.by_camera, aliases)
    return match


def suggest_take_name(
    by_camera: dict[str, Path], aliases: dict[str, list[str]] | None = None
) -> str:
    """What the file names share once the camera's name is taken out, else DEFAULT_TAKE.

    PS18_Align_cam1.png and PS18_Align_cam2.png give PS18_Align.
    """
    stems = set()
    for camera, path in by_camera.items():
        stem = path.stem
        for name in sorted([camera, *(aliases or {}).get(camera, [])], key=len, reverse=True):
            stem = re.sub(re.escape(name), "", stem, flags=re.IGNORECASE)
        stems.add(re.sub(r"[\s_.-]+", "_", stem).strip("_"))
    if len(stems) == 1 and (name := stems.pop()):
        return name
    return DEFAULT_TAKE


def taken_names(project: Project) -> set[str]:
    """Names of the active experiment's trials and takes."""
    exp = project.experiments[project.active_experiment]
    return {t.name for t in [*exp.trials, *exp.calibration_takes, *exp.axis_takes]}


def unique_take_name(project: Project, base: str = DEFAULT_TAKE) -> str:
    """base, or base2, base3, … if the experiment already has that name."""
    taken = taken_names(project)
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base}{n}"
    return name


def check_take_name(project: Project, take: str) -> None:
    if not take.strip():
        raise AxisTakeError("Give the axis take a name.")
    if take != take.strip() or "/" in take or "\\" in take:
        raise AxisTakeError(f"'{take}' can't be used as a take name.")
    if take in taken_names(project):
        raise AxisTakeError(
            f"Experiment '{project.active_experiment}' already has a trial or take named '{take}'."
        )


# -- Adding ------------------------------------------------------------------


def add_from_files(project: Project, take: str, files: dict[str, Path]) -> None:
    """Copy one image or video per camera into calibrations/{camera}/ as a new axis take.

    Nothing is changed when the take can't be added: a file already in
    calibrations/{camera}/ under the same name must be the same file.
    """
    check_take_name(project, take)
    files = {cam: Path(f) for cam, f in files.items()}
    _check_cameras(project, files)
    exp_dir = project.experiments[project.active_experiment].root
    copies = {}
    for cam, src in files.items():
        if not src.is_file():
            raise AxisTakeError(f"{src} doesn't exist.")
        dst = camera_dir(exp_dir, cam) / src.name
        if dst.exists() and not filecmp.cmp(src, dst, shallow=False):
            raise AxisTakeError(
                f"calibrations/{cam}/ already has a different {src.name}; rename the file."
            )
        copies[cam] = (src, dst)
    for cam, (src, dst) in copies.items():
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_file(str(src), str(dst))
    _append_axis_row(project, take, {cam: dst.name for cam, (_src, dst) in copies.items()})


def add_from_frame(project: Project, take: str, source: str, frame: int) -> AlignmentSource:
    """Extract one frame (0-based, synchronized) of a calibration take or trial as an axis take.

    Each camera's video is read at its start offset plus frame. The stills are
    {take}_{camera}.png in calibrations/{camera}/, and where they came from is
    recorded in [alignment.{exp}.extracted.{take}]. Returns that record.
    """
    check_take_name(project, take)
    info, source_type = _find_source(project, source)
    exp_dir = project.experiments[project.active_experiment].root
    plan = {}
    for cam, (filename, start, _end) in info.camera_videos.items():
        if _is_blank_cell(filename):
            continue
        video = project.video_path(source, cam)
        if start + frame >= frame_count(video):
            raise AxisTakeError(f"'{source}' has no frame {shown_frame(frame)} for {cam}.")
        still = camera_dir(exp_dir, cam) / axis_still_name(take, cam)
        if still.exists():
            raise AxisTakeError(f"calibrations/{cam}/{still.name} already exists.")
        plan[cam] = (video, start + frame, still)
    _check_cameras(project, plan)
    for video, video_frame, still in plan.values():
        extract_frame(video, video_frame, still)

    record = AlignmentSource(
        take=source, type=source_type, frame=frame,
        videos={cam: (video.name, video_frame) for cam, (video, video_frame, _) in plan.items()},
    )
    project.save_config_sections({
        alignment_record.TABLE: alignment_record.with_extracted(
            project.config, project.active_experiment, take, record
        )
    })
    _append_axis_row(project, take, {cam: still.name for cam, (_, _, still) in plan.items()})
    return record


def frame_sources(project: Project) -> list[tuple[str, str]]:
    """(name, type) of the active experiment's calibration takes, then its trials."""
    return (
        [(t.name, CALIBRATION) for t in project.calibration_takes]
        + [(t.name, TRIAL) for t in project.trials]
    )


def source_length(project: Project, source: str) -> int:
    """Synchronized frames of a take or trial: the shortest camera from its start."""
    info, _ = _find_source(project, source)
    lengths = []
    for cam, (filename, start, end) in info.camera_videos.items():
        if _is_blank_cell(filename):
            continue
        last = end if end is not None else frame_count(project.video_path(source, cam)) - 1
        lengths.append(last - start + 1)
    return max(min(lengths), 0) if lengths else 0


def _find_source(project: Project, source: str) -> tuple[TrialInfo, str]:
    for info in project.calibration_takes:
        if info.name == source:
            return info, CALIBRATION
    for info in project.trials:
        if info.name == source:
            return info, TRIAL
    raise AxisTakeError(
        f"Experiment '{project.active_experiment}' has no calibration take or trial '{source}'."
    )


def _check_cameras(project: Project, by_camera: dict) -> None:
    unknown = [cam for cam in by_camera if cam not in project.cameras]
    if unknown:
        raise AxisTakeError(f"Not cameras of this project: {', '.join(unknown)}.")
    if len(by_camera) < 2:
        raise AxisTakeError("An axis take needs at least two cameras to triangulate its points.")


def _append_axis_row(project: Project, take: str, cells: dict[str, str]) -> None:
    """Add an axis row to trials.csv, measure its files and refresh the project's takes."""
    path = project.trials_csv_path
    trials_df = read_trials_csv(path)
    if TYPE_COLUMN not in trials_df.columns:
        position = trials_df.columns.get_loc("trial") + 1
        trials_df.insert(position, TYPE_COLUMN, row_types(trials_df))
    row: dict = {"trial": take, TYPE_COLUMN: AXIS}
    if "experiment" in trials_df.columns:
        row["experiment"] = project.active_experiment
    for cam in project.cameras:
        row[cam] = cells.get(cam, "")
        row[f"{cam}_start"] = 0 if cam in cells else pd.NA
    trials_df = pd.concat([trials_df, pd.DataFrame([row])], ignore_index=True)
    video_roots = {
        str(name): Path(str(value)).expanduser()
        for name, value in project.config.get("video_roots", {}).items()
    }
    check_trial_videos(trials_df, project.root, video_roots, project.cameras)
    write_trials_csv(path, trials_df)
    project.refresh_takes()

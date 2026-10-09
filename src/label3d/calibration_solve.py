"""Charuco calibration solves, without Qt.

Shared by the Calibration tab's worker and the refinement (Phase 12): the
takes and their frame ranges, aniposelib's calibrate_rows with repeated runs,
and the [metadata] that records how a charuco solution was made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import Camera, CameraGroup

from label3d.calibration_compare import run_spread
from label3d.calibration_io import compute_frame_errors, corner_detections
from label3d.calibration_registry import camera_dir

if TYPE_CHECKING:
    from label3d.project import Project


@dataclass
class TakeVideo:
    """One camera's video within a calibration take, with optional trimming."""

    path: Path
    start: int = 0
    end: int | None = None


@dataclass
class CalibrationTake:
    """One synchronized calibration recording across cameras.

    start/end bound the synchronized frames used for calibration (inclusive;
    end None means through the end of the take).
    """

    name: str
    videos: dict[str, TakeVideo] = field(default_factory=dict)  # camera -> video
    start: int = 0
    end: int | None = None

    def contains(self, frame: int) -> bool:
        return frame >= self.start and (self.end is None or frame <= self.end)


JOINT_TAKE = "All takes (joint)"


@dataclass
class TakeCalibration:
    """One calibration solution — a single take, or all takes jointly."""

    name: str
    cgroup: object
    error: float
    frame_errors: dict
    n_rows: int
    # Largest (rotation_deg, baseline_pct) between the repeated runs.
    spread: tuple[float, float] = (0.0, 0.0)


def build_board(bp: dict) -> CharucoBoard:
    return CharucoBoard(
        squaresX=bp["squares_x"],
        squaresY=bp["squares_y"],
        square_length=bp["square_length"],
        marker_length=bp["marker_length"],
        marker_bits=bp.get("marker_bits", 4),
        dict_size=bp.get("dict_size", 50),
    )


def calibrate_rows(
    cam_names: list[str],
    cam_sizes: dict[str, tuple[int, int]],
    rows_by_cam: dict[str, list],
    board: CharucoBoard,
) -> tuple[object, float, dict]:
    """Run bundle adjustment on one set of rows. Returns (cgroup, error, errors)."""
    cameras = [Camera(name=n, size=cam_sizes.get(n)) for n in cam_names]
    cgroup = CameraGroup(cameras)
    rows_list = [rows_by_cam.get(cam, []) for cam in cam_names]
    error = cgroup.calibrate_rows(rows_list, board, verbose=False)
    _, frame_errors = compute_frame_errors(cgroup, board, rows_list, cam_names)
    return cgroup, float(error), frame_errors


def rows_for_take(rows_by_cam: dict[str, list], take: str) -> dict[str, list]:
    """Subset of detection rows belonging to one take."""
    return {
        cam: [
            r for r in rows
            if isinstance(r.get("framenum"), tuple) and r["framenum"][0] == take
        ]
        for cam, rows in rows_by_cam.items()
    }


def rows_in_range(
    rows_by_cam: dict[str, list], takes: list[CalibrationTake]
) -> dict[str, list]:
    """Drop detection rows outside their take's start/end range.

    Rows from a take not in `takes` are kept — there is no range to apply.
    """
    by_name = {take.name: take for take in takes}

    def keep(row: dict) -> bool:
        framenum = row.get("framenum")
        if not isinstance(framenum, tuple) or framenum[0] not in by_name:
            return True
        return by_name[framenum[0]].contains(framenum[1])

    return {cam: [r for r in rows if keep(r)] for cam, rows in rows_by_cam.items()}


def calibrate_repeated(
    cam_names: list[str],
    cam_sizes: dict[str, tuple[int, int]],
    rows: dict[str, list],
    board: CharucoBoard,
    n_repeats: int,
    name: str,
    progress: Callable[[str], None] | None = None,
) -> TakeCalibration:
    """Calibrate `rows` n_repeats times and keep the lowest-error run.

    aniposelib's bundle adjustment subsamples at random, so the spread between
    runs says how much of any disagreement is just noise.
    """
    say = progress or (lambda _msg: None)
    n_repeats = max(1, n_repeats)
    n_rows = sum(len(r) for r in rows.values())
    say(f"  Calibrating {name} ({n_rows} detections)...")
    runs = []
    for i in range(n_repeats):
        try:
            cgroup, error, frame_errors = calibrate_rows(cam_names, cam_sizes, rows, board)
        except Exception as exc:
            say(f"    run {i + 1}/{n_repeats} failed: {exc}")
            continue
        say(f"    run {i + 1}/{n_repeats}: mean error {error:.3f} px")
        runs.append((error, cgroup, frame_errors))
    if not runs:
        raise RuntimeError(f"every calibration run of {name} failed")

    error, cgroup, frame_errors = min(runs, key=lambda r: r[0])
    spread = run_spread([r[1] for r in runs])
    if len(runs) > 1:
        say(
            f"    kept {error:.3f} px; run-to-run spread "
            f"{spread[0]:.2f} deg rotation, {spread[1]:.1f}% baseline"
        )
    return TakeCalibration(name, cgroup, error, frame_errors, n_rows, spread)


def take_range(config: dict, experiment: str, take: str) -> tuple[int, int | None]:
    """A take's calibration frame range from project.toml (end None = to the end).

    Stored as [calibration.ranges."<experiment>"."<take>"] with start/end in
    synchronized frames.
    """
    ranges = config.get("calibration", {}).get("ranges", {})
    entry = ranges.get(experiment, {}).get(take, {})
    end = entry.get("end")
    return int(entry.get("start", 0)), (int(end) if end is not None else None)


def build_takes(project: Project) -> list[CalibrationTake]:
    """Build detection takes from the is_calibration rows of trials.csv."""
    takes: list[CalibrationTake] = []
    for info in project.calibration_takes:
        videos: dict[str, TakeVideo] = {}
        for cam, (filename, start, end) in info.camera_videos.items():
            if not filename or filename.lower() == "nan":
                continue
            videos[cam] = TakeVideo(
                path=project.video_path(info.name, cam), start=start, end=end
            )
        start, end = take_range(project.config, project.active_experiment, info.name)
        takes.append(CalibrationTake(name=info.name, videos=videos, start=start, end=end))
    return takes


def take_camera_sizes(
    takes: list[CalibrationTake], cam_names: list[str]
) -> dict[str, tuple[int, int]]:
    """(width, height) of each camera, from the first of its take videos that opens."""
    sizes: dict[str, tuple[int, int]] = {}
    for take in takes:
        for cam in cam_names:
            tv = take.videos.get(cam)
            if cam in sizes or tv is None or not tv.path.is_file():
                continue
            cap = cv2.VideoCapture(str(tv.path))
            if cap.isOpened():
                sizes[cam] = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            cap.release()
    return sizes


def take_ranges(takes: list[CalibrationTake], names: list[str]) -> dict[str, dict]:
    """{take: {start, end?}} for the named takes, as stored in [metadata.charuco]."""
    ranges: dict[str, dict] = {}
    for take in takes:
        if take.name in names:
            ranges[take.name] = {"start": take.start}
            if take.end is not None:
                ranges[take.name]["end"] = take.end
    return ranges


def take_detections(exp_dir: Path, cam_names: list[str], names: list[str]) -> dict[str, str]:
    """take -> latest detection timestamp over the cameras, for the named takes."""
    detected: dict[str, str] = {}
    for cam in cam_names:
        for take, stamp in corner_detections(camera_dir(exp_dir, cam)).items():
            if take in names:
                detected[take] = max(stamp, detected.get(take, stamp))
    return detected


def charuco_metadata(
    result: TakeCalibration,
    names: list[str],
    takes: list[CalibrationTake],
    exp_dir: Path,
    cam_names: list[str],
    board_params: dict,
    frame_skip: int,
) -> dict:
    """[metadata] for a charuco entry made from takes `names`: enough to redo the solve."""
    return {
        "charuco_takes": list(names),
        "charuco": {
            "board": dict(board_params),
            "frame_skip": frame_skip,
            "take_ranges": take_ranges(takes, names),
            "detected": take_detections(exp_dir, cam_names, names),
            "n_rows": result.n_rows,
        },
        "errors": {
            "charuco": float(result.error),
            "spread_rotation_deg": float(result.spread[0]),
            "spread_baseline_pct": float(result.spread[1]),
        },
    }

"""Background worker for Charuco corner detection and camera calibration.

Workflow:
  1. DetectionWorker: detects Charuco corners in each calibration take's
     videos (calibrations/{cam}/), emits per-camera progress.
  2. MultiTakeCalibrationWorker: calibrates each take, then all takes jointly.
  3. RefineWorker: refines a calibration with labelled points (Phase 12).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

import logging
_log = logging.getLogger(__name__)

from aniposelib.boards import merge_rows
from aniposelib.cameras import Camera, CameraGroup

from label3d.calibration_io import compute_frame_errors
from label3d.calibration_refine import (
    RefineOptions,
    RefineResult,
    calibrate_takes,
    charuco_points,
    charuco_rows,
    refine,
)
from label3d.calibration_registry import CalibrationEntry, load_calibration
from label3d.calibration_solve import (
    JOINT_TAKE,
    CalibrationTake,
    TakeCalibration,
    TakeVideo,
    build_board,
    calibrate_repeated,
    rows_for_take,
    take_camera_sizes,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".cine"}


def calibration_images_error(cam_name: str, n_images: int) -> str:
    """Message for the deprecated image-based calibration path."""
    return (
        f"Image-based calibration is deprecated: camera '{cam_name}' has "
        f"{n_images} calibration image(s) and no video. Calibration must come "
        "from videos, which are matched into takes in trials.csv."
    )


def _read_media_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) of an image or video file, or None on failure."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return (w, h)
    elif ext in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (w, h)
    return None


def _draw_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw detected Charuco corners on a BGR image copy."""
    out = image.copy()
    if corners is None or len(corners) == 0:
        return out
    pts = np.array(corners).reshape(-1, 2)
    for x, y in pts:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), -1)
    return out



class DetectionWorker(QThread):
    """Detects Charuco corners in the videos of each calibration take.

    Takes come from the is_calibration rows of trials.csv, so cross-camera
    correspondence is by take *name*. Detection rows are keyed
    ``(take_name, frame_in_video)``, which is what aniposelib's merge_rows
    groups on — a camera missing from a take leaves a gap instead of silently
    shifting every later index, the way positional folder ordering did.

    Emits:
        progress(str)                         — log message
        camera_done(cam, n)                   — camera name, detection count
        frame_progress(label, current, total)  — label, frames processed in current video
        no_corners(cam, take)                 — a take's video had no corners in any
                                                frame; detection waits for resume()
        cancelled()                           — resume(False) stopped detection
        finished(all_rows, cam_sizes)
            all_rows:  dict[cam_name, list[row]]        — detection rows per camera
            cam_sizes: dict[cam_name, (w, h)]           — frame size per camera
        error_occurred(str)                   — if a fatal error happens

    ``skip`` strides video input: exactly every Nth frame is decoded and
    searched for corners.

    With ``ask_on_empty`` a take with no corners in any frame emits
    ``no_corners`` and blocks until ``resume`` is called; otherwise it only
    logs a warning.
    """

    progress = Signal(str)
    camera_done = Signal(str, int)      # cam_name, n_dets
    frame_progress = Signal(str, int, int)  # label, current_frame, total_frames
    no_corners = Signal(str, str)       # cam_name, take_name
    cancelled = Signal()
    finished = Signal(dict, dict)               # all_rows, cam_sizes
    error_occurred = Signal(str)

    def __init__(
        self,
        cam_names: list[str],
        board_params: dict,
        takes: list[CalibrationTake],
        skip: int = 20,
        ask_on_empty: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._cam_names = cam_names
        self._board_params = board_params
        self._takes = takes
        self._skip = skip
        self._ask_on_empty = ask_on_empty
        self._resume = threading.Event()
        self._continue = True

    def resume(self, keep_going: bool) -> None:
        """Answer a ``no_corners`` pause: carry on, or stop and keep nothing."""
        self._continue = keep_going
        self._resume.set()

    def run(self):
        try:
            bp = self._board_params
            board = build_board(bp)

            all_rows: dict[str, list] = {cam: [] for cam in self._cam_names}
            cam_sizes: dict[str, tuple[int, int]] = {}

            if not self._takes:
                self.progress.emit("  No calibration takes found in trials.csv")
                self.finished.emit(all_rows, cam_sizes)
                return

            for take in self._takes:
                missing = [c for c in self._cam_names if c not in take.videos]
                if missing:
                    self.progress.emit(
                        f"  take '{take.name}': missing cameras {', '.join(missing)}"
                    )

                for cam_name in self._cam_names:
                    tv = take.videos.get(cam_name)
                    if tv is None:
                        continue
                    if not tv.path.is_file():
                        raise FileNotFoundError(f'missing video file "{tv.path}"')

                    if cam_name not in cam_sizes:
                        size = _read_media_size(tv.path)
                        if size is not None:
                            cam_sizes[cam_name] = size

                    self.progress.emit(f"  {cam_name} [{take.name}]: detecting corners...")
                    rows, n_searched = self._detect_video(board, tv, take)
                    all_rows[cam_name].extend(rows)

                    n_corners = sum(len(r["corners"]) for r in rows)
                    n_found = len(rows)
                    self.progress.emit(
                        f"  {cam_name} [{take.name}]: {n_corners} corners in "
                        f"{n_found}/{n_searched} frames "
                        f"({n_searched - n_found}/{n_searched} frames with no corners)"
                    )
                    if n_found == 0:
                        self.progress.emit(
                            f"  Warning: no corners detected in any frame of "
                            f"{cam_name} [{take.name}]"
                        )
                        if self._ask_on_empty:
                            self._resume.clear()
                            self.no_corners.emit(cam_name, take.name)
                            self._resume.wait()
                            if not self._continue:
                                self.cancelled.emit()
                                return

            for cam_name in self._cam_names:
                n = len(all_rows[cam_name])
                self.progress.emit(f"  {cam_name}: {n} detections")
                self.camera_done.emit(cam_name, n)

            self.finished.emit(all_rows, cam_sizes)

        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _detect_video(
        self, board, tv: TakeVideo, take: CalibrationTake
    ) -> tuple[list, int]:
        """Detect corners on every ``self._skip``-th frame of one take's video.

        Returns the detection rows and the number of frames searched.

        Frame numbers in the returned rows are *synchronized* take frames (video
        frame minus the camera's start offset), so merge_rows pairs frames that
        were recorded at the same instant even when cameras' offsets differ.
        Only the take's start/end range is scanned, striding from its start.
        """
        take_name = take.name
        cap = cv2.VideoCapture(str(tv.path))
        if not cap.isOpened():
            raise FileNotFoundError(f'missing video file "{tv.path}"')

        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        offset = max(0, tv.start)
        start = offset + take.start
        last = tv.end if tv.end is not None else (n_total - 1 if n_total > 0 else None)
        if take.end is not None:
            last = offset + take.end if last is None else min(last, offset + take.end)
        span = (last - start + 1) if last is not None and last >= start else 0
        update_every = max(1, span // 200) if span else 30
        label = f"Detecting corners: {take_name} / {tv.path.name}"

        self.frame_progress.emit(label, 0, span)

        v_rows: list = []
        n_searched = 0
        framenum = 0
        last_emit = -update_every
        while True:
            if last is not None and framenum > last:
                break
            if framenum < start or (framenum - start) % self._skip != 0:
                # Demux without decoding: skipped frames are never converted.
                if not cap.grab():
                    break
                framenum += 1
                continue
            ret, frame = cap.read()
            if not ret:
                break
            if framenum - last_emit >= update_every:
                self.frame_progress.emit(label, framenum - start, span)
                last_emit = framenum
            corners, ids = board.detect_image(frame)
            n_searched += 1
            if corners is not None and len(corners) > 0:
                v_rows.append(
                    {
                        "framenum": (take_name, framenum - offset),
                        "corners": corners,
                        "ids": ids,
                    }
                )
            framenum += 1

        cap.release()
        end = span or max(0, framenum - start)
        self.frame_progress.emit(label, end, end)
        return board.fill_points_rows(v_rows), n_searched




class MultiTakeCalibrationWorker(QThread):
    """Calibrates each take on its own, then all takes jointly.

    Comparing the per-take solutions is how a camera that moved during the
    experiment gets caught: the joint solution would otherwise average the
    before and after geometry into something that fits neither. With a single
    take the joint solution would be the same rows again, so it is skipped.

    aniposelib's bundle adjustment subsamples at random, so each solution is
    calibrated ``n_repeats`` times and the lowest-error run kept; the spread
    between runs tells drift checks how much disagreement is just noise.

    Emits:
        progress(str)
        finished(list[TakeCalibration])  — per take, joint last (if >1 take)
        error_occurred(str)
    """

    progress = Signal(str)
    finished = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        cam_names: list[str],
        cam_sizes: dict[str, tuple[int, int]],
        all_rows: dict[str, list],
        board_params: dict,
        take_names: list[str],
        n_repeats: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self._cam_names = cam_names
        self._cam_sizes = cam_sizes
        self._all_rows = all_rows
        self._board_params = board_params
        self._take_names = take_names
        self._n_repeats = max(1, n_repeats)

    def _calibrate_repeated(
        self, name: str, rows: dict[str, list], board
    ) -> TakeCalibration:
        """Calibrate `rows` n_repeats times; keep the lowest-error run."""
        return calibrate_repeated(
            self._cam_names, self._cam_sizes, rows, board, self._n_repeats, name,
            progress=self.progress.emit,
        )

    def run(self):
        try:
            board = build_board(self._board_params)
            results: list[TakeCalibration] = []

            for take in self._take_names:
                rows = rows_for_take(self._all_rows, take)
                if sum(len(r) for r in rows.values()) == 0:
                    self.progress.emit(f"  take '{take}': no detections, skipping")
                    continue
                try:
                    results.append(
                        self._calibrate_repeated(f"take '{take}'", rows, board)
                    )
                    results[-1].name = take
                except Exception as exc:
                    self.progress.emit(f"  take '{take}' failed: {exc}")

            if len(self._take_names) > 1:
                joint = self._calibrate_repeated("all takes jointly", self._all_rows, board)
                joint.name = JOINT_TAKE
                results.append(joint)

            self.finished.emit(results)

        except Exception as exc:
            self.error_occurred.emit(str(exc))


class CalibrationWorker(QThread):
    """Runs aniposelib bundle adjustment and computes per-corner errors.

    Emits:
        progress(str)                        — log message
        finished(CameraGroup, float, dict)   — calibrated group, mean reprojection
                                               error (px), frame_errors dict
            frame_errors: dict[int, dict[str, ndarray]]
                merged_frame_idx → cam_name → per-corner error array (NaN = undetected)
        error_occurred(str)                  — on failure
    """

    progress = Signal(str)
    finished = Signal(object, float, object)  # CameraGroup, mean_error_px, frame_errors
    error_occurred = Signal(str)

    def __init__(
        self,
        cam_names: list[str],
        cam_sizes: dict[str, tuple[int, int]],
        all_rows: dict[str, list],
        board_params: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._cam_names = cam_names
        self._cam_sizes = cam_sizes
        self._all_rows = all_rows
        self._board_params = board_params

    def run(self):
        try:
            bp = self._board_params
            board = build_board(bp)

            cameras = []
            for cam_name in self._cam_names:
                size = self._cam_sizes.get(cam_name)
                cam = Camera(name=cam_name, size=size)
                cameras.append(cam)

            cgroup = CameraGroup(cameras)

            rows_list = [self._all_rows.get(cam, []) for cam in self._cam_names]

            self.progress.emit("Running bundle adjustment...")
            error = cgroup.calibrate_rows(rows_list, board, verbose=False)
            self.progress.emit(f"Calibration complete — mean error: {error:.3f} px")

            # ── Per-corner reprojection errors ─────────────────────────────
            self.progress.emit("Computing per-corner reprojection errors...")
            _, frame_errors = compute_frame_errors(
                cgroup, board, rows_list, self._cam_names
            )

            self.finished.emit(cgroup, float(error), frame_errors)

        except Exception as exc:
            self.error_occurred.emit(str(exc))



@dataclass
class RefineOutcome:
    """What RefineWorker produced, ready for the dialog to show and save.

    parent is the registry entry the refinement started from, or None when
    Stage 1 calibrated the takes afresh; then stage1 holds that new solution,
    which is saved before the refinement.
    """

    result: RefineResult
    parent: CalibrationEntry | None
    stage1: TakeCalibration | None
    stage1_takes: list[str]


class RefineWorker(QThread):
    """Runs a calibration refinement (Phase 12) off the UI thread.

    Stage 1 either loads an existing entry or calibrates the chosen takes
    jointly. Stage 2 adjusts it with the selected points, aligns the result
    back onto it and measures errors on held-out frames and charuco corners.

    Emits:
        progress(str)
        finished(RefineOutcome)
        error_occurred(str)
    """

    progress = Signal(str)
    finished = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        exp_dir: Path,
        cam_names: list[str],
        p2ds: np.ndarray,
        index,
        options: RefineOptions,
        start_entry: CalibrationEntry | None = None,
        takes: list[CalibrationTake] | None = None,
        take_names: list[str] | None = None,
        board_params: dict | None = None,
        n_stage1_repeats: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self._exp_dir = exp_dir
        self._cam_names = cam_names
        self._p2ds = p2ds
        self._index = index
        self._options = options
        self._start_entry = start_entry
        self._takes = takes or []
        self._take_names = take_names or []
        self._board_params = board_params or {}
        self._n_stage1_repeats = n_stage1_repeats

    def run(self):
        try:
            stage1 = None
            if self._start_entry is not None:
                self.progress.emit(f"Stage 1: starting from {self._start_entry.label}")
                start = load_calibration(self._start_entry)
                names = self._start_entry.charuco_takes
            else:
                names = self._take_names
                self.progress.emit(f"Stage 1: calibrating {', '.join(names)}")
                rows = charuco_rows(self._exp_dir, self._cam_names, self._takes, names)
                sizes = take_camera_sizes(self._takes, self._cam_names)
                stage1 = calibrate_takes(
                    rows, self._cam_names, sizes, self._board_params,
                    self._n_stage1_repeats, self.progress.emit,
                )
                start = stage1.cgroup

            # Points follow the order of the calibration's cameras.
            order = start.get_names()
            if sorted(order) != sorted(self._cam_names):
                raise ValueError(
                    f"the calibration has cameras {order}, the project has {self._cam_names}"
                )
            p2ds = self._p2ds[[self._cam_names.index(n) for n in order]]

            charuco = None
            if names:
                rows = charuco_rows(self._exp_dir, order, self._takes, names)
                charuco = charuco_points(rows, order)

            self.progress.emit("Stage 2: bundle adjustment with the selected points")
            result = refine(
                start, p2ds, self._index, self._options,
                charuco_p2ds=charuco, progress=self.progress.emit,
            )
            self.finished.emit(RefineOutcome(result, self._start_entry, stage1, list(names)))
        except Exception as exc:
            _log.exception("Refinement failed")
            self.error_occurred.emit(str(exc))

"""Background worker for Charuco corner detection and camera calibration.

Two-phase workflow:
  1. DetectionWorker: scans calibration/{cam}/ folders, detects Charuco corners
     per camera using images or video, emits per-camera progress.
  2. CalibrationWorker: given detected rows, runs aniposelib bundle adjustment,
     computes per-corner reprojection errors, emits results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

import logging
_log = logging.getLogger(__name__)

from aniposelib.boards import CharucoBoard, merge_rows
from aniposelib.cameras import Camera, CameraGroup

from label3d.calibration_io import compute_frame_errors

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


@dataclass
class TakeVideo:
    """One camera's video within a calibration take, with optional trimming."""

    path: Path
    start: int = 0
    end: int | None = None


@dataclass
class CalibrationTake:
    """One synchronized calibration recording across cameras."""

    name: str
    videos: dict[str, TakeVideo] = field(default_factory=dict)  # camera -> video


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
        finished(all_rows, cam_sizes, cam_media)
            all_rows:  dict[cam_name, list[row]]        — detection rows per camera
            cam_sizes: dict[cam_name, (w, h)]           — frame size per camera
            cam_media: dict[cam_name, dict[take, Path]] — video per camera per take
        error_occurred(str)                   — if a fatal error happens

    ``skip`` strides video input: exactly every Nth frame is decoded and
    searched for corners.
    """

    progress = Signal(str)
    camera_done = Signal(str, int)      # cam_name, n_dets
    frame_progress = Signal(str, int, int)  # label, current_frame, total_frames
    finished = Signal(dict, dict, dict)         # all_rows, cam_sizes, cam_media
    error_occurred = Signal(str)

    def __init__(
        self,
        cam_names: list[str],
        board_params: dict,
        takes: list[CalibrationTake],
        skip: int = 20,
        parent=None,
    ):
        super().__init__(parent)
        self._cam_names = cam_names
        self._board_params = board_params
        self._takes = takes
        self._skip = skip

    def run(self):
        try:
            bp = self._board_params
            board = CharucoBoard(
                squaresX=bp["squares_x"],
                squaresY=bp["squares_y"],
                square_length=bp["square_length"],
                marker_length=bp["marker_length"],
                marker_bits=bp.get("marker_bits", 4),
                dict_size=bp.get("dict_size", 50),
            )

            all_rows: dict[str, list] = {cam: [] for cam in self._cam_names}
            cam_sizes: dict[str, tuple[int, int]] = {}
            cam_media: dict[str, dict[str, Path]] = {cam: {} for cam in self._cam_names}

            if not self._takes:
                self.progress.emit("  No calibration takes found in trials.csv")
                self.finished.emit(all_rows, cam_sizes, cam_media)
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

                    cam_media[cam_name][take.name] = tv.path
                    if cam_name not in cam_sizes:
                        size = _read_media_size(tv.path)
                        if size is not None:
                            cam_sizes[cam_name] = size

                    self.progress.emit(f"  {cam_name} [{take.name}]: detecting corners...")
                    rows = self._detect_video(board, tv, take.name)
                    all_rows[cam_name].extend(rows)

            for cam_name in self._cam_names:
                n = len(all_rows[cam_name])
                self.progress.emit(f"  {cam_name}: {n} detections")
                self.camera_done.emit(cam_name, n)

            self.finished.emit(all_rows, cam_sizes, cam_media)

        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _detect_video(self, board, tv: TakeVideo, take_name: str) -> list:
        """Detect corners on every ``self._skip``-th frame of one take's video.

        Frame numbers in the returned rows are absolute video frame indices, so
        they stay meaningful when a take is trimmed with start/end.
        """
        cap = cv2.VideoCapture(str(tv.path))
        if not cap.isOpened():
            raise FileNotFoundError(f'missing video file "{tv.path}"')

        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = max(0, tv.start)
        last = tv.end if tv.end is not None else (n_total - 1 if n_total > 0 else None)
        span = (last - start + 1) if last is not None and last >= start else 0
        update_every = max(1, span // 200) if span else 30
        label = f"Detecting corners: {take_name} / {tv.path.name}"

        self.frame_progress.emit(label, 0, span)

        v_rows: list = []
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
            if corners is not None and len(corners) > 0:
                v_rows.append(
                    {"framenum": (take_name, framenum), "corners": corners, "ids": ids}
                )
            framenum += 1

        cap.release()
        end = span or max(0, framenum - start)
        self.frame_progress.emit(label, end, end)
        return board.fill_points_rows(v_rows)


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
            board = CharucoBoard(
                squaresX=bp["squares_x"],
                squaresY=bp["squares_y"],
                square_length=bp["square_length"],
                marker_length=bp["marker_length"],
                marker_bits=bp.get("marker_bits", 4),
                dict_size=bp.get("dict_size", 50),
            )

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


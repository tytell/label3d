"""Background worker for Charuco corner detection and camera calibration.

Two-phase workflow:
  1. DetectionWorker: scans calibration/{cam}/ folders, detects Charuco corners
     per camera using images or video, emits per-camera progress.
  2. CalibrationWorker: given detected rows, runs aniposelib bundle adjustment,
     computes per-corner reprojection errors, emits results.
"""

from __future__ import annotations

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
    """Scans experiments/{active}/calibration/{cam}/ folders and detects Charuco corners.

    Emits:
        progress(str)                         — log message
        camera_done(cam, n, img)              — camera name, detection count, sample BGR image
        finished(all_rows, cam_sizes, cam_media)
            all_rows:  dict[cam_name, list[row]]  — raw detection rows per camera
            cam_sizes: dict[cam_name, (w, h)]     — image size per camera
            cam_media: dict[cam_name, list[Path]] — calibration media files per camera
        error_occurred(str)                   — if a fatal error happens
    """

    progress = Signal(str)
    camera_done = Signal(str, int, object)      # cam_name, n_dets, sample_image
    finished = Signal(dict, dict, dict)         # all_rows, cam_sizes, cam_media
    error_occurred = Signal(str)

    def __init__(
        self,
        project_root: Path,
        cam_names: list[str],
        board_params: dict,
        active_experiment: str = "default",
        skip: int = 20,
        parent=None,
    ):
        super().__init__(parent)
        self._root = Path(project_root)
        self._cam_names = cam_names
        self._board_params = board_params
        self._active_experiment = active_experiment
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

            all_rows: dict[str, list] = {}
            cam_sizes: dict[str, tuple[int, int]] = {}
            cam_media: dict[str, list[Path]] = {}

            for cam_name in self._cam_names:
                cam_dir = self._root / "experiments" / self._active_experiment / "calibration" / cam_name
                if not cam_dir.is_dir():
                    self.progress.emit(f"  {cam_name}: no calibration folder, skipping")
                    all_rows[cam_name] = []
                    cam_media[cam_name] = []
                    continue

                files = sorted(cam_dir.iterdir())
                image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
                video_files = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS]

                if not image_files and not video_files:
                    self.progress.emit(f"  {cam_name}: no media files found, skipping")
                    all_rows[cam_name] = []
                    cam_media[cam_name] = []
                    continue

                # Record size from the first file
                first_file = (image_files or video_files)[0]
                size = _read_media_size(first_file)
                if size is not None:
                    cam_sizes[cam_name] = size

                rows: list = []
                sample_img: np.ndarray | None = None

                if image_files:
                    cam_media[cam_name] = image_files
                    self.progress.emit(
                        f"  {cam_name}: detecting corners in {len(image_files)} images..."
                    )
                    image_paths = [str(f) for f in image_files]
                    rows = board.detect_images(image_paths)

                    # Build sample image from the detection with the most corners
                    best_row = None
                    best_count = 0
                    for row in rows:
                        pts = row.get("filled", np.array([]))
                        valid = int(np.sum(~np.any(np.isnan(pts.reshape(-1, 2)), axis=1)))
                        if valid > best_count:
                            best_count = valid
                            best_row = row
                    if best_row is not None:
                        fname = best_row.get("fname")
                        if fname:
                            img = cv2.imread(str(fname))
                            if img is not None:
                                sample_img = _draw_corners(img, best_row["filled"])
                    if sample_img is None and image_files:
                        img = cv2.imread(str(image_files[0]))
                        if img is not None:
                            sample_img = img

                elif video_files:
                    cam_media[cam_name] = video_files
                    self.progress.emit(
                        f"  {cam_name}: detecting corners in {len(video_files)} video(s)..."
                    )
                    # Use prefix=vi so framenum = (video_idx, frame_in_video),
                    # enabling correct video lookup in CalibFrameProvider.
                    for vi, vpath in enumerate(video_files):
                        v_rows = board.detect_video(
                            str(vpath), prefix=vi, skip=self._skip
                        )
                        rows.extend(v_rows)

                    # Sample: first frame of first video
                    cap = cv2.VideoCapture(str(video_files[0]))
                    ret, img = cap.read()
                    cap.release()
                    if ret and rows:
                        sample_img = _draw_corners(img, rows[0]["filled"])
                    elif ret:
                        sample_img = img

                all_rows[cam_name] = rows
                self.progress.emit(f"  {cam_name}: {len(rows)} detections")
                self.camera_done.emit(cam_name, len(rows), sample_img)

            self.finished.emit(all_rows, cam_sizes, cam_media)

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


"""Calibration view: detect Charuco corners, inspect detections, run bundle adjustment.

Layout:
  QSplitter (horizontal)
  ├── Left panel  — board settings, per-camera detection counts, buttons, log
  └── Right panel — CameraGrid showing calibration images + frame nav bar
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import logging
logger = logging.getLogger(__name__)

from aniposelib.boards import CharucoBoard, merge_rows
from aniposelib.cameras import CameraGroup

from label3d.calibration_io import compute_frame_errors, load_corners, save_corners
from label3d.geometry import EpipolarCache, cameras_from_cgroup
from label3d.project import Project
from label3d.video import VideoReader
from label3d.widgets.camera_grid import CameraGrid
from label3d.widgets.charuco_overlay import CharucoOverlay
from label3d.workers.calibration_worker import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    CalibrationTake,
    CalibrationWorker,
    DetectionWorker,
    TakeVideo,
    calibration_images_error,
)

# Default Charuco board params (from DESIGN.md)
_CHARUCO_DEFAULTS = {
    "squares_x": 6,
    "squares_y": 8,
    "square_length": 30.0,
    "marker_length": 22.0,
    "marker_bits": 4,
    "dict_size": 50,
}

# Detect Charuco corners on every Nth video frame (ignored for image folders).
_FRAME_SKIP_DEFAULT = 20


def _build_takes(project: Project) -> list[CalibrationTake]:
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
        takes.append(CalibrationTake(name=info.name, videos=videos))
    return takes


def _scan_take_media(
    takes: list[CalibrationTake], cam_names: list[str], skip: int
) -> tuple[list[dict], dict, dict]:
    """Build a browsable frame list from the takes, before any detection runs.

    Frames are keyed (take_name, frame) exactly as detection keys them, so the
    browser visits what detection will visit. Returns
    (merged_rows, cam_sizes, cam_media) ready for CalibFrameProvider.setup().
    """
    cam_sizes: dict[str, tuple[int, int]] = {}
    cam_media: dict[str, dict[str, Path]] = {cam: {} for cam in cam_names}
    cam_keys: dict[str, set] = {cam: set() for cam in cam_names}
    ordered_keys: list[tuple[str, int]] = []

    for take in takes:
        for cam in cam_names:
            tv = take.videos.get(cam)
            if tv is None or not tv.path.is_file():
                continue
            cam_media[cam][take.name] = tv.path

            cap = cv2.VideoCapture(str(tv.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            if cap.isOpened() and cam not in cam_sizes:
                cam_sizes[cam] = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            cap.release()

            start = max(0, tv.start)
            last = tv.end if tv.end is not None else total - 1
            for frame in range(start, last + 1, skip):
                key = (take.name, frame)
                if key not in cam_keys[cam]:
                    cam_keys[cam].add(key)
                    ordered_keys.append(key)

    # One merged row per distinct (take, frame), in take order then frame order.
    take_order = {take.name: i for i, take in enumerate(takes)}
    unique_keys = sorted(
        set(ordered_keys), key=lambda k: (take_order.get(k[0], 0), k[1])
    )
    merged = [
        {cam: {"framenum": key} for cam in cam_names if key in cam_keys[cam]}
        for key in unique_keys
    ]
    return merged, cam_sizes, cam_media


# ── Frame provider for calibration images / video frames ─────────────────────

class CalibFrameProvider:
    """Serves calibration images for each merged detection frame.

    row['framenum'] is (take_name, frame_in_video); the video is looked up as
    cam_media[camera][take_name]. Returns a black frame when a camera has no
    detection at the given index.
    """

    def __init__(self):
        self._merged: list[dict] = []
        self._cam_sizes: dict[str, tuple[int, int]] = {}
        self._cam_media: dict[str, list[Path]] = {}
        self._video_readers: dict[tuple, VideoReader] = {}

    @property
    def merged_rows(self) -> list[dict]:
        return self._merged

    def setup(
        self,
        merged_rows: list[dict],
        cam_sizes: dict[str, tuple[int, int]],
        cam_media: dict[str, dict[str, Path]],
    ):
        self.close()
        self._merged = merged_rows
        self._cam_sizes = cam_sizes
        self._cam_media = cam_media

    def get_frame(self, camera: str, frame_idx: int) -> np.ndarray | None:
        w, h = self._cam_sizes.get(camera, (640, 480))
        blank = np.zeros((h, w, 3), dtype=np.uint8)

        if not self._merged or frame_idx < 0 or frame_idx >= len(self._merged):
            return blank

        merged_row = self._merged[frame_idx]
        if camera not in merged_row:
            return blank

        row = merged_row[camera]

        # Image file path available
        fname = row.get("fname")
        if fname:
            img = cv2.imread(str(fname))
            if img is None:
                logger.warning(f"Could not read image {fname} for camera {camera}")
                return blank
            return img

        # Video — framenum is (video_idx, frame_in_video)
        framenum = row.get("framenum")
        if framenum is None:
            return blank

        if not isinstance(framenum, tuple):
            return blank
        take_name, frame_in_vid = framenum

        path = self._cam_media.get(camera, {}).get(take_name)
        if path is None:
            return blank

        reader_key = (camera, take_name)
        if reader_key not in self._video_readers:
            try:
                self._video_readers[reader_key] = VideoReader(Path(path))
            except IOError:
                return blank

        img = self._video_readers[reader_key].read_frame(frame_in_vid)
        return img if img is not None else blank

    def n_frames(self) -> int:
        return len(self._merged)

    def close(self):
        for reader in self._video_readers.values():
            reader.close()
        self._video_readers.clear()


# ── Main view ─────────────────────────────────────────────────────────────────

class CalibrationView(QWidget):
    """Full calibration panel: board settings, detect, inspect, calibrate, save."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        # ── State ────────────────────────────────────────────────────────
        self._all_rows: dict[str, list] | None = None
        self._takes: list[CalibrationTake] = []
        self._cam_sizes: dict[str, tuple[int, int]] = {}
        self._cam_media: dict[str, dict[str, Path]] = {}
        self._merged_rows: list[dict] = []
        self._frame_errors: dict[int, dict[str, np.ndarray]] = {}
        self._cgroup = None
        self._epipolar_cache: EpipolarCache | None = None

        self._current_frame = 0
        self._selected_cam: str | None = None
        self._selected_corner: int | None = None

        self._detection_worker: DetectionWorker | None = None
        self._calibration_worker: CalibrationWorker | None = None

        # ── Layout ───────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # ── Left panel ───────────────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Board params group
        board_group = QGroupBox("Charuco Board Parameters")
        form = QFormLayout(board_group)

        charuco_cfg = project.config.get("charuco", {})

        def _int_spin(key, lo, hi):
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(charuco_cfg.get(key, _CHARUCO_DEFAULTS[key]))
            return sb

        def _dbl_spin(key, lo, hi, step=1.0, decimals=1):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(decimals)
            sb.setValue(charuco_cfg.get(key, _CHARUCO_DEFAULTS[key]))
            return sb

        self._sq_x = _int_spin("squares_x", 2, 30)
        self._sq_y = _int_spin("squares_y", 2, 30)
        self._sq_len = _dbl_spin("square_length", 1.0, 1000.0, step=0.5, decimals=2)
        self._mk_len = _dbl_spin("marker_length", 1.0, 1000.0, step=0.5, decimals=2)
        self._mk_bits = _int_spin("marker_bits", 4, 7)
        self._dict_size = _int_spin("dict_size", 10, 1000)

        form.addRow("Squares X:", self._sq_x)
        form.addRow("Squares Y:", self._sq_y)
        form.addRow("Square length (mm):", self._sq_len)
        form.addRow("Marker length (mm):", self._mk_len)
        form.addRow("Marker bits:", self._mk_bits)
        form.addRow("Dict size:", self._dict_size)

        left_layout.addWidget(board_group)

        # Detection settings group
        detect_group = QGroupBox("Detection")
        detect_form = QFormLayout(detect_group)
        calib_cfg = project.config.get("calibration", {})
        self._skip_spin = QSpinBox()
        self._skip_spin.setRange(1, 500)
        self._skip_spin.setValue(calib_cfg.get("frame_skip", _FRAME_SKIP_DEFAULT))
        self._skip_spin.setToolTip("Detect corners on every Nth video frame.")
        detect_form.addRow("Video frame step:", self._skip_spin)
        left_layout.addWidget(detect_group)

        # Persist board + detection settings a moment after the last edit.
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(500)
        self._settings_timer.timeout.connect(self._save_settings)
        for sb in (
            self._sq_x, self._sq_y, self._sq_len, self._mk_len,
            self._mk_bits, self._dict_size,
        ):
            sb.valueChanged.connect(self._settings_timer.start)
        self._skip_spin.valueChanged.connect(self._on_skip_changed)

        # Per-camera detection count labels
        cam_group = QGroupBox("Detection counts")
        cam_form = QFormLayout(cam_group)
        self._cam_count_labels: dict[str, QLabel] = {}
        for cam in project.cameras:
            lbl = QLabel("—")
            self._cam_count_labels[cam] = lbl
            cam_form.addRow(f"{cam}:", lbl)
        left_layout.addWidget(cam_group)

        # Buttons
        btn_row = QHBoxLayout()
        self._detect_btn = QPushButton("Detect Corners")
        self._calibrate_btn = QPushButton("Run Calibration")
        self._calibrate_btn.setEnabled(False)
        btn_row.addWidget(self._detect_btn)
        btn_row.addWidget(self._calibrate_btn)
        left_layout.addLayout(btn_row)

        # Overall error label
        self._error_label = QLabel("Reprojection error: —")
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self._error_label)

        # Progress bar + label
        self._progress_label = QLabel()
        self._progress_label.setVisible(False)
        left_layout.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        left_layout.addWidget(self._progress_bar)

        # Progress log
        left_layout.addWidget(QLabel("Log:"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        left_layout.addWidget(self._log, stretch=1)

        splitter.addWidget(left)

        # ── Right panel — camera grid + frame nav ────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # Camera grid
        self._calib_grid = CameraGrid()
        self._calib_grid.setup_cameras(project.cameras)
        self._calib_provider = CalibFrameProvider()
        self._calib_grid.set_frame_provider(self._calib_provider)
        right_layout.addWidget(self._calib_grid, stretch=1)

        # Frame navigation bar
        nav = QWidget()
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(4, 2, 4, 2)

        self._first_btn = QPushButton("|<")
        self._prev_btn = QPushButton("<")
        self._frame_label = QLabel("—")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumWidth(200)
        self._next_btn = QPushButton(">")
        self._last_btn = QPushButton(">|")

        for btn in (self._first_btn, self._prev_btn, self._next_btn, self._last_btn):
            btn.setFixedWidth(32)
            btn.setEnabled(False)

        nav_layout.addWidget(self._first_btn)
        nav_layout.addWidget(self._prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self._frame_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self._next_btn)
        nav_layout.addWidget(self._last_btn)
        right_layout.addWidget(nav)

        splitter.addWidget(right)
        splitter.setSizes([280, 720])

        # ── Connect buttons ───────────────────────────────────────────────
        self._detect_btn.clicked.connect(self._on_detect)
        self._calibrate_btn.clicked.connect(self._on_calibrate)
        self._first_btn.clicked.connect(self._go_first)
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)
        self._last_btn.clicked.connect(self._go_last)

        # ── Connect camera panel signals ──────────────────────────────────
        for cam in project.cameras:
            panel = self._calib_grid.get_panel(cam)
            if panel is not None:
                panel.point_clicked.connect(self._on_corner_clicked)
                panel.deselect_requested.connect(self._on_deselect)
                panel.setup_legend(project.cameras)

        # ── Charuco overlays (one per panel, set up once) ─────────────────
        self._charuco_overlays: dict[str, CharucoOverlay] = {}
        for cam in project.cameras:
            panel = self._calib_grid.get_panel(cam)
            if panel is not None:
                overlay = CharucoOverlay(panel.scene(), cam)
                panel._charuco_overlay = overlay
                self._charuco_overlays[cam] = overlay

        # Pre-populate frame provider so frames appear before corner detection
        self._load_take_media()

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous() and self._calib_provider.n_frames() > 0:
            self._show_frame(self._current_frame)

    # ── Board params ──────────────────────────────────────────────────────

    def _board_params(self) -> dict:
        return {
            "squares_x": self._sq_x.value(),
            "squares_y": self._sq_y.value(),
            "square_length": self._sq_len.value(),
            "marker_length": self._mk_len.value(),
            "marker_bits": self._mk_bits.value(),
            "dict_size": self._dict_size.value(),
        }

    @property
    def _skip(self) -> int:
        """Stride between video frames offered for corner detection."""
        return self._skip_spin.value()

    def _on_skip_changed(self) -> None:
        self._settings_timer.start()
        # The browsable frame list is built from the same stride, so it has to
        # follow — but only while it still shows raw media, not detections.
        if self._all_rows is None:
            self._rescan_raw_media()

    def _no_takes_message(self) -> str:
        """Explain why there is nothing to detect."""
        exp_dir = self._project.root / "experiments" / self._project.active_experiment
        for cam in self._project.cameras:
            cam_dir = exp_dir / "calibration" / cam
            if not cam_dir.is_dir():
                continue
            images = [
                f for f in cam_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if images:
                return calibration_images_error(cam, len(images))
        return (
            "  No calibration takes for experiment "
            f"'{self._project.active_experiment}' — add is_calibration rows to "
            "trials.csv."
        )

    def _load_take_media(self) -> bool:
        """Rebuild the takes and the browsable frame list. True if usable."""
        self._takes = _build_takes(self._project)
        if not self._takes:
            self._log_msg(self._no_takes_message())
            self._detect_btn.setEnabled(False)
            self._cam_media = {}
            return False

        self._detect_btn.setEnabled(True)
        merged, sizes, media = _scan_take_media(
            self._takes, self._project.cameras, self._skip
        )
        self._cam_sizes = sizes
        self._cam_media = media
        if merged:
            self._calib_provider.setup(merged, sizes, media)
            self._current_frame = min(self._current_frame, len(merged) - 1)
            self._show_frame(self._current_frame)
            self._update_frame_nav()
        return True

    def _rescan_raw_media(self) -> None:
        self._load_take_media()

    def _save_settings(self) -> None:
        """Write board + detection settings back to project.toml."""
        try:
            self._project.save_config_sections({
                "charuco": self._board_params(),
                "calibration": {"frame_skip": self._skip},
            })
        except Exception as exc:
            self._log_msg(f"  Warning: could not save settings: {exc}")

    # ── Detection ─────────────────────────────────────────────────────────

    def _on_detect(self):
        if self._detection_worker and self._detection_worker.isRunning():
            return
        self._detect_btn.setEnabled(False)
        self._calibrate_btn.setEnabled(False)
        self._all_rows = None
        self._merged_rows = []
        self._frame_errors = {}
        self._cgroup = None
        self._epipolar_cache = None
        self._error_label.setText("Reprojection error: —")
        for lbl in self._cam_count_labels.values():
            lbl.setText("—")
        self._clear_overlays()
        self._update_frame_nav()
        self._log_msg("=== Corner detection started ===")

        self._takes = _build_takes(self._project)
        self._detection_worker = DetectionWorker(
            cam_names=self._project.cameras,
            board_params=self._board_params(),
            takes=self._takes,
            skip=self._skip,
        )
        self._detection_worker.progress.connect(self._log_msg)
        self._detection_worker.camera_done.connect(self._on_camera_done)
        self._detection_worker.frame_progress.connect(self._on_frame_progress)
        self._detection_worker.finished.connect(self._on_detect_finished)
        self._detection_worker.error_occurred.connect(self._on_error)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._progress_label.setVisible(True)
        self._detection_worker.start()

    def _on_camera_done(self, cam_name: str, n_dets: int):
        lbl = self._cam_count_labels.get(cam_name)
        if lbl:
            lbl.setText(str(n_dets))

    def _on_frame_progress(self, label: str, current: int, total: int):
        self._progress_label.setText(label)
        if total > 0:
            self._progress_bar.setValue(int(current * 100 / total))
        else:
            self._progress_bar.setValue(0)

    def _on_detect_finished(
        self, all_rows: dict, cam_sizes: dict, cam_media: dict
    ):
        self._all_rows = all_rows
        self._cam_sizes = cam_sizes
        self._cam_media = cam_media

        # Save corners for later reload: one file per take, plus the joint
        # corners.csv holding every take's detections.
        exp_dir = self._project.root / "experiments" / self._project.active_experiment
        take_names = [t.name for t in self._takes]
        for cam in self._project.cameras:
            cam_dir = exp_dir / "calibration" / cam
            if not cam_dir.is_dir():
                continue
            rows = all_rows.get(cam, [])
            try:
                for take in take_names:
                    take_rows = [
                        r for r in rows
                        if isinstance(r.get("framenum"), tuple)
                        and r["framenum"][0] == take
                    ]
                    save_corners(cam_dir, take_rows, take=take)
                save_corners(cam_dir, rows)
            except Exception as exc:
                self._log_msg(f"  Warning: could not save corners for {cam}: {exc}")

        # Merge rows across cameras
        try:
            rows_list = [all_rows.get(c, []) for c in self._project.cameras]
            self._merged_rows = merge_rows(rows_list, cam_names=self._project.cameras)
        except Exception as exc:
            self._log_msg(f"  Warning: could not merge rows: {exc}")
            self._merged_rows = []

        total = sum(len(v) for v in all_rows.values())
        self._log_msg(
            f"=== Detection done — {total} detections, "
            f"{len(self._merged_rows)} merged frames ==="
        )

        # Set up CalibFrameProvider and show first frame
        self._calib_provider.setup(self._merged_rows, cam_sizes, cam_media)
        self._current_frame = 0
        self._update_frame_nav()
        self._show_frame(0)

        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._detect_btn.setEnabled(True)
        has_rows = any(len(v) > 0 for v in all_rows.values())
        self._calibrate_btn.setEnabled(has_rows)

    # ── Calibration ───────────────────────────────────────────────────────

    def _on_calibrate(self):
        if self._all_rows is None:
            return
        if self._calibration_worker and self._calibration_worker.isRunning():
            return
        self._calibrate_btn.setEnabled(False)
        self._log_msg("=== Calibration started ===")

        self._calibration_worker = CalibrationWorker(
            cam_names=self._project.cameras,
            cam_sizes=self._cam_sizes,
            all_rows=self._all_rows,
            board_params=self._board_params(),
        )
        self._calibration_worker.progress.connect(self._log_msg)
        self._calibration_worker.finished.connect(self._on_calibrate_finished)
        self._calibration_worker.error_occurred.connect(self._on_error)
        self._progress_bar.setRange(0, 0)  # indeterminate spinner
        self._progress_bar.setVisible(True)
        self._calibration_worker.start()

    def _on_calibrate_finished(self, cgroup, mean_error: float, frame_errors: dict):
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._frame_errors = frame_errors
        self._error_label.setText(f"Reprojection error: {mean_error:.3f} px")
        self._apply_calibration(cgroup)
        self._log_msg("=== Calibration complete ===")

        out_path = self._project.calibration_path()
        try:
            cgroup.dump(str(out_path))
            self._log_msg(f"Saved: {out_path}")
        except Exception as exc:
            self._log_msg(f"  Warning: could not save calibration: {exc}")

        self._show_frame(self._current_frame)
        self._calibrate_btn.setEnabled(True)

    # ── Calibration helpers ───────────────────────────────────────────────

    def _apply_calibration(self, cgroup) -> None:
        """Store a calibrated CameraGroup and rebuild the epipolar cache."""
        self._cgroup = cgroup
        try:
            cam_params = cameras_from_cgroup(cgroup)
            self._epipolar_cache = EpipolarCache(cam_params)
        except Exception as exc:
            self._log_msg(f"  Warning: could not build epipolar cache: {exc}")

    def _apply_frame_errors(self, cgroup) -> None:
        """Compute per-frame reprojection errors from loaded corners + calibration."""
        if not self._all_rows or cgroup is None:
            return
        bp = self._board_params()
        board = CharucoBoard(
            squaresX=bp["squares_x"],
            squaresY=bp["squares_y"],
            square_length=bp["square_length"],
            marker_length=bp["marker_length"],
            marker_bits=bp["marker_bits"],
            dict_size=bp["dict_size"],
        )
        rows_list = [self._all_rows.get(cam, []) for cam in self._project.cameras]
        mean_error, self._frame_errors = compute_frame_errors(
            cgroup, board, rows_list, self._project.cameras
        )
        self._error_label.setText(f"Reprojection error: {mean_error:.3f} px")

    # ── Experiment reload ─────────────────────────────────────────────────

    def reload_for_experiment(self) -> None:
        """Reset all calibration state and reload media for the new active experiment."""
        for worker in (self._detection_worker, self._calibration_worker):
            if worker is not None and worker.isRunning():
                worker.quit()
                worker.wait()

        self._all_rows = None
        self._merged_rows = []
        self._frame_errors = {}
        self._cgroup = None
        self._epipolar_cache = None
        self._current_frame = 0

        self._error_label.setText("Reprojection error: —")
        for lbl in self._cam_count_labels.values():
            lbl.setText("—")
        self._calibrate_btn.setEnabled(False)
        self._clear_overlays()

        # Rebuild takes and the browsable frame list for the new experiment
        self._load_take_media()

        # Try loading saved corners per camera
        exp_dir = self._project.root / "experiments" / self._project.active_experiment
        loaded_rows: dict[str, list] = {}
        for cam in self._project.cameras:
            rows = load_corners(exp_dir / "calibration" / cam)
            if rows is not None:
                loaded_rows[cam] = rows

        has_corners = any(len(v) > 0 for v in loaded_rows.values())

        if has_corners:
            self._all_rows = loaded_rows
            # Recompute filled arrays (set during detection, not stored in CSV)
            bp = self._board_params()
            board = CharucoBoard(
                squaresX=bp["squares_x"],
                squaresY=bp["squares_y"],
                square_length=bp["square_length"],
                marker_length=bp["marker_length"],
                marker_bits=bp["marker_bits"],
                dict_size=bp["dict_size"],
            )
            for rows in loaded_rows.values():
                board.fill_points_rows(rows)
            try:
                rows_list = [loaded_rows.get(c, []) for c in self._project.cameras]
                self._merged_rows = merge_rows(rows_list, cam_names=self._project.cameras)
            except Exception as exc:
                self._log_msg(f"  Warning: could not merge loaded corners: {exc}")
                self._merged_rows = []
            for cam, rows in loaded_rows.items():
                lbl = self._cam_count_labels.get(cam)
                if lbl and rows:
                    lbl.setText(str(len(rows)))
            self._calibrate_btn.setEnabled(bool(self._merged_rows))
            # Show only detected frames, consistent with post-detection behavior
            self._calib_provider.setup(
                self._merged_rows, self._cam_sizes, self._cam_media
            )
            self._update_frame_nav()
            self._show_frame(0)
        # Without saved corners the provider already holds the raw take frames
        # that _load_take_media() set up.

        # Load existing calibration to show reprojection errors
        calib_path = self._project.calibration_path()
        if calib_path.exists() and has_corners:
            try:
                cgroup = CameraGroup.load(str(calib_path))  # type: ignore[arg-type]
                self._apply_calibration(cgroup)
                self._apply_frame_errors(cgroup)
            except Exception as exc:
                self._log_msg(f"  Warning: could not load calibration: {exc}")

        if self._calib_provider.n_frames() > 0:
            self._show_frame(0)
        self._update_frame_nav()

    # ── Frame navigation ──────────────────────────────────────────────────

    def _go_first(self):
        self._show_frame(0)

    def _go_prev(self):
        self._show_frame(max(0, self._current_frame - 1))

    def _go_next(self):
        n = self._calib_provider.n_frames()
        self._show_frame(min(n - 1, self._current_frame + 1))

    def _go_last(self):
        n = self._calib_provider.n_frames()
        self._show_frame(max(0, n - 1))

    def _show_frame(self, frame_idx: int):
        """Display calibration images and update overlays for frame_idx."""
        self._current_frame = frame_idx
        self._on_deselect()          # clear epipolar lines when changing frames

        # Display images
        self._calib_grid.update_frame(frame_idx)

        # Update charuco overlays
        merged_row = self._merged_rows[frame_idx] if 0 <= frame_idx < len(self._merged_rows) else {}
        cam_errors = self._frame_errors.get(frame_idx, {})

        for cam in self._project.cameras:
            overlay = self._charuco_overlays.get(cam)
            if overlay is None:
                continue
            if cam in merged_row:
                filled = merged_row[cam].get("filled")
                if filled is not None:
                    pts = np.asarray(filled, dtype=float).reshape(-1, 2)
                    errs = cam_errors.get(cam)
                    overlay.update(pts, errs)
                else:
                    overlay.clear()
            else:
                overlay.clear()

        self._update_frame_nav()

    def _update_frame_nav(self):
        n = self._calib_provider.n_frames()
        has_frames = n > 0
        self._first_btn.setEnabled(has_frames and self._current_frame > 0)
        self._prev_btn.setEnabled(has_frames and self._current_frame > 0)
        self._next_btn.setEnabled(has_frames and self._current_frame < n - 1)
        self._last_btn.setEnabled(has_frames and self._current_frame < n - 1)
        if has_frames:
            label, anchor = self._representative_frame_info(self._current_frame)
            if anchor is not None:
                self._frame_label.setText(f"{label}  ({self._current_frame + 1}/{n})")
            else:
                self._frame_label.setText(f"{label} / {n}")
        else:
            self._frame_label.setText("—")

    def _representative_frame_info(
        self, frame_idx: int
    ) -> tuple[str, tuple[int, int] | None]:
        """Return (display_label, (video_idx, frame_in_video)) for frame_idx.

        Falls back to the provider's raw rows when no corners are detected yet.
        Returns ("—", None) when frame_idx is out of range.
        For image-based calibration returns ("Image N", None).
        For single-video: ("Frame N", (0, N)).
        For multi-video: ("{stem}, frame N", (vi, N)).
        """
        rows = self._merged_rows if self._merged_rows else self._calib_provider.merged_rows
        if not rows or frame_idx < 0 or frame_idx >= len(rows):
            return "—", None

        merged_row = rows[frame_idx]
        for cam in self._project.cameras:
            if cam not in merged_row:
                continue
            row = merged_row[cam]

            if row.get("fname") is not None:
                return f"Image {frame_idx + 1}", None

            framenum = row.get("framenum")
            if framenum is not None:
                vi, fi = framenum if isinstance(framenum, tuple) else (0, int(framenum))
                cam_files = self._cam_media.get(cam, [])
                if len(cam_files) > 1 and vi < len(cam_files):
                    stem = cam_files[vi].stem
                    return f"{stem}, frame {fi}", (vi, fi)
                return f"Frame {fi}", (vi, fi)

        return f"{frame_idx + 1}", None

    # ── Corner selection + epipolar lines ─────────────────────────────────

    def _on_corner_clicked(self, cam: str, _instance: int, corner_id_str: str):
        """Select a corner and draw its epipolar lines in all other cameras."""
        try:
            corner_idx = int(corner_id_str)
        except ValueError:
            return

        # Look up the corner's pixel position
        if 0 <= self._current_frame < len(self._merged_rows):
            merged_row = self._merged_rows[self._current_frame]
        else:
            return
        
        if cam not in merged_row:
            return
        row = merged_row[cam]
        filled = row.get("filled")
        if filled is None:
            return
        pts = np.asarray(filled, dtype=float).reshape(-1, 2)
        if corner_idx >= len(pts):
            return
        x, y = pts[corner_idx]
        if np.isnan(x) or np.isnan(y):
            return

        self._selected_cam = cam
        self._selected_corner = corner_idx

        # Update selection highlight on all overlays
        for c, overlay in self._charuco_overlays.items():
            if c == cam:
                overlay.set_selected(corner_idx)
            else:
                overlay.set_selected(None)

        # Draw epipolar lines in all other camera panels
        cam_idx = self._project.cameras.index(cam)
        for target_cam in self._project.cameras:
            panel = self._calib_grid.get_panel(target_cam)
            if panel is None:
                continue
            if target_cam == cam:
                panel.epipolar_overlay.update_direct(None, cam, x, y, cam_idx)
            else:
                panel.epipolar_overlay.update_direct(
                    self._epipolar_cache, cam, x, y, cam_idx
                )
            panel.update_line_labels()

    def _on_deselect(self):
        """Clear selection and hide all epipolar lines."""
        self._selected_cam = None
        self._selected_corner = None

        for overlay in self._charuco_overlays.values():
            overlay.set_selected(None)

        for cam in self._project.cameras:
            panel = self._calib_grid.get_panel(cam)
            if panel is not None:
                panel.epipolar_overlay.update_direct(None, "", 0, 0, 0)
                panel.update_line_labels()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _clear_overlays(self):
        for overlay in self._charuco_overlays.values():
            overlay.clear()

    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)

    def _on_error(self, msg: str):
        self._log_msg(f"ERROR: {msg}")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._detect_btn.setEnabled(True)
        self._calibrate_btn.setEnabled(self._all_rows is not None)

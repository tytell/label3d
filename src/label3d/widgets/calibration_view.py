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
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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
from label3d.widgets.frame_slider import FrameSlider
from label3d.calibration_compare import (
    compare_calibrations,
    drift_thresholds,
    drift_warnings,
    noise_adjusted_thresholds,
)
from label3d.workers.calibration_worker import (
    IMAGE_EXTENSIONS,
    JOINT_TAKE,
    VIDEO_EXTENSIONS,
    CalibrationTake,
    DetectionWorker,
    MultiTakeCalibrationWorker,
    TakeVideo,
    rows_for_take,
    rows_in_range,
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

# The log shrinks to this before the settings panel starts to scroll
LOG_MIN_HEIGHT = 100

# Detect Charuco corners on every Nth video frame (ignored for image folders).
_FRAME_SKIP_DEFAULT = 20

# Calibrate each solution this many times and keep the lowest-error run.
_REPEATS_DEFAULT = 3


def row_key(row: dict) -> tuple[str, int] | None:
    """(take, synchronized_frame) of a merged detection row, from any camera."""
    for cell in row.values():
        framenum = cell.get("framenum")
        if isinstance(framenum, tuple):
            return framenum
    return None


def row_take(row: dict) -> str | None:
    """Which take a merged detection row belongs to, from any camera's key."""
    key = row_key(row)
    return key[0] if key else None


def nearest_row(rows: list[dict], take: str, frame: int) -> int | None:
    """Index of the row in `take` whose frame is closest to `frame`, or None."""
    best, best_dist = None, None
    for i, row in enumerate(rows):
        key = row_key(row)
        if key is None or key[0] != take:
            continue
        dist = abs(key[1] - frame)
        if best_dist is None or dist < best_dist:
            best, best_dist = i, dist
    return best


def take_row_indices(rows: list[dict], take: str | None) -> list[int]:
    """Positions in `rows` belonging to `take`; all of them for the joint view.

    Returning indices rather than rows keeps the mapping back to the unfiltered
    list, which reprojection errors are keyed by.
    """
    if not take or take == JOINT_TAKE:
        return list(range(len(rows)))
    return [i for i, r in enumerate(rows) if row_take(r) == take]


def take_range(config: dict, experiment: str, take: str) -> tuple[int, int | None]:
    """A take's calibration frame range from project.toml (end None = to the end).

    Stored as [calibration.ranges."<experiment>"."<take>"] with start/end in
    synchronized frames.
    """
    ranges = config.get("calibration", {}).get("ranges", {})
    entry = ranges.get(experiment, {}).get(take, {})
    end = entry.get("end")
    return int(entry.get("start", 0)), (int(end) if end is not None else None)


def rekey_frame_errors(
    frame_errors: dict[int, dict],
    rows_by_cam: dict[str, list],
    cam_names: list[str],
    full_rows: list[dict],
) -> dict[int, dict]:
    """Re-key errors from positions in merge_rows(rows_by_cam) to full_rows.

    compute_frame_errors numbers frames by position in whatever rows it was
    given; the view looks them up by index into the full merged list. Matching
    on (take, frame) keeps the two aligned however the rows were filtered.
    """
    merged = merge_rows([rows_by_cam.get(c, []) for c in cam_names], cam_names=cam_names)
    full_index = {row_key(row): i for i, row in enumerate(full_rows)}
    rekeyed = {}
    for pos, errs in frame_errors.items():
        if 0 <= pos < len(merged):
            idx = full_index.get(row_key(merged[pos]))
            if idx is not None:
                rekeyed[idx] = errs
    return rekeyed


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
        start, end = take_range(project.config, project.active_experiment, info.name)
        takes.append(CalibrationTake(name=info.name, videos=videos, start=start, end=end))
    return takes


def _scan_take_media(
    takes: list[CalibrationTake], cam_names: list[str], skip: int
) -> tuple[list[dict], dict, dict]:
    """Build a browsable frame list from the takes, before any detection runs.

    Frames are keyed (take_name, synchronized_frame) exactly as detection keys
    them, so the browser visits what detection will visit. Returns
    (merged_rows, cam_sizes, take_lengths); take_lengths is the number of
    synchronized frames in each take, counting the longest camera.
    """
    cam_sizes: dict[str, tuple[int, int]] = {}
    cam_keys: dict[str, set] = {cam: set() for cam in cam_names}
    ordered_keys: list[tuple[str, int]] = []
    take_lengths: dict[str, int] = {take.name: 0 for take in takes}

    for take in takes:
        for cam in cam_names:
            tv = take.videos.get(cam)
            if tv is None or not tv.path.is_file():
                continue

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
            n_sync = last - start + 1
            take_lengths[take.name] = max(take_lengths[take.name], n_sync)
            stop = n_sync if take.end is None else min(n_sync, take.end + 1)
            for frame in range(take.start, stop, skip):
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
    return merged, cam_sizes, take_lengths


# ── Frame provider for calibration images / video frames ─────────────────────

class CalibFrameProvider:
    """Serves calibration images for each merged detection frame.

    row['framenum'] is (take_name, synchronized_frame); the camera's video and
    start offset come from that take, and the frame read is
    synchronized_frame + start. Every camera in the take is shown, whether or
    not it detected corners in that frame; black only when it has no video.
    """

    def __init__(self):
        self._merged: list[dict] = []
        self._cam_sizes: dict[str, tuple[int, int]] = {}
        self._takes: dict[str, CalibrationTake] = {}
        self._video_readers: dict[tuple, VideoReader] = {}

    @property
    def merged_rows(self) -> list[dict]:
        return self._merged

    def setup(
        self,
        merged_rows: list[dict],
        cam_sizes: dict[str, tuple[int, int]],
        takes: list[CalibrationTake],
    ):
        self.close()
        self._merged = merged_rows
        self._cam_sizes = cam_sizes
        self._takes = {take.name: take for take in takes}

    def get_frame(self, camera: str, frame_idx: int) -> np.ndarray | None:
        w, h = self._cam_sizes.get(camera, (640, 480))
        blank = np.zeros((h, w, 3), dtype=np.uint8)

        if not self._merged or frame_idx < 0 or frame_idx >= len(self._merged):
            return blank

        merged_row = self._merged[frame_idx]
        row = merged_row.get(camera, {})

        # Image file path available
        fname = row.get("fname")
        if fname:
            img = cv2.imread(str(fname))
            if img is None:
                logger.warning(f"Could not read image {fname} for camera {camera}")
                return blank
            return img

        # Video — framenum is (take_name, synchronized_frame). A camera with no
        # corners in this frame has no cell, but its video still has the frame.
        framenum = row.get("framenum") or row_key(merged_row)
        if not isinstance(framenum, tuple):
            return blank
        take_name, sync_frame = framenum

        take = self._takes.get(take_name)
        tv = take.videos.get(camera) if take is not None else None
        if tv is None:
            return blank

        reader_key = (camera, take_name)
        if reader_key not in self._video_readers:
            try:
                self._video_readers[reader_key] = VideoReader(tv.path)
            except IOError:
                return blank

        img = self._video_readers[reader_key].read_frame(sync_frame + max(0, tv.start))
        return img if img is not None else blank

    def n_frames(self) -> int:
        return len(self._merged)

    def close(self):
        for reader in self._video_readers.values():
            reader.close()
        self._video_readers.clear()


# ── Main view ─────────────────────────────────────────────────────────────────

class CalibrationView(QWidget):
    """Full calibration panel: board settings, detect, inspect, calibrate, save.

    Emits calibration_saved after writing the experiment's calibration.toml,
    so views that loaded the old one can reload it.
    """

    calibration_saved = Signal()

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        # ── State ────────────────────────────────────────────────────────
        self._all_rows: dict[str, list] | None = None
        self._takes: list[CalibrationTake] = []
        self._take_results: list = []   # list[TakeCalibration], joint last
        self._browse_rows: list[dict] = []  # every frame, before take filtering
        self._display_map: list[int] = []   # display index -> index in _browse_rows
        self._cam_sizes: dict[str, tuple[int, int]] = {}
        self._take_lengths: dict[str, int] = {}  # synchronized frames per take
        self._merged_rows: list[dict] = []
        self._calib_rows: dict[str, list] = {}  # detections the last calibration used
        self._frame_errors: dict[int, dict[str, np.ndarray]] = {}
        self._cgroup = None
        self._epipolar_cache: EpipolarCache | None = None

        self._current_frame = 0
        self._selected_cam: str | None = None
        self._selected_corner: int | None = None

        self._detection_worker: DetectionWorker | None = None
        self._calibration_worker: MultiTakeCalibrationWorker | None = None

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
        board_grid = QGridLayout(board_group)

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

        # Pairs side by side: label, spin, label, spin
        for row, pair in enumerate((
            (("Squares X:", self._sq_x), ("Squares Y:", self._sq_y)),
            (("Square length (mm):", self._sq_len), ("Marker length (mm):", self._mk_len)),
            (("Marker bits:", self._mk_bits), ("Dict size:", self._dict_size)),
        )):
            for half, (text, spin) in enumerate(pair):
                board_grid.addWidget(QLabel(text), row, 2 * half)
                board_grid.addWidget(spin, row, 2 * half + 1)
        board_grid.setColumnStretch(1, 1)
        board_grid.setColumnStretch(3, 1)

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

        # Per-take frame range; also shown as draggable brackets on the slider.
        self._range_take_label = QLabel("—")
        self._start_spin = QSpinBox()
        self._end_spin = QSpinBox()
        for sb in (self._start_spin, self._end_spin):
            sb.setRange(0, 0)
            sb.setKeyboardTracking(False)
        self._start_spin.setToolTip(
            "First synchronized frame of this take used for calibration."
        )
        self._end_spin.setToolTip(
            "Last synchronized frame of this take used for calibration."
        )
        range_row = QHBoxLayout()
        range_row.addWidget(self._start_spin, stretch=1)
        range_row.addWidget(QLabel("–"))
        range_row.addWidget(self._end_spin, stretch=1)
        detect_form.addRow("Range of take:", self._range_take_label)
        detect_form.addRow("Frames:", range_row)
        left_layout.addWidget(detect_group)

        # Calibration takes group
        takes_group = QGroupBox("Calibration takes")
        takes_layout = QVBoxLayout(takes_group)

        self._take_combo = QComboBox()
        self._take_combo.setToolTip(
            "Which calibration solution to use. Each take is calibrated on its "
            "own and all takes jointly."
        )
        self._take_combo.currentTextChanged.connect(self._on_take_selected)
        takes_layout.addWidget(self._take_combo)

        repeat_row = QHBoxLayout()
        repeat_row.addWidget(QLabel("Repeat calibration:"))
        self._repeat_spin = QSpinBox()
        self._repeat_spin.setRange(1, 20)
        self._repeat_spin.setValue(int(calib_cfg.get("repeats", _REPEATS_DEFAULT)))
        self._repeat_spin.setToolTip(
            "Calibrate each take this many times and keep the lowest error. "
            "Runs differ because bundle adjustment samples at random; their "
            "spread keeps that noise from being reported as camera drift."
        )
        repeat_row.addWidget(self._repeat_spin, stretch=1)
        takes_layout.addLayout(repeat_row)

        self._take_table = QTableWidget(0, 5)
        self._take_table.setHorizontalHeaderLabels(
            ["Take", "Dets", "Error px", "Rot °", "Base %"]
        )
        self._take_table.verticalHeader().setVisible(False)
        self._take_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._take_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._take_table.setMaximumHeight(140)
        takes_layout.addWidget(self._take_table)

        self._drift_label = QLabel()
        self._drift_label.setWordWrap(True)
        self._drift_label.setVisible(False)
        self._drift_label.setStyleSheet("color: #b45309; font-weight: bold;")
        takes_layout.addWidget(self._drift_label)

        self._use_take_btn = QPushButton("Use selected take")
        self._use_take_btn.setEnabled(False)
        self._use_take_btn.clicked.connect(self._on_use_take)
        takes_layout.addWidget(self._use_take_btn)

        left_layout.addWidget(takes_group)

        # Persist board + detection settings a moment after the last edit.
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(500)
        self._settings_timer.timeout.connect(self._save_settings)
        for sb in (
            self._sq_x, self._sq_y, self._sq_len, self._mk_len,
            self._mk_bits, self._dict_size, self._repeat_spin,
        ):
            sb.valueChanged.connect(self._settings_timer.start)
        self._skip_spin.valueChanged.connect(self._on_skip_changed)

        # Rebuilding the raw frame list reopens every video, so wait for a
        # range drag or edit to settle first.
        self._range_timer = QTimer(self)
        self._range_timer.setSingleShot(True)
        self._range_timer.setInterval(300)
        self._range_timer.timeout.connect(self._on_range_settled)
        self._start_spin.valueChanged.connect(self._on_range_spin_changed)
        self._end_spin.valueChanged.connect(self._on_range_spin_changed)

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
        self._log.setMinimumHeight(LOG_MIN_HEIGHT)
        left_layout.addWidget(self._log, stretch=1)

        # Settings scroll when the window is short, so they don't set the
        # whole window's minimum height
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(
            left.minimumSizeHint().width() + left_scroll.verticalScrollBar().sizeHint().width()
        )
        splitter.addWidget(left_scroll)

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

        # Seekbar over one take's synchronized frames; rows mark detections.
        self._slider = FrameSlider()
        self._slider.set_cameras(project.cameras)
        self._slider.frame_changed.connect(self._on_slider_seek)
        self._slider.bounds_changed.connect(self._on_slider_bounds_changed)

        nav_layout.addWidget(self._first_btn)
        nav_layout.addWidget(self._prev_btn)
        nav_layout.addWidget(self._frame_label)
        nav_layout.addWidget(self._slider, stretch=1)
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

        self._setup_shortcuts()

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
            return False

        self._detect_btn.setEnabled(True)
        merged, sizes, lengths = _scan_take_media(
            self._takes, self._project.cameras, self._skip
        )
        self._cam_sizes = sizes
        self._take_lengths = lengths
        self._populate_take_combo()
        self._set_browse_rows(merged)
        return True

    def _populate_take_combo(self) -> None:
        """List the takes as soon as they are known, joint entry last.

        With a single take the joint solution is that take, so it is not offered.
        """
        names = [t.name for t in self._takes]
        if len(names) > 1:
            names.append(JOINT_TAKE)
        current = self._take_combo.currentText()
        self._take_combo.blockSignals(True)
        self._take_combo.clear()
        self._take_combo.addItems(names)
        self._take_combo.setCurrentText(current if current in names else names[-1])
        self._take_combo.blockSignals(False)

    def _set_browse_rows(self, rows: list[dict]) -> None:
        """Replace the full frame list and show the selected take's slice."""
        self._browse_rows = rows
        self._current_frame = 0
        self._refresh_display()

    def _full_index(self, display_idx: int) -> int:
        """Translate a displayed frame position back to the merged-row index."""
        if 0 <= display_idx < len(self._display_map):
            return self._display_map[display_idx]
        return display_idx

    def _refresh_display(self) -> None:
        """Show only the frames of the selected take (all of them if joint)."""
        # Reprojection errors are keyed by position in the *unfiltered* merged
        # list, so filtering the display needs a map back to it.
        keep = take_row_indices(self._browse_rows, self._take_combo.currentText())
        self._display_map = keep
        rows = [self._browse_rows[i] for i in keep]

        self._calib_provider.setup(rows, self._cam_sizes, self._takes)
        if rows:
            self._current_frame = min(self._current_frame, len(rows) - 1)
            self._show_frame(self._current_frame)
        else:
            self._current_frame = 0
            self._clear_overlays()
        self._update_frame_nav()

    def _rescan_raw_media(self) -> None:
        """Rebuild the raw frame list, staying near the frame being shown."""
        rows = self._calib_provider.merged_rows
        key = row_key(rows[self._current_frame]) if self._current_frame < len(rows) else None
        self._load_take_media()
        if key is not None:
            idx = nearest_row(self._calib_provider.merged_rows, key[0], key[1])
            if idx is not None:
                self._show_frame(idx)

    def _save_settings(self) -> None:
        """Write board + detection settings back to project.toml.

        [calibration] also holds the per-take ranges and drift thresholds, so
        it is merged into rather than replaced.
        """
        calibration = dict(self._project.config.get("calibration", {}))
        calibration["frame_skip"] = self._skip
        calibration["repeats"] = self._repeat_spin.value()
        try:
            self._project.save_config_sections({
                "charuco": self._board_params(),
                "calibration": calibration,
            })
        except Exception as exc:
            self._log_msg(f"  Warning: could not save settings: {exc}")

    # ── Per-take frame range ──────────────────────────────────────────────

    def _take_by_name(self, name: str | None) -> CalibrationTake | None:
        return next((t for t in self._takes if t.name == name), None)

    def _on_range_spin_changed(self) -> None:
        start, end = self._start_spin.value(), self._end_spin.value()
        # Keep start <= end by moving whichever spin box was not edited.
        if start > end:
            other = self._end_spin if self.sender() is self._start_spin else self._start_spin
            other.blockSignals(True)
            other.setValue(start if other is self._end_spin else end)
            other.blockSignals(False)
            start, end = self._start_spin.value(), self._end_spin.value()
        self._set_take_range(start, end)

    def _on_slider_bounds_changed(self, start: int, end: int) -> None:
        for sb, value in ((self._start_spin, start), (self._end_spin, end)):
            sb.blockSignals(True)
            sb.setValue(value)
            sb.blockSignals(False)
        self._set_take_range(start, end)

    def _set_take_range(self, start: int, end: int) -> None:
        """Apply a new range to the slider's take and remember it in the config."""
        take = self._take_by_name(self._slider_take())
        if take is None:
            return
        take.start, take.end = start, end
        self._slider.set_bounds(start, end)

        # Update the in-memory config now (detection rebuilds takes from it);
        # the file write is debounced with the other settings.
        calibration = self._project.config.setdefault("calibration", {})
        exp_ranges = calibration.setdefault("ranges", {}).setdefault(
            self._project.active_experiment, {}
        )
        exp_ranges[take.name] = {"start": start, "end": end}
        self._settings_timer.start()
        self._range_timer.start()

    def _on_range_settled(self) -> None:
        if self._all_rows is None:
            self._rescan_raw_media()
            return
        take = self._take_by_name(self._slider_take())
        if take is not None:
            self._log_msg(
                f"Range of {take.name}: frames {take.start}-{take.end}. Detected "
                "frames outside it are left out of calibration; to cover frames "
                "that were not detected, run Detect Corners again."
            )

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

    def _on_detect_finished(self, all_rows: dict, cam_sizes: dict):
        self._all_rows = all_rows
        self._cam_sizes = cam_sizes

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

        # Browse the detected frames, filtered to the selected take
        self._populate_take_combo()
        self._set_browse_rows(self._merged_rows)

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

        self._calib_rows = rows_in_range(self._all_rows, self._takes)
        n_dropped = sum(len(r) for r in self._all_rows.values()) - sum(
            len(r) for r in self._calib_rows.values()
        )
        if n_dropped:
            self._log_msg(f"  Leaving out {n_dropped} detections outside the frame ranges")

        self._calibration_worker = MultiTakeCalibrationWorker(
            cam_names=self._project.cameras,
            cam_sizes=self._cam_sizes,
            all_rows=self._calib_rows,
            board_params=self._board_params(),
            take_names=[t.name for t in self._takes],
            n_repeats=self._repeat_spin.value(),
        )
        self._calibration_worker.progress.connect(self._log_msg)
        self._calibration_worker.finished.connect(self._on_calibrate_finished)
        self._calibration_worker.error_occurred.connect(self._on_error)
        self._progress_bar.setRange(0, 0)  # indeterminate spinner
        self._progress_bar.setVisible(True)
        self._calibration_worker.start()

    def _on_calibrate_finished(self, results: list):
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._take_results = results
        self._calibrate_btn.setEnabled(True)

        if not results:
            self._log_msg("=== Calibration produced no solutions ===")
            return

        # Errors come keyed by position in each solution's own rows.
        for result in results:
            rows = self._calib_rows
            if result.name != JOINT_TAKE:
                rows = rows_for_take(rows, result.name)
            result.frame_errors = rekey_frame_errors(
                result.frame_errors, rows, self._project.cameras, self._merged_rows
            )

        self._save_take_calibrations(results)
        self._populate_take_table(results)
        self._log_msg("=== Calibration complete ===")

        # Default to the joint solution, which the worker emits last.
        self._use_take_btn.setEnabled(True)
        self._take_combo.setCurrentText(results[-1].name)
        self._on_take_selected(results[-1].name)
        self._on_use_take()

    def _save_take_calibrations(self, results: list) -> None:
        """Write each take's solution to calibration_takes/{take}.toml."""
        exp_dir = self._project.root / "experiments" / self._project.active_experiment
        takes_dir = exp_dir / "calibration_takes"
        try:
            takes_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._log_msg(f"  Warning: could not create {takes_dir}: {exc}")
            return
        for result in results:
            safe = result.name.replace(" ", "_").replace("(", "").replace(")", "")
            try:
                result.cgroup.dump(str(takes_dir / f"{safe}.toml"))
            except Exception as exc:
                self._log_msg(f"  Warning: could not save take '{result.name}': {exc}")

    def _populate_take_table(self, results: list) -> None:
        """Fill the comparison table and flag takes that drifted.

        Takes are compared with the joint solution, when there is one. A drift
        only counts if it also exceeds the run-to-run spread of the two
        solutions (see noise_adjusted_thresholds), since repeated runs on the
        same rows already disagree.
        """
        joint = results[-1] if results[-1].name == JOINT_TAKE else None
        rot_cfg, base_cfg = drift_thresholds(self._project.config)

        self._take_table.setRowCount(len(results))
        all_warnings: list[str] = []

        for row, result in enumerate(results):
            if joint is None or result is joint:
                rot_txt, base_txt, flagged = "—", "—", False
            else:
                rot_thresh, base_thresh = noise_adjusted_thresholds(
                    rot_cfg, base_cfg, [result.spread, joint.spread]
                )
                if (rot_thresh, base_thresh) != (rot_cfg, base_cfg):
                    self._log_msg(
                        f"  {result.name}: drift thresholds raised to "
                        f"{rot_thresh:.2f} deg / {base_thresh:.1f}% by run-to-run spread"
                    )
                drifts = compare_calibrations(joint.cgroup, result.cgroup)
                max_rot = max((d.rotation_deg for d in drifts), default=0.0)
                max_base = max((d.baseline_pct for d in drifts), default=0.0)
                rot_txt, base_txt = f"{max_rot:.2f}", f"{max_base:.1f}"
                warnings = drift_warnings(
                    result.name, drifts, rot_thresh, base_thresh
                )
                all_warnings.extend(warnings)
                flagged = bool(warnings)

            cells = [
                result.name,
                str(result.n_rows),
                f"{result.error:.3f}",
                rot_txt,
                base_txt,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if flagged:
                    item.setForeground(Qt.GlobalColor.red)
                self._take_table.setItem(row, col, item)

        self._take_table.resizeColumnsToContents()

        if all_warnings:
            self._drift_label.setText(
                "Calibration differs between takes — a camera may have moved "
                "during the experiment. Consider using a single take.\n"
                + "\n".join(all_warnings)
            )
            self._drift_label.setVisible(True)
            for msg in all_warnings:
                self._log_msg(f"  ! {msg}")
        else:
            self._drift_label.setVisible(False)

    def _selected_take(self):
        """The TakeCalibration currently chosen in the combo, or None."""
        name = self._take_combo.currentText()
        for result in self._take_results:
            if result.name == name:
                return result
        return None

    def _on_take_selected(self, name: str) -> None:
        """Switch the browsed frames to this take, and preview its solution.

        The combo is filled as soon as the takes load, so this runs before any
        calibration exists — in that case it only changes what is displayed.
        """
        result = self._selected_take()
        self._current_frame = 0
        self._refresh_display()   # builds _display_map for this take

        if result is None:
            return

        # _on_calibrate_finished re-keyed these to the full merged list.
        self._frame_errors = result.frame_errors
        self._error_label.setText(f"Reprojection error: {result.error:.3f} px")
        self._apply_calibration(result.cgroup)
        self._show_frame(self._current_frame)

    def _on_use_take(self) -> None:
        """Write the chosen solution to calibration.toml and corners.csv."""
        result = self._selected_take()
        if result is None:
            return

        out_path = self._project.calibration_path()
        try:
            result.cgroup.dump(str(out_path))
            self._log_msg(f"Saved [{result.name}]: {out_path}")
        except Exception as exc:
            self._log_msg(f"  Warning: could not save calibration: {exc}")
            return
        self._project.reload_calibration()
        self.calibration_saved.emit()

        # corners.csv holds whatever produced the saved calibration.
        exp_dir = self._project.root / "experiments" / self._project.active_experiment
        rows_by_cam = self._calib_rows
        if result.name != JOINT_TAKE:
            rows_by_cam = rows_for_take(rows_by_cam, result.name)
        for cam in self._project.cameras:
            cam_dir = exp_dir / "calibration" / cam
            if not cam_dir.is_dir():
                continue
            try:
                save_corners(cam_dir, rows_by_cam.get(cam, []))
            except Exception as exc:
                self._log_msg(f"  Warning: could not save corners for {cam}: {exc}")

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
        self._calib_rows = {}
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
            self._set_browse_rows(self._merged_rows)
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

    def _setup_shortcuts(self) -> None:
        """Arrow keys step through listed frames, mirroring the Label tab's keys.

        Qt ignores shortcuts whose parent is hidden, so these only fire while
        the Calibrate tab is shown, and the Label tab's only while it is.
        """
        shortcuts = [
            (Qt.Key.Key_Right, lambda: self._step(1)),
            (Qt.Key.Key_Left, lambda: self._step(-1)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Right), lambda: self._step(4)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Left), lambda: self._step(-4)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.ALT | Qt.Key.Key_Right),
             lambda: self._step(100)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.ALT | Qt.Key.Key_Left),
             lambda: self._step(-100)),
            (Qt.Key.Key_Home, self._go_first),
            (Qt.Key.Key_End, self._go_last),
        ]
        for key, callback in shortcuts:
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(callback)

    def _step(self, delta: int) -> None:
        n = self._calib_provider.n_frames()
        if n == 0:
            return
        target = max(0, min(n - 1, self._current_frame + delta))
        if target != self._current_frame:
            self._show_frame(target)

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

        # Update charuco overlays. frame_idx is a position in the filtered
        # display; corners and errors are keyed by the merged-row index.
        full_idx = self._full_index(frame_idx)
        merged_row = (
            self._merged_rows[full_idx]
            if 0 <= full_idx < len(self._merged_rows)
            else {}
        )
        cam_errors = self._frame_errors.get(full_idx, {})

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
        self._update_slider()

    def _slider_take(self) -> str | None:
        """The take the slider spans: the current frame's, else the selected one."""
        rows = self._calib_provider.merged_rows
        if 0 <= self._current_frame < len(rows):
            take = row_take(rows[self._current_frame])
            if take is not None:
                return take
        name = self._take_combo.currentText()
        if name and name != JOINT_TAKE:
            return name
        return self._takes[0].name if self._takes else None

    def _update_slider(self) -> None:
        """Point the slider at the current frame's take, position, and detections."""
        take = self._slider_take()
        last = max(0, self._take_lengths.get(take, 0) - 1)
        self._slider.set_range(last)
        self._update_range_controls(self._take_by_name(take), last)

        rows = self._calib_provider.merged_rows
        key = row_key(rows[self._current_frame]) if self._current_frame < len(rows) else None
        self._slider.set_value(key[1] if key and key[0] == take else 0)

        # Rows only carry corners once detection has run (or corners loaded).
        marks: dict[str, dict[int, str]] = {cam: {} for cam in self._project.cameras}
        for row in rows:
            key = row_key(row)
            if key is None or key[0] != take:
                continue
            for cam, cell in row.items():
                if cell.get("corners") is not None and cam in marks:
                    marks[cam][key[1]] = "user"
        self._slider.set_marks(marks)

    def _update_range_controls(self, take: CalibrationTake | None, last: int) -> None:
        """Show `take`'s range in the spin boxes and as slider brackets."""
        enabled = take is not None and self._take_lengths.get(take.name, 0) > 0
        self._range_take_label.setText(take.name if take is not None else "—")
        self._start_spin.setEnabled(enabled)
        self._end_spin.setEnabled(enabled)
        if not enabled:
            self._slider.clear_bounds()
            return

        start = min(take.start, last)
        end = last if take.end is None else max(start, min(take.end, last))
        for sb, value in ((self._start_spin, start), (self._end_spin, end)):
            sb.blockSignals(True)
            sb.setRange(0, last)
            sb.setValue(value)
            sb.blockSignals(False)
        self._slider.set_bounds(start, end)

    def _on_slider_seek(self, frame: int) -> None:
        """Jump to the listed frame nearest the clicked synchronized frame."""
        take = self._slider_take()
        idx = nearest_row(self._calib_provider.merged_rows, take, frame)
        if idx is None:
            self._update_slider()
        elif idx != self._current_frame:
            self._show_frame(idx)
        else:
            self._update_slider()   # snap the handle back onto the listed frame

    def _representative_frame_info(
        self, frame_idx: int
    ) -> tuple[str, tuple[int, int] | None]:
        """Return (display_label, (take_name, frame_in_video)) for frame_idx.

        Falls back to the provider's raw rows when no corners are detected yet.
        Returns ("—", None) when frame_idx is out of range.
        With one take: ("Frame N", (take, N)).
        With several: ("{take}, frame N", (take, N)).
        """
        rows = self._calib_provider.merged_rows
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
            if isinstance(framenum, tuple):
                take_name, fi = framenum
                if len(self._takes) > 1:
                    return f"{take_name}, frame {fi}", (take_name, fi)
                return f"Frame {fi}", (take_name, fi)

        return f"{frame_idx + 1}", None

    # ── Corner selection + epipolar lines ─────────────────────────────────

    def _on_corner_clicked(self, cam: str, _instance: int, corner_id_str: str):
        """Select a corner and draw its epipolar lines in all other cameras."""
        try:
            corner_idx = int(corner_id_str)
        except ValueError:
            return

        # Look up the corner's pixel position
        full_idx = self._full_index(self._current_frame)
        if 0 <= full_idx < len(self._merged_rows):
            merged_row = self._merged_rows[full_idx]
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

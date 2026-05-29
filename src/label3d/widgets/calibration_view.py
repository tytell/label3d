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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from label3d.geometry import EpipolarCache, cameras_from_cgroup
from label3d.project import Project
from label3d.video import VideoReader
from label3d.widgets.camera_grid import CameraGrid
from label3d.widgets.charuco_overlay import CharucoOverlay
from label3d.workers.calibration_worker import CalibrationWorker, DetectionWorker

# Default Charuco board params (from DESIGN.md)
_CHARUCO_DEFAULTS = {
    "squares_x": 6,
    "squares_y": 8,
    "square_length": 30.0,
    "marker_length": 22.0,
    "marker_bits": 4,
    "dict_size": 50,
}


# ── Frame provider for calibration images / video frames ─────────────────────

class CalibFrameProvider:
    """Serves calibration images for each merged detection frame.

    For image-based calibration: reads the file named in row['fname'].
    For video-based calibration: seeks to row['framenum'] = (video_idx, frame_num).
    Returns a black frame when a camera has no detection at the given index.
    """

    def __init__(self):
        self._merged: list[dict] = []
        self._cam_sizes: dict[str, tuple[int, int]] = {}
        self._cam_media: dict[str, list[Path]] = {}
        self._video_readers: dict[tuple, VideoReader] = {}

    def setup(
        self,
        merged_rows: list[dict],
        cam_sizes: dict[str, tuple[int, int]],
        cam_media: dict[str, list[Path]],
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
            return img if img is not None else blank

        # Video — framenum is (video_idx, frame_in_video)
        framenum = row.get("framenum")
        if framenum is None:
            return blank

        if isinstance(framenum, tuple):
            vid_idx, frame_in_vid = framenum
        else:
            vid_idx, frame_in_vid = 0, int(framenum)

        files = self._cam_media.get(camera, [])
        if vid_idx >= len(files):
            return blank

        reader_key = (camera, vid_idx)
        if reader_key not in self._video_readers:
            try:
                self._video_readers[reader_key] = VideoReader(Path(files[vid_idx]))
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
        self._cam_sizes: dict[str, tuple[int, int]] = {}
        self._cam_media: dict[str, list[Path]] = {}
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

        self._skip_spin = QSpinBox()
        self._skip_spin.setRange(1, 100)
        self._skip_spin.setValue(20)
        self._skip_spin.setToolTip("For video: process every Nth frame")
        form.addRow("Video skip (frames):", self._skip_spin)

        left_layout.addWidget(board_group)

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
        self._save_btn = QPushButton("Save")
        self._calibrate_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        btn_row.addWidget(self._detect_btn)
        btn_row.addWidget(self._calibrate_btn)
        btn_row.addWidget(self._save_btn)
        left_layout.addLayout(btn_row)

        # Overall error label
        self._error_label = QLabel("Reprojection error: —")
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self._error_label)

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
        self._frame_label.setMinimumWidth(80)
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
        self._save_btn.clicked.connect(self._on_save)
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
                panel.charuco_overlay = overlay
                self._charuco_overlays[cam] = overlay

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

    # ── Detection ─────────────────────────────────────────────────────────

    def _on_detect(self):
        if self._detection_worker and self._detection_worker.isRunning():
            return
        self._detect_btn.setEnabled(False)
        self._calibrate_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
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

        self._detection_worker = DetectionWorker(
            project_root=self._project.root,
            cam_names=self._project.cameras,
            board_params=self._board_params(),
            active_experiment=self._project.active_experiment,
            skip=self._skip_spin.value(),
        )
        self._detection_worker.progress.connect(self._log_msg)
        self._detection_worker.camera_done.connect(self._on_camera_done)
        self._detection_worker.finished.connect(self._on_detect_finished)
        self._detection_worker.error_occurred.connect(self._on_error)
        self._detection_worker.start()

    def _on_camera_done(self, cam_name: str, n_dets: int, _sample_img):
        lbl = self._cam_count_labels.get(cam_name)
        if lbl:
            lbl.setText(str(n_dets))

    def _on_detect_finished(
        self, all_rows: dict, cam_sizes: dict, cam_media: dict
    ):
        self._all_rows = all_rows
        self._cam_sizes = cam_sizes
        self._cam_media = cam_media

        # Merge rows across cameras
        try:
            from aniposelib.boards import merge_rows
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
        self._save_btn.setEnabled(False)
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
        self._calibration_worker.start()

    def _on_calibrate_finished(self, cgroup, mean_error: float, frame_errors: dict):
        self._cgroup = cgroup
        self._frame_errors = frame_errors
        self._error_label.setText(f"Reprojection error: {mean_error:.3f} px")

        # Build epipolar cache from calibrated CameraGroup
        try:
            cam_params = cameras_from_cgroup(cgroup)
            self._epipolar_cache = EpipolarCache(cam_params)
        except Exception as exc:
            self._log_msg(f"  Warning: could not build epipolar cache: {exc}")

        self._log_msg("=== Calibration complete ===")

        # Refresh current frame with error coloring
        self._show_frame(self._current_frame)

        self._calibrate_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

    # ── Save ──────────────────────────────────────────────────────────────

    def _on_save(self):
        if self._cgroup is None:
            return
        out_path = self._project.root / "experiments" / self._project.active_experiment / "calibration.toml"
        try:
            self._cgroup.dump(str(out_path))
            self._log_msg(f"Saved: {out_path}")
            self._save_btn.setEnabled(False)
        except Exception as exc:
            self._on_error(str(exc))

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
            self._frame_label.setText(f"{self._current_frame + 1} / {n}")
        else:
            self._frame_label.setText("—")

    # ── Corner selection + epipolar lines ─────────────────────────────────

    def _on_corner_clicked(self, cam: str, _instance: int, corner_id_str: str):
        """Select a corner and draw its epipolar lines in all other cameras."""
        try:
            corner_idx = int(corner_id_str)
        except ValueError:
            return

        # Look up the corner's pixel position
        merged_row = (
            self._merged_rows[self._current_frame]
            if 0 <= self._current_frame < len(self._merged_rows)
            else {}
        )
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
        self._detect_btn.setEnabled(True)
        self._calibrate_btn.setEnabled(self._all_rows is not None)

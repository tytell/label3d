"""Calibration view: detect Charuco corners and run bundle adjustment.

Layout:
  QSplitter (horizontal)
  ├── Left panel  — board settings, buttons, progress log
  └── Right panel — scrollable per-camera status widgets
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from label3d.project import Project
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


def _ndarray_to_pixmap(bgr: np.ndarray, max_width: int = 320) -> QPixmap:
    """Convert a BGR numpy image to a QPixmap, scaled to max_width."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    q_img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img)
    if w > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
    return pixmap


class _CameraStatusWidget(QWidget):
    """Shows one camera's calibration status: name, detection count, sample image."""

    def __init__(self, cam_name: str, parent=None):
        super().__init__(parent)
        self._cam_name = cam_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Sample image
        self._image_label = QLabel()
        self._image_label.setFixedSize(160, 90)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background: #111; border: 1px solid #444;")
        self._image_label.setText("No image")
        layout.addWidget(self._image_label)

        # Text info
        info_layout = QVBoxLayout()
        self._name_label = QLabel(f"<b>{cam_name}</b>")
        self._status_label = QLabel("No detections")
        self._error_label = QLabel("")
        info_layout.addWidget(self._name_label)
        info_layout.addWidget(self._status_label)
        info_layout.addWidget(self._error_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

    def set_detection(self, n_detections: int, sample_image: np.ndarray | None):
        self._status_label.setText(f"{n_detections} detections")
        if sample_image is not None:
            pix = _ndarray_to_pixmap(sample_image, max_width=160)
            self._image_label.setPixmap(
                pix.scaled(
                    self._image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._image_label.setText("")

    def set_error(self, error_px: float):
        self._error_label.setText(f"Error: {error_px:.3f} px")

    def clear_error(self):
        self._error_label.setText("")


class CalibrationView(QWidget):
    """Full calibration panel: board settings, detect, calibrate, save."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._all_rows: dict[str, list] | None = None
        self._cam_sizes: dict[str, tuple[int, int]] = {}
        self._cgroup = None
        self._detection_worker: DetectionWorker | None = None
        self._calibration_worker: CalibrationWorker | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────────
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

        # Skip spinbox
        self._skip_spin = QSpinBox()
        self._skip_spin.setRange(1, 100)
        self._skip_spin.setValue(20)
        self._skip_spin.setToolTip("For video: process every Nth frame")
        form.addRow("Video skip (frames):", self._skip_spin)

        left_layout.addWidget(board_group)

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

        # ── Right panel — per-camera status ──────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        cam_container = QWidget()
        self._cam_layout = QVBoxLayout(cam_container)
        self._cam_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._cam_widgets: dict[str, _CameraStatusWidget] = {}
        for cam in project.cameras:
            w = _CameraStatusWidget(cam)
            self._cam_widgets[cam] = w
            self._cam_layout.addWidget(w)

        self._cam_layout.addStretch()
        scroll.setWidget(cam_container)
        splitter.addWidget(scroll)

        splitter.setSizes([320, 480])

        # Connect buttons
        self._detect_btn.clicked.connect(self._on_detect)
        self._calibrate_btn.clicked.connect(self._on_calibrate)
        self._save_btn.clicked.connect(self._on_save)

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
        self._cgroup = None
        self._error_label.setText("Reprojection error: —")
        for w in self._cam_widgets.values():
            w.clear_error()
        self._log_msg("=== Corner detection started ===")

        self._detection_worker = DetectionWorker(
            project_root=self._project.root,
            cam_names=self._project.cameras,
            board_params=self._board_params(),
            skip=self._skip_spin.value(),
        )
        self._detection_worker.progress.connect(self._log_msg)
        self._detection_worker.camera_done.connect(self._on_camera_done)
        self._detection_worker.finished.connect(self._on_detect_finished)
        self._detection_worker.error_occurred.connect(self._on_error)
        self._detection_worker.start()

    def _on_camera_done(self, cam_name: str, n_dets: int, sample_img):
        w = self._cam_widgets.get(cam_name)
        if w:
            w.set_detection(n_dets, sample_img)

    def _on_detect_finished(self, all_rows: dict, cam_sizes: dict):
        self._all_rows = all_rows
        self._cam_sizes = cam_sizes
        total = sum(len(v) for v in all_rows.values())
        self._log_msg(f"=== Detection done — {total} total detections ===")
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

    def _on_calibrate_finished(self, cgroup, mean_error: float):
        self._cgroup = cgroup
        self._error_label.setText(f"Reprojection error: {mean_error:.3f} px")

        # Per-camera errors
        try:
            cam_names = cgroup.get_names()
            rows_list = [self._all_rows.get(c, []) for c in cam_names]
            self._log_msg("=== Calibration complete ===")
        except Exception:
            pass

        self._calibrate_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

    # ── Save ──────────────────────────────────────────────────────────────

    def _on_save(self):
        if self._cgroup is None:
            return
        out_path = self._project.root / "calibration.toml"
        try:
            self._cgroup.dump(str(out_path))
            self._log_msg(f"Saved: {out_path}")
            self._save_btn.setEnabled(False)
        except Exception as exc:
            self._on_error(str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)

    def _on_error(self, msg: str):
        self._log_msg(f"ERROR: {msg}")
        self._detect_btn.setEnabled(True)
        self._calibrate_btn.setEnabled(self._all_rows is not None)

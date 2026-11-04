"""Align tab (Phase 16.6): click an object's origin and axes, compute and apply an alignment.

The user picks an axis take and frame, then clicks each target (origin and
one or two points per axis) in at least two cameras. Clicks are saved as they
are made (alignment_picks). Compute triangulates them with the active
calibration (alignment.compute_alignment) and draws the new axes on every
camera; Apply saves the alignment in project.toml. Recompute (Phase 16.8)
goes back to the take and frame the alignment in use was clicked on and
computes again from its saved clicks with the active calibration, for the user
to Apply. The calibration and the points file are never changed.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPen, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsSimpleTextItem,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from label3d import alignment as alignment_record
from label3d.alignment import AlignmentSettings, Computation, compute_alignment
from label3d.alignment_picks import ORIGIN, Picks, read_picks, targets_for, write_picks
from label3d.calibration_registry import find_calibration
from label3d.frames import shown_frame, stored_frame
from label3d.geometry import (
    AXES,
    AlignmentError,
    CameraParams,
    EpipolarCache,
    load_cameras,
    unalign_points,
)
from label3d.project import Project, TrialInfo, _is_blank_cell
from label3d.video import FrameProvider, frame_count
from label3d.widgets.add_axis_take_dialog import AddAxisTakeDialog
from label3d.widgets.camera_grid import CameraGrid
from label3d.widgets.epipolar_overlay import camera_color

AXIS_COLORS = {"x": QColor(240, 80, 80), "y": QColor(80, 220, 80), "z": QColor(90, 150, 255)}
ORIGIN_COLOR = QColor(255, 255, 255)
SCALE_WARNING_PCT = 2.0
# Recompute warns when the clicks' largest error grows past both of these
RECOMPUTE_ERROR_FACTOR = 2.0   # times the alignment's max_error_px
RECOMPUTE_ERROR_MIN_PX = 1.0   # px above it
REMOVE_RADIUS_PX = 12      # screen pixels around a pick that a right click removes
MARKER_RADIUS_PX = 6

MODES = [(1, "1 point per axis"), (2, "2 points per axis")]
THIRD_AXIS = [("", "Clicked"), ("x", "X from Y × Z"), ("y", "Y from Z × X"), ("z", "Z from X × Y")]
PRIMARY = [("longest", "Longest (auto)"), ("lowest_error", "Lowest error (auto)"),
           ("x", "X"), ("y", "Y"), ("z", "Z")]


def target_label(target: str, points_per_axis: int) -> str:
    """Origin, X or X near / X far, … for the pick table."""
    if target == ORIGIN:
        return "Origin"
    axis, n = target[0].upper(), target[1]
    if points_per_axis == 1:
        return axis
    return f"{axis} {'near' if n == '1' else 'far'}"


def marker_text(target: str) -> str:
    return "O" if target == ORIGIN else target.upper()


def target_color(target: str) -> QColor:
    return ORIGIN_COLOR if target == ORIGIN else AXIS_COLORS[target[0]]


def rotation_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation taking R a to R b."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class _Overlay:
    """The Align tab's drawing on one camera panel: picks, epipolar lines, axes."""

    def __init__(self, panel):
        self._panel = panel
        self._items: list[QGraphicsItem] = []

    def clear(self) -> None:
        scene = self._panel.scene()
        for item in self._items:
            scene.removeItem(item)
        self._items.clear()

    def _add(self, item: QGraphicsItem, z: float = 50) -> None:
        item.setZValue(z)
        self._panel.scene().addItem(item)
        self._items.append(item)

    def marker(self, x: float, y: float, text: str, color: QColor, active: bool, hollow: bool = False):
        r = MARKER_RADIUS_PX
        dot = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        dot.setPos(x, y)
        pen = QPen(QColor(0, 0, 0) if not hollow else color, 2.5 if active else 1.5)
        pen.setCosmetic(True)
        dot.setPen(pen)
        dot.setBrush(QBrush(Qt.BrushStyle.NoBrush) if hollow else QBrush(color))
        self._add(dot, 60)
        if text:
            label = QGraphicsSimpleTextItem(text)
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setBrush(color)
            font = QFont(label.font())
            font.setBold(True)
            font.setPointSize(11)
            label.setFont(font)
            label.setPos(x, y)
            label.setTransform(label.transform().translate(r + 2, -r - 14))
            self._add(label, 61)

    def line(self, p1, p2, color: QColor, width: float = 1.5, alpha: int = 180) -> None:
        item = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
        c = QColor(color)
        c.setAlpha(alpha)
        pen = QPen(c, width)
        pen.setCosmetic(True)
        item.setPen(pen)
        self._add(item, 55)


class AxisAlignmentView(QWidget):
    """The Align tab. Emits alignment_applied after Apply saved an alignment."""

    alignment_applied = Signal()

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._cameras: dict[str, CameraParams] = {}
        self._epipolar: EpipolarCache | None = None
        self._provider = FrameProvider()
        self._take: TrialInfo | None = None
        self._frame = 0
        self._picks: Picks = {}
        self._active: str | None = None
        self._computation: Computation | None = None
        self._overlays: dict[str, _Overlay] = {}
        self._loaded = False
        self._settings = AlignmentSettings()

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        self._grid = CameraGrid()
        self._grid.set_frame_provider(self._provider)
        splitter.addWidget(self._grid)
        splitter.addWidget(self._side_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([900, 360])

        escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        escape.activated.connect(lambda: self.set_active_target(None))
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            QShortcut(QKeySequence(key), self).activated.connect(self.next_target)

    # -- Layout ----------------------------------------------------------------

    def _side_panel(self) -> QWidget:
        side = QWidget()
        side.setMinimumWidth(320)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(8, 8, 8, 8)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        take_box = QGroupBox("Axis take")
        take_form = QFormLayout(take_box)
        take_row = QHBoxLayout()
        self._take_combo = QComboBox()
        self._take_combo.currentTextChanged.connect(self._on_take_changed)
        take_row.addWidget(self._take_combo, stretch=1)
        add = QPushButton("Add take…")
        add.setToolTip("Add images or videos of the axis object, or a frame of a take or trial")
        add.clicked.connect(self._on_add_take)
        take_row.addWidget(add)
        take_form.addRow("Take:", take_row)
        self._frame_spin = QSpinBox()
        self._frame_spin.valueChanged.connect(self._on_frame_changed)
        self._frame_label = QLabel("Frame:")
        take_form.addRow(self._frame_label, self._frame_spin)
        layout.addWidget(take_box)

        settings_box = QGroupBox("Axes")
        form = QFormLayout(settings_box)
        self._mode = QComboBox()
        for value, text in MODES:
            self._mode.addItem(text, value)
        form.addRow("Clicks:", self._mode)
        self._third = QComboBox()
        for value, text in THIRD_AXIS:
            self._third.addItem(text, value)
        form.addRow("Third axis:", self._third)
        self._primary = QComboBox()
        for value, text in PRIMARY:
            self._primary.addItem(text, value)
        self._primary.setToolTip("The axis kept exactly; the others are made perpendicular to it")
        form.addRow("Primary axis:", self._primary)
        flips = QHBoxLayout()
        self._flips: dict[str, QCheckBox] = {}
        for axis in AXES:
            box = QCheckBox(axis.upper())
            flips.addWidget(box)
            self._flips[axis] = box
        flips.addStretch()
        form.addRow("Flip:", flips)
        lengths = QHBoxLayout()
        self._lengths: dict[str, QDoubleSpinBox] = {}
        for axis in AXES:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10000.0)
            spin.setDecimals(2)
            spin.setSpecialValueText("—")
            spin.setToolTip(f"Known distance between the {axis} points, cm (— for none)")
            lengths.addWidget(QLabel(axis.upper()))
            lengths.addWidget(spin)
            self._lengths[axis] = spin
        form.addRow("Known cm:", lengths)
        layout.addWidget(settings_box)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Point", "Cameras", "Error (px)"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self._table)

        hint = QLabel("Click a point in each camera; right-click a point to remove it. "
                      "Enter: next point, Esc: stop clicking.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QGridLayout()
        self._next_button = QPushButton("Next point")
        self._next_button.clicked.connect(self.next_target)
        self._clear_button = QPushButton("Clear clicks")
        self._clear_button.clicked.connect(self.clear_picks)
        self._compute_button = QPushButton("Compute")
        self._compute_button.clicked.connect(self.compute)
        self._apply_button = QPushButton("Apply")
        self._apply_button.setToolTip("Save this alignment; exported 3D points use it")
        self._apply_button.clicked.connect(self.apply)
        self._recompute_button = QPushButton("Recompute")
        self._recompute_button.setToolTip(
            "Compute the alignment in use again from its saved clicks with the active "
            "calibration; Apply to keep it"
        )
        self._recompute_button.clicked.connect(self.recompute)
        buttons.addWidget(self._next_button, 0, 0)
        buttons.addWidget(self._clear_button, 0, 1)
        buttons.addWidget(self._compute_button, 1, 0)
        buttons.addWidget(self._apply_button, 1, 1)
        buttons.addWidget(self._recompute_button, 2, 0, 1, 2)
        layout.addLayout(buttons)

        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._results.setMinimumHeight(160)
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._results.setFont(font)
        layout.addWidget(self._results, stretch=1)

        for widget in (self._mode, self._third, self._primary):
            widget.currentIndexChanged.connect(self._on_settings_changed)
        for box in self._flips.values():
            box.toggled.connect(self._on_settings_changed)
        for spin in self._lengths.values():
            spin.valueChanged.connect(self._on_settings_changed)
        return side

    # -- Loading ---------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._loaded:
            self.reload_for_experiment()
        # Hidden panels skip frame updates; draw the frame once the layout settles
        QTimer.singleShot(0, lambda: self._grid.update_frame(self._frame))

    def reload_for_experiment(self) -> None:
        """Pick up the active experiment: cameras, calibration, settings and takes."""
        self._loaded = True
        self._grid.setup_cameras(self._project.cameras)
        self._overlays = {
            cam: _Overlay(self._grid.get_panel(cam)) for cam in self._project.cameras
        }
        for cam in self._project.cameras:
            panel = self._grid.get_panel(cam)
            panel.raw_clicked.connect(self.add_pick)
            panel.empty_right_clicked.connect(self.remove_pick_near)
        self._load_settings()
        self.reload_calibration()
        self.refresh_takes()

    def reload_calibration(self) -> None:
        """Use the active calibration for triangulation and epipolar lines."""
        path = self._project.calibration_path()
        self._cameras = load_cameras(path) if path is not None else {}
        self._epipolar = EpipolarCache(self._cameras) if self._cameras else None
        self._computation = None
        self._results.clear()
        self.update_status()
        self._redraw()

    def refresh_takes(self, select: str | None = None) -> None:
        names = [t.name for t in self._project.axis_takes]
        current = select or self._take_combo.currentText()
        self._take_combo.blockSignals(True)
        self._take_combo.clear()
        self._take_combo.addItems(names)
        if current in names:
            self._take_combo.setCurrentText(current)
        self._take_combo.blockSignals(False)
        self._on_take_changed(self._take_combo.currentText())

    def _load_settings(self) -> None:
        settings = self._project.alignment_settings()
        widgets = [self._mode, self._third, self._primary, *self._flips.values(), *self._lengths.values()]
        for w in widgets:
            w.blockSignals(True)
        self._mode.setCurrentIndex(self._mode.findData(settings.points_per_axis))
        self._third.setCurrentIndex(self._third.findData(settings.cross_product))
        self._primary.setCurrentIndex(self._primary.findData(settings.primary))
        for axis in AXES:
            self._flips[axis].setChecked(axis in settings.flip)
            self._lengths[axis].setValue(settings.known_length_cm.get(axis, 0.0))
        for w in widgets:
            w.blockSignals(False)
        self._settings = settings
        self._fill_table()

    def settings(self) -> AlignmentSettings:
        return AlignmentSettings(
            points_per_axis=int(self._mode.currentData()),
            cross_product=str(self._third.currentData()),
            primary=str(self._primary.currentData()),
            flip=[axis for axis in AXES if self._flips[axis].isChecked()],
            known_length_cm={axis: s.value() for axis, s in self._lengths.items() if s.value() > 0},
        )

    def _on_settings_changed(self, *_args) -> None:
        primary = self._primary.currentData()
        if primary == self._third.currentData():
            # A derived axis can't be the primary one
            self._primary.blockSignals(True)
            self._primary.setCurrentIndex(0)
            self._primary.blockSignals(False)
        self._settings = self.settings()
        self._project.save_alignment_settings(self._settings)
        self._computation = None
        self._fill_table()
        self._redraw()

    # -- Take and frame -------------------------------------------------------

    def _on_take_changed(self, name: str) -> None:
        self._take = next((t for t in self._project.axis_takes if t.name == name), None)
        self._provider.close()
        length = 1
        if self._take is not None:
            paths, offsets = {}, {}
            for cam, (filename, start, _end) in self._take.camera_videos.items():
                if _is_blank_cell(filename):
                    continue
                paths[cam] = self._project.video_path(self._take.name, cam)
                offsets[cam] = start
            try:
                self._provider.open_trial(paths, offsets)
            except IOError as e:
                self._results.setPlainText(str(e))
            length = self._take_length(self._take)
        self._frame_spin.blockSignals(True)
        self._frame_spin.setRange(shown_frame(0), shown_frame(max(length - 1, 0)))
        self._frame_spin.setValue(shown_frame(0))
        self._frame_spin.blockSignals(False)
        self._frame_spin.setVisible(length > 1)
        self._frame_label.setVisible(length > 1)
        self.set_frame(0)

    def _take_length(self, take: TrialInfo) -> int:
        lengths = []
        for cam, (filename, start, end) in take.camera_videos.items():
            if _is_blank_cell(filename):
                continue
            last = end if end is not None else frame_count(self._project.video_path(take.name, cam)) - 1
            lengths.append(last - start + 1)
        return max(min(lengths), 1) if lengths else 1

    def _on_frame_changed(self, value: int) -> None:
        self.set_frame(stored_frame(value))

    def set_frame(self, frame: int) -> None:
        """Show a frame (0-based) of the take with its saved clicks."""
        self._frame = frame
        self._picks = (
            read_picks(self._exp_dir(), self._take.name, frame) if self._take is not None else {}
        )
        self._computation = None
        self._grid.update_frame(frame)
        self._fill_table()
        targets = self._targets()
        first_open = next((t for t in targets if len(self._picks.get(t, {})) < 2), None)
        self.set_active_target(first_open if self._take is not None else None)

    def _exp_dir(self):
        return self._project.experiments[self._project.active_experiment].root

    def _take_cameras(self) -> list[str]:
        if self._take is None:
            return []
        return [cam for cam, (f, _s, _e) in self._take.camera_videos.items()
                if not _is_blank_cell(f) and cam in self._project.cameras]

    def _on_add_take(self) -> None:
        dialog = AddAxisTakeDialog(self._project, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_takes(select=dialog.take())

    # -- Picking -------------------------------------------------------------

    def _targets(self) -> list[str]:
        return targets_for(self._settings.points_per_axis, self._settings.cross_product)

    def active_target(self) -> str | None:
        return self._active

    def picks(self) -> Picks:
        return {t: dict(c) for t, c in self._picks.items()}

    def set_active_target(self, target: str | None) -> None:
        """Click into this target; None stops clicking (the panels pan again)."""
        if target is not None and target not in self._targets():
            target = None
        self._active = target
        for cam in self._project.cameras:
            panel = self._grid.get_panel(cam)
            if panel is not None:
                panel.set_pick_mode(target is not None)
        self._select_table_row(target)
        self._redraw()

    def add_pick(self, camera: str, x: float, y: float) -> None:
        """Set this camera's click for the active target; on to the next target once all cameras have it."""
        if self._active is None or self._take is None:
            return
        self._picks.setdefault(self._active, {})[camera] = (float(x), float(y))
        self._save_picks()
        if set(self._take_cameras()) <= set(self._picks[self._active]):
            self.next_target()
        else:
            self._fill_table()
            self._redraw()

    def remove_pick_near(self, camera: str, x: float, y: float) -> None:
        """Remove the click in this camera nearest (x, y), if one is close on screen."""
        panel = self._grid.get_panel(camera)
        scale = panel.transform().m11() if panel is not None else 1.0
        radius = REMOVE_RADIUS_PX / max(scale, 1e-9)
        best, best_d = None, radius
        for target, per_cam in self._picks.items():
            if camera in per_cam:
                px, py = per_cam[camera]
                d = math.hypot(px - x, py - y)
                if d <= best_d:
                    best, best_d = target, d
        if best is None:
            return
        del self._picks[best][camera]
        if not self._picks[best]:
            del self._picks[best]
        self._save_picks()
        self._fill_table()
        self._redraw()

    def next_target(self) -> None:
        """The next target in the list after the active one (wrapping), else stop."""
        targets = self._targets()
        if not targets:
            return
        start = targets.index(self._active) + 1 if self._active in targets else 0
        order = targets[start:] + targets[:start]
        cams = set(self._take_cameras())
        nxt = next((t for t in order if not cams <= set(self._picks.get(t, {}))), None)
        self._fill_table()
        self.set_active_target(nxt)

    def clear_picks(self) -> None:
        """Remove every click on this frame of the take."""
        self._picks = {}
        self._save_picks()
        self._computation = None
        self._fill_table()
        self.set_active_target(self._targets()[0] if self._take is not None else None)

    def _save_picks(self) -> None:
        self._computation = None
        if self._take is not None:
            write_picks(self._exp_dir(), self._take.name, self._frame, self._picks)

    def _on_table_selection(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        target = self._table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        if target != self._active:
            self.set_active_target(target)

    def _select_table_row(self, target: str | None) -> None:
        self._table.blockSignals(True)
        self._table.clearSelection()
        for row in range(self._table.rowCount()):
            if self._table.item(row, 0).data(Qt.ItemDataRole.UserRole) == target:
                self._table.selectRow(row)
        self._table.blockSignals(False)

    def _fill_table(self) -> None:
        targets = self._targets()
        self._table.blockSignals(True)
        self._table.setRowCount(len(targets))
        n_cams = len(self._take_cameras())
        for row, target in enumerate(targets):
            name = QTableWidgetItem(target_label(target, self._settings.points_per_axis))
            name.setData(Qt.ItemDataRole.UserRole, target)
            name.setForeground(target_color(target) if target != ORIGIN else QColor(120, 120, 120))
            self._table.setItem(row, 0, name)
            self._table.setItem(row, 1, QTableWidgetItem(f"{len(self._picks.get(target, {}))} / {n_cams}"))
            error = self._computation.mean_error(target) if self._computation else None
            self._table.setItem(row, 2, QTableWidgetItem("" if error is None else f"{error:.2f}"))
        self._table.blockSignals(False)
        self._select_table_row(self._active)

    # -- Drawing -----------------------------------------------------------------

    def _redraw(self) -> None:
        cams = self._project.cameras
        for cam, overlay in self._overlays.items():
            overlay.clear()
            for target, per_cam in self._picks.items():
                if cam in per_cam:
                    x, y = per_cam[cam]
                    overlay.marker(x, y, marker_text(target), target_color(target), target == self._active)
            # Epipolar lines from the active target's clicks in the other cameras
            if self._active is not None and self._epipolar is not None and cam in self._cameras:
                for other, (x, y) in self._picks.get(self._active, {}).items():
                    if other == cam or other not in self._cameras:
                        continue
                    ends = self._epipolar.get_epipolar_line(other, cam, (x, y))
                    if ends is not None:
                        overlay.line(ends[0], ends[1], camera_color(cams.index(other)))
            if self._computation is not None and cam in self._cameras:
                self._draw_computation(cam, overlay)

    def _draw_computation(self, cam: str, overlay: _Overlay) -> None:
        computation, camera = self._computation, self._cameras[cam]
        for target, point in computation.points.items():
            x, y = camera.project(point[np.newaxis])[0]
            overlay.marker(float(x), float(y), "", target_color(target), False, hollow=True)
        result = computation.result
        lengths = {
            axis: self._settings.known_length_cm.get(axis) or float(np.linalg.norm(vec))
            for axis, vec in computation.axes.items()
        }
        default = float(np.mean(list(lengths.values())))
        tips = np.array([np.eye(3)[i] * lengths.get(axis, default) for i, axis in enumerate(AXES)])
        ends = unalign_points(np.vstack([np.zeros((1, 3)), tips]), result.R_align, result.origin)
        pixels = camera.project(ends)
        origin = pixels[0]
        for i, axis in enumerate(AXES):
            overlay.line(origin, pixels[i + 1], AXIS_COLORS[axis], width=3.0, alpha=230)
            overlay.marker(float(pixels[i + 1][0]), float(pixels[i + 1][1]), axis, AXIS_COLORS[axis], False, hollow=True)

    # -- Compute and Apply -------------------------------------------------------

    def computation(self) -> Computation | None:
        return self._computation

    def compute(self) -> Computation | None:
        """Triangulate the clicks with the active calibration and show the alignment."""
        self._computation = None
        if not self._cameras:
            self._results.setPlainText("The experiment has no active calibration.")
            return None
        try:
            self._computation = compute_alignment(self._cameras, self._picks, self._settings)
        except AlignmentError as e:
            self._results.setPlainText(str(e))
            self._fill_table()
            self._redraw()
            return None
        self._results.setPlainText(self._report(self._computation))
        self._fill_table()
        self._redraw()
        return self._computation

    def _report(self, computation: Computation) -> str:
        result = computation.result
        n = self._settings.points_per_axis
        lines = ["Reprojection error (px) by point:"]
        for target in self._targets():
            per_cam = computation.errors.get(target, {})
            cams = ", ".join(f"{c} {e:.2f}" for c, e in sorted(per_cam.items()))
            lines.append(f"  {target_label(target, n):8s} {cams}")
        lines.append("")
        if result.axis_angles_deg:
            angles = ", ".join(f"{p} {a:.1f}°" for p, a in result.axis_angles_deg.items())
            lines.append(f"Angles between clicked axes: {angles}")
        moved = ", ".join(f"{a} {d:.2f}°" for a, d in result.moved_deg.items())
        lines.append(f"Moved to make them perpendicular: {moved}")
        rule = {"longest": "the longest", "lowest_error": "the lowest error"}.get(
            self._settings.primary, "chosen")
        lines.append(f"Primary axis: {result.primary} ({rule}); then {result.secondary}"
                     + (f"; {result.derived} from the cross product" if result.derived else ""))
        for axis, check in computation.scale.items():
            flag = "  !" if abs(check["error_pct"]) > SCALE_WARNING_PCT else ""
            lines.append(f"Scale {axis}: measured {check['measured_cm']:.2f} cm, known "
                         f"{check['known_cm']:.2f} cm ({check['error_pct']:+.1f}%){flag}")
        lines.append("")
        lines.append("R_align (columns are the new axes):")
        for row in result.R_align:
            lines.append("  " + "  ".join(f"{v:+.4f}" for v in row))
        lines.append("Origin (cm): " + "  ".join(f"{v:.2f}" for v in result.origin))
        old = self._project.alignment()
        if old is not None:
            angle = rotation_between_deg(old.R_align, result.R_align)
            shift = float(np.linalg.norm(old.origin - result.origin))
            lines.append(f"Change from the alignment in use: {angle:.2f}° rotation, {shift:.2f} cm origin shift")
        return "\n".join(lines)

    def recompute(self) -> Computation | None:
        """Compute the alignment in use again from its saved clicks, with the active calibration.

        Shows the take and frame it was clicked on and the result, as after
        Compute; nothing is saved until Apply. Says so when the clicks fit the
        calibration much worse than when the alignment was made.
        """
        if not self._loaded:
            self.reload_for_experiment()
        alignment = self._project.alignment()
        source = self._project.alignment_source()
        if alignment is None or source is None:
            self._results.setPlainText(
                "There is no alignment to recompute: click the points, then Compute."
            )
            return None
        if source.take not in [t.name for t in self._project.axis_takes]:
            self._results.setPlainText(
                f"The axis take '{source.take}' the alignment was clicked on is gone; "
                "click the points again on another take."
            )
            return None
        self.show_take_frame(source.take, source.frame)
        if not self._picks:
            self._results.setPlainText(
                f"No clicks are saved for '{source.take}', frame {shown_frame(source.frame)}; "
                "click the points again."
            )
            return None
        computation = self.compute()
        if computation is None:
            return None
        lines = [
            f"Recomputed from the clicks saved for '{source.take}', frame "
            f"{shown_frame(source.frame)}, with '{self._calibration_label()}'. "
            "Apply to use it."
        ]
        old = alignment.max_error_px
        new = computation.max_error_px
        if old is not None and new > max(RECOMPUTE_ERROR_FACTOR * old, old + RECOMPUTE_ERROR_MIN_PX):
            lines.append(
                f"!  The clicks' largest reprojection error is {new:.2f} px, up from "
                f"{old:.2f} px when the alignment was made. The new calibration may not "
                "fit them; check the axes drawn on the cameras, and click again if they are off."
            )
        self._results.setPlainText("\n".join(lines) + "\n\n" + self._results.toPlainText())
        return computation

    def show_take_frame(self, take: str, frame: int) -> None:
        """Select an axis take and a frame (0-based) of it, with its saved clicks."""
        if self._take_combo.currentText() != take:
            self._take_combo.setCurrentText(take)
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(shown_frame(frame))
        self._frame_spin.blockSignals(False)
        self.set_frame(frame)

    def _calibration_label(self) -> str:
        entry_id = self._project.active_calibration_id()
        entry = find_calibration(self._exp_dir(), entry_id) if entry_id else None
        return entry.label if entry else (entry_id or "no calibration")

    def apply(self) -> bool:
        """Save the computed alignment against the active calibration."""
        if self._computation is None and self.compute() is None:
            return False
        calibration = self._project.active_calibration_id()
        if calibration is None or self._take is None:
            return False
        self._project.save_alignment(
            self._computation.alignment(calibration),
            self._project.alignment_source_for(self._take.name, self._frame),
        )
        self.update_status()
        self.alignment_applied.emit()
        return True

    def update_status(self) -> None:
        """Say whether there is an alignment and whether it is current."""
        project = self._project
        self._recompute_button.setEnabled(
            project.alignment() is not None and project.alignment_source() is not None
        )
        status = project.alignment_status()
        alignment = project.alignment()
        if status == alignment_record.NONE:
            text = "No alignment yet: exported 3D points are in the calibration's frame."
        else:
            exp_dir = self._exp_dir()
            computed = find_calibration(exp_dir, alignment.calibration)
            label = computed.label if computed else alignment.calibration
            if status == alignment_record.CURRENT:
                text = f"Alignment in use, computed with '{label}'."
            else:
                used = project.triangulated_with() or project.active_calibration_id()
                entry = find_calibration(exp_dir, used) if used else None
                now = entry.label if entry else (used or "no calibration")
                text = (f"<span style='color:#b45309'>The alignment was computed with '{label}'; "
                        f"the 3D points now come from '{now}'. Recompute it from the saved "
                        "clicks, or click again, then Apply.</span>")
        self._status.setText(text)

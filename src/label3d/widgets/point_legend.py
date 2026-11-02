"""PointLegend: sidebar key to point shapes, colors, marks and frame-slider ticks."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from label3d.points import SOURCE_LABEL3D, SOURCE_SLEAP_PRED, SOURCE_SLEAP_USER
from label3d.widgets.frame_slider import (
    SLIDER_COLOR_NEW_USER,
    SLIDER_COLOR_PREDICTED,
    SLIDER_COLOR_USER,
)
from label3d.widgets.point_item import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_NONE,
    COLOR_OK,
    COLOR_SOURCE,
    COLORMAP_NAMES,
    DEFAULT_POINT_COLOR,
    ERROR_BAD_PX,
    ERROR_OK_PX,
    MARK_COLORS,
    NEW_HALF_SIZE,
    NEW_PEN,
    SCORE_HIGH,
    SCORE_LOW,
    ColorMode,
    point_style,
    score_color,
)

ICON_SIZE = 28
# Icons sit on a dark swatch like the video background, so white marks show
SWATCH_COLOR = QColor(40, 42, 40)
SHAPE_COLOR = QColor(200, 200, 200)  # neutral color for the shape section

SHAPES = [
    ("set", "User label"),
    ("predicted", "Prediction"),
    ("not_visible", "Not visible"),
    ("not_set", "Not placed yet"),
]

ERROR_COLORS = [
    (COLOR_GOOD, f"Error < {ERROR_OK_PX} px"),
    (COLOR_OK, f"Error {ERROR_OK_PX}–{ERROR_BAD_PX} px"),
    (COLOR_BAD, f"Error > {ERROR_BAD_PX} px"),
    (COLOR_NONE, "Not triangulated"),
]

SOURCE_COLORS = [
    (COLOR_SOURCE[SOURCE_LABEL3D], "Labeled in label3d"),
    (COLOR_SOURCE[SOURCE_SLEAP_USER], "Labeled in SLEAP"),
    (COLOR_SOURCE[SOURCE_SLEAP_PRED], "Predicted by SLEAP"),
]

SCORE_STEPS = 5
SCORE_COLORS = [
    (score_color(score), f"Score {score:.2f}")
    for score in np.linspace(SCORE_HIGH, SCORE_LOW, SCORE_STEPS)
] + [(COLOR_NONE, "No score (user label)")]

SLIDER_TICKS = [
    (SLIDER_COLOR_USER, "User labels"),
    (SLIDER_COLOR_PREDICTED, "Predictions only"),
    (SLIDER_COLOR_NEW_USER, "New row: SLEAP labels"),
    (SLIDER_COLOR_PREDICTED, "New row: predictions"),
]


def point_icon(state: str, color: QColor, new: bool = False, selected: bool = False) -> QPixmap:
    """Draw a point marker exactly as PointItem does, on a dark swatch."""
    pixmap = QPixmap(ICON_SIZE, ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(SWATCH_COLOR)
    painter.drawRoundedRect(QRectF(0, 0, ICON_SIZE, ICON_SIZE), 4, 4)

    center = QPointF(ICON_SIZE / 2, ICON_SIZE / 2)
    brush, pen, radius = point_style(state, color, selected)
    painter.setBrush(brush)
    painter.setPen(pen)
    painter.drawEllipse(center, radius, radius)
    if new:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(NEW_PEN)
        h = min(NEW_HALF_SIZE, ICON_SIZE / 2 - 3)
        painter.drawRect(QRectF(center.x() - h, center.y() - h, 2 * h, 2 * h))
    painter.end()
    return pixmap


TICK_ICON_HEIGHT = 14


def tick_icon(color: QColor) -> QPixmap:
    """A frame-slider tick on the slider's grey row."""
    pixmap = QPixmap(ICON_SIZE, TICK_ICON_HEIGHT)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    top = TICK_ICON_HEIGHT // 2 - 3
    painter.fillRect(2, top, ICON_SIZE - 4, 6, QColor(220, 220, 220))
    painter.fillRect(ICON_SIZE // 2 - 1, top, 3, 6, color)
    painter.end()
    return pixmap


class PointLegend(QWidget):
    """Key to the Label tab's markers.

    Shows shapes (point state), the colors for the current View > Point color
    choice, the new/selected marks, and the frame-slider ticks.  The color
    section has a drop-down with the same choices; picking one emits
    coloring_chosen.  set_coloring() shows a choice made elsewhere.
    """

    # (ColorMode value, fixed color key or "") picked in the drop-down
    coloring_chosen = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color_mode = ColorMode.ERROR
        self._point_color = DEFAULT_POINT_COLOR

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QLabel("Legend")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self._sections: dict[str, QWidget] = {}
        self._icons: dict[str, list[QLabel]] = {}

        layout.addWidget(self._section(
            "shape", "Shape",
            [(point_icon(state, SHAPE_COLOR), text) for state, text in SHAPES],
        ))

        # Color: the colormaps, then the fixed colors
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 4, 0, 0)
        color_title = QLabel("Color:")
        color_title.setStyleSheet("font-size: 11px; font-weight: 600;")
        color_row.addWidget(color_title)
        self._color_combo = QComboBox()
        self._color_combo.setStyleSheet("font-size: 11px;")
        for mode, name in COLORMAP_NAMES.items():
            self._color_combo.addItem(name, (mode.value, ""))
        self._color_combo.insertSeparator(self._color_combo.count())
        for key, (name, _color) in MARK_COLORS.items():
            self._color_combo.addItem(name, (ColorMode.FIXED.value, key))
        self._color_combo.activated.connect(self._on_color_activated)
        color_row.addWidget(self._color_combo, stretch=1)
        layout.addLayout(color_row)

        layout.addWidget(self._section(
            ColorMode.ERROR.value, None,
            [(point_icon("set", color), text) for color, text in ERROR_COLORS],
        ))
        layout.addWidget(self._section(
            ColorMode.SOURCE.value, None,
            [(point_icon("set", color), text) for color, text in SOURCE_COLORS],
        ))
        layout.addWidget(self._section(
            ColorMode.SCORE.value, None,
            [(point_icon("set", color), text) for color, text in SCORE_COLORS],
        ))
        layout.addWidget(self._section(
            ColorMode.FIXED.value, None,
            [(point_icon("set", COLOR_NONE), "All points")],
        ))
        layout.addWidget(self._section(
            "marks", "Marks",
            [
                (point_icon("set", SHAPE_COLOR, new=True), "New since last sync"),
                (point_icon("set", SHAPE_COLOR, selected=True), "Selected"),
            ],
        ))
        layout.addWidget(self._section(
            "slider", "Frame slider",
            [(tick_icon(color), text) for color, text in SLIDER_TICKS],
        ))
        layout.addStretch(0)

        self.set_coloring(ColorMode.ERROR, DEFAULT_POINT_COLOR)

    @property
    def color_mode(self) -> ColorMode:
        return self._color_mode

    def set_coloring(self, mode: ColorMode, point_color: str) -> None:
        """Show mode's color section, with point_color (a MARK_COLORS key) for
        every point under FIXED and for the points a colormap leaves out."""
        self._color_mode = mode
        self._point_color = point_color
        choice = (mode.value, point_color if mode is ColorMode.FIXED else "")
        for i in range(self._color_combo.count()):
            if self._color_combo.itemData(i) == choice:
                self._color_combo.setCurrentIndex(i)
                break
        for m in ColorMode:
            self._sections[m.value].setVisible(m is mode)
        # "Not triangulated", "No score" and "All points" take the point color
        icon = point_icon("set", MARK_COLORS[point_color][1])
        for m in (ColorMode.ERROR, ColorMode.SCORE, ColorMode.FIXED):
            self._icons[m.value][-1].setPixmap(icon)

    def _on_color_activated(self, index: int) -> None:
        mode, key = self._color_combo.itemData(index)
        self.coloring_chosen.emit(mode, key)

    def visible_entries(self) -> list[str]:
        """Labels of the entries currently shown, in order (for tests)."""
        entries = []
        for section in self._sections.values():
            if section.isHidden():
                continue
            entries += [
                lbl.text() for lbl in section.findChildren(QLabel)
                if lbl.objectName() == "entry"
            ]
        return entries

    def _section(
        self, key: str, title: str | None, rows: list[tuple[QPixmap, str]]
    ) -> QWidget:
        section = QWidget()
        grid = QGridLayout(section)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        if title is not None:
            title_label = QLabel(title)
            title_label.setStyleSheet("font-size: 11px; font-weight: 600;")
            grid.addWidget(title_label, 0, 0, 1, 2)
        self._icons[key] = []
        for i, (pixmap, text) in enumerate(rows, start=1):
            icon = QLabel()
            icon.setPixmap(pixmap)
            icon.setFixedSize(pixmap.size())
            label = QLabel(text)
            label.setObjectName("entry")
            label.setStyleSheet("font-size: 11px;")
            grid.addWidget(icon, i, 0)
            self._icons[key].append(icon)
            grid.addWidget(label, i, 1)
        grid.setColumnStretch(1, 1)
        self._sections[key] = section
        return section

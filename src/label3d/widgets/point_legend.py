"""PointLegend: sidebar key to point shapes, colors, marks and frame-slider ticks."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

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
    ERROR_BAD_PX,
    ERROR_OK_PX,
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

    Shows shapes (point state), the colors for the current View > Color points
    by mode, the new/selected marks, and the frame-slider ticks.
    set_color_mode() swaps the color section.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color_mode = ColorMode.ERROR

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QLabel("Legend")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self._sections: dict[str, QWidget] = {}
        self._color_titles: dict[ColorMode, str] = {
            ColorMode.ERROR: "Color: reprojection error",
            ColorMode.SOURCE: "Color: source",
            ColorMode.SCORE: "Color: prediction score",
        }

        layout.addWidget(self._section(
            "shape", "Shape",
            [(point_icon(state, SHAPE_COLOR), text) for state, text in SHAPES],
        ))
        layout.addWidget(self._section(
            ColorMode.ERROR.value, self._color_titles[ColorMode.ERROR],
            [(point_icon("set", color), text) for color, text in ERROR_COLORS],
        ))
        layout.addWidget(self._section(
            ColorMode.SOURCE.value, self._color_titles[ColorMode.SOURCE],
            [(point_icon("set", color), text) for color, text in SOURCE_COLORS],
        ))
        layout.addWidget(self._section(
            ColorMode.SCORE.value, self._color_titles[ColorMode.SCORE],
            [(point_icon("set", color), text) for color, text in SCORE_COLORS],
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

        self.set_color_mode(ColorMode.ERROR)

    @property
    def color_mode(self) -> ColorMode:
        return self._color_mode

    def set_color_mode(self, mode: ColorMode) -> None:
        """Show only the color section for mode."""
        self._color_mode = mode
        for m in ColorMode:
            self._sections[m.value].setVisible(m is mode)

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

    def _section(self, key: str, title: str, rows: list[tuple[QPixmap, str]]) -> QWidget:
        section = QWidget()
        grid = QGridLayout(section)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 11px; font-weight: 600;")
        grid.addWidget(title_label, 0, 0, 1, 2)
        for i, (pixmap, text) in enumerate(rows, start=1):
            icon = QLabel()
            icon.setPixmap(pixmap)
            icon.setFixedSize(pixmap.size())
            label = QLabel(text)
            label.setObjectName("entry")
            label.setStyleSheet("font-size: 11px;")
            grid.addWidget(icon, i, 0)
            grid.addWidget(label, i, 1)
        grid.setColumnStretch(1, 1)
        self._sections[key] = section
        return section

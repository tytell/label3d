"""PointItem: QGraphicsEllipseItem subclass representing one labeled point."""

from __future__ import annotations

import math
from enum import Enum

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsRectItem

from label3d.points import SOURCE_LABEL3D, SOURCE_SLEAP_PRED, SOURCE_SLEAP_USER

# Point radius in pixels (screen-space, ignores view transform)
POINT_RADIUS = 8

# Reprojection error thresholds (px) for the error colors
ERROR_OK_PX = 5
ERROR_BAD_PX = 15

# Colors for reprojection error ranges
COLOR_GOOD = QColor(0, 200, 0)        # green: error < ERROR_OK_PX
COLOR_OK = QColor(255, 200, 0)        # yellow: ERROR_OK_PX to ERROR_BAD_PX
COLOR_BAD = QColor(255, 50, 50)       # red: error > ERROR_BAD_PX
COLOR_NONE = QColor(160, 160, 160)    # grey: no error info
COLOR_NOT_SET = QColor(100, 100, 100) # dim grey for points not yet positioned
COLOR_NOT_VISIBLE = QColor(80, 80, 80, 120)  # grayed out for not-visible points

# Colors by point source (View > Color points by > Source).  SLEAP predictions
# use SLEAP's prediction blue, as on the frame slider.
COLOR_SOURCE = {
    SOURCE_LABEL3D: QColor(230, 60, 230),      # magenta: labeled in label3d
    SOURCE_SLEAP_USER: QColor(255, 140, 0),    # orange: labeled in SLEAP
    SOURCE_SLEAP_PRED: QColor(1, 170, 247),    # light blue: predicted by SLEAP
}

SOURCE_NAMES = {
    SOURCE_LABEL3D: "label3d user",
    SOURCE_SLEAP_USER: "SLEAP user",
    SOURCE_SLEAP_PRED: "SLEAP prediction",
}

# Square outline around points that are new since the latest sync
NEW_PEN = QPen(QColor(255, 255, 255), 1.5)
NEW_HALF_SIZE = POINT_RADIUS + 4


class ColorMode(Enum):
    ERROR = "error"
    SOURCE = "source"


# Selection outline
SELECT_PEN = QPen(Qt.GlobalColor.white, 2)
NORMAL_PEN = QPen(Qt.GlobalColor.transparent, 0)


def _error_color(error: float | None) -> QColor:
    """Map reprojection error to a color."""
    if error is None:
        return COLOR_NONE
    if error < ERROR_OK_PX:
        return COLOR_GOOD
    if error < ERROR_BAD_PX:
        return COLOR_OK
    return COLOR_BAD


def point_style(state: str, color: QColor, selected: bool = False) -> tuple[QBrush, QPen, float]:
    """(brush, pen, radius) for a point marker.  Shared by PointItem and the legend.

    Shape encodes state: set is a filled circle, predicted a ring, not_visible a
    grey circle, not_set a small dim dot.  color applies to set and predicted.
    """
    r = POINT_RADIUS
    if state == "set":
        return QBrush(color), SELECT_PEN if selected else QPen(color.darker(150), 1), r
    if state == "predicted":
        pen_color = QColor(Qt.GlobalColor.white) if selected else color
        return QBrush(Qt.GlobalColor.transparent), QPen(pen_color, 2), r
    if state == "not_visible":
        # Grayed out — excluded from epipolar and triangulation
        pen = SELECT_PEN if selected else QPen(COLOR_NOT_VISIBLE.darker(120), 1)
        return QBrush(COLOR_NOT_VISIBLE), pen, r
    # not_set — small dim dot at default position
    pen = SELECT_PEN if selected else QPen(COLOR_NOT_SET.darker(130), 1)
    return QBrush(COLOR_NOT_SET), pen, r * 0.5


class PointItem(QGraphicsEllipseItem):
    """Visual representation of one labeled point in a camera view.

    Renders differently based on state:
    - set: filled circle, colored by reprojection error
    - predicted: ring (outline only), colored by error
    - not_set: small dim dot (always visible at default position)
    - not_visible: grayed-out filled circle (excluded from epipolar/triangulation)
    """

    def __init__(self, parent=None):
        r = POINT_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r, parent)
        self._bodypart: str = ""
        self._instance: int = 0
        self._camera: str = ""
        self._state: str = "not_set"
        self._error: float | None = None
        self._xr: float | None = None
        self._yr: float | None = None
        self._selected: bool = False
        self._source: str = ""
        self._score: float | None = None
        self._is_new: bool = False
        self._new_label: str = ""
        self._color_mode: ColorMode = ColorMode.ERROR

        # Square shown around new points; a child, so it moves and hides with the point
        self._new_marker = QGraphicsRectItem(
            -NEW_HALF_SIZE, -NEW_HALF_SIZE, 2 * NEW_HALF_SIZE, 2 * NEW_HALF_SIZE, self
        )
        self._new_marker.setPen(NEW_PEN)
        self._new_marker.setBrush(QBrush(Qt.GlobalColor.transparent))
        self._new_marker.setVisible(False)

        # Constant screen size regardless of zoom
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptHoverEvents(True)
        self.setZValue(50)

        self._apply_style()

    @property
    def bodypart(self) -> str:
        return self._bodypart

    @property
    def instance(self) -> int:
        return self._instance

    @property
    def state(self) -> str:
        return self._state

    @property
    def has_reprojection(self) -> bool:
        return self._xr is not None and self._yr is not None

    @property
    def source(self) -> str:
        return self._source

    @property
    def is_new(self) -> bool:
        return self._is_new

    def configure(
        self,
        bodypart: str,
        instance: int,
        state: str,
        x: float,
        y: float,
        camera: str = "",
        xr: float | None = None,
        yr: float | None = None,
        error: float | None = None,
        selected: bool = False,
        source: str = "",
        score: float | None = None,
        is_new: bool = False,
        new_label: str = "",
        color_mode: ColorMode = ColorMode.ERROR,
    ):
        """Update all properties and reposition."""
        self._bodypart = bodypart
        self._instance = instance
        self._camera = camera
        self._state = state
        self._error = error
        self._xr = xr
        self._yr = yr
        self._selected = selected
        self._source = source
        self._score = score if score is not None and not math.isnan(score) else None
        self._is_new = is_new
        self._new_label = new_label
        self._color_mode = color_mode
        self.setPos(x, y)
        self._apply_style()

    def set_selected(self, selected: bool):
        if self._selected != selected:
            self._selected = selected
            self._apply_style()

    def _apply_style(self):
        """Update visual appearance based on state and selection."""
        if self._color_mode is ColorMode.SOURCE:
            color = COLOR_SOURCE.get(self._source, COLOR_NONE)
        else:
            color = _error_color(self._error)
        self._new_marker.setVisible(
            self._is_new and self._state in ("set", "predicted", "not_visible")
        )

        brush, pen, radius = point_style(self._state, color, self._selected)
        self.setBrush(brush)
        self.setPen(pen)
        self.setRect(-radius, -radius, 2 * radius, 2 * radius)
        self.setVisible(True)

    def hoverEnterEvent(self, event):
        """Show tooltip with point info."""
        cam_label = f"  —  {self._camera}" if self._camera else ""
        lines = [f"{self._bodypart}{cam_label}"]

        if self._state == "not_visible":
            lines.append("Marked not visible")
        if self._state in ("set", "predicted", "not_visible") and self._source:
            source = SOURCE_NAMES.get(self._source, self._source)
            if self._source == SOURCE_LABEL3D and self._state == "predicted":
                source = "label3d guess"
            lines.append(f"Source: {source}")
        if self._score is not None:
            lines.append(f"Score: {self._score:.2f}")
        if self._is_new:
            lines.append(f"New since {self._new_label}" if self._new_label else "New")

        pos = self.pos()
        lines.append(f"Clicked:     ({pos.x():.1f}, {pos.y():.1f})")
        if self._xr is not None and self._yr is not None:
            lines.append(f"Reprojected: ({self._xr:.1f}, {self._yr:.1f})")
        if self._error is not None:
            lines.append(f"Error:        {self._error:.2f} px")

        self.setToolTip("\n".join(lines))
        super().hoverEnterEvent(event)

"""PointItem: QGraphicsEllipseItem subclass representing one labeled point."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem

# Point radius in pixels (screen-space, ignores view transform)
POINT_RADIUS = 8

# Colors for reprojection error ranges
COLOR_GOOD = QColor(0, 200, 0)        # green: error < 5px
COLOR_OK = QColor(255, 200, 0)        # yellow: 5-15px
COLOR_BAD = QColor(255, 50, 50)       # red: error > 15px
COLOR_NONE = QColor(160, 160, 160)    # grey: no error info
COLOR_NOT_SET = QColor(100, 100, 100) # dim grey for points not yet positioned
COLOR_NOT_VISIBLE = QColor(80, 80, 80, 120)  # grayed out for not-visible points

# Selection outline
SELECT_PEN = QPen(Qt.GlobalColor.white, 2)
NORMAL_PEN = QPen(Qt.GlobalColor.transparent, 0)


def _error_color(error: float | None) -> QColor:
    """Map reprojection error to a color."""
    if error is None:
        return COLOR_NONE
    if error < 5:
        return COLOR_GOOD
    if error < 15:
        return COLOR_OK
    return COLOR_BAD


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
        self._state: str = "not_set"
        self._error: float | None = None
        self._selected: bool = False

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

    def configure(
        self,
        bodypart: str,
        instance: int,
        state: str,
        x: float,
        y: float,
        error: float | None = None,
        selected: bool = False,
    ):
        """Update all properties and reposition."""
        self._bodypart = bodypart
        self._instance = instance
        self._state = state
        self._error = error
        self._selected = selected
        self.setPos(x, y)
        self._apply_style()

    def set_selected(self, selected: bool):
        if self._selected != selected:
            self._selected = selected
            self._apply_style()

    def _apply_style(self):
        """Update visual appearance based on state and selection."""
        r = POINT_RADIUS
        color = _error_color(self._error)

        if self._state == "set":
            self.setBrush(QBrush(color))
            self.setPen(SELECT_PEN if self._selected else QPen(color.darker(150), 1))
            self.setRect(-r, -r, 2 * r, 2 * r)
            self.setVisible(True)

        elif self._state == "predicted":
            self.setBrush(QBrush(Qt.GlobalColor.transparent))
            pen_color = Qt.GlobalColor.white if self._selected else color
            self.setPen(QPen(pen_color, 2))
            self.setRect(-r, -r, 2 * r, 2 * r)
            self.setVisible(True)

        elif self._state == "not_visible":
            # Grayed out — excluded from epipolar and triangulation
            self.setBrush(QBrush(COLOR_NOT_VISIBLE))
            self.setPen(SELECT_PEN if self._selected else QPen(COLOR_NOT_VISIBLE.darker(120), 1))
            self.setRect(-r, -r, 2 * r, 2 * r)
            self.setVisible(True)

        else:
            # not_set — small dim dot at default position
            hr = r * 0.5
            self.setBrush(QBrush(COLOR_NOT_SET))
            self.setPen(SELECT_PEN if self._selected else QPen(COLOR_NOT_SET.darker(130), 1))
            self.setRect(-hr, -hr, 2 * hr, 2 * hr)
            self.setVisible(True)

    def hoverEnterEvent(self, event):
        """Show tooltip with point info."""
        lines = [f"{self._bodypart} (inst {self._instance})"]
        lines.append(f"State: {self._state}")
        pos = self.pos()
        lines.append(f"Pos: ({pos.x():.1f}, {pos.y():.1f})")
        if self._error is not None:
            lines.append(f"Error: {self._error:.2f}px")
        self.setToolTip("\n".join(lines))
        super().hoverEnterEvent(event)

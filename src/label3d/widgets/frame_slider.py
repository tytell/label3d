"""Custom frame seekbar with per-camera labeled-frame indicators."""

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


_COLOR_USER = QColor(0, 0, 0)          # black — user-placed points
_COLOR_PREDICTED = QColor(1, 170, 247) # light blue — SLEAP predictions
_COLOR_ROW_BG = QColor(220, 220, 220)  # row background
_COLOR_GROOVE = QColor(160, 160, 160)  # groove line
_COLOR_HANDLE = QColor(80, 80, 80)     # handle triangle

_ROW_H = 4      # height of each camera row (px)
_ROW_GAP = 1    # gap between rows (px)
_GROOVE_H = 2   # groove line height (px)
_HANDLE_H = 8   # triangle height (px)
_HANDLE_W = 10  # triangle base width (px)
_V_PAD = 3      # top/bottom padding (px)
_H_MARGIN = 6   # left/right margin so handle doesn't clip (px)


class FrameSlider(QWidget):
    """Seekbar that shows per-camera labeled-frame tick marks.

    Camera rows stack above a groove line; a triangle handle marks the current
    frame.  Colors match SLEAP: black for user labels, light blue for
    predicted-only frames.
    """

    frame_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._max_frame: int = 0
        self._current_frame: int = 0
        self._camera_order: list[str] = []
        # {camera: {frame_index: 'user'|'predicted'}}
        self._marks: dict[str, dict[int, str]] = {}

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._refresh_height()

    # -- Public API ------------------------------------------------------------

    def set_cameras(self, cameras: list[str]) -> None:
        """Set ordered camera names (determines row order)."""
        self._camera_order = list(cameras)
        self._marks = {cam: {} for cam in cameras}
        self._refresh_height()
        self.update()

    def set_marks(self, marks: dict[str, dict[int, str]]) -> None:
        """Replace all labeled-frame marks."""
        self._marks = marks
        self.update()

    def set_range(self, max_frame: int) -> None:
        self._max_frame = max(0, max_frame)
        self.update()

    def set_value(self, frame: int) -> None:
        self._current_frame = max(0, min(frame, self._max_frame))
        self.update()

    def value(self) -> int:
        return self._current_frame

    # -- Geometry helpers ------------------------------------------------------

    def _refresh_height(self) -> None:
        n = len(self._camera_order)
        rows_h = n * _ROW_H + max(0, n - 1) * _ROW_GAP if n > 0 else 0
        spacer = 4 if n > 0 else 0
        h = _V_PAD + rows_h + spacer + _GROOVE_H + _HANDLE_H + _V_PAD
        self.setFixedHeight(max(h, 20))

    def _groove_y(self) -> int:
        n = len(self._camera_order)
        rows_h = n * _ROW_H + max(0, n - 1) * _ROW_GAP if n > 0 else 0
        spacer = 4 if n > 0 else 0
        return _V_PAD + rows_h + spacer

    def _frame_to_x(self, frame: int) -> float:
        if self._max_frame <= 0:
            return float(_H_MARGIN)
        return _H_MARGIN + frame / self._max_frame * (self.width() - 2 * _H_MARGIN)

    def _x_to_frame(self, x: float) -> int:
        w = self.width() - 2 * _H_MARGIN
        if w <= 0 or self._max_frame <= 0:
            return 0
        t = (x - _H_MARGIN) / w
        return round(max(0.0, min(1.0, t)) * self._max_frame)

    # -- Painting --------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        content_w = self.width() - 2 * _H_MARGIN
        groove_y = self._groove_y()

        # Camera rows
        for i, cam in enumerate(self._camera_order):
            row_y = _V_PAD + i * (_ROW_H + _ROW_GAP)
            painter.fillRect(_H_MARGIN, row_y, content_w, _ROW_H, _COLOR_ROW_BG)
            if self._max_frame > 0:
                for frame, mark_type in self._marks.get(cam, {}).items():
                    color = _COLOR_USER if mark_type == "user" else _COLOR_PREDICTED
                    x = int(self._frame_to_x(frame))
                    painter.fillRect(x, row_y, 2, _ROW_H, color)

        # Groove
        painter.fillRect(_H_MARGIN, groove_y, content_w, _GROOVE_H, _COLOR_GROOVE)

        # Handle triangle (points downward from groove)
        hx = self._frame_to_x(self._current_frame)
        hy = groove_y + _GROOVE_H
        half_w = _HANDLE_W / 2
        triangle = QPolygonF([
            QPointF(hx - half_w, hy),
            QPointF(hx + half_w, hy),
            QPointF(hx, hy + _HANDLE_H),
        ])
        painter.setBrush(QBrush(_COLOR_HANDLE))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(triangle)

        painter.end()

    # -- Mouse interaction -----------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            frame = self._x_to_frame(event.position().x())
            self._current_frame = frame
            self.update()
            self.frame_changed.emit(frame)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            frame = self._x_to_frame(event.position().x())
            if frame != self._current_frame:
                self._current_frame = frame
                self.update()
                self.frame_changed.emit(frame)

    def sizeHint(self) -> QSize:
        return QSize(200, self.height())

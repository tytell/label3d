"""Custom frame seekbar with per-camera labeled-frame indicators."""

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


_COLOR_USER = QColor(0, 0, 0)          # black — user-placed points
_COLOR_PREDICTED = QColor(1, 170, 247) # light blue — SLEAP predictions
_COLOR_NEW_USER = QColor(255, 140, 0)  # orange — new SLEAP user labels
_COLOR_ROW_BG = QColor(220, 220, 220)  # row background
_COLOR_GROOVE = QColor(160, 160, 160)  # groove line
_COLOR_HANDLE = QColor(80, 80, 80)     # handle triangle
_COLOR_BOUND = QColor(0, 110, 200)     # start/end markers and the span between
_COLOR_OUTSIDE = QColor(255, 255, 255, 170)  # veil over frames outside the bounds

_ROW_H = 4      # height of each camera row (px)
_ROW_GAP = 1    # gap between rows (px)
_GROOVE_H = 2   # groove line height (px)
_HANDLE_H = 8   # triangle height (px)
_HANDLE_W = 10  # triangle base width (px)
_V_PAD = 3      # top/bottom padding (px)
_H_MARGIN = 6   # left/right margin so handle doesn't clip (px)
_BOUND_W = 4    # width of a start/end bracket's foot (px)
_BOUND_GRAB = 6 # how close (px) a press must be to grab a bound marker


class FrameSlider(QWidget):
    """Seekbar that shows per-camera labeled-frame tick marks.

    Camera rows stack above a groove line; a triangle handle marks the current
    frame.  Colors match SLEAP: black for user labels, light blue for
    predicted-only frames.

    Optional start/end bounds (set_bounds) draw as draggable brackets; frames
    outside them are veiled.  Dragging a bracket emits bounds_changed.

    Frames with points new since the latest sync (set_new_marks) get an extra
    row above the camera rows: orange when a new point is a SLEAP user label,
    blue when all of them are predictions.  The row is hidden when empty.
    """

    frame_changed = Signal(int)
    bounds_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._max_frame: int = 0
        self._current_frame: int = 0
        self._camera_order: list[str] = []
        # {camera: {frame_index: 'user'|'predicted'}}
        self._marks: dict[str, dict[int, str]] = {}
        self._new_marks: dict[int, str] = {}  # frame → 'user' | 'predicted'
        self._bounds: tuple[int, int] | None = None
        self._dragging: str | None = None  # 'start' | 'end' while dragging a bound

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)  # hover cursor over bound markers
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

    def set_new_marks(self, marks: dict[int, str]) -> None:
        """Replace the frames marked as having new points ('user' or 'predicted')."""
        had_row = bool(self._new_marks)
        self._new_marks = dict(marks)
        if had_row != bool(self._new_marks):
            self._refresh_height()
        self.update()

    def set_range(self, max_frame: int) -> None:
        self._max_frame = max(0, max_frame)
        if self._bounds is not None:
            self.set_bounds(*self._bounds)  # re-clamp to the new range
        self.update()

    def set_value(self, frame: int) -> None:
        self._current_frame = max(0, min(frame, self._max_frame))
        self.update()

    def value(self) -> int:
        return self._current_frame

    def set_bounds(self, start: int, end: int) -> None:
        """Show draggable start/end markers (inclusive frames)."""
        start = max(0, min(start, self._max_frame))
        self._bounds = (start, max(start, min(end, self._max_frame)))
        self.update()

    def clear_bounds(self) -> None:
        self._bounds = None
        self._dragging = None
        self.update()

    def bounds(self) -> tuple[int, int] | None:
        return self._bounds

    # -- Geometry helpers ------------------------------------------------------

    def _n_rows(self) -> int:
        """Camera rows plus the "new" row when any frame has new points."""
        return len(self._camera_order) + (1 if self._new_marks else 0)

    def _refresh_height(self) -> None:
        self.setFixedHeight(max(self._groove_y() + _GROOVE_H + _HANDLE_H + _V_PAD, 20))

    def _groove_y(self) -> int:
        n = self._n_rows()
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

    def _bound_at(self, x: float) -> str | None:
        """Which bound marker (if any) is within grabbing distance of x."""
        if self._bounds is None:
            return None
        d_start = abs(x - self._frame_to_x(self._bounds[0]))
        d_end = abs(x - self._frame_to_x(self._bounds[1]))
        if min(d_start, d_end) > _BOUND_GRAB:
            return None
        # Coincident markers: pick the one the drag can move away from.
        if d_start == d_end:
            return "start" if x < self._frame_to_x(self._bounds[0]) else "end"
        return "start" if d_start < d_end else "end"

    def _drag_bound_to(self, x: float) -> None:
        frame = self._x_to_frame(x)
        start, end = self._bounds
        if self._dragging == "start":
            new = (min(frame, end), end)
        else:
            new = (start, max(frame, start))
        if new != self._bounds:
            self._bounds = new
            self.update()
            self.bounds_changed.emit(*new)

    # -- Painting --------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        content_w = self.width() - 2 * _H_MARGIN
        groove_y = self._groove_y()

        # "New" row on top, when any frame has new points
        first_cam_row = 0
        if self._new_marks:
            first_cam_row = 1
            painter.fillRect(_H_MARGIN, _V_PAD, content_w, _ROW_H, _COLOR_ROW_BG)
            if self._max_frame > 0:
                for frame, kind in self._new_marks.items():
                    color = _COLOR_NEW_USER if kind == "user" else _COLOR_PREDICTED
                    x = int(self._frame_to_x(frame))
                    painter.fillRect(x, _V_PAD, 2, _ROW_H, color)

        # Camera rows
        for i, cam in enumerate(self._camera_order):
            row_y = _V_PAD + (i + first_cam_row) * (_ROW_H + _ROW_GAP)
            painter.fillRect(_H_MARGIN, row_y, content_w, _ROW_H, _COLOR_ROW_BG)
            if self._max_frame > 0:
                for frame, mark_type in self._marks.get(cam, {}).items():
                    color = _COLOR_USER if mark_type == "user" else _COLOR_PREDICTED
                    x = int(self._frame_to_x(frame))
                    painter.fillRect(x, row_y, 2, _ROW_H, color)

        # Groove
        painter.fillRect(_H_MARGIN, groove_y, content_w, _GROOVE_H, _COLOR_GROOVE)

        if self._bounds is not None:
            self._paint_bounds(painter, groove_y)

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

    def _paint_bounds(self, painter: QPainter, groove_y: int) -> None:
        """Veil frames outside the bounds and draw [ ] brackets at them."""
        x0 = int(self._frame_to_x(self._bounds[0]))
        x1 = int(self._frame_to_x(self._bounds[1]))
        top = _V_PAD
        bottom = groove_y + _GROOVE_H

        painter.fillRect(_H_MARGIN, top, x0 - _H_MARGIN, bottom - top, _COLOR_OUTSIDE)
        right = self.width() - _H_MARGIN
        painter.fillRect(x1 + 1, top, right - x1 - 1, bottom - top, _COLOR_OUTSIDE)

        painter.fillRect(x0, groove_y, x1 - x0 + 1, _GROOVE_H, _COLOR_BOUND)
        # Each bracket: a 2px upright plus feet at top and bottom pointing inward.
        for upright_x, foot_x in ((x0, x0), (x1 - 1, x1 - _BOUND_W + 1)):
            painter.fillRect(upright_x, top, 2, bottom - top, _COLOR_BOUND)
            painter.fillRect(foot_x, top, _BOUND_W, 2, _COLOR_BOUND)
            painter.fillRect(foot_x, bottom - 2, _BOUND_W, 2, _COLOR_BOUND)

    # -- Mouse interaction -----------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            self._dragging = self._bound_at(x)
            if self._dragging is not None:
                return
            frame = self._x_to_frame(x)
            self._current_frame = frame
            self.update()
            self.frame_changed.emit(frame)

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if not event.buttons() & Qt.MouseButton.LeftButton:
            over = self._bound_at(x) is not None
            if over:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.unsetCursor()
            return
        if self._dragging is not None:
            self._drag_bound_to(x)
            return
        frame = self._x_to_frame(x)
        if frame != self._current_frame:
            self._current_frame = frame
            self.update()
            self.frame_changed.emit(frame)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = None

    def sizeHint(self) -> QSize:
        return QSize(200, self.height())

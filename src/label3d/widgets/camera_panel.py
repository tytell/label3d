"""QGraphicsView for one camera view — displays video frame with zoom/pan.

Phase 3 additions: skeleton/epipolar overlays, mouse interaction for
point selection, dragging, and right-click not-visible toggle.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)
import numpy as np

from label3d.widgets.skeleton_overlay import SkeletonOverlay
from label3d.widgets.epipolar_overlay import EpipolarOverlay, camera_color

# "Zoom to point" box size, as a fraction of the image dimensions
POINT_ZOOM_FRAC = 0.05


class _CameraLegend(QWidget):
    """Small overlay widget showing colored dots + camera names."""

    def __init__(self, entries: list[tuple[QColor, str]], parent=None):
        super().__init__(parent)
        self._entries = entries  # [(color, name), ...]
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._update_size()

    def _update_size(self):
        h = max(len(self._entries) * 18 + 4, 10)
        self.setFixedHeight(h)
        self.setFixedWidth(130)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = 4
        for color, name in self._entries:
            # Dot
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(6, y + 2, 10, 10)
            # Text
            p.setPen(color)
            p.drawText(22, y + 12, name)
            y += 18
        p.end()


class _RotatedLabel(QWidget):
    """Small widget that draws rotated text at a given angle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._color = QColor(255, 255, 255)
        self._angle = 0.0  # degrees
        self._font = QFont()
        self._font.setPointSize(9)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(80, 30)

    def configure(self, text: str, color: QColor, angle_deg: float):
        self._text = text
        self._color = color
        self._angle = angle_deg
        # Size to fit rotated text
        fm = QFontMetrics(self._font)
        tw = fm.horizontalAdvance(text) + 4
        th = fm.height() + 4
        rad = math.radians(abs(angle_deg))
        w = int(tw * abs(math.cos(rad)) + th * abs(math.sin(rad))) + 2
        h = int(tw * abs(math.sin(rad)) + th * abs(math.cos(rad))) + 2
        self.setFixedSize(w, h)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(self.width() / 2, self.height() / 2)
        p.rotate(self._angle)
        p.setFont(self._font)
        p.setPen(self._color)
        fm = QFontMetrics(self._font)
        tw = fm.horizontalAdvance(self._text)
        p.drawText(-tw // 2, fm.ascent() // 2, self._text)
        p.end()


class CameraPanel(QGraphicsView):
    """Displays a single camera's video frame with zoom, pan, and overlays.

    Left-click on a point to select it. Click-and-drag to move.
    Click on empty space to deselect. Right-click toggles not-visible.
    """

    ZOOM_FACTOR = 1.15
    # Smallest zoom box, as a fraction of the image's shorter side
    MIN_ZOOM_FRAC = 0.05

    # (camera_name, instance, bodypart) — clicked on existing point
    point_clicked = Signal(str, int, str)
    # (camera_name, instance, bodypart, scene_x, scene_y) — drag finished
    point_dragged = Signal(str, int, str, float, float)
    # (camera_name, instance, bodypart) — right-click on point
    point_right_clicked = Signal(str, int, str)
    # (camera_name, scene_x, scene_y) — right-click on empty space (no point hit)
    empty_right_clicked = Signal(str, float, float)
    # clicked on empty space
    deselect_requested = Signal()
    # (camera_name, x, y) — raw pixel click in pick mode
    raw_clicked = Signal(str, float, float)

    def __init__(self, camera_name: str, parent=None):
        super().__init__(parent)
        self._camera_name = camera_name

        # Scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Background pixmap
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # Camera name label (top-left, overlaid)
        self._label = QGraphicsSimpleTextItem(camera_name)
        self._label.setBrush(Qt.GlobalColor.white)
        font = self._label.font()
        font.setPointSize(12)
        font.setBold(True)
        self._label.setFont(font)
        self._label.setFlag(
            QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self._label.setZValue(100)
        self._scene.addItem(self._label)

        # Overlays
        self._skeleton_overlay = SkeletonOverlay(self._scene, camera_name)
        self._epipolar_overlay = EpipolarOverlay(self._scene, camera_name)

        # View settings
        self.setRenderHints(
            self.renderHints()
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.black)

        self._fitted = False
        self._saved_scene_rect: QRectF | None = None

        # Interaction state
        self._has_selection = False  # whether a bodypart is selected
        self._dragging: tuple[int, str] | None = None  # (instance, bodypart) being dragged
        self._drag_start_pos = None
        self._legend: _CameraLegend | None = None
        self._charuco_overlay = None  # CharucoOverlay — set externally for calibration mode
        self._pick_mode = False       # raw click-to-pick mode for axis alignment

        # Epipolar line labels (viewport-fixed, rotated)
        self._line_labels: list[_RotatedLabel] = []
        self._label_debounce = QTimer(self)
        self._label_debounce.setSingleShot(True)
        self._label_debounce.setInterval(150)
        self._label_debounce.timeout.connect(self._reposition_line_labels)

    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def skeleton_overlay(self) -> SkeletonOverlay:
        return self._skeleton_overlay

    @property
    def epipolar_overlay(self) -> EpipolarOverlay:
        return self._epipolar_overlay

    @property
    def charuco_overlay(self):
        """CharucoOverlay for calibration mode, or None in label mode."""
        return self._charuco_overlay

    @charuco_overlay.setter
    def charuco_overlay(self, overlay):
        self._charuco_overlay = overlay

    def setup_legend(self, all_cameras: list[str]):
        """Create a viewport-fixed legend identifying this camera's color."""
        own_idx = all_cameras.index(self._camera_name) if self._camera_name in all_cameras else 0
        entries = [(camera_color(own_idx), self._camera_name)]
        if self._legend is not None:
            self._legend.deleteLater()
        self._legend = _CameraLegend(entries, parent=self.viewport())
        self._position_legend()
        self._legend.show()

    def _position_legend(self):
        """Place the legend in the bottom-left of the viewport."""
        if self._legend is None:
            return
        vp = self.viewport()
        self._legend.move(4, vp.height() - self._legend.height() - 4)

    def update_line_labels(self):
        """Schedule a line label reposition (called after overlay update)."""
        self._label_debounce.start()

    def _hide_line_labels(self):
        """Immediately hide all line labels during view transforms."""
        for lbl in self._line_labels:
            lbl.setVisible(False)

    def _reposition_line_labels(self):
        """Compute viewport-edge positions for epipolar line labels."""
        active = self._epipolar_overlay.active_lines
        vp = self.viewport()
        vp_rect = vp.rect()

        # Ensure we have enough label widgets
        while len(self._line_labels) < len(active):
            lbl = _RotatedLabel(parent=vp)
            self._line_labels.append(lbl)

        for idx, info in enumerate(active):
            # Map scene endpoints to viewport coordinates
            vp1 = self.mapFromScene(QPointF(info.x1, info.y1))
            vp2 = self.mapFromScene(QPointF(info.x2, info.y2))

            # Find intersection with viewport edge — pick the endpoint
            # closer to the viewport edge
            edge_pt = self._clip_to_edge(vp_rect, vp1, vp2)
            if edge_pt is None:
                self._line_labels[idx].setVisible(False)
                continue

            # Angle of the line in viewport space
            dx = vp2.x() - vp1.x()
            dy = vp2.y() - vp1.y()
            angle_deg = math.degrees(math.atan2(dy, dx))
            # Keep text readable (avoid upside-down text)
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            color = camera_color(info.color_index)
            lbl = self._line_labels[idx]
            lbl.configure(info.source_cam, color, angle_deg)

            # Position: center the label on the edge point, clamped to viewport
            lx = int(edge_pt.x()) - lbl.width() // 2
            ly = int(edge_pt.y()) - lbl.height() // 2
            lx = max(0, min(lx, vp_rect.width() - lbl.width()))
            ly = max(0, min(ly, vp_rect.height() - lbl.height()))
            lbl.move(lx, ly)
            lbl.setVisible(True)

        # Hide unused labels
        for idx in range(len(active), len(self._line_labels)):
            self._line_labels[idx].setVisible(False)

    @staticmethod
    def _clip_to_edge(rect, p1, p2):
        """Find where the line from p1 to p2 intersects the viewport edge.

        Returns the intersection point closest to the edge, or None if the
        line doesn't cross the viewport.
        """
        edges = [
            (QPointF(rect.left(), rect.top()), QPointF(rect.right(), rect.top())),      # top
            (QPointF(rect.left(), rect.bottom()), QPointF(rect.right(), rect.bottom())), # bottom
            (QPointF(rect.left(), rect.top()), QPointF(rect.left(), rect.bottom())),     # left
            (QPointF(rect.right(), rect.top()), QPointF(rect.right(), rect.bottom())),   # right
        ]

        margin = 8  # inset from edge
        best = None
        best_t = float("inf")

        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()

        for ea, eb in edges:
            # Line-segment intersection using parametric form
            ex = eb.x() - ea.x()
            ey = eb.y() - ea.y()

            denom = dx * ey - dy * ex
            if abs(denom) < 1e-10:
                continue

            t = ((ea.x() - p1.x()) * ey - (ea.y() - p1.y()) * ex) / denom
            u = ((ea.x() - p1.x()) * dy - (ea.y() - p1.y()) * dx) / denom

            if 0 <= u <= 1:  # intersection is on the edge segment
                ix = p1.x() + t * dx
                iy = p1.y() + t * dy
                # Prefer intersections closest to the midpoint of the edge
                # (better label placement), but any valid intersection works
                if abs(t) < best_t:
                    best_t = abs(t)
                    # Inset slightly from the edge
                    cx = rect.center().x()
                    cy = rect.center().y()
                    # Nudge toward center
                    nudge_x = margin if ix < cx else -margin
                    nudge_y = margin if iy < cy else -margin
                    best = QPointF(ix + nudge_x, iy + nudge_y)

        return best

    def set_pick_mode(self, enabled: bool):
        """Enable/disable raw pick mode for axis alignment.

        When enabled, left-click emits raw_clicked(camera, x, y) and
        suppresses all normal point interaction.
        """
        self._pick_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_interaction_mode(self, has_selection: bool):
        """Update cursor based on selection state.

        Points are always clickable/draggable regardless of selection.
        Pan is available via middle-click or scroll-hand-drag on empty space.
        """
        self._has_selection = has_selection

    def save_view_state(self):
        """Save the visible scene rect so it can be restored after re-show."""
        self._saved_scene_rect = QRectF(
            self.mapToScene(self.viewport().rect()).boundingRect()
        )

    def restore_view_state(self):
        """Restore previously saved visible scene rect, or fit if none saved."""
        if self._saved_scene_rect is not None:
            self.fitInView(self._saved_scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._saved_scene_rect = None
        else:
            self.fit_view()

    def get_scene_rect(self) -> QRectF | None:
        """Return current visible scene rect; falls back to saved rect for hidden panels."""
        if self._fitted and self.isVisible():
            return QRectF(self.mapToScene(self.viewport().rect()).boundingRect())
        return self._saved_scene_rect

    def set_scene_rect(self, rect: QRectF) -> None:
        """Apply a scene rect (zoom/pan). If not yet shown, stores for deferred restore."""
        if self._fitted and self.isVisible():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self._saved_scene_rect = QRectF(rect)

    def set_frame(self, frame: np.ndarray):
        """Update the displayed frame from a BGR numpy array."""
        # BGR -> RGB
        rgb = frame[..., ::-1].copy()
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item.setPixmap(pixmap)

        # Position label at top-left of the image
        self._label.setPos(5, 5)

        if not self._fitted:
            self.fit_view()
            self._fitted = True

    def fit_view(self):
        """Fit the entire image in the viewport."""
        rect = self._pixmap_item.boundingRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_to_scene_rect(self, rect: QRectF, margin_frac: float = 0.25):
        """Zoom to a scene rect, padded by margin_frac and clamped to the image."""
        img_rect = self._pixmap_item.boundingRect()
        if img_rect.isEmpty():
            return

        rect = QRectF(rect).normalized()
        # Pad, with a floor so a degenerate (single-point) rect still zooms sanely
        min_size = min(img_rect.width(), img_rect.height()) * self.MIN_ZOOM_FRAC
        pad_x = max(rect.width() * margin_frac, min_size / 2.0)
        pad_y = max(rect.height() * margin_frac, min_size / 2.0)
        rect.adjust(-pad_x, -pad_y, pad_x, pad_y)

        rect = rect.intersected(img_rect)
        if rect.isEmpty():
            return

        self._hide_line_labels()
        self._label_debounce.start()
        self.set_scene_rect(rect)

    def zoom_to_points(self, points: list[QPointF], margin_frac: float = 0.25):
        """Zoom to the bounding box of the given scene points."""
        if not points:
            return
        xs = [p.x() for p in points]
        ys = [p.y() for p in points]
        rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        self.zoom_to_scene_rect(rect, margin_frac)

    def zoom_to_point(self, point: QPointF, frac: float = POINT_ZOOM_FRAC):
        """Zoom to a box frac of the image size, centred on one scene point."""
        img_rect = self._pixmap_item.boundingRect()
        if img_rect.isEmpty():
            return

        w = img_rect.width() * frac
        h = img_rect.height() * frac
        rect = QRectF(point.x() - w / 2.0, point.y() - h / 2.0, w, h)

        # Keep the box inside the image rather than shrinking it at the edges
        rect.translate(
            max(0.0, img_rect.left() - rect.left()) + min(0.0, img_rect.right() - rect.right()),
            max(0.0, img_rect.top() - rect.top()) + min(0.0, img_rect.bottom() - rect.bottom()),
        )
        rect = rect.intersected(img_rect)
        if rect.isEmpty():
            return

        self._hide_line_labels()
        self._label_debounce.start()
        self.set_scene_rect(rect)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with scroll wheel, clamped to image bounds."""
        self._hide_line_labels()
        self._label_debounce.start()
        if event.angleDelta().y() > 0:
            self.scale(self.ZOOM_FACTOR, self.ZOOM_FACTOR)
        else:
            # Don't zoom out past the full image
            img_rect = self._pixmap_item.boundingRect()
            if img_rect.isEmpty():
                return
            vp = self.viewport().size()
            visible = self.mapToScene(self.viewport().rect()).boundingRect()
            # If the visible area would exceed the image, clamp to fit
            if (visible.width() * (1.0 / self.ZOOM_FACTOR) >= img_rect.width()
                    or visible.height() * (1.0 / self.ZOOM_FACTOR) >= img_rect.height()):
                self.fit_view()
            else:
                self.scale(1.0 / self.ZOOM_FACTOR, 1.0 / self.ZOOM_FACTOR)

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset zoom."""
        self._hide_line_labels()
        self._label_debounce.start()
        self.fit_view()
        event.accept()

    def scrollContentsBy(self, dx, dy):
        """Hide labels during pan scroll."""
        super().scrollContentsBy(dx, dy)
        self._hide_line_labels()
        self._label_debounce.start()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle point interaction on left/right click."""
        scene_pos = self.mapToScene(event.position().toPoint())

        # Pick mode: emit raw coordinates and skip all other interaction
        if self._pick_mode and event.button() == Qt.MouseButton.LeftButton:
            self.raw_clicked.emit(self._camera_name, scene_pos.x(), scene_pos.y())
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Calibration mode: charuco corners, no drag
            if self._charuco_overlay is not None:
                corner_id = self._charuco_overlay.hit_test(scene_pos)
                if corner_id is not None:
                    self.point_clicked.emit(self._camera_name, 0, corner_id)
                    event.accept()
                else:
                    self.deselect_requested.emit()
                    super().mousePressEvent(event)
                return

            # Label mode: skeleton points with drag support
            hit = self._skeleton_overlay.hit_test(scene_pos)
            if hit is not None:
                inst, bp = hit
                self.point_clicked.emit(self._camera_name, inst, bp)
                self._dragging = hit
                self._drag_start_pos = scene_pos
                event.accept()
            else:
                # Clicked on empty space — deselect and allow pan
                self.deselect_requested.emit()
                super().mousePressEvent(event)

        elif event.button() == Qt.MouseButton.RightButton:
            hit = self._skeleton_overlay.hit_test(scene_pos)
            if hit is not None:
                inst, bp = hit
                self.point_right_clicked.emit(self._camera_name, inst, bp)
            else:
                self.empty_right_clicked.emit(
                    self._camera_name, scene_pos.x(), scene_pos.y()
                )
            event.accept()

        elif event.button() == Qt.MouseButton.MiddleButton:
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Update point position during drag (visual only)."""
        if self._dragging is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            inst, bp = self._dragging
            item = self._skeleton_overlay.get_point_item(inst, bp)
            if item is not None:
                item.setPos(scene_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Commit drag position to the data store."""
        if self._dragging is not None and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            inst, bp = self._dragging
            self.point_dragged.emit(
                self._camera_name, inst, bp, scene_pos.x(), scene_pos.y()
            )
            self._dragging = None
            self._drag_start_pos = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_legend()
        self._hide_line_labels()
        self._label_debounce.start()

"""QGraphicsView for one camera view — displays video frame with zoom/pan."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)
import numpy as np


class CameraPanel(QGraphicsView):
    """Displays a single camera's video frame with zoom and pan."""

    ZOOM_FACTOR = 1.15

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

    @property
    def camera_name(self) -> str:
        return self._camera_name

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

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with scroll wheel."""
        if event.angleDelta().y() > 0:
            self.scale(self.ZOOM_FACTOR, self.ZOOM_FACTOR)
        else:
            self.scale(1.0 / self.ZOOM_FACTOR, 1.0 / self.ZOOM_FACTOR)

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset zoom."""
        self.fit_view()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)

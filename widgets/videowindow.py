import os
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import logging

import qtpy
from qtpy import QtCore, QtGui
from qtpy.QtCore import (
    QEvent, Qt,
    QRectF, Slot
)
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QMessageBox, 
    QGraphicsView, QGraphicsScene,
    QAction, QVBoxLayout
)
from qtpy.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)

logger = logging.getLogger(__name__)

def get_package_file(filename: str) -> str:
    """Returns full path to specified file within sleap package."""
    
    rootpath = os.path.join(os.path.dirname(__file__), "..")
    filepath = os.path.abspath(os.path.join(rootpath, filename))
    return filepath

class GraphicsView(QGraphicsView):
  
    updatedViewer = QtCore.Signal()
    updatedSelection = QtCore.Signal()
    areaSelected = QtCore.Signal(float, float, float, float)
    pointSelected = QtCore.Signal(float, float)
    leftMouseButtonPressed = QtCore.Signal(float, float)
    rightMouseButtonPressed = QtCore.Signal(float, float)
    leftMouseButtonReleased = QtCore.Signal(float, float)
    rightMouseButtonReleased = QtCore.Signal(float, float)
    leftMouseButtonDoubleClicked = QtCore.Signal(float, float)
    rightMouseButtonDoubleClicked = QtCore.Signal(float, float)
    zoomModeChanged = QtCore.Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setAcceptDrops(True)

        self.scene.setBackgroundBrush(QBrush(QColor(Qt.black)))

        self._pixmapHandle = None

        self.setRenderHint(QPainter.Antialiasing)

        self.aspectRatioMode = Qt.KeepAspectRatio
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.canZoom = True
        self.canPan = True
        self.click_mode = ""
        self.zoom_mode = ""

        self.zoomFactor = 1
        anchor_mode = QGraphicsView.AnchorUnderMouse
        self.setTransformationAnchor(anchor_mode)

        # Set icon as default background.

        bgimg = QImage(get_package_file("resources/bluegill.jpg"))
        assert(not bgimg.isNull())
        self.setImage(bgimg)

    # def dragEnterEvent(self, event):
    #     if self.parentWidget():
    #         self.parentWidget().dragEnterEvent(event)

    # def dropEvent(self, event):
    #     if self.parentWidget():
    #         self.parentWidget().dropEvent(event)

    @Slot(bool)
    def set_zoom(self, on):
        if on:
            self.zoom_mode = "button"
            self.click_mode = "area"
        else:
            self.zoom_mode = ""

        logger.debug("Zoom {}".format(on))

    def hasImage(self) -> bool:
        """Returns whether or not the scene contains an image pixmap."""
        return self._pixmapHandle is not None

    def clear(self):
        """Clears the displayed frame from the scene."""

        if self._pixmapHandle:
            # get the pixmap currently shown
            pixmap = self._pixmapHandle.pixmap()

        self.scene.clear()

        if self._pixmapHandle:
            # add the pixmap back
            self._pixmapHandle = self._add_pixmap(pixmap)

    def _add_pixmap(self, pixmap):
        """Adds a pixmap to the scene and transforms it to midpoint coordinates."""
        pixmap_graphics_item = self.scene.addPixmap(pixmap)

        transform = pixmap_graphics_item.transform()
        transform.translate(-0.5, -0.5)
        pixmap_graphics_item.setTransform(transform)

        return pixmap_graphics_item

    def setImage(self, image: Union[QImage, QPixmap, np.ndarray]):
        """
        Set the scene's current image pixmap to the input QImage or QPixmap.

        Args:
            image: The QPixmap or QImage to display.

        Raises:
            RuntimeError: If the input image is not QImage or QPixmap

        Returns:
            None.
        """
        # if type(image) is np.ndarray:
        #     # Convert numpy array of frame image to QImage
        #     image = qimage2ndarray.array2qimage(image)

        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap(image)
        else:
            raise RuntimeError(
                "ImageViewer.setImage: Argument must be a QImage or QPixmap."
            )
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self._add_pixmap(pixmap)

            # Ensure that image is behind everything else
            self._pixmapHandle.setZValue(-1)

        # Set scene size to image size, translated to midpoint coordinates.
        # (If we don't translate the rect, the image will be cut off by
        # 1/2 pixel at the top left and have a 1/2 pixel border at bottom right)
        rect = QRectF(pixmap.rect())
        rect.translate(-0.5, -0.5)
        self.setSceneRect(rect)
        self.updateViewer()

    def updateViewer(self):
        """Apply current zoom."""
        if not self.hasImage():
            return

        base_w_scale = self.width() / self.sceneRect().width()
        base_h_scale = self.height() / self.sceneRect().height()
        base_scale = min(base_w_scale, base_h_scale)

        transform = QTransform()
        transform.scale(base_scale * self.zoomFactor, base_scale * self.zoomFactor)
        self.setTransform(transform)
        # self.updatedViewer.emit()

    def resizeEvent(self, event):
        """Maintain current zoom on resize."""
        self.updateViewer()

    def mousePressEvent(self, event):
        """Start mouse pan or zoom mode."""
        scenePos = self.mapToScene(event.pos())
        # keep track of click location
        self._down_pos = event.pos()
        # behavior depends on which button is pressed
        if event.button() == Qt.LeftButton:

            if event.modifiers() == Qt.NoModifier:
                if self.click_mode == "area":
                    self.setDragMode(QGraphicsView.RubberBandDrag)
                elif self.click_mode == "point":
                    self.setDragMode(QGraphicsView.NoDrag)
                elif self.canPan:
                    self.setDragMode(QGraphicsView.ScrollHandDrag)

            elif event.modifiers() == Qt.AltModifier:
                if self.canZoom:
                    self.zoom_mode = "alt"
                    self.zoomModeChanged.emit(True)
                    self.setDragMode(QGraphicsView.RubberBandDrag)

            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())

        elif event.button() == Qt.RightButton:
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """Stop mouse pan or zoom mode (apply zoom if valid)."""
        QGraphicsView.mouseReleaseEvent(self, event)
        scenePos = self.mapToScene(event.pos())

        # check if mouse moved during click
        has_moved = event.pos() != self._down_pos
        if event.button() == Qt.LeftButton:

            if has_moved and self.zoom_mode != "":
                # if self.zoom_mode == "alt":
                self.zoom_mode = ""
                self.click_mode = ""
                self.zoomModeChanged.emit(False)

                zoom_rect = self.scene.selectionArea().boundingRect()
                self.scene.setSelectionArea(QPainterPath())  # clear selection
                self.zoomToRect(zoom_rect)

            elif self.click_mode == "":
                # Check if this was just a tap (not a drag)
                pass

            elif self.click_mode == "area":
                # Check if user was selecting rectangular area
                selection_rect = self.scene.selectionArea().boundingRect()

                self.areaSelected.emit(
                    selection_rect.left(),
                    selection_rect.top(),
                    selection_rect.right(),
                    selection_rect.bottom(),
                )
            elif self.click_mode == "point":
                self.pointSelected.emit(scenePos.x(), scenePos.y())

            self.unsetCursor()

            # finish drag
            self.setDragMode(QGraphicsView.NoDrag)
            # pass along event
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:

            self.setDragMode(QGraphicsView.NoDrag)
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

    def mouseMoveEvent(self, event):
        QGraphicsView.mouseMoveEvent(self, event)

    def zoomToRect(self, zoom_rect: QRectF):
        """
        Method to zoom scene to a given rectangle.

        The rect can either be given relative to the current zoom
        (this is useful if it's the rect drawn by user) or it can be
        given in absolute coordinates for displayed frame.

        Args:
            zoom_rect: The `QRectF` to which we want to zoom.
        """

        if zoom_rect.isNull():
            return

        scale_h = self.scene.height() / zoom_rect.height()
        scale_w = self.scene.width() / zoom_rect.width()
        scale = min(scale_h, scale_w)

        self.zoomFactor = scale
        self.updateViewer()
        self.centerOn(zoom_rect.center())

    def clearZoom(self):
        """Clear zoom stack. Doesn't update display."""
        self.zoomFactor = 1

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Custom event handler, clears zoom."""
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:

            if self.zoom_mode != "":
                self.clearZoom()
                self.updateViewer()

            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)

    # def wheelEvent(self, event):
    #     """Custom event handler. Zoom in/out based on scroll wheel change."""
    #     # zoom on wheel when no mouse buttons are pressed
    #     if event.buttons() == Qt.NoButton:
    #         angle = event.angleDelta().y()
    #         factor = 1.1 if angle > 0 else 0.9

    #         self.zoomFactor = max(factor * self.zoomFactor, 1)
    #         self.updateViewer()

    #     # Trigger wheelEvent for all child elements. This is a bit of a hack.
    #     # We can't use QGraphicsView.wheelEvent(self, event) since that will scroll
    #     # view.
    #     # We want to trigger for all children, since wheelEvent should continue rotating
    #     # an skeleton even if the skeleton node/node label is no longer under the
    #     # cursor.
    #     # Note that children expect a QGraphicsSceneWheelEvent event, which is why we're
    #     # explicitly ignoring TypeErrors. Everything seems to work fine since we don't
    #     # care about the mouse position; if we did, we'd need to map pos to scene.
    #     for child in self.items():
    #         try:
    #             child.wheelEvent(event)
    #         except TypeError:
    #             pass

    # def keyPressEvent(self, event):
    #     """Custom event hander, disables default QGraphicsView behavior."""
    #     event.ignore()  # Kicks the event up to parent

    # def keyReleaseEvent(self, event):
    #     """Custom event hander, disables default QGraphicsView behavior."""
    #     event.ignore()  # Kicks the event up to parent

class VideoWindow(QWidget):
    def __init__(self, filename: str,
                 main_window: QWidget):
        super().__init__()
        self.filename = filename
        self.name = os.path.basename(filename)
        self.main_window = main_window

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.setObjectName(self.name + "VideoWindow")
        self.setWindowTitle(self.name)

        self.view = GraphicsView()
        self.view.setObjectName(self.name + "VideoView")

        l = QVBoxLayout()
        l.addWidget(self.view)

        self.setLayout(l)


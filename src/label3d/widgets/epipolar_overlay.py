"""EpipolarOverlay: draws epipolar lines in one camera for selected points."""

from __future__ import annotations

import math
from collections.abc import Collection

from typing import NamedTuple

from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsScene,
)

from label3d.geometry import EpipolarCache
from label3d.points import PointsStore

# Camera colors for epipolar lines — consistent palette, indexed by camera
# position in the global camera list
CAMERA_COLORS = [
    QColor(255, 100, 100),   # red
    QColor(100, 180, 255),   # blue
    QColor(100, 255, 100),   # green
    QColor(255, 200, 50),    # yellow
    QColor(200, 100, 255),   # purple
    QColor(255, 150, 50),    # orange
]


def camera_color(index: int) -> QColor:
    """Return the color for a camera by its index in the global list."""
    return CAMERA_COLORS[index % len(CAMERA_COLORS)]


class EpipolarLineInfo(NamedTuple):
    """Metadata for one active epipolar line, used by CameraPanel for labels."""

    source_cam: str
    color_index: int
    x1: float
    y1: float
    x2: float
    y2: float


class EpipolarOverlay:
    """Manages epipolar line items for one camera panel.

    For the selected bodypart: in each *other* camera that has a set/predicted
    point, computes the epipolar line and draws it in this camera.
    """

    def __init__(self, scene: QGraphicsScene, camera_name: str):
        self._scene = scene
        self._camera_name = camera_name
        self._line_items: list[QGraphicsLineItem] = []
        self._active_lines: list[EpipolarLineInfo] = []

    @property
    def active_lines(self) -> list[EpipolarLineInfo]:
        """Active epipolar line metadata for viewport label positioning."""
        return self._active_lines

    def update(
        self,
        epipolar_cache: EpipolarCache | None,
        points_store: PointsStore | None,
        all_cameras: list[str],
        trial: str,
        frame: int,
        selected_instance: int,
        selected_bodypart: str,
        available: dict[str, Collection[str]] | None = None,
    ):
        """Recompute and draw epipolar lines for the current selection.

        available maps each camera to the nodes its skeleton has (None: all).
        No lines are drawn in a camera without the selected node, and a camera
        without it is no line source.
        """
        line_idx = 0
        self._active_lines.clear()

        if (
            epipolar_cache is None
            or points_store is None
            or selected_instance < 0
            or not selected_bodypart
            or not trial
        ):
            self._hide_lines()
            return

        target_cam = self._camera_name

        def has(cam: str) -> bool:
            return available is None or selected_bodypart in available.get(cam, ())

        if not has(target_cam):
            self._hide_lines()
            return

        for i, source_cam in enumerate(all_cameras):
            if source_cam == target_cam or not has(source_cam):
                continue

            pt = points_store.get_point(
                trial, frame, selected_instance, selected_bodypart, source_cam
            )
            if pt.state not in ("set", "predicted"):
                continue  # skip not_set and not_visible points

            if math.isnan(pt.x) or math.isnan(pt.y):
                continue

            endpoints = epipolar_cache.get_epipolar_line(
                source_cam, target_cam, (pt.x, pt.y)
            )
            if endpoints is None:
                continue

            (x1, y1), (x2, y2) = endpoints

            # Get or create line item
            line = self._get_or_create_line(line_idx)
            line.setLine(x1, y1, x2, y2)

            # Color by source camera's global index
            color = camera_color(i)
            line_color = QColor(color)
            line_color.setAlpha(180)
            pen = QPen(line_color, 1.5)
            pen.setCosmetic(True)
            line.setPen(pen)
            line.setVisible(True)

            # Store metadata for viewport label positioning
            self._active_lines.append(
                EpipolarLineInfo(source_cam, i, x1, y1, x2, y2)
            )

            line_idx += 1

        # Hide unused lines
        for j in range(line_idx, len(self._line_items)):
            self._line_items[j].setVisible(False)

    def _get_or_create_line(self, index: int) -> QGraphicsLineItem:
        while index >= len(self._line_items):
            line = QGraphicsLineItem()
            line.setZValue(30)
            self._scene.addItem(line)
            self._line_items.append(line)
        return self._line_items[index]

    def update_direct(
        self,
        epipolar_cache: "EpipolarCache | None",
        source_cam: str,
        x: float,
        y: float,
        source_cam_index: int,
    ):
        """Draw a single epipolar line from a raw (x, y) point in source_cam.

        Used in calibration mode where points come directly from detection
        rows rather than from a PointsStore.
        """
        self._active_lines.clear()

        if epipolar_cache is None or source_cam == self._camera_name:
            self._hide_lines()
            return

        endpoints = epipolar_cache.get_epipolar_line(
            source_cam, self._camera_name, (x, y)
        )
        if endpoints is None:
            self._hide_lines()
            return

        (x1, y1), (x2, y2) = endpoints
        line = self._get_or_create_line(0)
        color = camera_color(source_cam_index)
        line_color = QColor(color)
        line_color.setAlpha(180)
        pen = QPen(line_color, 1.5)
        pen.setCosmetic(True)
        line.setPen(pen)
        line.setLine(x1, y1, x2, y2)
        line.setVisible(True)

        self._active_lines.append(
            EpipolarLineInfo(source_cam, source_cam_index, x1, y1, x2, y2)
        )

        for j in range(1, len(self._line_items)):
            self._line_items[j].setVisible(False)

    def _hide_lines(self):
        self._active_lines.clear()
        for item in self._line_items:
            item.setVisible(False)

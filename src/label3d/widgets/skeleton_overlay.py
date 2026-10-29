"""SkeletonOverlay: manages PointItems and edge lines for one camera panel."""

from __future__ import annotations

import math
from collections.abc import Collection

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

from label3d.points import PointData, PointsStore
from label3d.skeleton import Skeleton
from label3d.widgets.point_item import ColorMode, PointItem

# Edge line style
EDGE_PEN = QPen(QColor(200, 200, 200, 160), 1.5)
EDGE_PEN.setCosmetic(True)  # constant width regardless of zoom

# Label style
LABEL_COLOR = QColor(220, 220, 220)


class SkeletonOverlay:
    """Manages point items, edge lines, and node labels for one camera.

    Uses item pooling: items are reused across frames rather than
    created/destroyed each time.
    """

    def __init__(self, scene: QGraphicsScene, camera_name: str):
        self._scene = scene
        self._camera_name = camera_name

        # Item pools
        self._point_items: dict[tuple[int, str], PointItem] = {}  # (instance, bodypart)
        self._edge_items: list[QGraphicsLineItem] = []
        self._label_items: dict[tuple[int, str], QGraphicsSimpleTextItem] = {}

        # Track which items are active this frame
        self._active_points: set[tuple[int, str]] = set()
        # Instances that are predicted in this camera: shown, but not editable
        self._locked: set[int] = set()
        self._show_non_selected: bool = True
        self.color_mode: ColorMode = ColorMode.ERROR

    @property
    def show_non_selected(self) -> bool:
        return self._show_non_selected

    @show_non_selected.setter
    def show_non_selected(self, value: bool):
        self._show_non_selected = value

    def update(
        self,
        points_store: PointsStore | None,
        skeleton: Skeleton | None,
        trial: str,
        frame: int,
        selected_instance: int,
        selected_bodypart: str,
        available: Collection[str] | None = None,
    ):
        """Rebuild the overlay for the current frame.

        Every bodypart of every instance is shown, except those not in
        available (the nodes this camera's skeleton has; None: all of them),
        which are not drawn and can't be clicked. Points without stored
        coordinates are placed at a default position.
        """
        self._active_points.clear()
        self._locked.clear()

        if points_store is None or skeleton is None or not trial:
            self._hide_all()
            return

        instances = points_store.get_instances(trial, frame)
        cam = self._camera_name
        latest_sync = points_store.latest_sync
        new_label = points_store.latest_sync_label

        # Update point items for each instance and bodypart
        for inst in instances:
            if points_store.is_predicted_instance(trial, frame, inst, cam):
                self._locked.add(inst)
            # Skip non-selected instances if toggled off
            if not self._show_non_selected and inst != selected_instance and selected_instance >= 0:
                continue

            for bp_idx, bp in enumerate(skeleton.nodes):
                if available is not None and bp not in available:
                    continue
                pt = points_store.get_point(trial, frame, inst, bp, cam)
                key = (inst, bp)
                self._active_points.add(key)

                # Compute reprojection error if we have reprojected coords
                error = None
                if not math.isnan(pt.xr) and not math.isnan(pt.x):
                    error = math.sqrt((pt.x - pt.xr) ** 2 + (pt.y - pt.yr) ** 2)

                is_selected = (inst == selected_instance and bp == selected_bodypart)

                # Determine display position
                if not math.isnan(pt.x):
                    x, y = pt.x, pt.y
                elif not math.isnan(pt.xr):
                    x, y = pt.xr, pt.yr
                else:
                    # Default position for unset points (stacked in top-left)
                    x = 30.0
                    y = 30.0 + bp_idx * 20.0

                # Get or create point item
                item = self._get_or_create_point(key)
                item.configure(
                    bodypart=bp,
                    instance=inst,
                    state=pt.state,
                    x=x,
                    y=y,
                    camera=cam,
                    xr=pt.xr if not math.isnan(pt.xr) else None,
                    yr=pt.yr if not math.isnan(pt.yr) else None,
                    error=error,
                    selected=is_selected,
                    source=pt.source,
                    score=pt.score,
                    is_new=latest_sync is not None and pt.sync == latest_sync,
                    new_label=new_label,
                    color_mode=self.color_mode,
                )

                # Node name label
                label = self._get_or_create_label(key, bp)
                label.setPos(x + 10, y - 5)
                label.setVisible(item.isVisible())

        # Hide unused point items
        for key, item in self._point_items.items():
            if key not in self._active_points:
                item.setVisible(False)
            if key in self._label_items and key not in self._active_points:
                self._label_items[key].setVisible(False)

        # Update skeleton edges
        self._update_edges(points_store, skeleton, trial, frame, instances, available)

    def _update_edges(
        self,
        points_store: PointsStore,
        skeleton: Skeleton,
        trial: str,
        frame: int,
        instances: list[int],
        available: Collection[str] | None = None,
    ):
        """Draw skeleton edge lines between connected bodyparts (both available)."""
        cam = self._camera_name
        edge_idx = 0

        for inst in instances:
            for src_bp, dst_bp in skeleton.edges:
                if available is not None and not (
                    src_bp in available and dst_bp in available
                ):
                    continue
                src_pt = points_store.get_point(trial, frame, inst, src_bp, cam)
                dst_pt = points_store.get_point(trial, frame, inst, dst_bp, cam)

                # Need valid positions for both endpoints
                src_x = src_pt.x if not math.isnan(src_pt.x) else src_pt.xr
                src_y = src_pt.y if not math.isnan(src_pt.y) else src_pt.yr
                dst_x = dst_pt.x if not math.isnan(dst_pt.x) else dst_pt.xr
                dst_y = dst_pt.y if not math.isnan(dst_pt.y) else dst_pt.yr

                if any(math.isnan(v) for v in (src_x, src_y, dst_x, dst_y)):
                    continue

                # Get or create edge line
                line = self._get_or_create_edge(edge_idx)
                line.setLine(src_x, src_y, dst_x, dst_y)
                line.setVisible(True)
                edge_idx += 1

        # Hide unused edge items
        for i in range(edge_idx, len(self._edge_items)):
            self._edge_items[i].setVisible(False)

    def _get_or_create_point(self, key: tuple[int, str]) -> PointItem:
        if key not in self._point_items:
            item = PointItem()
            self._scene.addItem(item)
            self._point_items[key] = item
        return self._point_items[key]

    def _get_or_create_label(self, key: tuple[int, str], text: str) -> QGraphicsSimpleTextItem:
        if key not in self._label_items:
            item = QGraphicsSimpleTextItem(text)
            item.setBrush(LABEL_COLOR)
            font = item.font()
            font.setPointSize(9)
            item.setFont(font)
            item.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setZValue(49)
            self._scene.addItem(item)
            self._label_items[key] = item
        return self._label_items[key]

    def _get_or_create_edge(self, index: int) -> QGraphicsLineItem:
        while index >= len(self._edge_items):
            line = QGraphicsLineItem()
            line.setPen(EDGE_PEN)
            line.setZValue(40)
            self._scene.addItem(line)
            self._edge_items.append(line)
        return self._edge_items[index]

    def _hide_all(self):
        for item in self._point_items.values():
            item.setVisible(False)
        for item in self._edge_items:
            item.setVisible(False)
        for item in self._label_items.values():
            item.setVisible(False)

    def placed_points(self, instance: int) -> list[QPointF]:
        """Scene positions of one instance's points that are placed in this
        camera (set or predicted); not_visible and not_set points are left out."""
        return [pos for _key, pos in self._placed_items(instance)]

    def instance_points(self, instance: int) -> list[QPointF]:
        """Scene positions of one instance's placed points plus its unset points
        that are shown at their reprojection from the other cameras."""
        return [
            item.pos()
            for key, item in self._point_items.items()
            if key[0] == instance
            and key in self._active_points
            and item.isVisible()
            and (
                item.state in ("set", "predicted")
                or (item.state == "not_set" and item.has_reprojection)
            )
        ]

    def placed_point(self, instance: int, bodypart: str) -> QPointF | None:
        """Scene position of one point if it is placed in this camera, else None."""
        return dict(self._placed_items(instance)).get((instance, bodypart))

    def _placed_items(self, instance: int) -> list[tuple[tuple[int, str], QPointF]]:
        return [
            (key, item.pos())
            for key, item in self._point_items.items()
            if key[0] == instance
            and key in self._active_points
            and item.isVisible()
            and item.state in ("set", "predicted")
        ]

    def is_locked(self, instance: int) -> bool:
        """True when instance is predicted in this camera (double-click to edit)."""
        return instance in self._locked

    def get_point_item(self, instance: int, bodypart: str) -> PointItem | None:
        """Get the PointItem for a given (instance, bodypart), if it exists."""
        return self._point_items.get((instance, bodypart))

    def hit_test(self, scene_pos, threshold: float = 15.0) -> tuple[int, str] | None:
        """Find the nearest visible point item within threshold pixels.

        Args:
            scene_pos: QPointF in scene coordinates.
            threshold: Maximum distance in pixels.

        Returns:
            (instance, bodypart) tuple, or None if nothing close enough.
        """
        best = None
        best_dist = threshold

        for key, item in self._point_items.items():
            if not item.isVisible():
                continue
            dx = scene_pos.x() - item.pos().x()
            dy = scene_pos.y() - item.pos().y()
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best = key

        return best

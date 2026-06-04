"""LabelView: labeling mode — camera grid + bodypart sidebar + keyboard shortcuts.

Orchestrates skeleton overlays, epipolar overlays, point interaction,
and selection state across all camera panels.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QWidget

import logging
logger = logging.getLogger(__name__)

from label3d.geometry import (
    EpipolarCache,
    load_cameras,
    triangulate_all_instances,
    triangulate_frame,
)
from label3d.project import Project
from label3d.state import ProjectState
from label3d.widgets.bodypart_list import BodypartList
from label3d.widgets.camera_grid import CameraGrid
from label3d.widgets.view_3d import View3D


class LabelView(QWidget):
    """Labeling mode: camera grid with overlays + bodypart sidebar.

    Wires together:
    - SkeletonOverlay + EpipolarOverlay on each CameraPanel
    - Mouse interaction signals from CameraPanels
    - Selection state from ProjectState
    - Keyboard shortcuts for bodypart/instance cycling and point ops
    """

    def __init__(
        self,
        project: Project,
        state: ProjectState,
        camera_grid: CameraGrid,
        parent=None,
    ):
        super().__init__(parent)
        self._project = project
        self._state = state
        self._camera_grid = camera_grid
        self._epipolar_cache: EpipolarCache | None = None
        self._cameras: dict = {}
        self._tri_worker = None  # TriangulationWorker (batch)

        # Load calibration for epipolar geometry and triangulation
        calib_path = project.calibration_path()
        if calib_path.exists():
            self._cameras = load_cameras(calib_path)
            self._epipolar_cache = EpipolarCache(self._cameras)

        # Layout: sidebar | camera grid | 3D view
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Bodypart sidebar
        self._bodypart_list = BodypartList(state)
        self._bodypart_list.set_skeleton(state.skeleton)
        splitter.addWidget(self._bodypart_list)

        # Camera grid (reparented into this splitter)
        splitter.addWidget(camera_grid)

        # 3D view
        self._view_3d = View3D(self._cameras if self._cameras else None)
        splitter.addWidget(self._view_3d)

        # Set initial sizes: sidebar ~150px, grid gets the rest, 3D ~300px
        splitter.setSizes([150, 800, 300])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        layout.addWidget(splitter)

        # Connect camera panel signals and set up legends
        for cam_name in camera_grid.camera_names:
            panel = camera_grid.get_panel(cam_name)
            if panel is None:
                continue
            panel.point_clicked.connect(self._on_point_clicked)
            panel.point_dragged.connect(self._on_point_dragged)
            panel.point_right_clicked.connect(self._on_point_right_clicked)
            panel.deselect_requested.connect(self._state.deselect)
            # Set up colored legend showing other cameras (fixed in viewport)
            panel.setup_legend(camera_grid.camera_names)

        # Connect state signals
        state.selection_changed.connect(self._on_selection_changed)
        state.points_changed.connect(self._on_points_changed)
        state.frame_changed.connect(self._on_frame_changed)
        state.trial_changed.connect(self._on_trial_changed)

        # Register keyboard shortcuts
        self._setup_shortcuts()

        # Initial overlay update
        self._update_all_overlays()
        self._update_3d_view()

    # -- Overlay updates -------------------------------------------------------

    def _update_all_overlays(self):
        """Refresh skeleton and epipolar overlays on all camera panels."""
        for cam_name in self._camera_grid.camera_names:
            panel = self._camera_grid.get_panel(cam_name)
            if panel is None:
                continue
            self._update_panel_overlays(panel)

    def _update_panel_overlays(self, panel):
        """Refresh overlays for a single camera panel."""
        panel.skeleton_overlay.update(
            self._state.points_store,
            self._state.skeleton,
            self._state.current_trial,
            self._state.current_frame,
            self._state.selected_instance,
            self._state.selected_bodypart,
        )
        panel.epipolar_overlay.update(
            self._epipolar_cache,
            self._state.points_store,
            self._camera_grid.camera_names,
            self._state.current_trial,
            self._state.current_frame,
            self._state.selected_instance,
            self._state.selected_bodypart,
        )
        panel.update_line_labels()

    # -- Signal handlers -------------------------------------------------------

    def _on_selection_changed(self, instance: int, bodypart: str):
        """Update interaction mode and overlays when selection changes."""
        has_selection = instance >= 0 and bodypart != ""
        for cam_name in self._camera_grid.camera_names:
            panel = self._camera_grid.get_panel(cam_name)
            if panel is not None:
                panel.set_interaction_mode(has_selection)
        self._update_all_overlays()
        self._update_3d_view()

    def _on_points_changed(self, trial: str, frame: int, instance: int, bodypart: str):
        """Refresh overlays after a point edit; auto-triangulate if enabled."""
        if trial == self._state.current_trial and frame == self._state.current_frame:
            if self._state.auto_triangulate and self._cameras:
                self._triangulate_current_frame()
            self._update_all_overlays()
            self._update_3d_view()

    def _on_frame_changed(self, frame: int):
        self._update_all_overlays()
        self._update_3d_view()

    def _on_trial_changed(self, trial: str):
        self._update_all_overlays()
        self._update_3d_view()

    def _on_point_clicked(self, camera: str, instance: int, bodypart: str):
        """Select the clicked point."""
        self._state.select(instance, bodypart)

    def _on_point_dragged(
        self, camera: str, instance: int, bodypart: str, x: float, y: float
    ):
        """Commit dragged point position — dragging always promotes to 'set'."""
        store = self._state.points_store
        if store is None:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame

        store.set_point(trial, frame, instance, bodypart, camera, x, y, state="set")
        self._state.points_changed.emit(trial, frame, instance, bodypart)

    def _on_point_right_clicked(self, camera: str, instance: int, bodypart: str):
        """Toggle point between normal and not_visible.

        Not-visible points are grayed out and excluded from epipolar
        line computation and triangulation.
        """
        store = self._state.points_store
        if store is None:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame
        pt = store.get_point(trial, frame, instance, bodypart, camera)

        if pt.state == "not_visible":
            # Restore to set — keep existing coordinates
            x = pt.x if not math.isnan(pt.x) else 0.0
            y = pt.y if not math.isnan(pt.y) else 0.0
            store.set_point(trial, frame, instance, bodypart, camera, x, y, state="set")
        else:
            # Mark as not visible — keep coordinates but change state
            store.set_not_visible(trial, frame, instance, bodypart, camera)

        self._state.points_changed.emit(trial, frame, instance, bodypart)

    # -- Triangulation ---------------------------------------------------------

    def _update_3d_view(self):
        """Refresh the 3D view for the current frame."""
        self._view_3d.update_frame(
            self._state.points_store,
            self._state.skeleton,
            self._state.current_trial,
            self._state.current_frame,
            self._state.selected_instance,
            self._state.selected_bodypart,
        )

    def _triangulate_current_frame(self):
        """Triangulate all instances in the current frame."""
        store = self._state.points_store
        if store is None or not self._cameras:
            return
        trial = self._state.current_trial
        frame = self._state.current_frame
        instances = store.get_instances(trial, frame)
        for inst in instances:
            triangulate_frame(self._cameras, store, trial, frame, inst)
        # Invalidate the frame cache after bulk 3D writes, then redraw directly.
        # Do NOT emit points_changed here — that would re-trigger auto-triangulation.
        store._invalidate_cache(trial, frame)
        self._update_all_overlays()
        self._update_3d_view()

    def _triangulate_all_frames(self):
        """Launch batch triangulation of all frames in the current trial (background)."""
        store = self._state.points_store
        if store is None or not self._cameras:
            return

        # Cancel any running worker
        if self._tri_worker is not None and self._tri_worker.isRunning():
            self._tri_worker.quit()
            self._tri_worker.wait()

        from label3d.workers.triangulation_worker import TriangulationWorker
        self._tri_worker = TriangulationWorker(
            self._cameras, store, self._state.current_trial, parent=self
        )
        self._tri_worker.frame_done.connect(self._on_batch_frame_done)
        self._tri_worker.finished.connect(self._on_batch_done)
        self._tri_worker.start()

    def _on_batch_frame_done(self, frame: int):
        if frame == self._state.current_frame:
            store = self._state.points_store
            if store is not None:
                store._invalidate_cache(self._state.current_trial, frame)
            self._update_all_overlays()
            self._update_3d_view()

    def _on_batch_done(self):
        self._update_all_overlays()
        self._update_3d_view()

    # -- Keyboard shortcuts ----------------------------------------------------

    def _setup_shortcuts(self):
        """Register all keyboard shortcuts."""
        shortcuts = [
            # Bodypart cycling
            (Qt.Key.Key_Down, self._state.select_next_bodypart),
            (Qt.Key.Key_Up, self._state.select_prev_bodypart),
            # Instance cycling
            (Qt.Key.Key_QuoteLeft, self._state.select_next_instance),  # backtick
            # Instance by index (1-9)
            *[
                (getattr(Qt.Key, f"Key_{i}"),
                 lambda idx=i - 1: self._state.select_instance_by_index(idx))
                for i in range(1, 10)
            ],
            # Deselect
            (Qt.Key.Key_Escape, self._state.deselect),
            # Delete selected point
            (Qt.Key.Key_Backspace, self._delete_selected_point),
            # Instance management
            (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_I), self._add_instance),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Backspace), self._delete_instance),
            # Toggle non-selected instance visibility
            (Qt.Key.Key_H, self._toggle_instance_visibility),
            # Triangulation
            (Qt.Key.Key_T, self._triangulate_current_frame),
            (
                QKeySequence(
                    Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_T
                ),
                self._triangulate_all_frames,
            ),
        ]

        for key, callback in shortcuts:
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(callback)

    def _delete_selected_point(self):
        """Reset the selected point to not_set in all cameras."""
        store = self._state.points_store
        if store is None:
            return

        inst = self._state.selected_instance
        bp = self._state.selected_bodypart
        if inst < 0 or not bp:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame

        for cam in self._camera_grid.camera_names:
            store.delete_point(trial, frame, inst, bp, cam)
        self._state.points_changed.emit(trial, frame, inst, bp)

    def _add_instance(self):
        """Add a new empty instance."""
        store = self._state.points_store
        skeleton = self._state.skeleton
        if store is None or skeleton is None:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame
        new_idx = store.add_instance(trial, frame, skeleton.nodes)

        # Select the new instance
        bp = self._state.selected_bodypart or skeleton.nodes[0]
        self._state.select(new_idx, bp)
        self._state.points_changed.emit(trial, frame, new_idx, bp)

    def _delete_instance(self):
        """Delete the selected instance."""
        store = self._state.points_store
        if store is None:
            return

        inst = self._state.selected_instance
        if inst < 0:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame
        store.delete_instance(trial, frame, inst)
        self._state.deselect()
        self._state.points_changed.emit(trial, frame, inst, "")

    def _toggle_instance_visibility(self):
        """Toggle visibility of non-selected instances."""
        for cam_name in self._camera_grid.camera_names:
            panel = self._camera_grid.get_panel(cam_name)
            if panel is not None:
                overlay = panel.skeleton_overlay
                overlay.show_non_selected = not overlay.show_non_selected
        self._update_all_overlays()

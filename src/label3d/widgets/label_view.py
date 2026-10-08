"""LabelView: labeling mode — camera grid + bodypart sidebar + keyboard shortcuts.

Orchestrates skeleton overlays, epipolar overlays, point interaction,
and selection state across all camera panels.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMenu,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import logging
logger = logging.getLogger(__name__)

from label3d.geometry import (
    EpipolarCache,
    load_cameras,
    triangulate_all_instances,
    triangulate_frame,
)
from label3d.instance_init import (
    available_strategies,
    init_average,
    init_random,
    init_reprojected,
)
from label3d.project import Project
from label3d.skeleton import Skeleton
from label3d.state import ProjectState
from label3d.widgets.bodypart_list import BodypartList
from label3d.widgets.camera_grid import CameraGrid
from label3d.widgets.point_item import ColorMode
from label3d.widgets.point_legend import PointLegend
from label3d.widgets.view_3d import View3D

# The 3D view (widgets/view_3d.py) is kept but not shown; set True to add it
# back to the right of the camera grid.
SHOW_3D_VIEW = False

# Below this the Points list stops shrinking and the sidebar scrolls instead
SIDEBAR_LIST_MIN_HEIGHT = 160


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
        if calib_path is not None:
            self._cameras = load_cameras(calib_path)
            self._epipolar_cache = EpipolarCache(self._cameras)

        # Layout: sidebar | camera grid | 3D view
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Sidebar: bodypart list above the point legend, in a scroll area so
        # a short window scrolls instead of squeezing the list
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        self._bodypart_list = BodypartList(state)
        self._bodypart_list.set_skeleton(state.skeleton)
        self._bodypart_list.setMinimumHeight(SIDEBAR_LIST_MIN_HEIGHT)
        sidebar_layout.addWidget(self._bodypart_list, stretch=1)
        self._legend = PointLegend()
        sidebar_layout.addWidget(self._legend, stretch=0)

        self._sidebar_scroll = QScrollArea()
        self._sidebar_scroll.setWidget(sidebar)
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll.setMinimumWidth(sidebar.minimumSizeHint().width() + 16)
        self._sidebar_scroll.setMaximumWidth(240)
        splitter.addWidget(self._sidebar_scroll)

        # Camera grid (reparented into this splitter)
        splitter.addWidget(camera_grid)

        # 3D view (hidden: not built unless SHOW_3D_VIEW)
        self._view_3d: View3D | None = None
        if SHOW_3D_VIEW:
            self._view_3d = View3D(self._cameras if self._cameras else None)
            splitter.addWidget(self._view_3d)

        # Set initial sizes: sidebar ~150px, grid gets the rest, 3D ~300px
        splitter.setSizes([150, 800, 300] if SHOW_3D_VIEW else [150, 1100])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        if SHOW_3D_VIEW:
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
            panel.empty_right_clicked.connect(self._on_empty_right_clicked)
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

    # -- Experiment reload -----------------------------------------------------

    def reload_for_experiment(self) -> None:
        """Reload calibration data for the new active experiment."""
        calib_path = self._project.calibration_path()
        if calib_path is not None:
            self._cameras = load_cameras(calib_path)
            self._epipolar_cache = EpipolarCache(self._cameras)
        else:
            self._cameras = {}
            self._epipolar_cache = None
        if self._view_3d is not None:
            self._view_3d.set_cameras(self._cameras)

    def reload_calibration(self) -> None:
        """Pick up a change of active calibration and redraw."""
        self.reload_for_experiment()
        self._update_all_overlays()
        self._update_3d_view()

    def set_skeleton(self, skeleton: Skeleton | None) -> None:
        """Show a new skeleton: bodypart list, selection and overlays."""
        self._bodypart_list.set_skeleton(skeleton)
        state = self._state
        bodypart = state.selected_bodypart
        if skeleton is not None and bodypart and bodypart not in skeleton.nodes:
            state.select(state.selected_instance, skeleton.nodes[0])
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
            logger.debug(f"Points changed: {self._state.auto_triangulate=}, cameras={list(self._cameras.keys())}")
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
        if self._view_3d is None:
            return
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
        logger.debug(f"Triangulating frame {frame} in trial {trial}: instances={instances}")
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
        """Register keyboard shortcuts not already handled by menu QActions."""
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
            # Toggle non-selected instance visibility
            (Qt.Key.Key_H, self._toggle_instance_visibility),
            # Triangulation
            (Qt.Key.Key_T, self._triangulate_current_frame),
            (
                QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_T),
                self._triangulate_all_frames,
            ),
        ]
        # Ctrl+I, Ctrl+Shift+I and Ctrl+Backspace are registered as QAction
        # shortcuts in MainWindow's Instances menu, not as QShortcuts here,
        # to avoid double-firing.

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

    def accept_selected_point(self) -> None:
        """Mark the selected bodypart as reviewed in every camera (clears its new mark)."""
        store = self._state.points_store
        inst = self._state.selected_instance
        bp = self._state.selected_bodypart
        if store is None or inst < 0 or not bp:
            self._show_status("Select a point to accept")
            return
        n = store.accept_points(self._state.current_trial, self._state.current_frame, inst, bp)
        self._show_status(f"Accepted {bp}" if n else f"{bp} has no new points")
        self._update_all_overlays()

    def accept_frame(self) -> None:
        """Mark every point in the current frame as reviewed."""
        store = self._state.points_store
        if store is None:
            return
        n = store.accept_points(self._state.current_trial, self._state.current_frame)
        self._show_status(f"Accepted {n} point(s)" if n else "No new points in this frame")
        self._update_all_overlays()

    def set_color_mode(self, mode: ColorMode) -> None:
        """Color points by reprojection error or by source."""
        for cam_name in self._camera_grid.camera_names:
            panel = self._camera_grid.get_panel(cam_name)
            if panel is not None:
                panel.skeleton_overlay.color_mode = mode
        self._legend.set_color_mode(mode)
        self._update_all_overlays()

    def set_legend_visible(self, visible: bool) -> None:
        self._legend.setVisible(visible)

    def legend_visible(self) -> bool:
        return not self._legend.isHidden()

    def _toggle_instance_visibility(self):
        """Toggle visibility of non-selected instances."""
        for cam_name in self._camera_grid.camera_names:
            panel = self._camera_grid.get_panel(cam_name)
            if panel is not None:
                overlay = panel.skeleton_overlay
                overlay.show_non_selected = not overlay.show_non_selected
        self._update_all_overlays()

    # -- Instance initialisation (right-click context menu + shortcuts) --------

    def _show_status(self, msg: str, timeout: int = 3000) -> None:
        """Show msg in the main window's status bar, if available."""
        win = self.window()
        if hasattr(win, "statusBar") and callable(win.statusBar):
            try:
                win.statusBar().showMessage(msg, timeout)
            except Exception:
                pass

    def _on_empty_right_clicked(
        self, camera_name: str, scene_x: float, scene_y: float
    ) -> None:
        """Show 'Add instance' context menu when right-clicking on empty space."""
        store = self._state.points_store
        if store is None:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame

        if bool(store.get_instances_for_camera(trial, frame, camera_name)):
            menu = QMenu(self)
            add_menu = menu.addMenu("Add instance")
            add_menu.setEnabled(False)
            menu.exec(QCursor.pos())
            return

        strats = available_strategies(
            store,
            store.experiment_name,
            trial,
            frame,
            self._state.selected_instance if self._state.selected_instance >= 0 else None,
            camera_name,
        )

        menu = QMenu(self)
        add_menu = menu.addMenu("Add instance")

        random_act = add_menu.addAction("Random")
        random_act.triggered.connect(
            lambda: self._do_add_instance_single_camera(
                camera_name, "random", (scene_x, scene_y)
            )
        )

        avg_act = add_menu.addAction("Average")
        avg_act.setEnabled("average" in strats)
        if "average" not in strats:
            avg_act.setToolTip("No labeled instances to average")
        avg_act.triggered.connect(
            lambda: self._do_add_instance_single_camera(
                camera_name, "average", (scene_x, scene_y)
            )
        )

        repr_act = add_menu.addAction("Reprojected")
        repr_act.setEnabled("reprojected" in strats)
        if "reprojected" not in strats:
            repr_act.setToolTip(
                "Select an instance labeled in ≥2 other cameras first"
            )
        repr_act.triggered.connect(
            lambda: self._do_add_instance_single_camera(
                camera_name, "reprojected", (scene_x, scene_y)
            )
        )

        menu.exec(QCursor.pos())

    def _do_add_instance_single_camera(
        self,
        camera_name: str,
        strategy: str,
        center_px: tuple[float, float] | None,
    ) -> None:
        """Execute an add-instance strategy writing into one specific camera.

        random/average  — create a new instance; write coordinates for camera_name.
        reprojected     — fill missing bodyparts in camera_name for the selected
                          instance; no new instance is created.
        """
        store = self._state.points_store
        skeleton = self._state.skeleton
        if store is None or skeleton is None:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame
        cam = self._cameras.get(camera_name)
        frame_size = cam.size if cam is not None and cam.size[0] > 0 else (10000, 10000)

        if strategy == "reprojected":
            instance = self._state.selected_instance
            if instance < 0 or not self._cameras:
                return
            coords = init_reprojected(
                store,
                store.experiment_name,
                trial,
                frame,
                instance,
                camera_name,
                self._cameras,
            )
            if not coords:
                self._show_status("Reprojected init: insufficient data")
                return
            for bp, (x, y) in coords.items():
                pt = store.get_point(trial, frame, instance, bp, camera_name)
                if pt.state == "not_set":
                    store.set_point(
                        trial, frame, instance, bp, camera_name, x, y, state="set"
                    )
            self._state.points_changed.emit(trial, frame, instance, "")
            return

        # Reuse instance 0 if it already exists in another camera; otherwise create it.
        existing = store.get_instances(trial, frame)
        if existing:
            new_instance = existing[0]
        else:
            new_instance = store.add_instance(trial, frame, skeleton.nodes)

        if strategy == "average":
            coords, used_fallback = init_average(
                store,
                store.experiment_name,
                trial,
                camera_name,
                skeleton,
                center_px,
                frame_size,
            )
            if used_fallback:
                self._show_status(
                    "Not enough labeled instances to compute average"
                    " — used random placement"
                )
        else:
            coords = init_random(
                skeleton, camera_name, self._cameras, center_px, frame_size
            )

        for bp, (x, y) in coords.items():
            store.set_point(
                trial, frame, new_instance, bp, camera_name, x, y, state="set"
            )

        bp0 = skeleton.nodes[0] if skeleton.nodes else ""
        self._state.select(new_instance, bp0)
        self._state.points_changed.emit(trial, frame, new_instance, "")

    def _do_add_instance_all_cameras(self, strategy: str) -> None:
        """Add a new instance across every camera using strategy.

        Used by Ctrl+I (average), Ctrl+Shift+I (random), and the Instances menu.
        Each camera's coordinates are computed independently.
        """
        store = self._state.points_store
        skeleton = self._state.skeleton
        if store is None or skeleton is None:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame
        cam_names = self._camera_grid.camera_names

        existing = store.get_instances(trial, frame)
        new_instance = existing[0] if existing else store.add_instance(trial, frame, skeleton.nodes)

        any_fallback = False
        for cam_name in cam_names:
            cam = self._cameras.get(cam_name)
            frame_size = (
                cam.size if cam is not None and cam.size[0] > 0 else (10000, 10000)
            )

            if strategy == "average":
                coords, used_fallback = init_average(
                    store,
                    store.experiment_name,
                    trial,
                    cam_name,
                    skeleton,
                    None,       # center_px=None → frame centre
                    frame_size,
                )
                if used_fallback:
                    any_fallback = True
            else:
                coords = init_random(
                    skeleton, cam_name, self._cameras, None, frame_size
                )

            for bp, (x, y) in coords.items():
                store.set_point(
                    trial, frame, new_instance, bp, cam_name, x, y, state="set"
                )

        if any_fallback:
            self._show_status(
                "Not enough labeled instances to compute average"
                " — used random placement for some cameras"
            )

        bp0 = skeleton.nodes[0] if skeleton.nodes else ""
        self._state.select(new_instance, bp0)
        self._state.points_changed.emit(trial, frame, new_instance, "")

    def _do_reprojected_all_cameras(self) -> None:
        """Fill all cameras that are missing data for the selected instance.

        Used by Instances > Add > Reprojected when triggered from the menu
        (no single camera context).
        """
        store = self._state.points_store
        if store is None or not self._cameras:
            return

        instance = self._state.selected_instance
        if instance < 0:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame
        filled_any = False

        for cam_name in self._camera_grid.camera_names:
            coords = init_reprojected(
                store,
                store.experiment_name,
                trial,
                frame,
                instance,
                cam_name,
                self._cameras,
            )
            if not coords:
                continue
            for bp, (x, y) in coords.items():
                pt = store.get_point(trial, frame, instance, bp, cam_name)
                if pt.state == "not_set":
                    store.set_point(
                        trial, frame, instance, bp, cam_name, x, y, state="set"
                    )
                    filled_any = True

        if filled_any:
            self._state.points_changed.emit(trial, frame, instance, "")
        else:
            self._show_status(
                "Reprojected: insufficient camera data to fill any bodyparts"
            )

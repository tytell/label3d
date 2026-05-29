"""Main application window with tab bar and label view."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from label3d.project import Project
from label3d.state import ProjectState
from label3d.video import FrameProvider
from label3d.widgets.calibration_view import CalibrationView
from label3d.widgets.camera_grid import CameraGrid, LayoutMode
from label3d.widgets.frame_controls import FrameControls
from label3d.widgets.label_view import LabelView


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, project: Project, state: ProjectState, parent=None):
        super().__init__(parent)
        self._project = project
        self._state = state
        self._frame_provider = FrameProvider()

        self.setWindowTitle(f"label3d — {project.root.name}")
        self.resize(1200, 800)

        # Initialise state's experiment field silently (no signal yet)
        state._current_experiment = project.active_experiment

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar: tabs + selectors
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(4, 2, 4, 2)

        # Tab bar
        self._tabs = QTabBar()
        self._tabs.addTab("Calibrate")
        self._tabs.addTab("Label")
        self._tabs.addTab("Export")
        self._tabs.setCurrentIndex(1)  # Start on Label tab
        top_layout.addWidget(self._tabs)

        top_layout.addStretch()

        # Experiment selector (hidden when only one experiment)
        self._exp_label = QLabel("Experiment:")
        self._exp_combo = QComboBox()
        for exp_name in sorted(project.experiments.keys()):
            self._exp_combo.addItem(exp_name)
        self._exp_combo.setCurrentText(project.active_experiment)
        _single_experiment = len(project.experiments) <= 1
        self._exp_label.setVisible(not _single_experiment)
        self._exp_combo.setVisible(not _single_experiment)
        self._exp_combo.currentTextChanged.connect(self._on_experiment_combo_changed)
        top_layout.addWidget(self._exp_label)
        top_layout.addWidget(self._exp_combo)

        top_layout.addSpacing(12)

        # Camera view selector
        top_layout.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItem("All cameras")
        for cam in project.cameras:
            self._view_combo.addItem(cam)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        top_layout.addWidget(self._view_combo)

        top_layout.addSpacing(12)

        # Trial selector
        top_layout.addWidget(QLabel("Trial:"))
        self._trial_combo = QComboBox()
        for name in project.trial_names():
            self._trial_combo.addItem(name)
        self._trial_combo.currentTextChanged.connect(self._on_trial_changed)
        top_layout.addWidget(self._trial_combo)

        main_layout.addWidget(top_bar)

        # Camera grid (main area)
        self._camera_grid = CameraGrid()
        self._camera_grid.setup_cameras(project.cameras)
        self._camera_grid.set_frame_provider(self._frame_provider)

        # Wire skeleton and points into state (before LabelView so sidebar populates)
        state.skeleton = project.skeleton
        state.points_store = project.points

        # Label view wraps the camera grid with overlays + sidebar
        self._label_view = LabelView(project, state, self._camera_grid)

        # Calibration view
        self._calibration_view = CalibrationView(project)

        # Export stub
        _export_stub = QLabel("Export (coming soon)")
        _export_stub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Stacked widget: index matches tab index (0=Calibrate, 1=Label, 2=Export)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._calibration_view)  # 0
        self._stack.addWidget(self._label_view)        # 1
        self._stack.addWidget(_export_stub)            # 2
        self._stack.setCurrentIndex(1)
        main_layout.addWidget(self._stack, stretch=1)

        # Frame controls (bottom) — only relevant in Label tab
        self._frame_controls = FrameControls(state)
        main_layout.addWidget(self._frame_controls)

        # Connect tab switching
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

        # Menus
        self._setup_menus()

        # Connect state signals
        state.experiment_changed.connect(self._on_experiment_state_changed)
        state.frame_changed.connect(self._on_frame_changed)
        state.points_changed.connect(self._on_points_changed_status)

        # Load first trial
        if project.trial_names():
            self._load_trial(project.trial_names()[0])

    # -- Menus ---------------------------------------------------------------

    def _setup_menus(self):
        menu_bar = self.menuBar()

        # Label3D menu
        app_menu = menu_bar.addMenu("Label3D")
        quit_action = app_menu.addAction("Quit")
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)

        # File menu
        file_menu = menu_bar.addMenu("File")
        new_action = file_menu.addAction("New project...")
        new_action.setEnabled(False)
        open_action = file_menu.addAction("Open project...")
        open_action.setEnabled(False)
        file_menu.addSeparator()
        export_action = file_menu.addAction("Export points...")
        export_action.setEnabled(False)

        # View menu
        view_menu = menu_bar.addMenu("View")

        # View > Cameras
        cameras_menu = view_menu.addMenu("Cameras")
        cam_group = QActionGroup(self)
        cam_group.setExclusive(True)

        all_cam_action = cameras_menu.addAction("All cameras")
        all_cam_action.setCheckable(True)
        all_cam_action.setChecked(True)
        cam_group.addAction(all_cam_action)
        all_cam_action.triggered.connect(lambda: self._set_camera_view(0))

        cameras_menu.addSeparator()

        self._camera_actions: list[QAction] = [all_cam_action]
        for i, cam in enumerate(self._project.cameras):
            action = cameras_menu.addAction(cam)
            action.setCheckable(True)
            cam_group.addAction(action)
            action.triggered.connect(lambda checked, idx=i + 1: self._set_camera_view(idx))
            self._camera_actions.append(action)

        # View > Layout
        layout_menu = view_menu.addMenu("Layout")
        layout_group = QActionGroup(self)
        layout_group.setExclusive(True)

        self._layout_actions: dict[LayoutMode, QAction] = {}
        for mode, label in [
            (LayoutMode.GRID, "Grid"),
            (LayoutMode.HORIZONTAL, "Horizontal"),
            (LayoutMode.VERTICAL, "Vertical"),
        ]:
            action = layout_menu.addAction(label)
            action.setCheckable(True)
            layout_group.addAction(action)
            action.triggered.connect(
                lambda checked, m=mode: self._set_layout_mode(m)
            )
            self._layout_actions[mode] = action

        self._layout_actions[LayoutMode.GRID].setChecked(True)

    # -- Experiment switching ------------------------------------------------

    def _on_experiment_combo_changed(self, name: str):
        """Called when the user picks a different experiment from the combo."""
        if not name or name == self._project.active_experiment:
            return
        self._state.current_experiment = name  # auto-saves dirty store + emits signal

    def _on_experiment_state_changed(self, name: str):
        """Reload everything after an experiment switch."""
        self._project.set_experiment(name)
        self._state.points_store = self._project.points

        # Repopulate trial combo without triggering _on_trial_changed
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        for trial_name in self._project.trial_names():
            self._trial_combo.addItem(trial_name)
        self._trial_combo.blockSignals(False)

        if self._project.trial_names():
            self._load_trial(self._project.trial_names()[0])
            self._trial_combo.setCurrentIndex(0)

    # -- Tab switching -------------------------------------------------------

    def _on_tab_changed(self, index: int):
        """Switch main content area when tab changes."""
        self._stack.setCurrentIndex(index)
        # Frame controls are only meaningful in the Label tab (index 1)
        self._frame_controls.setVisible(index == 1)

    def _set_camera_view(self, index: int):
        """Switch camera from menu — keeps combo box in sync."""
        self._view_combo.setCurrentIndex(index)

    def _set_layout_mode(self, mode: LayoutMode):
        """Switch layout from menu."""
        self._camera_grid.set_layout_mode(mode)

    # -- Trial loading -------------------------------------------------------

    def _load_trial(self, trial_name: str):
        """Open video readers for a trial and set up frame range."""
        trial = self._project.get_trial(trial_name)

        # Build camera paths and offsets
        camera_paths = {}
        offsets = {}
        for cam in self._project.cameras:
            camera_paths[cam] = self._project.video_path(trial_name, cam)
            offsets[cam] = self._project.video_start_frame(trial_name, cam)

        self._frame_provider.open_trial(camera_paths, offsets)

        # Set frame range
        length = trial.length or 0
        max_frame = max(0, length - 1)
        self._state.max_frame = max_frame
        self._frame_controls.set_range(max_frame)

        # Set state (this triggers frame_changed -> display)
        self._state._current_trial = trial_name
        self._state.current_frame = 0

        # Force first frame display
        self._camera_grid.update_frame(0)

        self._status_bar.showMessage(
            f"Trial: {trial_name} — {length} frames, "
            f"{len(self._project.cameras)} cameras"
        )

    def _on_view_changed(self, index: int):
        """Switch between all-camera grid and single-camera view."""
        # Keep menu checkmarks in sync
        if 0 <= index < len(self._camera_actions):
            self._camera_actions[index].setChecked(True)

        if index == 0:
            self._camera_grid.show_single_camera(None)
        else:
            cam = self._project.cameras[index - 1]
            self._camera_grid.show_single_camera(cam)

    def _on_trial_changed(self, trial_name: str):
        if trial_name:
            self._load_trial(trial_name)

    def _on_frame_changed(self, frame: int):
        self._camera_grid.update_frame(frame)

    def _on_points_changed_status(self, trial: str, frame: int, _instance: int, _bodypart: str):
        """Update status bar with per-frame mean reprojection error after edits."""
        store = self._state.points_store
        if store is None or trial != self._state.current_trial:
            return
        if frame != self._state.current_frame:
            return
        errors = []
        for inst in store.get_instances(trial, frame):
            try:
                inst_df = store.df.loc[trial].loc[frame].loc[inst]
            except KeyError:
                continue
            for bp in inst_df.index:
                for cam in store.cameras:
                    pt = store.get_point(trial, frame, inst, bp, cam)
                    if pt.state in ("set", "predicted") and not math.isnan(pt.x):
                        if not math.isnan(pt.xr):
                            errors.append(math.hypot(pt.x - pt.xr, pt.y - pt.yr))
        if errors:
            mean_err = float(np.mean(errors))
            self._status_bar.showMessage(
                f"Trial: {trial} — frame {frame} — reprojection error: {mean_err:.2f} px"
            )

    def showEvent(self, event):
        super().showEvent(event)
        # Force first frame display now that widgets have their final sizes
        if not event.spontaneous():
            self._camera_grid.update_frame(self._state.current_frame)

    def closeEvent(self, event):
        # Save points if dirty
        if self._state.points_store is not None:
            self._state.points_store.save_if_dirty()
        self._frame_provider.close()
        super().closeEvent(event)

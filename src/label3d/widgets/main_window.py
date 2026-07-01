"""Main application window with tab bar and label view."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from label3d.workers.import_worker import ImportWorker
from label3d.workers.save_worker import SaveWorker

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
        self._experiment_positions: dict[str, tuple[str, int]] = {}
        self._pending_session: tuple[int, dict[str, QRectF], list[int], list[list[int]]] | None = None

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
        self._frame_controls.set_cameras(project.cameras)
        main_layout.addWidget(self._frame_controls)

        # Connect tab switching
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

        # Menus
        self._setup_menus()

        # Track the currently-connected PointsStore for marks updates
        self._connected_store = None
        self._connect_store(project.points)

        # Connect state signals
        state.experiment_changed.connect(self._on_experiment_state_changed)
        state.frame_changed.connect(self._on_frame_changed)
        state.points_changed.connect(self._on_points_changed_status)

        # Load first trial (or restore from session)
        self._restore_and_apply_session()

        # Offer SLP import if points.csv is empty
        self._maybe_offer_slp_import()

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
        save_action = file_menu.addAction("Save")
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._do_save_slp)
        slp_files_exist = (
            self._project.slp_dir.exists()
            and any(self._project.slp_dir.glob("*.slp"))
        )
        save_action.setEnabled(slp_files_exist)
        self._save_action = save_action
        file_menu.addSeparator()
        import_slp_action = file_menu.addAction("Import from SLP files…")
        import_slp_action.triggered.connect(self._do_import_slp)
        file_menu.addSeparator()
        export_action = file_menu.addAction("Export points...")
        export_action.setEnabled(False)

        # Navigate menu
        nav_menu = menu_bar.addMenu("Navigate")
        prev_labeled = nav_menu.addAction("Previous labeled frame")
        prev_labeled.setShortcut(QKeySequence("["))
        prev_labeled.triggered.connect(self._state.jump_to_prev_labeled)
        next_labeled = nav_menu.addAction("Next labeled frame")
        next_labeled.setShortcut(QKeySequence("]"))
        next_labeled.triggered.connect(self._state.jump_to_next_labeled)

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

        # Instances menu
        inst_menu = menu_bar.addMenu("Instances")

        add_menu = inst_menu.addMenu("Add")
        add_menu.aboutToShow.connect(self._update_instances_menu)

        self._action_add_average = add_menu.addAction("Average / Random")
        self._action_add_average.setShortcut(
            QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_I)
        )
        self._action_add_average.triggered.connect(
            lambda: self._label_view._do_add_instance_all_cameras("average")
        )

        self._action_add_random = add_menu.addAction("Random")
        self._action_add_random.setShortcut(
            QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_I)
        )
        self._action_add_random.triggered.connect(
            lambda: self._label_view._do_add_instance_all_cameras("random")
        )

        self._action_add_reprojected = add_menu.addAction("Reprojected")
        self._action_add_reprojected.setEnabled(False)
        self._action_add_reprojected.triggered.connect(
            self._label_view._do_reprojected_all_cameras
        )

        inst_menu.addSeparator()

        self._action_delete_instance = inst_menu.addAction("Delete instance")
        self._action_delete_instance.setShortcut(
            QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Backspace)
        )
        self._action_delete_instance.triggered.connect(
            self._label_view._delete_instance
        )

    def _update_instances_menu(self) -> None:
        """Enable/disable menu items based on current labeling state."""
        store = self._state.points_store
        instance = self._state.selected_instance
        trial = self._state.current_trial
        frame = self._state.current_frame

        # Disable add actions only when every camera already has an instance.
        all_cameras_have_instance = (
            store is not None
            and bool(store.get_instances(trial, frame))
            and all(
                bool(store.get_instances_for_camera(trial, frame, cam))
                for cam in store.cameras
            )
        )
        # "Average" is available when any labeled point exists in this trial
        avg_available = False
        if store is not None and trial and not all_cameras_have_instance:
            try:
                trial_df = store.df.loc[(store.experiment_name, trial)]
                for cam in store.cameras:
                    if (cam, "state") in trial_df.columns:
                        if trial_df[(cam, "state")].isin(["set", "predicted"]).any():
                            avg_available = True
                            break
            except KeyError:
                pass

        # "Reprojected" is available when selected instance has ≥2 cameras with data
        repr_available = False
        if store is not None and trial and instance >= 0:
            try:
                inst_df = store.df.loc[(store.experiment_name, trial, frame, instance)]
                n_cams = 0
                for cam in store.cameras:
                    if (cam, "state") in inst_df.columns:
                        if inst_df[(cam, "state")].isin(["set", "predicted"]).any():
                            n_cams += 1
                if n_cams >= 2:
                    repr_available = True
            except KeyError:
                pass

        self._action_add_average.setEnabled(avg_available and not all_cameras_have_instance)
        self._action_add_random.setEnabled(not all_cameras_have_instance)
        self._action_add_reprojected.setEnabled(repr_available and not all_cameras_have_instance)

    # -- Slider marks ----------------------------------------------------------

    def _connect_store(self, store) -> None:
        """Wire PointsStore.data_changed → slider marks refresh, replacing any old connection."""
        if self._connected_store is not None:
            try:
                self._connected_store.data_changed.disconnect(self._update_slider_marks)
            except RuntimeError:
                pass
        self._connected_store = store
        if store is not None:
            store.data_changed.connect(self._update_slider_marks)

    def _update_slider_marks(self) -> None:
        store = self._state.points_store
        trial = self._state.current_trial
        if store is None or not trial:
            self._frame_controls.update_marks({})
            return
        marks = store.get_labeled_frames_by_camera(trial)
        self._frame_controls.update_marks(marks)

    # -- Experiment switching ------------------------------------------------

    def _on_experiment_combo_changed(self, name: str):
        """Called when the user picks a different experiment from the combo."""
        if not name or name == self._project.active_experiment:
            return
        self._state.current_experiment = name  # auto-saves dirty store + emits signal

    def _on_experiment_state_changed(self, name: str):
        """Reload everything after an experiment switch."""
        # Save current (trial, frame) before the project switches
        old_exp = self._project.active_experiment
        if old_exp and self._state.current_trial:
            self._experiment_positions[old_exp] = (
                self._state.current_trial,
                self._state.current_frame,
            )

        self._project.set_experiment(name)
        self._state.points_store = self._project.points
        self._connect_store(self._project.points)
        self._label_view.reload_for_experiment()
        self._calibration_view.reload_for_experiment()

        # Repopulate trial combo without triggering _on_trial_changed
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        for trial_name in self._project.trial_names():
            self._trial_combo.addItem(trial_name)
        self._trial_combo.blockSignals(False)

        if self._project.trial_names():
            trial_names = self._project.trial_names()
            saved = self._experiment_positions.get(name)
            if saved and saved[0] in trial_names:
                saved_trial, saved_frame = saved
                self._load_trial(saved_trial, start_frame=saved_frame)
                self._trial_combo.setCurrentText(saved_trial)
            else:
                self._load_trial(trial_names[0])
                self._trial_combo.setCurrentIndex(0)

        self._maybe_offer_slp_import()

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

    def _load_trial(self, trial_name: str, start_frame: int = 0):
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

        frame = max(0, min(start_frame, max_frame))

        # Set state; always force-emit frame_changed so overlays refresh even
        # when switching experiments with frame already at 0.
        self._state._current_trial = trial_name
        self._state._current_frame = frame
        self._state.frame_changed.emit(frame)

        # Force first frame display (direct call covers pre-signal-connection startup)
        self._camera_grid.update_frame(frame)

        self._update_slider_marks()

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
                inst_df = store.df.loc[(store.experiment_name, trial, frame, inst)]
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

    # -- Settings persistence ------------------------------------------------

    @staticmethod
    def _app_settings() -> QSettings:
        return QSettings(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            "label3d",
            "label3d",
        )

    def _save_app_settings(self) -> None:
        s = self._app_settings()
        s.setValue("geometry", self.saveGeometry())

    def _restore_app_settings(self) -> None:
        s = self._app_settings()
        geom = s.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)

    def _save_session_settings(self) -> None:
        ini_path = self._project.root / ".label3d_session.ini"
        s = QSettings(str(ini_path), QSettings.Format.IniFormat)

        s.beginGroup("session")
        s.setValue("trial", self._state.current_trial or "")
        s.setValue("frame", self._state.current_frame)
        s.setValue("layout_mode", self._camera_grid.layout_mode.value)
        s.setValue("view_index", self._view_combo.currentIndex())
        s.endGroup()

        splitter_sizes = self._camera_grid.get_grid_splitter_sizes()
        if splitter_sizes is not None:
            v_sizes, row_sizes = splitter_sizes
            s.beginGroup("splitter")
            s.setValue("v", ",".join(str(x) for x in v_sizes))
            for i, row in enumerate(row_sizes):
                s.setValue(f"row_{i}", ",".join(str(x) for x in row))
            s.endGroup()

        s.beginGroup("zoom")
        for cam, rect in self._camera_grid.get_all_scene_rects().items():
            cam_key = cam.replace("/", "_")
            s.beginGroup(cam_key)
            s.setValue("x", rect.x())
            s.setValue("y", rect.y())
            s.setValue("w", rect.width())
            s.setValue("h", rect.height())
            s.endGroup()
        s.endGroup()

    def _restore_and_apply_session(self) -> None:
        """Read .label3d_session.ini and apply trial/frame/layout immediately.

        View-combo and per-camera zoom are deferred to showEvent via
        _pending_session because panels need their final sizes first.
        """
        ini_path = self._project.root / ".label3d_session.ini"
        trial_names = self._project.trial_names()

        if not ini_path.exists():
            if trial_names:
                self._load_trial(trial_names[0])
            return

        s = QSettings(str(ini_path), QSettings.Format.IniFormat)

        s.beginGroup("session")
        saved_trial = s.value("trial", "")
        saved_frame = int(s.value("frame", 0) or 0)
        layout_val = s.value("layout_mode", "grid")
        view_index = int(s.value("view_index", 0) or 0)
        s.endGroup()

        # Apply layout mode
        try:
            mode = LayoutMode(layout_val)
        except ValueError:
            mode = LayoutMode.GRID
        self._camera_grid.set_layout_mode(mode)
        self._layout_actions[mode].setChecked(True)

        # Load saved trial/frame
        if saved_trial and saved_trial in trial_names:
            self._trial_combo.setCurrentText(saved_trial)
            self._load_trial(saved_trial, start_frame=saved_frame)
        elif trial_names:
            self._load_trial(trial_names[0])

        # Read zoom rects for deferred restore
        zoom_rects: dict[str, QRectF] = {}
        s.beginGroup("zoom")
        for cam_key in s.childGroups():
            s.beginGroup(cam_key)
            x, y, w, h = s.value("x"), s.value("y"), s.value("w"), s.value("h")
            s.endGroup()
            if all(v is not None for v in (x, y, w, h)):
                cam = next(
                    (c for c in self._project.cameras if c.replace("/", "_") == cam_key),
                    cam_key,
                )
                try:
                    zoom_rects[cam] = QRectF(float(x), float(y), float(w), float(h))
                except (TypeError, ValueError):
                    pass
        s.endGroup()

        # Read splitter sizes for deferred restore
        v_sizes: list[int] = []
        row_sizes: list[list[int]] = []
        s.beginGroup("splitter")
        v_str = s.value("v", "")
        if v_str:
            try:
                v_sizes = [int(x) for x in str(v_str).split(",") if x.strip()]
            except ValueError:
                pass
        i = 0
        while True:
            row_str = s.value(f"row_{i}")
            if row_str is None:
                break
            try:
                row_sizes.append([int(x) for x in str(row_str).split(",") if x.strip()])
            except ValueError:
                pass
            i += 1
        s.endGroup()

        self._pending_session = (view_index, zoom_rects, v_sizes, row_sizes)

    def _apply_deferred_session(self) -> None:
        """Apply splitters, view-combo, and per-camera zoom once the window is fully shown."""
        if self._pending_session is None:
            return
        view_index, zoom_rects, v_sizes, row_sizes = self._pending_session
        self._pending_session = None

        # Restore all-cameras splitter sizes first, while all panels are visible.
        # If view_index then switches to solo mode, show_single_camera() will snapshot
        # these sizes so they survive the round-trip back to all-cameras view.
        if v_sizes and row_sizes:
            self._camera_grid.set_grid_splitter_sizes(v_sizes, row_sizes)

        # Restore view combo (show_single_camera runs processEvents internally)
        if 0 <= view_index < self._view_combo.count():
            self._view_combo.setCurrentIndex(view_index)

        # Restore zoom after any view change has settled
        if zoom_rects:
            self._camera_grid.set_all_scene_rects(zoom_rects)

    # -- SLP import ----------------------------------------------------------

    def _maybe_offer_slp_import(self) -> None:
        """Prompt the user to import SLP predictions if points.csv is empty."""
        if not self._project.slp_import_available:
            return
        slp_count = len(list(self._project.slp_dir.glob("*.slp")))
        reply = QMessageBox.question(
            self,
            "Import SLP predictions?",
            f"Found {slp_count} SLP file(s) in {self._project.slp_dir.name}/\n"
            "and the project has no labeled points yet.\n\n"
            "Import predicted instances now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._do_import_slp()

    def _do_import_slp(self) -> None:
        """Load SLP files in a background thread; merge into PointsStore on main thread.

        The worker only does the I/O-heavy SleapAdapter.load() call.  All
        PointsStore creation and mutation (QTimer-bearing objects) stays on the
        main thread to avoid "Timers cannot be stopped from another thread".
        """
        # Ensure PointsStore exists in the main thread before the worker starts.
        # This avoids creating QTimer-bearing objects in the background thread.
        if self._project.points is None:
            exp = self._project.experiments[self._project.active_experiment]
            from label3d.points import PointsStore
            self._project.points = PointsStore.create_empty(
                cameras=self._project.cameras,
                bodyparts=[],
                experiment_name=self._project.active_experiment,
            )
            self._project.points._save_path = exp.root / "points.csv"
            self._state.points_store = self._project.points
            self._connect_store(self._project.points)

        progress = QProgressDialog(
            "Importing SLP files…", "Cancel", 0, 0, self
        )
        progress.setWindowTitle("Importing")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setValue(0)

        self._import_worker = ImportWorker(self._project, parent=self)
        # `loaded` delivers the raw DataFrame to the main thread for merging.
        self._import_worker.loaded.connect(
            lambda df: self._on_import_loaded(df, progress)
        )
        self._import_worker.error.connect(
            lambda msg: self._on_import_error(msg, progress)
        )
        progress.canceled.connect(self._import_worker.requestInterruption)
        self._import_worker.start()

    def _on_import_loaded(self, adapter_df, progress: QProgressDialog) -> None:
        """Receive the adapter DataFrame in the main thread and merge into PointsStore."""
        progress.close()

        # All PointsStore mutation happens here, on the main thread.
        count = self._project.merge_slp_adapter_df(adapter_df)
        slp_count = len(list(self._project.slp_dir.glob("*.slp")))

        # Wire the store into state if it was newly created.
        if self._state.points_store is not self._project.points:
            self._state.points_store = self._project.points
            self._connect_store(self._project.points)
            self._label_view.reload_for_experiment()

        self._update_slider_marks()
        self._status_bar.showMessage(
            f"Imported {count} rows from {slp_count} SLP file(s)"
        )

    def _on_import_error(self, msg: str, progress: QProgressDialog) -> None:
        progress.close()
        QMessageBox.critical(self, "Import failed", f"SLP import error:\n{msg}")

    # -- SLP save ------------------------------------------------------------

    def _do_save_slp(self) -> None:
        """Write current labels back to SLP files in a background thread."""
        progress = QProgressDialog("Saving SLP files…", None, 0, 0, self)
        progress.setWindowTitle("Saving")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setValue(0)

        self._save_worker = SaveWorker(self._project, parent=self)
        self._save_worker.finished.connect(
            lambda n: self._on_save_done(n, progress)
        )
        self._save_worker.error.connect(
            lambda msg: self._on_save_error(msg, progress)
        )
        self._save_worker.start()

    def _on_save_done(self, n: int, progress: QProgressDialog) -> None:
        progress.close()
        self._status_bar.showMessage(f"Saved — wrote {n} SLP file(s)")

    def _on_save_error(self, msg: str, progress: QProgressDialog) -> None:
        progress.close()
        QMessageBox.critical(self, "Save failed", f"SLP save error:\n{msg}")

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            self._camera_grid.update_frame(self._state.current_frame)
            self._restore_app_settings()
            if self._pending_session is not None:
                QTimer.singleShot(0, self._apply_deferred_session)

    def closeEvent(self, event):
        if self._state.points_store is not None:
            self._state.points_store.save_if_changed()
        self._save_session_settings()
        self._save_app_settings()
        self._frame_provider.close()
        super().closeEvent(event)

"""Main application window with tab bar and label view."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
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

from label3d.workers.slp_sync_worker import (
    PredictionImportWorker,
    SlpReadWorker,
    SlpWriteWorker,
)
from label3d.workers.triangulation_worker import (
    AllExperimentsTriangulationWorker,
    TriangulationWorker,
)

from label3d.calibration_registry import entry_summary, entry_tree, find_calibration
from label3d.geometry import load_cameras
from label3d.points import PointsStore
from label3d.project import Project
from label3d.skeleton import Skeleton
from label3d.state import ProjectState
from label3d.video import FrameProvider
from label3d.widgets.calibration_view import CalibrationView
from label3d.widgets.camera_grid import CameraGrid, LayoutMode
from label3d.widgets.frame_controls import FrameControls
from label3d.widgets.label_view import LabelView
from label3d.widgets.point_item import ColorMode
from label3d.widgets.slp_setup_dialog import SlpSetupDialog
from label3d.widgets.sync_summary_dialog import SyncSummaryDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, project: Project, state: ProjectState, parent=None):
        super().__init__(parent)
        self._project = project
        self._state = state
        self._frame_provider = FrameProvider()
        self._experiment_positions: dict[str, tuple[str, int]] = {}
        self._pending_session: tuple[int, dict[str, QRectF], list[int], list[list[int]]] | None = None
        self._tri_queue: list[tuple[str, list[int]]] = []
        self._merge_tri_worker = None
        self._all_tri_worker = None
        self._switch_tri_worker = None
        self._offered_slp_setup = False

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

        # Calibration selector: the experiment's registry, active one selected
        top_layout.addWidget(QLabel("Calibration:"))
        self._calib_combo = QComboBox()
        self._calib_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._calib_combo.activated.connect(self._on_calibration_combo_activated)
        top_layout.addWidget(self._calib_combo)

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
        self._calibration_view.calibrations_changed.connect(
            self._on_active_calibration_changed
        )

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
        # An experiment with no active calibration needs calibrating before labeling
        if not project.has_calibration:
            self._tabs.setCurrentIndex(0)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")
        # Shown while the 3D points come from another calibration than the active one
        self._calib_status = QLabel()
        self._calib_status.setStyleSheet("color: #b45309;")
        self._calib_status.linkActivated.connect(lambda _link: self._retriangulate_experiment())
        self._calib_status.setVisible(False)
        self._status_bar.addPermanentWidget(self._calib_status)

        # Menus
        self._setup_menus()
        self._refresh_calibration_combo()
        self._update_calibration_status()

        # Track the currently-connected PointsStore for marks updates
        self._connected_store = None
        self._connect_store(project.points)

        # Connect state signals
        state.experiment_changed.connect(self._on_experiment_state_changed)
        state.frame_changed.connect(self._on_frame_changed)
        state.points_changed.connect(self._on_points_changed_status)

        # Load first trial (or restore from session)
        self._restore_and_apply_session()

        self._update_sync_action()
        # The startup SLP read runs once the window is on screen (showEvent):
        # its dialogs (conflict warning, sync summary) are application-modal,
        # and one opened before the window appears ends up hidden behind it,
        # blocking every click and key press.
        self._startup_read_pending = True

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
        sync_action = file_menu.addAction("Sync with Sleap...")
        sync_action.setShortcut(QKeySequence.StandardKey.Save)
        sync_action.triggered.connect(self._do_sync_slp)
        self._sync_action = sync_action
        setup_slp_action = file_menu.addAction("Set up Sleap files...")
        setup_slp_action.triggered.connect(self._do_setup_slp)
        import_action = file_menu.addAction("Import predictions...")
        import_action.setToolTip(
            "Read .slp prediction files (e.g. from sleap-nn track) into label3d"
        )
        import_action.triggered.connect(self._do_import_predictions)
        self._last_summary_action = file_menu.addAction("Last sync summary...")
        self._last_summary_action.triggered.connect(self._show_last_sync_summary)
        file_menu.aboutToShow.connect(
            lambda: self._last_summary_action.setEnabled(
                self._project.latest_sync_id() is not None
            )
        )
        self._update_sync_action()
        file_menu.addSeparator()
        load_skeleton_action = file_menu.addAction("Load skeleton...")
        load_skeleton_action.setToolTip(
            "Replace the skeleton from a skeleton.json; points of nodes with the "
            "same name are kept"
        )
        load_skeleton_action.triggered.connect(self._do_load_skeleton)
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
        nav_menu.addSeparator()
        prev_new = nav_menu.addAction("Previous new frame")
        prev_new.setShortcuts([QKeySequence("{"), QKeySequence("Shift+[")])
        prev_new.triggered.connect(lambda: self._jump_to_new(-1))
        next_new = nav_menu.addAction("Next new frame")
        next_new.setShortcuts([QKeySequence("}"), QKeySequence("Shift+]")])
        next_new.triggered.connect(lambda: self._jump_to_new(1))

        # View menu
        view_menu = menu_bar.addMenu("View")

        # View > Color points by
        color_menu = view_menu.addMenu("Color points by")
        color_group = QActionGroup(self)
        color_group.setExclusive(True)
        self._color_actions: dict[ColorMode, QAction] = {}
        for mode, label in (
            (ColorMode.ERROR, "Reprojection error"),
            (ColorMode.SOURCE, "Source"),
        ):
            action = color_menu.addAction(label)
            action.setCheckable(True)
            color_group.addAction(action)
            action.triggered.connect(lambda _checked, m=mode: self._set_color_mode(m))
            self._color_actions[mode] = action
        self._color_actions[ColorMode.ERROR].setChecked(True)
        self._color_mode = ColorMode.ERROR

        self._legend_action = view_menu.addAction("Show legend")
        self._legend_action.setCheckable(True)
        self._legend_action.setChecked(True)
        self._legend_action.toggled.connect(self._label_view.set_legend_visible)

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

        # View > Zoom
        # Zoom in/out act on the camera under the mouse; the rest on all cameras
        zoom_menu = view_menu.addMenu("Zoom")
        zoom_in = zoom_menu.addAction("Zoom in")
        zoom_in.setShortcuts([
            QKeySequence("+"),
            QKeySequence("Shift+="),
            QKeySequence(Qt.KeyboardModifier.KeypadModifier | Qt.Key.Key_Plus),
        ])
        zoom_in.triggered.connect(self._camera_grid.zoom_in_under_mouse)
        zoom_out = zoom_menu.addAction("Zoom out")
        zoom_out.setShortcuts([
            QKeySequence("-"),
            QKeySequence(Qt.KeyboardModifier.KeypadModifier | Qt.Key.Key_Minus),
        ])
        zoom_out.triggered.connect(self._camera_grid.zoom_out_under_mouse)
        zoom_menu.addSeparator()
        self._action_zoom_point = zoom_menu.addAction("Zoom to point")
        self._action_zoom_point.setShortcut(QKeySequence("P"))
        self._action_zoom_point.triggered.connect(self._zoom_to_point)
        self._action_zoom_instance = zoom_menu.addAction("Zoom to instance")
        self._action_zoom_instance.setShortcut(QKeySequence("I"))
        self._action_zoom_instance.triggered.connect(self._zoom_to_instance)
        zoom_all = zoom_menu.addAction("Zoom to all")
        zoom_all.setShortcuts([
            QKeySequence("="),
            QKeySequence(Qt.KeyboardModifier.KeypadModifier | Qt.Key.Key_Equal),
        ])
        zoom_all.triggered.connect(self._camera_grid.fit_targets)

        self._state.selection_changed.connect(self._update_zoom_actions)
        self._update_zoom_actions(
            self._state.selected_instance, self._state.selected_bodypart
        )

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

        accept_point = inst_menu.addAction("Accept point")
        accept_point.setShortcut(QKeySequence("A"))
        accept_point.setToolTip("Mark the selected point as reviewed (clears its new mark)")
        accept_point.triggered.connect(self._label_view.accept_selected_point)
        accept_frame = inst_menu.addAction("Accept frame")
        accept_frame.setShortcut(QKeySequence("Shift+A"))
        accept_frame.setToolTip("Mark every point in this frame as reviewed")
        accept_frame.triggered.connect(self._label_view.accept_frame)

        inst_menu.addSeparator()

        triangulate_all = inst_menu.addAction("Triangulate all experiments...")
        triangulate_all.triggered.connect(self._do_triangulate_all_experiments)

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
            self._frame_controls.update_new_marks({})
            return
        marks = store.get_labeled_frames_by_camera(trial)
        self._frame_controls.update_marks(marks)
        self._frame_controls.update_new_marks(store.get_new_frames(trial))

    # -- New points ------------------------------------------------------------

    def _set_color_mode(self, mode: ColorMode) -> None:
        self._color_mode = mode
        self._color_actions[mode].setChecked(True)
        self._label_view.set_color_mode(mode)

    def _jump_to_new(self, direction: int) -> None:
        """Go to the next (1) or previous (-1) frame with new points, crossing trials."""
        state = self._state
        if (state.jump_to_next_new() if direction > 0 else state.jump_to_prev_new()):
            return
        store = state.points_store
        if store is None:
            return
        with_new = set(store.trials_with_new_points())
        order = [t for t in self._project.trial_names() if t in with_new]
        if not order:
            self._status_bar.showMessage("No new points", 3000)
            return
        current = state.current_trial
        if current in order:
            i = order.index(current)
            candidates = order[i + 1:] if direction > 0 else order[:i][::-1]
        else:
            names = self._project.trial_names()
            pos = names.index(current) if current in names else -1
            candidates = [
                t for t in (names[pos + 1:] if direction > 0 else names[:max(pos, 0)][::-1])
                if t in with_new
            ]
        if not candidates:
            self._status_bar.showMessage(
                "No more new points" if direction > 0 else "No earlier new points", 3000
            )
            return
        trial = candidates[0]
        frames = sorted(store.get_new_frames(trial))
        self._go_to_trial_frame(trial, frames[0] if direction > 0 else frames[-1])

    def _go_to_trial_frame(self, trial: str, frame: int) -> None:
        if trial != self._state.current_trial:
            self._trial_combo.blockSignals(True)
            self._trial_combo.setCurrentText(trial)
            self._trial_combo.blockSignals(False)
            self._load_trial(trial, start_frame=frame)
        else:
            self._state.current_frame = frame

    def _go_to_first_new(self) -> None:
        store = self._state.points_store
        if store is None:
            return
        with_new = set(store.trials_with_new_points())
        for trial in self._project.trial_names():
            if trial in with_new:
                self._go_to_trial_frame(trial, min(store.get_new_frames(trial)))
                return

    def _show_sync_summary(self, results=None, unmatched: list[str] | None = None) -> None:
        """Show what the latest sync brought in, if the merge just applied recorded one."""
        latest = self._project.latest_sync_id()
        if results is not None:
            ids = {r.sync_id for r in results.values() if r.sync_id is not None}
            if latest not in ids:
                if unmatched:
                    QMessageBox.information(
                        self, "Import predictions",
                        "No new predictions were imported.\n\n"
                        f"{len(unmatched)} video(s) match no trial: "
                        + ", ".join(unmatched[:10]),
                    )
                return
        if latest is None:
            return
        history = self._project.sync_history()
        rows = history[history["id"] == latest]
        dialog = SyncSummaryDialog(
            rows, self._project.active_experiment, unmatched, parent=self
        )
        dialog.exec()
        if dialog.go_to_first_new():
            self._go_to_first_new()

    def _show_last_sync_summary(self) -> None:
        self._show_sync_summary()

    # -- Import predictions ---------------------------------------------------

    def _do_import_predictions(self) -> None:
        """Read .slp files from anywhere, e.g. sleap-nn track output, and merge them."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import predictions", str(self._project.root), "SLEAP files (*.slp)"
        )
        if not paths:
            return
        if self._project.points is not None:
            self._project.points.save_if_changed()
        progress = self._busy_dialog("Reading prediction files…", "Import predictions")
        self._import_worker = PredictionImportWorker(
            self._project, [Path(p) for p in paths], parent=self
        )
        self._import_worker.done.connect(
            lambda results, unmatched: self._on_import_read(results, unmatched, progress)
        )
        self._import_worker.error.connect(
            lambda msg: self._on_slp_error("Import failed", msg, progress)
        )
        self._import_worker.start()

    def _on_import_read(self, results, unmatched: list[str], progress: QProgressDialog) -> None:
        progress.close()
        overwrite = self._confirm_merge(results)
        if overwrite is None:
            self._status_bar.showMessage("Import cancelled — nothing changed")
            return
        self._project.apply_merge(results, overwrite=overwrite, kind="import")
        self._refresh_after_merge()
        n = sum(r.n_predictions for r in results.values())
        message = f"Imported {n} prediction(s)"
        if self._sync_action.isEnabled():
            message += "; Sync to write them to the SLP files"
        self._status_bar.showMessage(message)
        self._retriangulate(results, overwrite)
        self._show_sync_summary(results, unmatched)

    # -- Skeleton --------------------------------------------------------------

    def _do_load_skeleton(self) -> None:
        """Merge a skeleton.json into the project, asking before deleting points."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load skeleton", str(self._project.root), "Skeleton files (*.json)"
        )
        if not path:
            return
        try:
            skeleton = Skeleton.from_json(Path(path))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Load skeleton", f"Could not read {path}:\n{exc}")
            return

        removals = self._project.skeleton_removals(skeleton)
        if removals:
            lines = "\n".join(
                f"  {node}: {n} point(s)" for node, n in sorted(removals.items())
            )
            reply = QMessageBox.warning(
                self,
                "Delete points?",
                "These nodes have labelled points but are not in the new skeleton:\n\n"
                f"{lines}\n\n"
                "Loading it deletes those points in every experiment. "
                "Delete them and load the skeleton?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._status_bar.showMessage("Load skeleton cancelled — nothing changed")
                return

        old_nodes = set(self._project.skeleton.nodes) if self._project.skeleton else set()
        self._project.set_skeleton(skeleton)
        self._state.skeleton = skeleton
        self._label_view.set_skeleton(skeleton)
        self._refresh_after_merge()

        added = [n for n in skeleton.nodes if n not in old_nodes]
        removed = sorted(old_nodes - set(skeleton.nodes))
        parts = [f"Loaded skeleton with {len(skeleton.nodes)} node(s)"]
        if added:
            parts.append(f"added {', '.join(added)}")
        if removed:
            parts.append(f"removed {', '.join(removed)}")
        self._status_bar.showMessage("; ".join(parts))

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
        if not self._project.has_calibration:
            self._tabs.setCurrentIndex(0)

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

        self._refresh_calibration_combo()
        self._update_calibration_status()
        self._update_sync_action()
        self._read_slp_on_open()

    # -- Calibration switching -------------------------------------------------

    def _refresh_calibration_combo(self) -> None:
        """Refill the calibration combo from the active experiment's registry."""
        combo = self._calib_combo
        active = self._project.active_calibration_id()
        combo.blockSignals(True)
        combo.clear()
        for entry, depth in entry_tree(self._project.calibrations()):
            indent = "    " * depth + ("↳ " if depth else "")
            combo.addItem(f"{indent}{entry.label} · {entry.created:%Y-%m-%d %H:%M}", entry.id)
            combo.setItemData(combo.count() - 1, entry_summary(entry), Qt.ItemDataRole.ToolTipRole)
        index = combo.findData(active) if active else -1
        if index < 0 and combo.count():
            combo.insertItem(0, "(none active)", None)
            index = 0
        combo.setCurrentIndex(index)
        combo.setEnabled(combo.count() > 0)
        combo.setToolTip(combo.itemData(index, Qt.ItemDataRole.ToolTipRole) or "")
        combo.blockSignals(False)

    def _on_calibration_combo_activated(self, index: int) -> None:
        entry_id = self._calib_combo.itemData(index)
        if not entry_id or entry_id == self._project.active_calibration_id():
            return
        if self._switch_tri_worker is not None or self._all_tri_worker is not None:
            self._status_bar.showMessage("Wait for the current triangulation to finish", 3000)
            self._refresh_calibration_combo()
            return
        self._project.set_active_calibration(None, entry_id)
        self._on_active_calibration_changed()
        self._calibration_view.reload_active_calibration()
        self._retriangulate_experiment()

    def _on_active_calibration_changed(self) -> None:
        """The active calibration or the registry changed: reload and redraw."""
        self._label_view.reload_calibration()
        self._refresh_calibration_combo()
        self._update_calibration_status()

    def _update_calibration_status(self) -> None:
        """Warn when the 3D points were computed with another calibration."""
        project = self._project
        store = project.points
        if project.triangulation_is_stale() and (store is None or store.df.empty):
            # Nothing to re-triangulate yet: new points use the active one.
            project.set_triangulated_with(None, project.active_calibration_id())
        if not project.triangulation_is_stale():
            self._calib_status.setVisible(False)
            return
        used = project.triangulated_with()
        if used is None:
            source = "an earlier calibration"
        else:
            entry = find_calibration(project.experiments[project.active_experiment].root, used)
            source = f"'{entry.label}'" if entry else "a deleted calibration"
        self._calib_status.setText(
            f"3D points are from {source}; <a href='retriangulate'>re-triangulate</a>"
        )
        self._calib_status.setToolTip(
            "Re-triangulate this experiment with the active calibration. "
            "Instances > Triangulate all experiments does every experiment."
        )
        self._calib_status.setVisible(True)

    def _retriangulate_experiment(self) -> None:
        """Re-triangulate the active experiment with the active calibration.

        The current frame is done at once, then every labeled frame, current
        trial first, behind a window-modal progress dialog: the worker writes
        into the live store, so edits wait until it finishes. Cancelling
        leaves the stale-3D warning up.
        """
        store = self._project.points
        cameras = self._label_view._cameras
        entry_id = self._project.active_calibration_id()
        if store is None or not cameras or entry_id is None:
            return
        if self._switch_tri_worker is not None or self._all_tri_worker is not None:
            return
        if self._state.current_trial:
            self._label_view._triangulate_current_frame()

        progress = QProgressDialog("Re-triangulating…", "Cancel", 0, 0, self)
        progress.setWindowTitle("Switching calibration")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)

        worker = AllExperimentsTriangulationWorker(
            [(self._project.active_experiment, cameras, store)],
            first_trial=self._state.current_trial or None,
            parent=self,
        )
        progress.canceled.connect(worker.requestInterruption)

        def on_progress(done: int, total: int) -> None:
            progress.setMaximum(total)
            progress.setValue(done)
            progress.setLabelText(f"Re-triangulating frame {done} of {total}…")

        worker.progress.connect(on_progress)
        worker.finished.connect(
            lambda n: self._on_retriangulate_done(worker, n, entry_id, progress)
        )
        worker.error_occurred.connect(
            lambda msg: self._on_retriangulate_error(msg, progress)
        )
        self._switch_tri_worker = worker
        worker.start()

    def _on_retriangulate_done(self, worker, n_frames: int, entry_id: str, progress) -> None:
        progress.close()
        self._switch_tri_worker = None
        store = self._project.points
        if store is not None:
            store.save_if_changed()
            store._invalidate_cache(self._state.current_trial, self._state.current_frame)
        if worker.completed and self._project.active_calibration_id() == entry_id:
            self._project.set_triangulated_with(None, entry_id)
            self._status_bar.showMessage(f"Re-triangulated {n_frames} frame(s)", 5000)
        else:
            self._status_bar.showMessage(
                f"Re-triangulation stopped after {n_frames} frame(s)", 5000
            )
        self._update_calibration_status()
        self._state.frame_changed.emit(self._state.current_frame)

    def _on_retriangulate_error(self, msg: str, progress) -> None:
        self._switch_tri_worker = None
        self._on_slp_error("Triangulation failed", msg, progress)
        self._update_calibration_status()

    # -- Tab switching -------------------------------------------------------

    def _on_tab_changed(self, index: int):
        """Switch main content area when tab changes."""
        self._stack.setCurrentIndex(index)
        # Frame controls are only meaningful in the Label tab (index 1)
        self._frame_controls.setVisible(index == 1)

    def _set_camera_view(self, index: int):
        """Switch camera from menu — keeps combo box in sync."""
        self._view_combo.setCurrentIndex(index)

    def _update_zoom_actions(self, instance: int, bodypart: str):
        """Enable the selection-dependent zoom actions only when something is selected."""
        self._action_zoom_instance.setEnabled(instance >= 0)
        self._action_zoom_point.setEnabled(instance >= 0 and bool(bodypart))

    def _zoom_to_instance(self):
        instance = self._state.selected_instance
        if instance < 0:
            self._status_bar.showMessage("No instance selected", 2000)
            return
        self._camera_grid.zoom_to_instance(instance)

    def _zoom_to_point(self):
        instance = self._state.selected_instance
        bodypart = self._state.selected_bodypart
        if instance < 0 or not bodypart:
            self._status_bar.showMessage("No point selected", 2000)
            return
        self._camera_grid.zoom_to_point(instance, bodypart)

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
        s.setValue("color_mode", self._color_mode.value)
        s.setValue("show_legend", self._legend_action.isChecked())
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
        color_val = s.value("color_mode", ColorMode.ERROR.value)
        show_legend = str(s.value("show_legend", "true")).lower() != "false"
        s.endGroup()

        self._legend_action.setChecked(show_legend)

        try:
            self._set_color_mode(ColorMode(color_val))
        except ValueError:
            pass

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

    # -- Triangulate all experiments ------------------------------------------

    def _do_triangulate_all_experiments(self) -> None:
        """Re-triangulate every labeled frame of every experiment and save points.csv."""
        reply = QMessageBox.question(
            self,
            "Triangulate all experiments?",
            "Re-triangulate every labeled frame in every experiment and write the "
            "results to each experiment's points.csv?\n\n"
            "Experiments without an active calibration are skipped.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self._project.points is not None:
            self._project.points.save_if_changed()

        # Stores and calibrations are built here, in the main thread.
        jobs = []
        skipped = []
        used: dict[str, str | None] = {}
        for name, exp in self._project.experiments.items():
            calib_path = self._project.calibration_path(name)
            csv_path = exp.root / "points.csv"
            if name == self._project.active_experiment and self._project.points is not None:
                store = self._project.points
            elif csv_path.exists():
                store = PointsStore.from_csv(csv_path)
            else:
                continue
            if calib_path is None:
                skipped.append(name)
                continue
            try:
                cameras = load_cameras(calib_path)
            except Exception as exc:
                logger.warning(f"Could not load calibration for {name}: {exc}")
                skipped.append(name)
                continue
            jobs.append((name, cameras, store))
            used[name] = self._project.active_calibration_id(name)

        if not jobs:
            QMessageBox.information(
                self, "Nothing to triangulate",
                "No experiment has both an active calibration and labeled points.",
            )
            return

        progress = self._busy_dialog("Triangulating…", "Triangulate all experiments")
        self._all_tri_worker = AllExperimentsTriangulationWorker(jobs, parent=self)
        self._all_tri_worker.progress.connect(
            lambda done, total: progress.setLabelText(f"Triangulating frame {done} of {total}…")
        )
        self._all_tri_worker.finished.connect(
            lambda n: self._on_triangulate_all_done(n, jobs, skipped, used, progress)
        )
        self._all_tri_worker.error_occurred.connect(
            lambda msg: self._on_slp_error("Triangulation failed", msg, progress)
        )
        self._all_tri_worker.start()

    def _on_triangulate_all_done(self, n_frames: int, jobs, skipped, used, progress) -> None:
        progress.close()
        for _name, _cameras, store in jobs:
            store.save_if_changed()
        if self._all_tri_worker is not None and self._all_tri_worker.completed:
            for name, entry_id in used.items():
                if entry_id is not None:
                    self._project.set_triangulated_with(name, entry_id)
        self._update_calibration_status()
        self._all_tri_worker = None
        self._offered_slp_setup = False
        message = f"Triangulated {n_frames} frame(s) in {len(jobs)} experiment(s)"
        if skipped:
            message += f"; skipped {', '.join(sorted(skipped))} (no calibration)"
        self._status_bar.showMessage(message)
        self._state.frame_changed.emit(self._state.current_frame)

    # -- SLP startup read ------------------------------------------------------

    def _read_slp_on_open(self) -> None:
        """Read the mapped SLP files into points.csv when a project or experiment opens.

        Same read-and-merge as Sync, without the write-back: labels made in
        SLEAP show up straight away, and the SLP files are left alone until the
        next Sync.
        """
        if not self._sync_action.isEnabled():
            self._offer_slp_setup()
            return

        progress = self._busy_dialog("Reading SLP files…", "Loading SLP files")
        self._slp_open_worker = SlpReadWorker(self._project, parent=self)
        self._slp_open_worker.done.connect(
            lambda results: self._on_open_read(results, progress)
        )
        self._slp_open_worker.error.connect(
            lambda msg: self._on_slp_error("Reading SLP files failed", msg, progress)
        )
        self._slp_open_worker.start()

    def _on_open_read(self, results, progress: QProgressDialog) -> None:
        progress.close()
        overwrite = self._confirm_merge(results)
        if overwrite is None:
            self._status_bar.showMessage("SLP files not loaded — nothing changed")
            return
        n_new = sum(r.n_new for r in results.values())
        n_predictions = sum(r.n_predictions for r in results.values())
        if not any(r.has_changes for r in results.values()) and not (
            overwrite and any(r.conflicts for r in results.values())
        ):
            return
        self._project.apply_merge(results, overwrite=overwrite, kind="open")
        self._refresh_after_merge()
        self._status_bar.showMessage(
            f"Loaded {n_new} new labels / {n_predictions} predictions from SLP files"
        )
        self._retriangulate(results, overwrite)
        self._show_sync_summary(results)

    def _offer_slp_setup(self) -> None:
        """Once per session: point the user at Set up when slp/ has unassigned files."""
        if self._offered_slp_setup:
            return
        if not self._project.slp_dir.exists() or not any(self._project.slp_dir.glob("*.slp")):
            return
        self._offered_slp_setup = True
        reply = QMessageBox.question(
            self,
            "SLP files not assigned",
            f"{self._project.slp_dir.name}/ contains SLP files that are not assigned "
            "to videos.\n\nOpen Set up Sleap files... now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._do_setup_slp()

    # -- SLP sync ------------------------------------------------------------

    def _update_sync_action(self) -> None:
        """Enable Sync only when every video is assigned to exactly one SLP file."""
        mapping = self._project.slp_mapping
        trials_df = self._project.trials_df
        cameras = self._project.cameras
        complete = trials_df is not None and mapping.is_complete(trials_df, cameras)
        self._sync_action.setEnabled(complete)
        if complete:
            self._sync_action.setToolTip("")
            return
        if not mapping.files:
            reason = "No SLP files assigned"
        else:
            missing = mapping.missing_cells(trials_df, cameras) if trials_df is not None else []
            duplicates = mapping.duplicate_cells()
            parts = []
            if missing:
                parts.append(f"{len(missing)} video(s) have no SLP file")
            if duplicates:
                parts.append(f"{len(duplicates)} video(s) are in more than one file")
            reason = "; ".join(parts) or "SLP file assignments are incomplete"
        self._sync_action.setToolTip(f"{reason} — use Set up Sleap files...")

    def _do_sync_slp(self) -> None:
        """Read the mapped SLP files, merge, then write them back."""
        if self._project.points is not None:
            self._project.points.save_if_changed()

        progress = self._busy_dialog("Reading SLP files…", "Sync with Sleap")
        self._slp_read_worker = SlpReadWorker(self._project, parent=self)
        self._slp_read_worker.done.connect(
            lambda results: self._on_sync_read(results, progress)
        )
        self._slp_read_worker.error.connect(
            lambda msg: self._on_slp_error("Sync failed", msg, progress)
        )
        self._slp_read_worker.start()

    def _on_sync_read(self, results, progress: QProgressDialog) -> None:
        progress.close()
        overwrite = self._confirm_merge(results)
        if overwrite is None:
            self._status_bar.showMessage("Sync cancelled — nothing changed")
            return
        self._project.apply_merge(results, overwrite=overwrite, kind="sync")
        self._refresh_after_merge()

        n_new = sum(r.n_new for r in results.values())
        n_predictions = sum(r.n_predictions for r in results.values())

        progress = self._busy_dialog("Writing SLP files…", "Sync with Sleap")
        self._slp_write_worker = SlpWriteWorker(self._project, parent=self)
        self._slp_write_worker.done.connect(
            lambda n, _backup: self._on_sync_written(
                n, n_new, n_predictions, results, overwrite, progress
            )
        )
        self._slp_write_worker.error.connect(
            lambda msg: self._on_slp_error("Sync failed", msg, progress)
        )
        self._slp_write_worker.start()

    def _on_sync_written(
        self, n: int, n_new: int, n_predictions: int, results, overwrite: bool,
        progress: QProgressDialog,
    ) -> None:
        progress.close()
        self._status_bar.showMessage(
            f"Synced — read {n_new} new, {n_predictions} predictions; "
            f"wrote {n} SLP file(s)"
        )
        # Triangulation runs after the write so the two never touch the
        # DataFrame at the same time; 3D coordinates are not written to SLP.
        self._retriangulate(results, overwrite)
        self._show_sync_summary(results)

    # -- Re-triangulation after a merge ---------------------------------------

    def _retriangulate(self, results, overwrite: bool) -> None:
        """Re-triangulate the active experiment's merged frames, trial by trial."""
        experiment = self._project.active_experiment
        result = results.get(experiment)
        store = self._project.points
        cameras = self._label_view._cameras
        if result is None or store is None or not cameras:
            return

        frames = set(result.changed_frames)
        if overwrite:
            frames |= result.conflict_frames

        by_trial: dict[str, list[int]] = {}
        for exp, trial, frame in frames:
            if exp != experiment:
                continue
            by_trial.setdefault(str(trial), []).append(int(frame))
        if not by_trial:
            return

        self._tri_queue = [(trial, sorted(f)) for trial, f in sorted(by_trial.items())]
        self._run_next_triangulation()

    def _run_next_triangulation(self) -> None:
        if not self._tri_queue:
            self._merge_tri_worker = None
            self._refresh_after_merge()
            return
        trial, frames = self._tri_queue.pop(0)
        self._merge_tri_worker = TriangulationWorker(
            self._label_view._cameras, self._project.points, trial, frames, parent=self
        )
        self._merge_tri_worker.finished.connect(self._run_next_triangulation)
        self._merge_tri_worker.error_occurred.connect(self._on_triangulation_error)
        self._merge_tri_worker.start()

    def _on_triangulation_error(self, msg: str) -> None:
        self._tri_queue = []
        self._merge_tri_worker = None
        self._status_bar.showMessage(f"Triangulation failed: {msg}")

    # -- SLP setup -----------------------------------------------------------

    def _busy_dialog(self, text: str, title: str) -> QProgressDialog:
        progress = QProgressDialog(text, None, 0, 0, self)
        progress.setWindowTitle(title)
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setValue(0)
        return progress

    def _do_setup_slp(self) -> None:
        """Edit SLP file assignments, then read old files, merge, and write the new files."""
        dialog = SlpSetupDialog(self._project, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_mapping = dialog.mapping()
        old_mapping = self._project.slp_mapping
        if self._project.points is not None:
            self._project.points.save_if_changed()

        # Read the old files first so labels made in SLEAP are not lost
        progress = self._busy_dialog("Reading SLP files…", "Set up Sleap files")
        self._slp_read_worker = SlpReadWorker(self._project, old_mapping, parent=self)
        self._slp_read_worker.done.connect(
            lambda results: self._on_setup_read(results, new_mapping, progress)
        )
        self._slp_read_worker.error.connect(
            lambda msg: self._on_slp_error("Set up failed", msg, progress)
        )
        self._slp_read_worker.start()

    def _on_setup_read(self, results, new_mapping, progress: QProgressDialog) -> None:
        progress.close()
        self._setup_results = results
        overwrite = self._confirm_merge(results)
        if overwrite is None:
            self._status_bar.showMessage("Set up cancelled — nothing changed")
            return
        self._project.apply_merge(results, overwrite=overwrite, kind="setup")
        self._refresh_after_merge()
        self._project.save_slp_mapping(new_mapping)

        progress = self._busy_dialog("Writing SLP files…", "Set up Sleap files")
        self._slp_write_worker = SlpWriteWorker(
            self._project, new_mapping, backup=True, remove_unmapped=True, parent=self
        )
        self._slp_write_worker.done.connect(
            lambda n, backup_dir: self._on_setup_written(n, backup_dir, progress)
        )
        self._slp_write_worker.error.connect(
            lambda msg: self._on_slp_error("Set up failed", msg, progress)
        )
        self._slp_write_worker.start()

    def _on_setup_written(self, n: int, backup_dir: str, progress: QProgressDialog) -> None:
        progress.close()
        message = f"Set up {n} SLP file(s)"
        if backup_dir:
            message += f"; backup in {Path(backup_dir).relative_to(self._project.root).as_posix()}"
        self._status_bar.showMessage(message)
        self._update_sync_action()
        self._show_sync_summary(self._setup_results)

    def _on_slp_error(self, title: str, msg: str, progress: QProgressDialog) -> None:
        progress.close()
        QMessageBox.critical(self, title, msg)

    def _confirm_merge(self, results) -> bool | None:
        """Ask how to resolve conflicts.  Returns overwrite flag, or None to cancel."""
        conflicts = sum(len(r.conflicts) for r in results.values())
        if not conflicts:
            return False
        frames = sum(len(r.conflict_frames) for r in results.values())
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("SLEAP files changed")
        box.setText(
            f"{conflicts} point(s) in {frames} frame(s) were changed in the SLEAP files "
            "and differ from label3d."
        )
        box.setInformativeText(
            "Use SLEAP — overwrite label3d points with the SLEAP versions.\n"
            "Keep label3d — keep label3d points; SLEAP files will be overwritten."
        )
        use_sleap = box.addButton("Use SLEAP", QMessageBox.ButtonRole.AcceptRole)
        keep = box.addButton("Keep label3d", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is use_sleap:
            return True
        if clicked is keep:
            return False
        return None

    def _refresh_after_merge(self) -> None:
        """Rewire and redraw after merge results replaced the active store's data."""
        if self._state.points_store is not self._project.points:
            self._state.points_store = self._project.points
            self._connect_store(self._project.points)
            self._label_view.reload_for_experiment()
        self._state.frame_changed.emit(self._state.current_frame)
        self._update_slider_marks()

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            self._camera_grid.update_frame(self._state.current_frame)
            self._restore_app_settings()
            if self._pending_session is not None:
                QTimer.singleShot(0, self._apply_deferred_session)
            if self._startup_read_pending:
                self._startup_read_pending = False
                QTimer.singleShot(0, self._after_startup)

    def _after_startup(self) -> None:
        """Startup dialogs, one after another: a timer firing inside one
        dialog's event loop would stack the next one under it."""
        if self._project.video_warnings:
            self._warn_changed_videos()
        self._read_slp_on_open()

    def _warn_changed_videos(self) -> None:
        """Videos modified since their frame range was recorded in trials.csv."""
        warnings = self._project.video_warnings
        shown = warnings[:20]
        more = f"\n…and {len(warnings) - len(shown)} more" if len(warnings) > len(shown) else ""
        QMessageBox.warning(
            self,
            "Videos have changed",
            f"{len(warnings)} video(s) were modified after trials.csv recorded them. "
            "Their frame ranges and labels may no longer line up.\n\n"
            + "\n".join(shown) + more,
        )

    def _sync_on_close(self) -> bool:
        """Read, merge and write the SLP files on the main thread.  False = cancelled."""
        self._project.points.save_if_changed()
        results = self._project.read_slp_for_sync()
        overwrite = self._confirm_merge(results)
        if overwrite is None:
            return False
        self._project.apply_merge(results, overwrite=overwrite, kind="close")
        self._project.write_slp_files()
        return True

    def closeEvent(self, event):
        store = self._state.points_store
        if store is not None and store.changed:
            if self._sync_action.isEnabled():
                box = QMessageBox(self)
                box.setWindowTitle("Sync with Sleap before closing?")
                box.setText(
                    "You have unsaved changes. Sync with the SLEAP files before closing?"
                )
                sync = box.addButton("Sync", QMessageBox.ButtonRole.AcceptRole)
                box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
                cancel = box.addButton(QMessageBox.StandardButton.Cancel)
                box.setDefaultButton(sync)
                box.exec()
                clicked = box.clickedButton()
                if clicked is cancel:
                    event.ignore()
                    return
                if clicked is sync and not self._sync_on_close():
                    event.ignore()
                    return
            else:
                store.save_if_changed()
        self._save_session_settings()
        self._save_app_settings()
        self._frame_provider.close()
        super().closeEvent(event)

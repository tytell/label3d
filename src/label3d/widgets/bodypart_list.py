"""BodypartList: sidebar widget showing skeleton node names with selection."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget, QLabel

from label3d.points import PointsStore
from label3d.skeleton import Skeleton
from label3d.state import ProjectState

PARTIAL_MARK = "◐"      # node in some, not all, cameras' skeletons
ONE_CAMERA_MARK = "2D"  # node in one camera's skeleton: never triangulated
NO_CAMERA_MARK = "—"    # node in no camera's skeleton in this trial


class BodypartList(QWidget):
    """Sidebar list of bodypart names with selection highlighting.

    Shows per-bodypart status: [n/k], where n cameras have data for the
    selected instance on the current frame, out of the k cameras whose
    skeleton has the node in this trial. A node in some but not all cameras
    is marked PARTIAL_MARK; in one camera only (never triangulated), ONE_CAMERA_MARK.
    """

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self._state = state
        self._bodyparts: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel("Points")
        self._header.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._header)

        self._list = QListWidget()
        self._list.setMaximumWidth(180)
        self._list.setMinimumWidth(120)
        layout.addWidget(self._list)

        self._key = QLabel(f"{PARTIAL_MARK} some cameras · {ONE_CAMERA_MARK} one camera")
        self._key.setStyleSheet("color: gray;")
        self._key.setMaximumWidth(180)
        self._key.setWordWrap(True)
        self._key.hide()
        layout.addWidget(self._key)

        # Connect signals
        self._list.currentRowChanged.connect(self._on_row_changed)
        state.selection_changed.connect(self._on_selection_changed)
        state.points_changed.connect(self._on_points_changed)
        state.frame_changed.connect(self._on_frame_changed)
        state.trial_changed.connect(lambda _trial: self._update_status())
        state.skeleton_changed.connect(self._update_status)

    def set_skeleton(self, skeleton: Skeleton | None):
        """Populate the list with bodypart names from the skeleton."""
        self._list.blockSignals(True)
        self._list.clear()
        self._bodyparts.clear()

        if skeleton is not None:
            self._bodyparts = list(skeleton.nodes)
            for bp in self._bodyparts:
                self._list.addItem(bp)

        self._list.blockSignals(False)
        self._update_status()

    def _on_row_changed(self, row: int):
        """User clicked a bodypart in the list."""
        if 0 <= row < len(self._bodyparts):
            bp = self._bodyparts[row]
            inst = self._state.selected_instance
            if inst < 0:
                inst = 0
            self._state.select(inst, bp)

    def _on_selection_changed(self, instance: int, bodypart: str):
        """Sync list selection with state."""
        self._list.blockSignals(True)
        if bodypart and bodypart in self._bodyparts:
            self._list.setCurrentRow(self._bodyparts.index(bodypart))
        else:
            self._list.clearSelection()
        self._list.blockSignals(False)

    def _on_points_changed(self, trial: str, frame: int, instance: int, bodypart: str):
        self._update_status()

    def _on_frame_changed(self, frame: int):
        self._update_status()

    def _update_status(self):
        """Update each row's camera count, partial-camera mark and tooltip."""
        store = self._state.points_store
        if store is None:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame
        inst = self._state.selected_instance
        if inst < 0:
            inst = 0

        cameras = store.cameras
        any_partial = False
        for i, bp in enumerate(self._bodyparts):
            item = self._list.item(i)
            if item is None:
                continue
            node_cameras = self._state.node_cameras(bp, cameras)
            count = 0
            for cam in node_cameras:
                pt = store.get_point(trial, frame, inst, bp, cam)
                if pt.state in ("set", "predicted"):
                    count += 1
            suffix = f"  [{count}/{len(node_cameras)}]" if count > 0 else ""
            mark, tooltip = "", ""
            if len(node_cameras) < len(cameras):
                any_partial = True
                if not node_cameras:
                    mark, tooltip = f" {NO_CAMERA_MARK}", "In no camera's skeleton in this trial"
                elif len(node_cameras) == 1:
                    mark = f" {ONE_CAMERA_MARK}"
                    tooltip = f"Only in {node_cameras[0]} — not triangulated"
                else:
                    mark = f" {PARTIAL_MARK}"
                    tooltip = f"Only in {', '.join(node_cameras)}"
            item.setText(f"{bp}{mark}{suffix}")
            item.setToolTip(tooltip)
        self._key.setVisible(any_partial)

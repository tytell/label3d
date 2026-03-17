"""BodypartList: sidebar widget showing skeleton node names with selection."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget, QLabel

from label3d.points import PointsStore
from label3d.skeleton import Skeleton
from label3d.state import ProjectState


class BodypartList(QWidget):
    """Sidebar list of bodypart names with selection highlighting.

    Shows per-bodypart status (how many cameras have data for the
    currently selected instance on the current frame).
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

        # Connect signals
        self._list.currentRowChanged.connect(self._on_row_changed)
        state.selection_changed.connect(self._on_selection_changed)
        state.points_changed.connect(self._on_points_changed)
        state.frame_changed.connect(self._on_frame_changed)

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
        """Update display text to show camera count for each bodypart."""
        store = self._state.points_store
        if store is None:
            return

        trial = self._state.current_trial
        frame = self._state.current_frame
        inst = self._state.selected_instance
        if inst < 0:
            inst = 0

        cameras = store.cameras

        for i, bp in enumerate(self._bodyparts):
            count = 0
            for cam in cameras:
                pt = store.get_point(trial, frame, inst, bp, cam)
                if pt.state in ("set", "predicted"):
                    count += 1
            item = self._list.item(i)
            if item is not None:
                suffix = f"  [{count}/{len(cameras)}]" if count > 0 else ""
                item.setText(f"{bp}{suffix}")

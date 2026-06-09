"""Central application state with Qt signals.

All mutable state lives here. Views subscribe to signals and update themselves.
No view reads from another view directly.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from label3d.points import PointsStore
from label3d.skeleton import Skeleton


class ProjectState(QObject):
    """Holds all mutable application state and emits signals on changes."""

    # Signals
    frame_changed = Signal(int)          # global frame index
    trial_changed = Signal(str)          # trial name
    selection_changed = Signal(int, str) # (instance, bodypart) or (-1, "") for deselect
    points_changed = Signal(str, int, int, str)  # (trial, frame, instance, bodypart)
    experiment_changed = Signal(str)     # experiment name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_experiment: str = ""
        self._current_trial: str = ""
        self._current_frame: int = 0
        self._max_frame: int = 0
        self._selected_instance: int = -1
        self._selected_bodypart: str = ""
        self._auto_triangulate: bool = True
        self._points_store: PointsStore | None = None
        self._skeleton: Skeleton | None = None

    # -- Points and skeleton --

    @property
    def points_store(self) -> PointsStore | None:
        return self._points_store

    @points_store.setter
    def points_store(self, value: PointsStore | None):
        self._points_store = value

    @property
    def skeleton(self) -> Skeleton | None:
        return self._skeleton

    @skeleton.setter
    def skeleton(self, value: Skeleton | None):
        self._skeleton = value

    # -- Experiment --

    @property
    def current_experiment(self) -> str:
        return self._current_experiment

    @current_experiment.setter
    def current_experiment(self, value: str):
        if value != self._current_experiment:
            if self._points_store is not None:
                self._points_store.save_if_dirty()
            self._current_experiment = value
            self.experiment_changed.emit(value)

    # -- Trial --

    @property
    def current_trial(self) -> str:
        return self._current_trial

    @current_trial.setter
    def current_trial(self, value: str):
        if value != self._current_trial:
            self._current_trial = value
            self._current_frame = 0
            self.trial_changed.emit(value)
            self.frame_changed.emit(0)

    # -- Frame --

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @current_frame.setter
    def current_frame(self, value: int):
        value = max(0, min(value, self._max_frame))
        if value != self._current_frame:
            self._current_frame = value
            self.frame_changed.emit(value)

    @property
    def max_frame(self) -> int:
        return self._max_frame

    @max_frame.setter
    def max_frame(self, value: int):
        self._max_frame = max(0, value)
        if self._current_frame > self._max_frame:
            self.current_frame = self._max_frame

    # -- Selection --

    @property
    def selected_instance(self) -> int:
        return self._selected_instance

    @property
    def selected_bodypart(self) -> str:
        return self._selected_bodypart

    def select(self, instance: int, bodypart: str):
        if instance != self._selected_instance or bodypart != self._selected_bodypart:
            self._selected_instance = instance
            self._selected_bodypart = bodypart
            self.selection_changed.emit(instance, bodypart)

    def deselect(self):
        self.select(-1, "")

    # -- Auto-triangulate --

    @property
    def auto_triangulate(self) -> bool:
        return self._auto_triangulate

    @auto_triangulate.setter
    def auto_triangulate(self, value: bool):
        self._auto_triangulate = value

    # -- Selection helpers --

    def select_next_bodypart(self):
        """Cycle to the next bodypart in skeleton order (wraps)."""
        if self._skeleton is None or not self._skeleton.nodes:
            return
        inst = self._selected_instance if self._selected_instance >= 0 else 0
        if not self._selected_bodypart:
            self.select(inst, self._skeleton.nodes[0])
        else:
            self.select(inst, self._skeleton.next_node(self._selected_bodypart))

    def select_prev_bodypart(self):
        """Cycle to the previous bodypart in skeleton order (wraps)."""
        if self._skeleton is None or not self._skeleton.nodes:
            return
        inst = self._selected_instance if self._selected_instance >= 0 else 0
        if not self._selected_bodypart:
            self.select(inst, self._skeleton.nodes[-1])
        else:
            self.select(inst, self._skeleton.prev_node(self._selected_bodypart))

    def select_next_instance(self):
        """Cycle to the next instance (same bodypart)."""
        if self._points_store is None:
            return
        instances = self._points_store.get_instances(
            self._current_trial, self._current_frame
        )
        if not instances:
            return
        bp = self._selected_bodypart or (
            self._skeleton.nodes[0] if self._skeleton and self._skeleton.nodes else ""
        )
        if self._selected_instance in instances:
            idx = instances.index(self._selected_instance)
            next_inst = instances[(idx + 1) % len(instances)]
        else:
            next_inst = instances[0]
        self.select(next_inst, bp)

    def select_instance_by_index(self, index: int):
        """Select instance by 0-based index."""
        if self._points_store is None:
            return
        instances = self._points_store.get_instances(
            self._current_trial, self._current_frame
        )
        if 0 <= index < len(instances):
            bp = self._selected_bodypart or (
                self._skeleton.nodes[0]
                if self._skeleton and self._skeleton.nodes
                else ""
            )
            self.select(instances[index], bp)

    # -- Frame navigation helpers --

    def step_frame(self, delta: int):
        self.current_frame = self._current_frame + delta

    def go_to_start(self):
        self.current_frame = 0

    def go_to_end(self):
        self.current_frame = self._max_frame

    def jump_to_next_labeled(self):
        if self._points_store is None or not self._current_trial:
            return
        f = self._points_store.get_next_labeled_frame(self._current_trial, self._current_frame)
        if f is not None:
            self.current_frame = f

    def jump_to_prev_labeled(self):
        if self._points_store is None or not self._current_trial:
            return
        f = self._points_store.get_prev_labeled_frame(self._current_trial, self._current_frame)
        if f is not None:
            self.current_frame = f

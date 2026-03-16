"""Central application state with Qt signals.

All mutable state lives here. Views subscribe to signals and update themselves.
No view reads from another view directly.
"""

from PySide6.QtCore import QObject, Signal


class ProjectState(QObject):
    """Holds all mutable application state and emits signals on changes."""

    # Signals
    frame_changed = Signal(int)          # global frame index
    trial_changed = Signal(str)          # trial name
    selection_changed = Signal(int, str) # (instance, bodypart) or (-1, "") for deselect
    points_changed = Signal(str, int, int, str)  # (trial, frame, instance, bodypart)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_trial: str = ""
        self._current_frame: int = 0
        self._max_frame: int = 0
        self._selected_instance: int = -1
        self._selected_bodypart: str = ""
        self._auto_triangulate: bool = True

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

    # -- Frame navigation helpers --

    def step_frame(self, delta: int):
        self.current_frame = self._current_frame + delta

    def go_to_start(self):
        self.current_frame = 0

    def go_to_end(self):
        self.current_frame = self._max_frame

"""Background workers for batch triangulation (one batch per trial)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from label3d.geometry import CameraParams, triangulate_trial
from label3d.points import PointsStore

logger = logging.getLogger(__name__)


class TriangulationWorker(QThread):
    """Triangulates a trial (or some of its frames) in one batch in a background thread.

    Emits:
        finished()           — done
        error_occurred(str)  — on fatal error
    """

    finished = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        cameras: dict[str, CameraParams],
        points_store: PointsStore,
        trial: str,
        frames: list[int] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._cameras = cameras
        self._store = points_store
        self._trial = trial
        self._frames = frames  # None = every frame with data

    def run(self):
        try:
            triangulate_trial(self._cameras, self._store, self._trial, self._frames)
            self.finished.emit()
        except Exception as exc:
            logger.exception("Triangulating trial %s failed", self._trial)
            self.error_occurred.emit(str(exc))


class AllExperimentsTriangulationWorker(QThread):
    """Triangulates every trial of several experiments in a background thread.

    Each job is (experiment_name, cameras, points_store).  Stores are created in
    the main thread and only written to here; saving happens in the main thread.
    Each trial is one batch (triangulate_trial). The trial first_trial (if
    given) goes first. requestInterruption() stops the run after the current
    trial; `completed` is then False.

    Emits:
        progress(int, int)   — (trials_done, total_trials)
        finished(int)        — trials triangulated
        error_occurred(str)  — on fatal error
    """

    progress = Signal(int, int)
    finished = Signal(int)
    error_occurred = Signal(str)

    def __init__(
        self,
        jobs: list[tuple[str, dict[str, CameraParams], PointsStore]],
        first_trial: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._jobs = jobs
        self._first_trial = first_trial
        self.completed = False

    def run(self):
        try:
            trials = [
                (store, cameras, trial)
                for _name, cameras, store in self._jobs
                for trial in _labeled_trials(store)
            ]
            if self._first_trial is not None:
                trials.sort(key=lambda job: job[2] != self._first_trial)
            total = len(trials)
            done = 0
            for store, cameras, trial in trials:
                if self.isInterruptionRequested():
                    break
                triangulate_trial(cameras, store, trial)
                done += 1
                self.progress.emit(done, total)
            self.completed = done == total
            self.finished.emit(done)
        except Exception as exc:
            logger.exception("Triangulating all experiments failed")
            self.error_occurred.emit(str(exc))


def _labeled_trials(store: PointsStore) -> list[str]:
    """Every trial with data in a store, for its own experiment."""
    try:
        exp_df = store.df.loc[store.experiment_name]
    except KeyError:
        return []
    return [str(t) for t in exp_df.index.get_level_values("trial").unique()]

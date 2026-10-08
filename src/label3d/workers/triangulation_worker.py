"""Background worker for batch triangulation of all frames in a trial."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QThread, Signal

from label3d.geometry import CameraParams, triangulate_frame
from label3d.points import PointsStore

logger = logging.getLogger(__name__)


class TriangulationWorker(QThread):
    """Triangulates every frame in a trial (or a subset) in a background thread.

    Emits:
        progress(int, int)   — (frames_done, total_frames)
        frame_done(int)      — frame index just triangulated
        finished()           — all frames complete
        error_occurred(str)  — on fatal error
    """

    progress = Signal(int, int)   # done, total
    frame_done = Signal(int)
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
        self._frames = frames  # None = all labeled frames

    def run(self):
        try:
            trial = self._trial
            frames = self._frames

            if frames is None:
                # Collect all frames that have any data
                try:
                    trial_df = self._store.df.loc[(self._store.experiment_name, trial)]
                    frames = sorted(
                        trial_df.index.get_level_values("frame").unique().tolist()
                    )
                except KeyError:
                    frames = []

            total = len(frames)
            for done, frame in enumerate(frames):
                instances = self._store.get_instances(trial, frame)
                for inst in instances:
                    triangulate_frame(self._cameras, self._store, trial, frame, inst)
                self.frame_done.emit(frame)
                self.progress.emit(done + 1, total)

            self.finished.emit()
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class AllExperimentsTriangulationWorker(QThread):
    """Triangulates every labeled frame of several experiments in a background thread.

    Each job is (experiment_name, cameras, points_store).  Stores are created in
    the main thread and only written to here; saving happens in the main thread.
    Frames of first_trial (if given) go first. requestInterruption() stops the
    run after the current frame; `completed` is then False.

    Emits:
        progress(int, int)   — (frames_done, total_frames)
        finished(int)        — frames triangulated
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
            frames = [
                (store, cameras, trial, frame)
                for _name, cameras, store in self._jobs
                for trial, frame in _labeled_frames(store)
            ]
            if self._first_trial is not None:
                frames.sort(key=lambda job: job[2] != self._first_trial)
            total = len(frames)
            done = 0
            for store, cameras, trial, frame in frames:
                if self.isInterruptionRequested():
                    break
                for inst in store.get_instances(trial, frame):
                    triangulate_frame(cameras, store, trial, frame, inst)
                done += 1
                self.progress.emit(done, total)
            self.completed = done == total
            self.finished.emit(done)
        except Exception as exc:
            logger.exception("Triangulating all experiments failed")
            self.error_occurred.emit(str(exc))


def _labeled_frames(store: PointsStore) -> list[tuple[str, int]]:
    """Every (trial, frame) with data in a store, for its own experiment."""
    try:
        exp_df = store.df.loc[store.experiment_name]
    except KeyError:
        return []
    pairs = exp_df.index.droplevel(["instance", "bodypart"]).unique()
    return [(str(trial), int(frame)) for trial, frame in pairs]

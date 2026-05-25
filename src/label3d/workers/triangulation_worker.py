"""Background worker for batch triangulation of all frames in a trial."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from label3d.geometry import CameraParams, triangulate_frame
from label3d.points import PointsStore


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
                    trial_df = self._store.df.loc[trial]
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

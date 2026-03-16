"""Video reader with LRU frame cache and per-trial frame provider."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Wraps cv2.VideoCapture with an LRU frame cache."""

    def __init__(self, path: Path, cache_size: int = 8):
        self._path = Path(path)
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self._path}")
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._last_read: int = -1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read_frame(self, n: int) -> np.ndarray | None:
        """Read frame n (0-indexed). Returns BGR array or None on failure."""
        if n < 0 or n >= self._frame_count:
            return None

        # Check cache
        if n in self._cache:
            self._cache.move_to_end(n)
            return self._cache[n]

        # Sequential read is faster than seeking for n == last_read + 1
        if n != self._last_read + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, n)

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._last_read = n

        # Cache the frame
        self._cache[n] = frame
        self._cache.move_to_end(n)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.close()


class FrameProvider:
    """Manages VideoReaders for all cameras in the current trial.

    Maps global frame numbers to per-camera video frames using start offsets.
    """

    def __init__(self):
        self._readers: dict[str, VideoReader] = {}
        self._offsets: dict[str, int] = {}  # camera -> start frame offset

    def open_trial(self, camera_paths: dict[str, Path], offsets: dict[str, int]):
        """Open video files for a new trial, closing any previous ones.

        Args:
            camera_paths: camera_name -> absolute path to video file
            offsets: camera_name -> start frame offset
        """
        self.close()
        for cam, path in camera_paths.items():
            self._readers[cam] = VideoReader(path)
            self._offsets[cam] = offsets.get(cam, 0)

    def get_frame(self, camera: str, global_frame: int) -> np.ndarray | None:
        """Get a frame for a camera at a global frame index.

        Maps global_frame -> video_frame via start offset.
        """
        reader = self._readers.get(camera)
        if reader is None:
            return None
        video_frame = global_frame + self._offsets.get(camera, 0)
        return reader.read_frame(video_frame)

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._offsets.clear()

"""Video reader with LRU frame cache and per-trial frame provider.

A still image (an axis take is often one per camera) is read as a video of
one frame.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def frame_count(path: Path) -> int:
    """Frames in a video, 1 for a readable image, 0 if the file can't be read."""
    if is_image(path):
        return 1 if cv2.imread(str(path)) is not None else 0
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    finally:
        cap.release()


class VideoReader:
    """Wraps cv2.VideoCapture with an LRU frame cache; an image is one frame."""

    def __init__(self, path: Path, cache_size: int = 8):
        self._path = Path(path)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._last_read: int = -1
        self._cap = None
        if is_image(self._path):
            image = cv2.imread(str(self._path))
            if image is None:
                raise IOError(f"Cannot open image: {self._path}")
            self._frame_count = 1
            self._cache[0] = image
            self._cache_size = max(cache_size, 1)
            return
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self._path}")
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        if self._cap is None:
            return None

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

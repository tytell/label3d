import os
import cv2
from attrs import define, field
import numpy as np

import logging

@define(order=False)
class MediaVideo:
    filename: str = field()

    _reader_ = field(default=None)

    @property
    def __reader(self):
        # Load if not already loaded
        if self._reader_ is None:
            if not os.path.isfile(self.filename):
                raise FileNotFoundError(
                    f"Could not find filename video filename named {self.filename}"
                )

            # Try and open the file either locally in current directory or with full
            # path
            self._reader_ = cv2.VideoCapture(self.filename)

        # Return cached reader
        return self._reader_

    @property
    def fps(self) -> float:
        """Returns frames per second of video."""
        return self.__reader.get(cv2.CAP_PROP_FPS)

    @property
    def nframes(self) -> int:
        return int(self.__reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_frame(self, idx: int) -> np.ndarray:
        """See :class:`Video`."""

        # with self.__lock:
        if self.__reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self.__reader.set(cv2.CAP_PROP_POS_FRAMES, idx)

        success, frame = self.__reader.read()

        if not success or frame is None:
            raise KeyError(f"Unable to load frame {idx} from {self}.")

        return frame

@define(order=False)
class Video:
    backend = field()

    def __getattr__(self, item):
        return getattr(self.backend, item)
    
    @classmethod
    def from_media(cls, filename: str, *args, **kwargs) -> "Video":
        """Create an instance of a video object from a typical media file.

        For example, mp4, avi, or other types readable by FFMPEG.

        Args:
            filename: The name of the file
            args: Arguments to pass to :class:`MediaVideo`
            kwargs: Arguments to pass to :class:`MediaVideo`

        Returns:
            A Video object with a MediaVideo backend
        """

        backend = MediaVideo(filename=filename, *args, **kwargs)
        return cls(backend=backend)

    def __len__(self) -> int:
        return self.nframes
    
    def get_frame(self, idx: int) -> np.ndarray:
        return self.backend.get_frame(idx)
    
    
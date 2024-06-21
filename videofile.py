import os, sys
import subprocess
import json
import cv2
from attrs import define, field
import numpy as np
from datetime import datetime, time
import re

import logging

if sys.platform == 'darwin':
    FFPROBE = '/opt/homebrew/bin/ffprobe'
    FFMPEG = '/opt/homebrew/bin/ffmpeg'
elif sys.platform == 'win32':
    FFPROBE = ''
    FFMPEG = ''
else:
    raise(OSError("Could not find ffprobe or ffmpeg"))

if not os.path.exists(FFPROBE):
    raise(OSError("Could not find ffprobe at path {}".format(FFPROBE)))
if not os.path.exists(FFMPEG):
    raise(OSError("Could not find ffmpeg at path {}".format(FFMPEG)))

@define(order=False)
class MediaVideo:
    filename: str = field()

    _reader_ = field(default=None)
    _filedata_ = field(default=None)
    timecode = field(default=None)
    _frame = field(default=None)

    @property
    def __reader(self):
        # Get the file data before we open the reader
        _ = self.__filedata

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
    def __filedata(self):
        if self._filedata_ is None:
            cmd = [FFPROBE, '-print_format', 'json',
                '-show_format', '-show_streams',
                    self.filename]
            logging.debug("Command: {}".format(' '.join(cmd)))
            r = subprocess.run(cmd, capture_output=True)

            self._filedata_ = json.loads(r.stdout)

        return self._filedata_

    @property
    def fps(self) -> float:
        """Returns frames per second of video."""
        return self.__reader.get(cv2.CAP_PROP_FPS)

    @property
    def nframes(self) -> int:
        return int(self.__reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def frame(self):
        self._frame = self.__reader.get(cv2.CAP_PROP_POS_FRAMES)

    @frame.setter
    def frame(self, fr):
        if self._frame != fr:
            self.__reader.set(cv2.CAP_PROP_POS_FRAMES, fr)

    def __repr__(self):
        return os.path.basename(self.filename)
    
    def _get_video_stream(self):
        for s in self.__filedata['streams']:
            if s['codec_type'] == 'video':
                break
        else:
            raise KeyError("No video stream found")
        
        return s

    def _parse_timecode(self):
        frame_rate = self.fps

        try:
            video_stream = self._get_video_stream()
            creation_time = video_stream['tags']['creation_time']
            timecode = video_stream['tags']['timecode']
            
            creation_time = datetime.fromisoformat(str(creation_time).replace('Z', '+00:00'))

            m = re.fullmatch('(\d{2})[:;.](\d{2})[:;.](\d{2})[:;.](\d+)', timecode)
            if m is None:
                raise ValueError("Could not parse timecode {}".format(timecode))

            hmsf = [int(g) for g in m.groups()]

            us = int(float(hmsf[3]) / frame_rate * 1e6)
            timecode = time(hmsf[0], hmsf[1], hmsf[2], us)

            timecode = datetime(creation_time.year, creation_time.month, creation_time.day,
                                timecode.hour, timecode.minute, timecode.second, timecode.microsecond)

            self.timecode = timecode
        except KeyError:
            self.timecode = None

    def get_next_frame(self) -> np.ndarray:
        success, frame = self.__reader.read()

        if not success or frame is None:
            raise KeyError(f"Unable to load frame {idx} from {self}.")

        return frame

    def get_frame(self, idx: int) -> np.ndarray:
        """See :class:`Video`."""

        # with self.__lock:
        if self.__reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self.__reader.set(cv2.CAP_PROP_POS_FRAMES, idx)

        success, frame = self.__reader.read()

        if not success or frame is None:
            raise KeyError(f"Unable to load frame {idx} from {self}.")

        return frame
    
    def get_info_as_parameters(self):
        self._parse_timecode()
        tc = "{:%H:%M:%S}.{:03d}".format(self.timecode, int(round(self.timecode.microsecond/1000)))
        p = [{'name': 'Timecode', 'type': 'str', 'value': tc, 'readonly': True},
             {'name': 'Frame rate', 'type': 'float', 'value': self.fps, 'suffix': 'fps', 'readonly': True}]

        return p
    
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
    
    def __repr__(self):
        return self.backend.__repr__()
    
    def get_frame(self, idx: int) -> np.ndarray:
        return self.backend.get_frame(idx)
    
    def get_info_as_parameters(self):
        return self.backend.get_info_as_parameters()
    
    
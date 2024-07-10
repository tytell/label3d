import os, sys
from attrs import define, field, Factory
import aniposelib
import cv2
from time import sleep
import numpy as np

from qtpy import QtCore, QtGui
from qtpy.QtCore import (
    QEvent, Qt, QObject, QThread,
    Slot
)

import logging

from videofile import Video

from contextlib import contextmanager, redirect_stdout
import io

@contextmanager
def VideoCapture(filename, *args, **kwargs):
    cap = cv2.VideoCapture(filename, *args, **kwargs)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f'Could not find video file "{filename}"')
        yield cap
    finally:
        cap.release()
    
class Calibration(QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(int, int)

    def __init__(self, cameranames, videos, framestep, type,
                 nx, ny, square_size, marker_size, marker_bits, n_markers_in_dict, outputfile):
        super(Calibration, self).__init__()

        self.cameranames = cameranames
        self.videos = videos
        self.framestep = framestep
        
        self.type = type
        self.nx = nx
        self.ny = ny
        self.square_size = square_size
        self.marker_size = marker_size
        self.marker_bits = marker_bits
        self.n_markers_in_dict = n_markers_in_dict
        
        self.outputfile = outputfile
        logging.debug("In Calibration.__init__")

    @classmethod
    def from_parameters(cls, cameranames, videos, params):
        logging.debug(f"params['Type'] = {params['Type']}")

        return cls(cameranames, videos, framestep=params['Frame Step'], type=params['Type'], 
                   nx=params['Number of squares horizontally'], ny=params['Number of squares vertically'],
                   square_size=params['Size of square'], marker_size=params['Size of marker'],
                   marker_bits=params['Marker bits'], n_markers_in_dict=params['Number of markers'],
                   outputfile=params['Output file'])

    def save_calibration(self, outputfile):
        self.camgroup.dump(outputfile)
        
    @Slot()
    def run(self):
        try:
            board = aniposelib.boards.CharucoBoard(squaresX=self.nx,
                                                squaresY=self.ny,
                                                square_length=self.square_size,
                                                marker_length=self.marker_size,
                                                marker_bits=self.marker_bits,
                                                dict_size=self.n_markers_in_dict)
            logging.debug("Set up boards")
            self.camgroup = aniposelib.cameras.CameraGroup.from_names(self.cameranames)

            nframes_in_vid = self.videos[0].nframes // self.framestep
            n = nframes_in_vid * len(self.videos)

            logging.debug(f"Calibration: using {nframes_in_vid} frames in each video for {len(self.videos)}")

            # from aniposelib.CameraGroup.get_rows_videos
            all_rows = []
            for vnum, (cam, vid) in enumerate(zip(self.camgroup.cameras, self.videos)):
                logging.debug(f"Detecting board in video #{vnum}: {vid}")

                framenums = range(0, vid.nframes, self.framestep)
                self.progress.emit(vnum*nframes_in_vid, n)

                # from aniposelib.CalibrationObject.detect_video
                rows_cam = []
                rows_vid = []
                for i, framenum in enumerate(framenums):
                    frame = vid.get_frame(framenum)

                    corners, ids = board.detect_image(frame)

                    if corners is not None and len(corners) > 0:
                        # first element in key is the video group number - which would allow us, in principle to calibrate
                        # on multiple videos from each camera
                        key = (0, framenum)
                        rows_vid.append({'framenum': key, 'corners': corners, 'ids': ids})

                    self.progress.emit(vnum*nframes_in_vid + i, n)

                rows_vid = board.fill_points_rows(rows_vid)
                logging.debug(f"{len(rows_vid)} boards detected")

                rows_cam.extend(rows_vid)

                all_rows.append(rows_cam)

            logging.debug("Setting video sizes")

            # from aniposelib.CameraGroup.calibrate_videos
            for cam, vid in zip(self.camgroup.cameras, self.videos):
                cam.set_size(vid.frame_size)
            
            logging.debug("Running calibration!")

            # f = io.StringIO()            
            # with redirect_stdout(f):
            #     error = self.camgroup.calibrate_rows(all_rows, board, init_intrinsics=True, init_extrinsics=True)
            # output = f.getvalue()
            # logging.debug(output)

            error = self.camgroup.calibrate_rows(all_rows, board, init_intrinsics=True, init_extrinsics=True)

        except Exception as ex:
            logging.error(ex)

        finally:
            logging.debug("Thread done!")
            self.finished.emit()



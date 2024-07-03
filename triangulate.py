import os, sys
from attrs import define, field, Factory
# import aniposelib

from qtpy import QtCore, QtGui
from qtpy.QtCore import (
    QEvent, Qt, QObject, QThread,
    Slot
)

import logging

class Calibration(QObject):
    def __init__(self, cameranames, videofiles, framestep, type,
                 nx, ny, square_size, marker_size, marker_bits, n_markers_in_dict, outputfile):
        super(QObject, self).__init__()

        self.cameranames = cameranames
        self.videofiles = videofiles
        self.framestep = framestep
        
        self.type = type
        self.nx = nx
        self.ny = ny
        self.square_size = square_size
        self.marker_size = marker_size
        self.marker_bits = marker_bits
        self.n_markers_in_dict = n_markers_in_dict
        
        self.outputfile = outputfile

    @classmethod
    def from_parameters(cls, cameranames, videofiles, params):
        logging.debug(f"params['Type'] = {params['Type']}")

        return cls(cameranames, videofiles, framestep=params['Frame Step'], type=params['Type'], 
                   nx=params['Number of squares horizontally'], ny=params['Number of squares vertically'],
                   square_size=params['Size of square'], marker_size=params['Size of marker'],
                   marker_bits=params['Marker bits'], n_markers_in_dict=params['Number of markers'],
                   outputfile=params['Output file'])
    


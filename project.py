import os, sys

import pandas as pd

from pyqtgraph.parametertree import Parameter, ParameterTree, parameterTypes
import pyqtgraph as pg
import numpy as np
from string import ascii_uppercase

import logging
logger = logging.getLogger('label3d')

from qtpy import QtCore
from qtpy.QtCore import (
    QObject,
    Slot,
)

from points import Points

from ruamel.yaml import YAML, CommentedMap, CommentedSeq
yaml = YAML(typ='safe')

def dataframe_to_yaml(representer, node):
    return representer.represent_mapping(u'!pandas.DataFrame', node.to_dict(orient='tight'))    

def yaml_to_dataframe(constructor, node):
    return pd.DataFrame.from_dict(constructor.construct_mapping(node, deep=True), orient='tight')

yaml.representer.add_representer(pd.DataFrame, dataframe_to_yaml)
yaml.constructor.add_constructor(u'!pandas.DataFrame', yaml_to_dataframe)

class Project(QObject):
    parametersSet = QtCore.Signal(Parameter)
    parametersUpdated = QtCore.Signal()

    def __init__(self):
        super(Project, self).__init__()

        self._filename = None
        self._params = []
        self._points = None

        self.calibration = None

    @classmethod
    def from_file(cls, filename):
        proj = cls()
        proj.filename = filename

        with open(filename, 'r') as f:
            projdata = yaml.safe_load(f)
        
    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, fn):
        self._filename = fn

    @property
    def parameters(self):
        return self._params
    
    @property
    def camera_names(self):
        try:
            cn = [v.name() for v in self._params.child("Videos").children()]
        except (KeyError, AttributeError):
            cn = []
        
        return cn

    def set_videos(self, videos, cameranames=None):
        if cameranames is None:
            cameranames = []
            camletter = ascii_uppercase
            for i in range(len(videos)):
                cameranames.append(f"cam{camletter[i]}")
        
        cams = []
        for i, (fn1, camname1) in enumerate(zip(videos, cameranames)):
            cams1 = {'name': camname1, 'type': 'group', 'children': [
                {'name': 'File', 'type': 'str', 'value': fn1}
            ]}
            cams.append(cams1)

        p = [{'name': 'Videos', 'type': 'group', 'children': cams}]

        if len(videos) > 1:
            p.append({'name': 'Synchronization', 'type': 'group', 'children': [
                {'name': 'Method', 'type': 'list', 'limits': ['None', 'Timecode', 'Audio', 'Timecode+Audio'],
                    'value': 'Timecode'},
                {'name': 'Synchronize...', 'type': 'action'},
                ]})
                                                
            p.append({'name': 'Calibration', 'type': 'group', 'children': [
                {'name': 'Type', 'type': 'list', 'limits': ['Charuco', 'Checkboard'],
                    'value': 'Charuco'},
                {'name': 'Frame Step', 'type': 'int', 'value': 40,  
                    'tip': "Calibrate on every nth frame"},
                {'name': 'Number of squares horizontally', 'type': 'int', 'value': 6},
                {'name': 'Number of squares vertically', 'type': 'int', 'value': 6},
                {'name': 'Size of square', 'type': 'float', 'value':24.33, 'suffix': 'mm'},
                {'name': 'Size of marker', 'type': 'float', 'value':17, 'suffix': 'mm'},
                {'name': 'Marker bits', 'type': 'int', 'value':5, 'tip':'Information bits in the markers'},
                {'name': 'Number of markers', 'type': 'int', 'value':50, 'tip':'Number of markers in the dictionary'},
                {'name': 'Calibrate...', 'type': 'action'},
                {'name': 'Refine calibration...', 'type': 'action'}
                ]})
        
        self._params = Parameter.create(name='Parameters', type='group', children=p)
        self.parametersSet.emit(self._params)

    def add_videos(self, videos, cameranames=None):
        raise NotImplementedError("Can't add videos to the project yet")

    def add_video_info(self, cameraname, info):
        self._params.child('Videos', cameraname).addChildren(info)
        self.parametersUpdated.emit()

    def add_calibration(self, cal):
        self.calibration = cal

    def add_points(self, points: Points):
        self._points = points

    def save(self, overwrite=False):
        if not overwrite and os.path.exists(self._filename):
            logger.debug(f'File {self._filename} exists. Not overwriting')
            return
        
        def convert_parameters_to_list(params):
            d = []
            for p in params:
                if p.hasChildren():
                    val = convert_parameters_to_list(p.children())
                    d1 = {p.name(): val}
                elif p.hasValue():
                    if p.isType('str') or p.isType('float') or p.isType('int'): 
                        d1 = {p.name(): p.value()}
                    elif p.isType('list'):
                        d1 = {p.name(): {'value': p.value(), 'type': 'list', 'limits': p.opts['limits']}}
                    elif p.isType('file'):
                        d1 = {p.name(): {'value': p.value(), 'type': 'file'}}
                    else:
                        logger.debug(f'Unrecognized parameter type {p}')
                else:
                    continue
                d.append(d1)
            return d

        projdata = [{'parameters': convert_parameters_to_list(self._params)},
                    {'points': self._points.dataframe},
                    {'calibration': self.calibration.to_dict()}]

        with open(self._filename, mode='wt', encoding='utf-8') as file:
            yaml.dump(projdata, file)


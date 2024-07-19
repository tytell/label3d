import os, sys

import pandas as pd

from pyqtgraph.parametertree import Parameter, ParameterTree, parameterTypes
import pyqtgraph as pg
import numpy as np
from string import ascii_uppercase
from datetime import datetime
import tomlkit
from collections.abc import Iterable

import logging
logger = logging.getLogger('label3d')

from qtpy import QtCore
from qtpy.QtCore import (
    QObject,
    Slot,
)

from points import Points
from settings import VERSION

def dict_to_toml(d, tab):
    
    for k, v in d.items():
        if isinstance(v, dict):
            sub = tomlkit.table()
            dict_to_toml(sub, v)
            v = sub
        elif isinstance(v, list) and (len(v) == 1) and (v[0] is None):
            v = ["None"]
        elif isinstance(v, list):
            vlist = []
            multiline = False
            for v1 in v:
                if isinstance(v1, str):
                    pass
                elif isinstance(v1, Iterable):
                    v1 = list(v1)
                    multiline = True
                vlist.append(v1)

            v = tomlkit.array(vlist).multiline(multiline)

        elif v is None:
            v = "None"
        tab.add(k, v)

def dataframe_to_toml(df, tab):
    d = df.to_dict(orient='tight')
    ind = tomlkit.array(d['index']).multiline(True)
    tab.add("index", ind)

    cols = tomlkit.array(d['columns']).multiline(True)
    tab.add("columns", cols)

    dat = tomlkit.array(d['data']).multiline(True)
    tab.add("data", dat)

    if 'index_names' in d:
        tab.add("index_names", d['index_names'])
    if 'column_names' in d:
        tab.add("column_names", d['column_names'])

def parameters_to_toml(params, tab):
    for p in params:
        if p.hasChildren():
            sub = tomlkit.table()
            parameters_to_toml(p.children(), sub)

            tab.add(p.name(), sub)

        elif p.hasValue():
            if p.writable() and (p.isType('str') or p.isType('float') or p.isType('int')): 
                tab.add(p.name(), p.value())
            elif p.isType('list'):
                sub = tomlkit.inline_table()
                sub.add('value', p.value())
                sub.add('type', 'list')
                sub.add('limits', p.opts['limits'])

                tab.add(p.name(), sub)

            elif p.isType('file'):
                sub = tomlkit.inline_table()
                sub.add('value', p.value())
                sub.add('type', 'file')

                tab.add(p.name(), sub)
            else:
                sub = tomlkit.inline_table()
                sub.add('value', p.value())
                sub.add('type', p.type())
                sub.add('readonly', p.readonly())

                tab.add(p.name(), sub)
        else:
            continue

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

        # with open(filename, 'r') as f:
        #     projdata = yaml.safe_load(f)
        
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

        doc = tomlkit.document()
        doc.add(tomlkit.comment(tomlkit.string("Label3D project", multiline=True)))

        doc.add(tomlkit.nl())
        doc.add('save_date', datetime.now())
        
        try:
            doc.add(tomlkit.nl())
            doc_params = tomlkit.table(True)
            parameters_to_toml(self.parameters, doc_params)
            doc.add("Parameters", doc_params)

            if self.calibration is not None:
                doc_calibration = tomlkit.table(True)

                calib = self.calibration.to_dict()
                for i, calib1 in enumerate(calib):
                    sub = tomlkit.table()
                    dict_to_toml(calib1, sub)
                    doc_calibration.add(f"cam_{i+1}", sub)

                doc.add(tomlkit.nl())
                doc.add("Calibration", doc_calibration)

            if self._points is not None:
                pts = self._points.dataframe
                
                doc_points = tomlkit.table(True)
                dict_to_toml(pts.to_dict(orient="tight"), doc_points)

                doc.add(tomlkit.nl())
                doc.add("Points", doc_points)
        except Exception as err:
            logging.error(f"Caught exception {err}. Trying to save project anyway")
            doc.add(tomlkit.comment(f"Caught exception {err}. Saving incomplete project file"))
        
        with open(self._filename, mode='wt', encoding='utf-8') as file:
            tomlkit.dump(doc, file)


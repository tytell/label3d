import os, sys
import pandas as pd
import numpy as np

from qtpy.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
)

import logging
logger = logging.getLogger('label3d')

class Points:
    def __init__(self): 
        self.type = None
        self._points = None
        self.calibration = None

    @classmethod
    def from_calibration_rows(cls, rows, calibration):
        pts = cls()

        pts_all = []
        for camname, rows_cam in zip(calibration.cameranames, rows):
            col_ind = pd.MultiIndex.from_product([[camname], ['auto'], ['x', 'y']],
                                                names=['camera', 'type', 'axis'])
            
            pts_cam = []
            for r in rows_cam:
                ids = r['ids']
                xy = r['corners']

                setnum = [r['framenum'][0]] * len(ids)
                framenum = [r['framenum'][1]] * len(ids)

                row_ind = pd.MultiIndex.from_arrays([setnum, framenum, ids.flatten().tolist()],
                                                    names=['set', 'frame', 'id'])
                pts_fr = pd.DataFrame(index=row_ind, columns=col_ind, data=xy.squeeze())

                pts_cam.append(pts_fr)
            
            pts_all.append(pd.concat(pts_cam, axis=0))
        
        pts_all = pd.concat(pts_all, axis=1)

        pts._points = pts_all

        return(pts)
            
    @classmethod
    def from_csv(cls, csvname):
        pts = cls()

        pts_flat = pd.read_csv(csvname)
        col_tup = [a.split('_') for a in pts_flat.columns.names()]
        col_ind = pd.MultiIndex.from_tuples(col_tup)

        row_ind = pd.MultiIndex.from_frame(pts_flat[['set', 'frame', 'id']])

        pts._points = pd.DataFrame(index=row_ind, columns=col_ind, data=pts_flat.drop(['set', 'frame', 'id'], axis=1))

        return pts

    @property
    def dataframe(self):
        return self._points
    
    def to_csv(self, csvname):
        pts_flat = self.to_flat_dataframe()
        pts_flat.to_csv(csvname)
        
    def to_flat_dataframe(self):
        if self._points is None:
            col_ind = pd.Index(['set', 'frame', 'id'])
            return pd.DataFrame(columns=col_ind)
        
        pts_flat = self._points.reset_index()
        pts_flat.columns = ["_".join(a) for a in pts_flat.columns.to_flat_index()]

        return pts_flat

    def get_camera_points(self, camname, setnum=0):
        if self._points is None:
            return([])
        
        try:
            pts_fr = self._points.loc[(setnum, slice(None), slice(None)), (camname, slice(None), slice(None))]
            pts_fr = pts_fr.reset_index(col_level='axis')
            return pts_fr

        except KeyError:
            logger.debug(f"No points for {camname} in frame {frame} (set {setnum})")

        return []



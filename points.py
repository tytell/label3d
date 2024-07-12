import os, sys
import pandas as pd
import numpy as np

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
            


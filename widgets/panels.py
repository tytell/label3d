from typing import Callable, Dict, Iterable, List, Optional, Type, Union
import logging
import itertools

from qtpy import QtGui, QtCore
from qtpy.QtCore import (
    Qt,
    QAbstractTableModel,
    Slot
)
from qtpy.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QScrollBar,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph.parametertree import Parameter, ParameterTree, parameterTypes
import pyqtgraph as pg
import numpy as np

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class VideoControlPanel(QDockWidget):
    addedVideos = QtCore.Signal(list)
    syncVideos = QtCore.Signal()
    doCalibrate = QtCore.Signal()

    def __init__(self, main_window: QMainWindow):
        super().__init__("Video Control")
        self.name = "Video Control"
        self.main_window = main_window

        self.setObjectName(self.name + "Panel")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        dock_widget = QWidget()
        dock_widget.setObjectName(self.name + "Widget")

        self._create_widgets(dock_widget)

        self.setWidget(dock_widget)

        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self)
        self.main_window.viewMenu.addAction(self.toggleViewAction())

    def selectVideos(self):
        vids, filt = QFileDialog.getOpenFileNames(parent=self.main_window, 
                                            caption="Select videos",
                                         dir="", filter="Videos (*.mp4)")
        
        if vids is not None:
            cams = []
            for i, fn in enumerate(vids):
                cams1 = {'name': f"cam{i+1}", 'type': 'group', 'children': [
                    {'name': 'File', 'type': 'str', 'value': fn}
                ]}
                cams.append(cams1)

            p = [{'name': 'Videos', 'type': 'group', 'children': cams}]

            p.append({'name': 'Synchronization', 'type': 'group', 'children': [
                {'name': 'Method', 'type': 'list', 'limits': ['None', 'Timecode', 'Audio', 'Timecode+Audio'],
                 'value': 'Timecode'},
                {'name': 'Synchronize...', 'type': 'action'},
                ]})
                                                
            p.append({'name': 'Calibration', 'type': 'group', 'children': [
                {'name': 'Type', 'type': 'list', 'limits': ['Charuco', 'Checkboard'],
                 'value': 'Charuco'},
                {'name': 'Frame Step', 'type': 'int', 'value': 5,  
                 'tip': "Calibrate on every nth frame"},
                {'name': 'Number of squares horizontally', 'type': 'int', 'value': 6},
                {'name': 'Number of squares vertically', 'type': 'int', 'value': 6},
                {'name': 'Size of square', 'type': 'float', 'value':24.33, 'suffix': 'mm'},
                {'name': 'Size of marker', 'type': 'float', 'value':17, 'suffix': 'mm'},
                {'name': 'Marker bits', 'type': 'int', 'value':5, 'tip':'Information bits in the markers'},
                {'name': 'Number of markers', 'type': 'int', 'value':50, 'tip':'Number of markers in the dictionary'},
                {'name': 'Output file', 'type': 'file', 'value': '', 'acceptMode':'AcceptSave'},
                {'name': 'Calibrate...', 'type': 'action'}
                ]})
            
            self.cameraParams = Parameter.create(name='cameras', type='group', children=p)

            self.parameterTreeWidget.setParameters(self.cameraParams, showTop=False)
            self.cameraParams.child('Synchronization', 'Synchronize...').sigActivated.connect(self.syncVideos)
            
            self.cameraParams.child('Calibration', 'Calibrate...').sigActivated.connect(self.doCalibrate)

            self.addedVideos.emit(vids)

    def addVideoInfo(self, i, info):
        self.cameraParams.child('Videos').children()[i].addChildren(info)

    def get_camera_names(self):
        camnames = []
        videonames = []
        for p1 in self.cameraParams.child('Videos'):
            camnames.append(p1.name())
            videonames.append(p1['File'])

        return camnames, videonames
    
    @Slot(int, int)
    def show_calibration_progress(self, i,n):
        try:
            progress = self.cameraParams.child('Calibration', 'Progress')
        except KeyError:
            progress = parameterTypes.ProgressBarParameter(name="Progress")
            self.cameraParams.child('Calibration').addChild(progress)

            calibrate_button = self.cameraParams.child('Calibration', 'Calibrate...')
            calibrate_button.hide()
        
        if i < n:
            pct = int((i*100) / n)
            progress.setValue(pct)
        else:
            progress.remove()
            calibrate_button = self.cameraParams.child('Calibration', 'Calibrate...')
            calibrate_button.show()

    @Slot()
    def calibration_finished(self):
        try:
            progress = self.cameraParams.child('Calibration', 'Progress')
            progress.remove()
            calibrate_button = self.cameraParams.child('Calibration', 'Calibrate...')
            calibrate_button.show()
        except KeyError:
            pass

    def _create_widgets(self, parent):
        layout = QVBoxLayout()

        gp = QGroupBox("Videos")
        vbox = QVBoxLayout()

        h = QHBoxLayout()

        self.selectVideosButton = QPushButton("Select videos...", self)
        self.selectVideosButton.clicked.connect(self.selectVideos)
        h.addWidget(self.selectVideosButton)
        h.addStretch()

        vbox.addLayout(h)

        self.parameterTreeWidget = ParameterTree(parent)
        self.parameterTreeWidget.setObjectName("ParameterTree")
        self.cameraParams = Parameter.create(name='cameras', type='group', children=[])
        self.parameterTreeWidget.setParameters(self.cameraParams, showTop=False)

        vbox.addWidget(self.parameterTreeWidget)
        gp.setLayout(vbox)
        layout.addWidget(gp)

        gp = QGroupBox("Control")

        self.rewindButton = QPushButton("Rewind")
        self.playButton = QPushButton("Play")
        self.fastfwdButton = QPushButton("Fast forward")

        self.stepBackButton = QPushButton("Back")
        self.stepFwdButton = QPushButton("Forward")

        self.jumpBackButton = QPushButton("Jump back")
        self.jumpFramesSpin = QSpinBox()
        self.jumpFramesSpin.setMinimum(1)
        self.jumpFramesSpin.setValue(5)
        self.jumpFwdButton = QPushButton("Jump fwd")

        gridlayout = QGridLayout()
        gridlayout.addWidget(self.rewindButton, 0,0, Qt.AlignCenter | Qt.AlignVCenter)
        gridlayout.addWidget(self.playButton, 0,1, Qt.AlignCenter | Qt.AlignVCenter)
        gridlayout.addWidget(self.fastfwdButton, 0,2, Qt.AlignCenter | Qt.AlignVCenter)
        
        gridlayout.addWidget(self.stepBackButton, 1,0, Qt.AlignCenter | Qt.AlignVCenter)
        gridlayout.addWidget(self.stepFwdButton, 1,2, Qt.AlignCenter | Qt.AlignVCenter)
        
        gridlayout.addWidget(self.jumpBackButton, 2,0, Qt.AlignCenter | Qt.AlignVCenter)
        gridlayout.addWidget(self.jumpFramesSpin, 2,1, Qt.AlignCenter | Qt.AlignVCenter)
        gridlayout.addWidget(self.jumpFwdButton, 2,2, Qt.AlignCenter | Qt.AlignVCenter)

        gp.setLayout(gridlayout)
        layout.addWidget(gp)

        parent.setLayout(layout)

class VideoFramePanel(QDockWidget):
    set_frame = QtCore.Signal(int)

    def __init__(self, main_window: QMainWindow):
        super().__init__("Frame Control")
        self.name = "Frame Control"
        self.main_window = main_window

        self.setObjectName(self.name + "Panel")
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)

        dock_widget = QWidget()
        dock_widget.setObjectName(self.name + "Widget")

        self._create_widgets(dock_widget)
        self._set_slots()

        self.setWidget(dock_widget)

        self.main_window.addDockWidget(Qt.BottomDockWidgetArea, self)
        self.main_window.viewMenu.addAction(self.toggleViewAction())

    @Slot(int)
    def setNumFrames(self, nframes):
        self.nframes = nframes
        self.frameSlider.setMinimum(1)
        self.frameSlider.setMaximum(nframes)
        self.frameNumberBox.setMinimum(1)
        self.frameNumberBox.setMaximum(nframes)

    @Slot(list)
    def addAudio(self, vids):
        w = self.graphicsWidget
        w.clear()
        w.show()

        self._audio = []
        self._audio_t = []
        self._audio_plots = []

        for i, v1 in enumerate(vids):
            arate, a = v1.audio()
            self._audio.append(a)

            t1 = np.arange(0, len(a)) / arate
            self._audio_t.append(t1)

            p1 = w.addPlot(row=i, col=0)
            p1.plot(t1, a, pen=(i, len(vids)))
            p1.showAxis('left', False)
            p1.setMouseEnabled(x=True, y=False)
            self._audio_plots.append(p1)

        self.frame_dur = 1 / vids[0].fps

        tfr1 = 0
        tfr2 = tfr1 + self.frame_dur

        self._frame_rgns = []
        for p1 in self._audio_plots:
            fr1 = pg.LinearRegionItem([tfr1, tfr2], movable=True)
            fr1.setZValue(-10)

            p1.addItem(fr1)
            self._frame_rgns.append(fr1)
        
        # self._frame_rgns[0].sigRegionChanged.connect(self._frame_rgns[1].setRegion)
        
        for p1, p2 in zip(self._audio_plots[:-1], self._audio_plots[1:]):
            p1.setXLink(p2)
                    
    def _create_widgets(self, parent):
        self.layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameSlider.setTickPosition(QSlider.TicksBelow)
        self.frameSlider.setTickInterval(10)

        self.frameNumberBox = QSpinBox()
        fm = self.frameNumberBox.fontMetrics()
        # m = self.frameNumberBox.textMargins()
        # c = self.frameNumberBox.contentsMargins()
        w = 6*fm.width('9') #+m.left()+m.right()+c.left()+c.right()
        self.frameNumberBox.setFixedWidth(w)

        hlayout.addWidget(self.frameSlider)
        hlayout.addWidget(self.frameNumberBox)

        self.layout.addLayout(hlayout)

        self.graphicsWidget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphicsWidget)
        self.graphicsWidget.hide()

        parent.setLayout(self.layout)
    
    def _set_slots(self):
        self.frameSlider.valueChanged.connect(self.frameNumberBox.setValue)
        self.frameNumberBox.valueChanged.connect(self.frameSlider.setValue)

        self.frameSlider.valueChanged.connect(self.set_frame)

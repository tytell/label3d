from typing import Callable, Dict, Iterable, List, Optional, Type, Union
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
from string import ascii_uppercase

import logging
logger = logging.getLogger('label3d')

from project import Project

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class VideoControlPanel(QDockWidget):
    addedVideos = QtCore.Signal(list)
    syncVideos = QtCore.Signal()
    doCalibrate = QtCore.Signal()

    def __init__(self, main_window: QMainWindow, project: Project):
        super().__init__("Video Control")
        self.name = "Video Control"
        self.main_window = main_window
        self.project = project
        self.parameters = None

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
            self.addedVideos.emit(vids)

    @Slot(Parameter)
    def setParameters(self, params):
        self.parameterTreeWidget.setParameters(params, showTop=True)
        self.parameters = params

    @Slot()
    def updateParameters(self):
        logger.debug("Parameters should have updated...")

    @Slot(int, int)
    def show_calibration_progress(self, i,n):
        try:
            progress = self.parameters.child('Calibration', 'Progress')
        except KeyError:
            progress = parameterTypes.ProgressBarParameter(name="Progress")
            self.parameters.child('Calibration').addChild(progress)

            calibrate_button = self.parameters.child('Calibration', 'Calibrate...')
            calibrate_button.hide()
        
        if i < 0:
            try:
                logger.debug('Trying to replace parameter')
                new_progress = Parameter.create(name='Progress', type='str', value='Working...')
                logger.debug(f'New progress {new_progress}')

                progress.remove()

                self.parameters.child('Calibration').addChild(new_progress)
                logger.debug('Added new progress')

            except Exception as err:
                logger.debug(f'Error: {err}')

        elif i < n:
            pct = int((i*100) / n)
            progress.setValue(pct)

    @Slot(list)
    def calibration_finished(self, rows):
        try:
            logger.debug('VideoControlPanel.calibration_finished')

            progress = self.parameters.child('Calibration', 'Progress')
            progress.remove()
            calibrate_button = self.parameters.child('Calibration', 'Calibrate...')
            calibrate_button.show()
        except KeyError as err:
            logger.debug(f'KeyError! {err}')
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

        # self.parameters = Parameter.create(name='Root', type='group', children=[])
        # self.parameterTreeWidget.setParameters(self.parameters, showTop=False)

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

        self.frameSlider.valueChanged.connect(self._emit_frame_minus_one)

    @Slot(int)
    def _emit_frame_minus_one(self, fr: int):
        self.set_frame.emit(fr-1)

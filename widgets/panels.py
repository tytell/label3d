from typing import Callable, Dict, Iterable, List, Optional, Type, Union
import logging

from qtpy import QtGui, QtCore
from qtpy.QtCore import (
    Qt,
    QAbstractTableModel
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
    QScrollBar,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class VideoFilesModel(QAbstractTableModel):
    keynames = ["camera", "filename"]
    colnames = ["Camera Name", "File"]

    def __init__(self, camfiles: Optional[List[Dict]] = None):
        super().__init__()
        self._data = camfiles or []

    def rowCount(self, index):
        return len(self._data)
    
    def columnCount(self, parent=None):
        return 2
    
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole or role == Qt.EditRole:
                value = self._data[index.row()][self.keynames[index.column()]]
                return value
    
    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self._data[index.row()][self.keynames[index.column()]] = value
            return True
        return False
    
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.colnames[col]
    
    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled

class VideoControlPanel(QDockWidget):
    addedVideos = QtCore.Signal(list)

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
            camfiles = [{"camera": "cam{}".format(i+1),
                         "filename": fn} for i, fn in enumerate(vids)]
            
            self.videoFilesTable.reset()
            self.videoFilesModel = VideoFilesModel(camfiles)
            self.videoFilesTable.setModel(self.videoFilesModel)

            self.addedVideos.emit(vids)

    def _create_widgets(self, parent):
        layout = QVBoxLayout()

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

        layout.addLayout(gridlayout)

        hlayout = QHBoxLayout()
        self.frameScrollBar = QScrollBar(Qt.Horizontal)
        self.frameNumberEdit = QLineEdit()
        fm = self.frameNumberEdit.fontMetrics()
        m = self.frameNumberEdit.textMargins()
        c = self.frameNumberEdit.contentsMargins()
        w = 5*fm.width('9')+m.left()+m.right()+c.left()+c.right()
        self.frameNumberEdit.setFixedWidth(w)

        hlayout.addWidget(self.frameScrollBar)
        hlayout.addWidget(self.frameNumberEdit)

        layout.addLayout(hlayout)

        gp = QGroupBox("Videos")

        self.videoFilesModel = VideoFilesModel()
        self.videoFilesTable = QTableView(self)
        self.videoFilesTable.setModel(self.videoFilesModel)

        callayout = QVBoxLayout()
        callayout.addWidget(self.videoFilesTable)

        h = QHBoxLayout()
        h.addStretch()

        self.selectVideosButton = QPushButton("Select videos...", self)
        self.selectVideosButton.clicked.connect(self.selectVideos)
        h.addWidget(self.selectVideosButton)

        callayout.addLayout(h)
        gp.setLayout(callayout)

        layout.addWidget(gp)

        parent.setLayout(layout)

class CalibrationPanel(QWidget):
    def __init__(self, main_window: QWidget):
        super().__init__()
        self.name = "Calibration"
        self.main_window = main_window

        self.setObjectName(self.name + "Panel")

        self._create_widgets()

    def _create_widgets(self, parent):
        layout = QVBoxLayout()

        def makeLabeledWidget(widget: QWidget, name: str, label: Optional[str] = None):
            widget.setObjectName(name)

            if label is not None:
                hlay = QHBoxLayout()
                qlab = QLabel(label)
                qlab.setAlignment(Qt.AlignRight)
                qlab.setBuddy(widget)

                hlay.addWidget(qlab)
                hlay.addWidget(widget)

                return hlay
            else:
                return widget
        
        self.calibrationFileNameEdit = QLineEdit(parent)
        h = makeLabeledWidget(self.calibrationFileNameEdit, "calibrationFileNameEdit", "Calibration file:")
        self.calibrationFileBrowse = QPushButton("...", parent)
        h.addWidget(self.calibrationFileBrowse)

        self.loadCalibrationButton = QPushButton("Load calibration...", parent)
        h.addWidget(self.loadCalibrationButton)

        layout.addLayout(h)

        layout.addStretch()                
        parent.setLayout(layout)
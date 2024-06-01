import os, sys
import platform
import logging
from typing import Callable, List, Optional, Tuple

import qtpy
from qtpy import QtCore, QtGui
from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, 
    QDockWidget, QAction, QToolBar, QLabel, QStatusBar,
)
from qtpy.QtGui import (
    QKeySequence
)

from widgets.docks import ParameterDock
from widgets.video import VideoDock

# from sleap.io.video import Video
# from sleap.gui.widgets.video import QtVideoPlayer

parameterDefinitions = [
    {'name': 'Calibration', 'type': 'group', 'children': [
        {'name': 'Get calibration...', 'type': 'action'},
        {'name': 'Calibration file', 'type': 'str', 'readonly': True},
    ]},
    {'name':'thing', 'type':'int', 'value':12},
    {'name':'stuff', 'type':'int', 'value':13},
]

class MainWindow(QMainWindow):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Label 3D")

        self._create_actions()
        self._create_menus()

        self._create_toolbars()

        self._create_video_windows()
        self._create_dock_windows()

    def _create_actions(self):
        self.newProjectAction = QAction("&New Project", self)
        self.openProjectAction = QAction("&Open Project...", self)
        
        self.quitAction = QAction("&Quit", self)
        self.quitAction.triggered.connect(self.close)

        self.zoomAction = QAction("&Zoom", self)
        self.zoomAction.setCheckable(True)
        self.zoomAction.setShortcut(QKeySequence("z"))

    def _create_toolbars(self):
        vidtoolbar = QToolBar("Video")
        vidtoolbar.addAction(self.zoomAction)

        self.addToolBar(vidtoolbar)
        
    def _create_menus(self):

        ### File Menu ###

        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction(self.newProjectAction)
        fileMenu.addAction(self.openProjectAction)
        
        fileMenu.addSeparator()
        self.addAction(self.quitAction)

        ### View Menu ###

        viewMenu = self.menuBar().addMenu("View")
        self.viewMenu = viewMenu  # store as attribute so docks can add items

    def _create_video_windows(self):
        self.videodock = []
        self.videodock.append(VideoDock(name="test", main_window=self))

        self.activeVideo = 0
        for vid in self.videodock:
            self.zoomAction.toggled.connect(vid.view.set_zoom)
            vid.view.zoomModeChanged.connect(self.zoomAction.setChecked)

    def _create_dock_windows(self):
        self.parameterdock = ParameterDock("Parameters", self, parameterDefinitions)
        self.params = self.parameterdock.parameters

    def newProject(self):
        pass

    def openProject(self):
        pass

def create_app():
    """Creates Qt application."""

    app = QApplication([])
    app.setApplicationName(f"label3d")
    # app.setWindowIcon(QtGui.QIcon(sleap.util.get_package_file("gui/icon.png")))

    return app


def main(args: Optional[list] = None):
    """Starts new instance of app."""

    # parser = create_sleap_label_parser()
    # args = parser.parse_args(args)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.debug("Started!")
    logging.debug("qtpy API: {}".format(qtpy.API_NAME))
    logging.debug("Qt: v {}".format(QtCore._qt_version))

    if platform.system() == "Darwin":
        # TODO: Remove this workaround when we update to qtpy >= 5.15.
        # https://bugreports.qt.io/browse/QTBUG-87014
        # https://stackoverflow.com/q/64818879
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

    app = create_app()

    window = MainWindow()
    window.showMaximized()

    app.exec_()

    pass

if __name__ == "__main__":
    main()

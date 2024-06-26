import os, sys
import platform
import logging
from typing import Callable, List, Optional, Tuple
from functools import partial

import qtpy
from qtpy import QtCore, QtGui
from qtpy.QtCore import (
    QEvent, Qt,
    Slot
)
from qtpy.QtWidgets import (
    QApplication, QWidget, QMainWindow, QMdiArea, QMessageBox, 
    QDockWidget, QAction, QToolBar, QLabel, QStatusBar,
    QVBoxLayout
)
from qtpy.QtGui import (
    QKeySequence
)

from widgets.panels import VideoControlPanel, VideoFramePanel, CalibrationPanel
from widgets.videowindow import VideoWindow
from videofile import Video

from settings import SETTINGS_FILE

parameterDefinitions = [
    {'name': 'Calibration', 'type': 'group', 'children': [
        {'name': 'Get calibration...', 'type': 'action'},
        {'name': 'Calibration file', 'type': 'str', 'readonly': True},
    ]},
    {'name':'thing', 'type':'int', 'value':12},
    {'name':'stuff', 'type':'int', 'value':13},
]

class MainWindow(QMainWindow):
    _cameraParams = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Label 3D")
        self.setObjectName("Label3DMainWindow")

        self._mdi_area = QMdiArea()
        self._mdi_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._mdi_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(self._mdi_area)

        # self._mdi_area.subWindowActivated.connect(self.update_menus)

        self._create_actions()
        self._create_menus()

        self._create_toolbars()

        self.videowindows = []
        self._create_panels()

        self.readSettings()

    @property
    def cameraParams(self):
        return self.videoControlPanel.cameraParams
    
    def _create_actions(self):
        self._newProject_act = QAction("&New Project", self)
        self._openProject_act = QAction("&Open Project...", self)
        
        self._quit_act = QAction("&Quit", self, triggered=self.close)

        self._zoom_act = QAction("&Zoom", self, shortcut=QKeySequence('z'))
        self._zoom_act.setCheckable(True)

        self._close_act = QAction("Cl&ose", self,
                                  statusTip="Close the active window",
                                  triggered=self._mdi_area.closeActiveSubWindow)

        self._close_all_act = QAction("Close &All", self,
                                      statusTip="Close all the windows",
                                      triggered=self._mdi_area.closeAllSubWindows)

        self._tile_act = QAction("&Tile", self, statusTip="Tile the windows",
                                 triggered=self._mdi_area.tileSubWindows)

        self._cascade_act = QAction("&Cascade", self,
                                    statusTip="Cascade the windows",
                                    triggered=self._mdi_area.cascadeSubWindows)

        self._next_act = QAction("Ne&xt", self, shortcut=QKeySequence.NextChild,
                                 statusTip="Move the focus to the next window",
                                 triggered=self._mdi_area.activateNextSubWindow)

        self._previous_act = QAction("Pre&vious", self,
                                     shortcut=QKeySequence.PreviousChild,
                                     statusTip="Move the focus to the previous window",
                                     triggered=self._mdi_area.activatePreviousSubWindow)
        
        self._separator_act = QAction(self)
        self._separator_act.setSeparator(True)

    def _create_toolbars(self):
        vidtoolbar = QToolBar("Video")
        vidtoolbar.addAction(self._zoom_act)
        vidtoolbar.setObjectName("VideoToolBar")

        self.addToolBar(vidtoolbar)
        
    def _create_menus(self):

        ### File Menu ###

        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction(self._newProject_act)
        fileMenu.addAction(self._openProject_act)
        
        fileMenu.addSeparator()
        self.addAction(self._quit_act)

        ### View Menu ###

        viewMenu = self.menuBar().addMenu("View")
        self.viewMenu = viewMenu  # store as attribute so docks can add items

        ### Window Menu ###

        self._window_menu = self.menuBar().addMenu("&Window")
        self.update_window_menu()
        self._window_menu.aboutToShow.connect(self.update_window_menu)

    @Slot(list)
    def setVideos(self, videofiles):
        for vw in self.videowindows:
            vw.close()

        exists = [os.path.exists(f) for f in videofiles]
        if not all(exists):
            QMessageBox.critical(None, "File not found", "Could not find some videos")
            return

        self.videowindows = []
        self.videos = []
        nfr = []
        for i, f in enumerate(videofiles):
            vid = Video.from_media(f)
            nfr1 = vid.nframes
            nfr.append(nfr1)

            vw = VideoWindow(filename=f, video=vid, main_window=self)

            self._zoom_act.toggled.connect(vw.view.set_zoom)
            vw.view.zoomModeChanged.connect(self._zoom_act.setChecked)
            
            self.videoFramePanel.set_frame.connect(vw.set_frame)

            self.videowindows.append(vw)
            self._mdi_area.addSubWindow(vw)
            vw.show()

            logging.debug(f"{f}: nframes = {nfr1}")
            self.videoControlPanel.addVideoInfo(i, [{'name': 'Number of frames', 'type': 'int', 'value': nfr1, 'readonly': True}])
            
            info = vid.get_info_as_parameters()
            if len(info) > 0:
                self.videoControlPanel.addVideoInfo(i, info)
            
            self.videos.append(vid)

        maxframes = max(nfr)
        
        self.activeVideo = 0

        self.videoFramePanel.setNumFrames(maxframes)

        # handle audio
        isaudio = [vid.is_audio for vid in self.videos]
        if all(isaudio):
            self.videoFramePanel.addAudio(self.videos)
            
    def sync_videos(self):
        logging.debug('Syncing')
        if self.cameraParams['Synchronization', 'Method'] == 'Timecode':
            logging.debug('Syncing by timecode!')

    def _create_panels(self):
        # self.parameterdock = ParameterDock("Parameters", self, parameterDefinitions)
        # self.params = self.parameterdock.parameters
        # self.calibrationDock = CalibrationDock(self)
        self.videoControlPanel = VideoControlPanel(self)
        self.videoControlPanel.addedVideos.connect(self.setVideos)
        self.videoControlPanel.syncVideos.connect(self.sync_videos)

        self.videoFramePanel = VideoFramePanel(self)

    @Slot()
    def update_window_menu(self):
        self._window_menu.clear()
        self._window_menu.addAction(self._close_act)
        self._window_menu.addAction(self._close_all_act)
        self._window_menu.addSeparator()
        self._window_menu.addAction(self._tile_act)
        self._window_menu.addAction(self._cascade_act)
        self._window_menu.addSeparator()
        self._window_menu.addAction(self._next_act)
        self._window_menu.addAction(self._previous_act)
        self._window_menu.addAction(self._separator_act)

        windows = self._mdi_area.subWindowList()
        self._separator_act.setVisible(len(windows) != 0)

        for i, window in enumerate(windows):
            child = window.widget()

            f = child.name
            text = f'{i + 1} {f}'
            if i < 9:
                text = '&' + text

            action = self._window_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(child is self._active_mdi_child())
            slot_func = partial(self._set_active_sub_window, window=window)
            action.triggered.connect(slot_func)

    def newProject(self):
        pass

    def openProject(self):
        pass

    def readSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        settings.beginGroup("MainWindow")
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))
        settings.endGroup()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        logging.debug('Writing settings!')

        settings.beginGroup("MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.endGroup()

    def closeEvent(self, event):
        self._mdi_area.closeAllSubWindows()
        self.writeSettings()
        event.accept()

    def _active_mdi_child(self):
        active_sub_window = self._mdi_area.activeSubWindow()
        if active_sub_window:
            return active_sub_window.widget()
        return None
    
    def _set_active_sub_window(self, window):
        if window:
            self._mdi_area.setActiveSubWindow(window)

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
    window.show()
    # window.showMaximized()

    app.exec_()

    pass

if __name__ == "__main__":
    main()

import os, sys
import platform
from typing import Callable, List, Optional, Tuple
from functools import partial
import yaml

import qtpy
from qtpy import QtCore, QtGui
from qtpy.QtCore import (
    QEvent, Qt, QThread,
    Slot
)
from qtpy.QtWidgets import (
    QApplication, QWidget, QMainWindow, QMdiArea, QMessageBox, 
    QDockWidget, QAction, QToolBar, QLabel, QStatusBar,
    QVBoxLayout, QFileDialog
)
from qtpy.QtGui import (
    QKeySequence
)

import logging
logger = logging.getLogger('label3d')

from widgets.panels import VideoControlPanel, VideoFramePanel
from widgets.videowindow import VideoWindow
from videofile import Video
from triangulate import Calibration
from points import Points
from project import Project

from settings import SETTINGS_FILE, DEBUG_CALIBRATION

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

        self.project = Project()
        self.project.videosUpdated.connect(self.showVideos)

        self._create_actions()
        self._create_menus()

        self._create_toolbars()

        self.videowindows = []
        self._create_panels()

        self.readSettings()

    @property
    def parameters(self):
        return self.project.parameters
    
    def _create_actions(self):
        self._newProject_act = QAction("&New Project", self)
        self._openProject_act = QAction("&Open Project...", self,
                                        triggered=self.openProject)
        self._saveProject_act = QAction("&Save Project...", self,
                                        triggered=self.saveProject)
        self._saveProjectAs_act = QAction("Save Project &As...", self,
                                        triggered=self.saveProjectAs)

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

        self._param_act = QAction("Add parameter", self,
                                  triggered=self.add_parameter)

    def _create_toolbars(self):
        vidtoolbar = QToolBar("Video")
        vidtoolbar.addAction(self._zoom_act)
        vidtoolbar.addAction(self._param_act)
        vidtoolbar.setObjectName("VideoToolBar")

        self.addToolBar(vidtoolbar)
        
    def _create_menus(self):

        ### File Menu ###

        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction(self._newProject_act)
        fileMenu.addAction(self._openProject_act)
        fileMenu.addAction(self._saveProject_act)
        fileMenu.addAction(self._saveProjectAs_act)

        fileMenu.addSeparator()
        self.addAction(self._quit_act)

        ### View Menu ###

        viewMenu = self.menuBar().addMenu("View")
        self.viewMenu = viewMenu  # store as attribute so docks can add items

        ### Window Menu ###

        self._window_menu = self.menuBar().addMenu("&Window")
        self.update_window_menu()
        self._window_menu.aboutToShow.connect(self.update_window_menu)

    @Slot()
    def add_parameter(self):
        self.parameters.child("Videos").addChild({'name': 'thing', 'type': 'int', 'value': 40},
                                 autoIncrementName=True)

    @Slot()
    def showVideos(self):
        for vw in self.videowindows:
            vw.close()

        self.videowindows = []
        nfr = []
        for i, (vid, cn) in enumerate(zip(self.project.videos, self.project.camera_names)):
            nfr1 = vid.nframes
            nfr.append(nfr1)

            vw = VideoWindow(filename=vid.filename, camera_name=cn, video=vid, main_window=self, project=self.project)

            self._zoom_act.toggled.connect(vw.view.set_zoom)
            vw.view.zoomModeChanged.connect(self._zoom_act.setChecked)
            
            self.videoFramePanel.set_frame.connect(vw.set_frame)

            self.videowindows.append(vw)
            self._mdi_area.addSubWindow(vw)
            vw.show()

        maxframes = max(nfr)
        
        self.activeVideo = 0

        self.videoFramePanel.setNumFrames(maxframes)

        for camnm1, vw1 in zip(self.project.camera_names, self.videowindows):
            vw1.set_camera_name(camnm1)

        # handle audio
        isaudio = [vid.is_audio for vid in self.project.videos]
        if all(isaudio):
            self.videoFramePanel.addAudio(self.videos)

    @Slot(int, str, int, int)
    def selectPoint(self, setnum, camname, frame, id):
        for camnm1, vw1 in zip(self.project.camera_names, self.videowindows):
            if camnm1 != camname:
                
    def setParameterCallbacks(self):
        try:
            self.project.parameters.child('Calibration', 'Calibrate...').sigActivated.connect(self.do_calibrate)
            self.project.parameters.child('Synchronization', 'Synchronize...').sigActivated.connect(self.sync_videos)
        except KeyError as err:
            logging.debug(f"Error connection parameter slots: {err}")

    def sync_videos(self):
        logger.debug('Syncing')
        if self.parameters['Synchronization', 'Method'] == 'Timecode':
            logger.debug('Syncing by timecode!')

    def do_calibrate(self):
        logger.debug('MainWindow.do_calibrate')

        camnames = self.project.camera_names
        sz = self.videos[0].frame_size
        logger.debug(f"{sz=}")
        
        self.calibration = Calibration.from_parameters(cameranames=camnames, videos=self.videos, 
                                            params=self.parameters.child('Calibration'))
        self.project.add_calibration(self.calibration)

        if DEBUG_CALIBRATION:
            self._calibration_worker = self.calibration
            
            self._calibration_worker.progress.connect(self.videoControlPanel.show_calibration_progress)
            self._calibration_worker.finished.connect(self.videoControlPanel.calibration_finished)
            self._calibration_worker.finished.connect(self.finish_calibration)

            # run the calibration in the main thread so that we can debug more easily
            self.calibration.run()

        else:
            self._calibration_thread = QThread()
            self._calibration_worker = self.calibration
            self._calibration_worker.moveToThread(self._calibration_thread)

            self._calibration_thread.started.connect(self._calibration_worker.run)
            self._calibration_worker.finished.connect(self._calibration_thread.quit)
            self._calibration_worker.finished.connect(self._calibration_worker.deleteLater)
            self._calibration_worker.finished.connect(self._calibration_thread.deleteLater)
            
            self._calibration_worker.progress.connect(self.videoControlPanel.show_calibration_progress)
            self._calibration_worker.finished.connect(self.videoControlPanel.calibration_finished)
            self._calibration_worker.finished.connect(self.finish_calibration)

            self._calibration_thread.start()

    @Slot(list)
    def finish_calibration(self, rows):
        logger.debug('finish_calibration')
        self.project.add_points(Points.from_calibration_rows(rows, self.calibration))

        # for vw1 in self.videowindows:
        #     vw1.set_points(self.points)
        
    def _create_panels(self):
        # self.parameterdock = ParameterDock("Parameters", self, parameterDefinitions)
        # self.params = self.parameterdock.parameters
        # self.calibrationDock = CalibrationDock(self)
        self.videoControlPanel = VideoControlPanel(self, self.project)
        self.videoControlPanel.addedVideos.connect(self.project.set_videos)
        self.videoControlPanel.syncVideos.connect(self.sync_videos)
        self.videoControlPanel.doCalibrate.connect(self.do_calibrate)

        self.project.parametersSet.connect(self.videoControlPanel.setParameters)
        self.project.parametersUpdated.connect(self.videoControlPanel.updateParameters)

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
        filename, ok = QFileDialog.getOpenFileName(self, "Project file", filter="TOML files (*.toml)")
        if ok:
            self.project.load(filename)
            self.setParameterCallbacks()

    def saveProject(self):
        if self.project.filename is None:
            self.saveProjectAs()
        else:        
            self.project.save(overwrite=True)

    def saveProjectAs(self):
        filename, ok = QFileDialog.getSaveFileName(self, "Project file", filter="TOML files (*.toml)")
        if not ok:
            return
        self.project.filename = filename
        
        self.project.save(overwrite=True)

    def readSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        settings.beginGroup("MainWindow")
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))
        settings.endGroup()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        logger.debug('Writing settings!')

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

    logging.basicConfig(stream=sys.stdout)
    logger.setLevel(level=logging.DEBUG)
    logger.debug("Started!")
    logger.debug("qtpy API: {}".format(qtpy.API_NAME))
    logger.debug("Qt: v {}".format(QtCore._qt_version))

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

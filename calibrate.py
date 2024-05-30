import os, sys
import platform
import logging
from typing import Callable, List, Optional, Tuple

from qtpy import QtCore, QtGui
from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QApplication, QMainWindow, QMessageBox

class MainWindow(QMainWindow):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the app.

        Args:
            labels_path: Path to saved :class:`Labels` dataset.
            reset: If `True`, reset preferences to default (including window state).
            no_usage_data: If `True`, launch GUI without sharing usage data regardless
                of stored preferences.
        """
        super(MainWindow, self).__init__(*args, **kwargs)

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

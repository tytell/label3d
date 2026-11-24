"""Run video discovery and staged registration without blocking the GUI."""

import logging

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class AddVideosWorker(QThread):
    progress = Signal(int, int, str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = ""

    def run(self):
        try:
            self.result = self.operation(self.progress.emit)
        except Exception as exc:
            logger.exception("Adding SLEAP videos failed")
            self.error = str(exc)

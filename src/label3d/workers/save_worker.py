"""SaveWorker: background thread for SLP file write-back."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from label3d.project import Project


class SaveWorker(QThread):
    """Write current labels back to SLP files in a background thread."""

    finished = Signal(int)  # number of files written
    error = Signal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

    def run(self):
        try:
            n = self._project.sync_slp_files()
            self.finished.emit(n)
        except Exception as exc:
            self.error.emit(str(exc))

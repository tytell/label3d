"""ImportWorker: background thread for SLP import."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from label3d.project import Project


class ImportWorker(QThread):
    """Run Project.import_from_slp() in a background thread."""

    finished = Signal(int)   # emits row count on success
    error = Signal(str)      # emits error message on failure

    def __init__(self, project: Project, experiment: str | None = None, parent=None):
        super().__init__(parent)
        self._project = project
        self._experiment = experiment

    def run(self):
        try:
            count = self._project.import_from_slp(self._experiment)
            self.finished.emit(count)
        except Exception as exc:
            self.error.emit(str(exc))

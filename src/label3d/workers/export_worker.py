"""Background worker for File > Export points... (Phase 13.3)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from label3d.export import (
    ExportOptions,
    export_points,
    pending_slp_changes,
    slp_mapping_complete,
)
from label3d.project import Project

logger = logging.getLogger(__name__)


class ExportWorker(QThread):
    """Check the SLP files for new points, then export the CSV and write the SLP files.

    If the SLP files hold points the store doesn't have, nothing is written
    and `pending` is emitted instead of `done`.
    """

    pending = Signal(object)  # dict[str, MergeResult]
    done = Signal(object)     # ExportResult
    error = Signal(str)

    def __init__(self, project: Project, options: ExportOptions, parent=None):
        super().__init__(parent)
        self._project = project
        self._options = options

    def run(self):
        try:
            if self._options.write_slp and slp_mapping_complete(self._project):
                pending = pending_slp_changes(self._project)
                if pending:
                    self.pending.emit(pending)
                    return
            self.done.emit(export_points(self._project, self._options))
        except Exception as exc:
            logger.exception("Export failed")
            self.error.emit(str(exc))

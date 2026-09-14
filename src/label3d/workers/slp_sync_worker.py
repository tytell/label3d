"""Background workers for reading and writing SLP files during setup and sync.

Only file I/O and merge computation run in the background.  Applying merge
results to the PointsStore happens on the main thread.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from label3d.project import Project
from label3d.slp_sync import SlpMapping

logger = logging.getLogger(__name__)


class SlpReadWorker(QThread):
    """Read mapped SLP files and compute merges for every experiment."""

    done = Signal(object)  # dict[str, MergeResult]
    error = Signal(str)

    def __init__(self, project: Project, mapping: SlpMapping | None = None, parent=None):
        super().__init__(parent)
        self._project = project
        self._mapping = mapping

    def run(self):
        try:
            self.done.emit(self._project.read_slp_for_sync(self._mapping))
        except Exception as exc:
            logger.exception("SLP read failed")
            self.error.emit(str(exc))


class SlpWriteWorker(QThread):
    """Write all points to the mapped SLP files, optionally backing up and cleaning slp/."""

    done = Signal(int, str)  # files written, backup folder ("" if none)
    error = Signal(str)

    def __init__(
        self,
        project: Project,
        mapping: SlpMapping | None = None,
        backup: bool = False,
        remove_unmapped: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._project = project
        self._mapping = mapping
        self._backup = backup
        self._remove_unmapped = remove_unmapped

    def run(self):
        try:
            backup_dir = self._project.backup_slp_files() if self._backup else None
            n = self._project.write_slp_files(self._mapping)
            if self._remove_unmapped:
                mapping = self._mapping if self._mapping is not None else self._project.slp_mapping
                self._project.remove_unmapped_slp_files(mapping)
            self.done.emit(n, str(backup_dir) if backup_dir else "")
        except Exception as exc:
            logger.exception("SLP write failed")
            self.error.emit(str(exc))

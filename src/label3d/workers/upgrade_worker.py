"""Background worker that upgrades a project made by an older label3d (Phase 20)."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from label3d.upgrade import run_upgrade

logger = logging.getLogger(__name__)


class UpgradeWorker(QThread):
    """Run upgrade.run_upgrade; the project it loads lives and ends in this thread."""

    done = Signal(object)  # UpgradeResult
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, root: Path, sources: dict[Path, Path | None], parent=None):
        super().__init__(parent)
        self._root = root
        self._sources = sources

    def run(self):
        try:
            self.done.emit(run_upgrade(self._root, self._sources, progress=self.progress.emit))
        except Exception as exc:
            logger.exception("Upgrade failed")
            self.error.emit(str(exc))

"""ImportWorker: background thread for SLP file loading.

Only the I/O-heavy SleapAdapter.load() call runs in the background.
All PointsStore creation and mutation happens in the main thread via the
`loaded` signal, ensuring QTimers are always created and stopped on the
correct thread.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QThread, Signal

from label3d.project import Project


class ImportWorker(QThread):
    """Load SLP files in a background thread; emit the raw adapter DataFrame."""

    loaded = Signal(object)  # pd.DataFrame — received and merged in main thread
    error = Signal(str)

    def __init__(self, project: Project, experiment: str | None = None, parent=None):
        super().__init__(parent)
        self._project = project
        self._experiment = experiment

    def run(self):
        try:
            df = self._project.load_slp_adapter_df(self._experiment)
            self.loaded.emit(df)
        except Exception as exc:
            self.error.emit(str(exc))

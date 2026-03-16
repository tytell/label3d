"""Grid layout of CameraPanels — one per camera."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QGridLayout, QWidget

from label3d.video import FrameProvider
from label3d.widgets.camera_panel import CameraPanel


class CameraGrid(QWidget):
    """Arranges CameraPanels in a grid and updates them with video frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setSpacing(2)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._panels: dict[str, CameraPanel] = {}
        self._frame_provider: FrameProvider | None = None

    def setup_cameras(self, camera_names: list[str]):
        """Create camera panels for the given camera names."""
        # Clear existing
        for panel in self._panels.values():
            self._layout.removeWidget(panel)
            panel.deleteLater()
        self._panels.clear()

        n = len(camera_names)
        if n == 0:
            return

        # Compute grid: try to be as square as possible
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        for i, cam_name in enumerate(camera_names):
            panel = CameraPanel(cam_name)
            row, col = divmod(i, cols)
            self._layout.addWidget(panel, row, col)
            self._panels[cam_name] = panel

    def set_frame_provider(self, provider: FrameProvider):
        self._frame_provider = provider

    def update_frame(self, global_frame: int):
        """Fetch and display the given global frame for all cameras."""
        if self._frame_provider is None:
            return
        for cam_name, panel in self._panels.items():
            frame = self._frame_provider.get_frame(cam_name, global_frame)
            if frame is not None:
                panel.set_frame(frame)

    def fit_all(self):
        """Reset zoom on all panels."""
        for panel in self._panels.values():
            panel.fit_view()

    def get_panel(self, camera: str) -> CameraPanel | None:
        return self._panels.get(camera)

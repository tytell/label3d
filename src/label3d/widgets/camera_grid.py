"""Grid of CameraPanels using nested QSplitters for draggable dividers."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from label3d.video import FrameProvider
from label3d.widgets.camera_panel import CameraPanel


class CameraGrid(QWidget):
    """Arranges CameraPanels in a splitter grid with draggable dividers.

    Layout is a vertical QSplitter of horizontal QSplitters (one per row).
    User-adjusted splitter sizes are preserved across view-mode toggles.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._v_splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(self._v_splitter)

        self._panels: dict[str, CameraPanel] = {}
        self._row_splitters: list[QSplitter] = []
        self._frame_provider: FrameProvider | None = None
        self._solo_camera: str | None = None
        self._cols: int = 0

        # Saved splitter state for restore after single-camera toggle
        self._saved_row_sizes: list[list[int]] | None = None
        self._saved_v_sizes: list[int] | None = None

    def setup_cameras(self, camera_names: list[str]):
        """Create camera panels arranged in a splitter grid."""
        # Tear down existing
        for panel in self._panels.values():
            panel.setParent(None)
            panel.deleteLater()
        self._panels.clear()
        for sp in self._row_splitters:
            sp.setParent(None)
            sp.deleteLater()
        self._row_splitters.clear()

        n = len(camera_names)
        if n == 0:
            return

        # Compute grid dimensions
        self._cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / self._cols)

        for r in range(rows):
            row_splitter = QSplitter(Qt.Orientation.Horizontal)
            start = r * self._cols
            end = min(start + self._cols, n)
            for i in range(start, end):
                cam_name = camera_names[i]
                panel = CameraPanel(cam_name)
                row_splitter.addWidget(panel)
                self._panels[cam_name] = panel
            self._row_splitters.append(row_splitter)
            self._v_splitter.addWidget(row_splitter)

        # Start with equal sizes
        self._set_equal_sizes()

    @property
    def camera_names(self) -> list[str]:
        return list(self._panels.keys())

    def set_frame_provider(self, provider: FrameProvider):
        self._frame_provider = provider

    def show_single_camera(self, camera: str | None):
        """Show only one camera, or all if camera is None.

        Saves panel view states and splitter sizes before hiding,
        restores them when returning to all-camera view.
        """
        # Save view state for all currently visible panels
        for panel in self._panels.values():
            if panel.isVisible():
                panel.save_view_state()

        if camera is not None:
            # Switching to single-camera mode — save splitter sizes
            self._save_splitter_sizes()
            self._solo_camera = camera
            for cam_name, panel in self._panels.items():
                panel.setVisible(cam_name == camera)
            for sp in self._row_splitters:
                sp.setVisible(any(
                    self._panels[cam].isVisible()
                    for cam in self._panels
                    if self._panels[cam].parent() is sp
                ))
            # Defer view-state restore until after layout recalculation
            QTimer.singleShot(0, self._restore_visible_panels)
        else:
            # Returning to all-camera view — the previously solo panel
            # changes size drastically, so discard its saved state and
            # re-fit instead.
            prev_solo = self._solo_camera
            self._solo_camera = None
            if prev_solo is not None:
                panel = self._panels.get(prev_solo)
                if panel is not None:
                    panel._saved_scene_rect = None
            for sp in self._row_splitters:
                sp.setVisible(True)
            for panel in self._panels.values():
                panel.setVisible(True)
            # Restore splitter sizes, then view states
            QTimer.singleShot(0, self._restore_after_show_all)

    def _save_splitter_sizes(self):
        """Save current splitter proportions."""
        self._saved_v_sizes = self._v_splitter.sizes()
        self._saved_row_sizes = [sp.sizes() for sp in self._row_splitters]

    def _restore_splitter_sizes(self):
        """Restore saved splitter proportions, or reset to equal."""
        if self._saved_v_sizes is not None and self._saved_row_sizes is not None:
            self._v_splitter.setSizes(self._saved_v_sizes)
            for sp, sizes in zip(self._row_splitters, self._saved_row_sizes):
                sp.setSizes(sizes)
            self._saved_v_sizes = None
            self._saved_row_sizes = None
        else:
            self._set_equal_sizes()

    def _set_equal_sizes(self):
        """Set all splitters to equal proportions."""
        n_rows = len(self._row_splitters)
        if n_rows > 0:
            total_h = self._v_splitter.height() or 800
            self._v_splitter.setSizes([total_h // n_rows] * n_rows)
        for sp in self._row_splitters:
            n_cols = sp.count()
            if n_cols > 0:
                total_w = sp.width() or 1200
                sp.setSizes([total_w // n_cols] * n_cols)

    def _restore_after_show_all(self):
        """Restore splitter sizes then panel view states."""
        self._restore_splitter_sizes()
        self._restore_visible_panels()

    def _restore_visible_panels(self):
        for panel in self._panels.values():
            if panel.isVisible():
                panel.restore_view_state()

    def update_frame(self, global_frame: int):
        """Fetch and display the given global frame for all cameras."""
        if self._frame_provider is None:
            return
        for cam_name, panel in self._panels.items():
            if not panel.isVisible():
                continue
            frame = self._frame_provider.get_frame(cam_name, global_frame)
            if frame is not None:
                panel.set_frame(frame)

    def fit_all(self):
        """Reset zoom on all panels."""
        for panel in self._panels.values():
            panel.fit_view()

    def get_panel(self, camera: str) -> CameraPanel | None:
        return self._panels.get(camera)

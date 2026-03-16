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
        self._last_global_frame: int = 0

        # Saved splitter state for restore after single-camera toggle
        self._saved_row_sizes: list[list[int]] | None = None
        self._saved_v_sizes: list[int] | None = None

        # Generation counter: stale deferred callbacks are skipped
        self._deferred_gen: int = 0

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

    # -- View mode switching -------------------------------------------------

    def show_single_camera(self, camera: str | None):
        """Show only one camera, or all if camera is None.

        On the first transition from all→solo: saves every panel's
        view state and the splitter sizes.  Switching between solo
        cameras does NOT re-save — the grid-mode snapshot is kept
        until we return to all-camera view.

        Frame update + fit are handled in a single deferred callback
        after Qt has recalculated panel sizes.  A generation counter
        ensures only the latest request takes effect when clicks
        arrive faster than the event loop processes them.
        """
        was_solo = self._solo_camera is not None

        if camera is not None:
            if not was_solo:
                # All → solo: snapshot the grid-mode state
                for panel in self._panels.values():
                    panel.save_view_state()
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
            self._deferred(self._on_solo_settled)
        else:
            # Solo → all: restore the grid-mode snapshot
            self._solo_camera = None
            for sp in self._row_splitters:
                sp.setVisible(True)
            for panel in self._panels.values():
                panel.setVisible(True)
            self._deferred(self._on_grid_settled)

    def _deferred(self, callback):
        """Post a callback to run after the current event loop iteration.

        Bumps the generation counter so earlier pending callbacks are
        skipped when they fire.
        """
        self._deferred_gen += 1
        gen = self._deferred_gen
        QTimer.singleShot(0, lambda: self._run_if_current(gen, callback))

    def _run_if_current(self, gen: int, callback):
        if gen == self._deferred_gen:
            callback()

    def _on_solo_settled(self):
        """After layout settles for a solo camera: fit view then draw frame."""
        for panel in self._panels.values():
            if panel.isVisible():
                panel.fit_view()
        self._update_visible_frames()

    def _on_grid_settled(self):
        """After layout settles for all-cameras: restore splits, views, frames."""
        self._restore_splitter_sizes()
        for panel in self._panels.values():
            if panel.isVisible():
                panel.restore_view_state()
        self._update_visible_frames()

    def _update_visible_frames(self):
        """Decode and display the current frame for all visible panels."""
        if self._frame_provider is None:
            return
        for cam_name, panel in self._panels.items():
            if not panel.isVisible():
                continue
            frame = self._frame_provider.get_frame(
                cam_name, self._last_global_frame
            )
            if frame is not None:
                panel.set_frame(frame)

    # -- Frame display -------------------------------------------------------

    def update_frame(self, global_frame: int):
        """Fetch and display the given global frame for all visible cameras."""
        self._last_global_frame = global_frame
        if self._frame_provider is None:
            return
        for cam_name, panel in self._panels.items():
            if not panel.isVisible():
                continue
            frame = self._frame_provider.get_frame(cam_name, global_frame)
            if frame is not None:
                panel.set_frame(frame)

    # -- Splitter state ------------------------------------------------------

    def _save_splitter_sizes(self):
        self._saved_v_sizes = self._v_splitter.sizes()
        self._saved_row_sizes = [sp.sizes() for sp in self._row_splitters]

    def _restore_splitter_sizes(self):
        if self._saved_v_sizes is not None and self._saved_row_sizes is not None:
            self._v_splitter.setSizes(self._saved_v_sizes)
            for sp, sizes in zip(self._row_splitters, self._saved_row_sizes):
                sp.setSizes(sizes)
            self._saved_v_sizes = None
            self._saved_row_sizes = None
        else:
            self._set_equal_sizes()

    def _set_equal_sizes(self):
        n_rows = len(self._row_splitters)
        if n_rows > 0:
            total_h = self._v_splitter.height() or 800
            self._v_splitter.setSizes([total_h // n_rows] * n_rows)
        for sp in self._row_splitters:
            n_cols = sp.count()
            if n_cols > 0:
                total_w = sp.width() or 1200
                sp.setSizes([total_w // n_cols] * n_cols)

    # -- Utility -------------------------------------------------------------

    def fit_all(self):
        for panel in self._panels.values():
            panel.fit_view()

    def get_panel(self, camera: str) -> CameraPanel | None:
        return self._panels.get(camera)

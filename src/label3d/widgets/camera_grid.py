"""Grid of CameraPanels using nested QSplitters for draggable dividers."""

from __future__ import annotations

import enum
import math

from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QApplication, QSplitter, QVBoxLayout, QWidget

from label3d.video import FrameProvider
from label3d.widgets.camera_panel import CameraPanel


class LayoutMode(enum.Enum):
    GRID = "grid"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class CameraGrid(QWidget):
    """Arranges CameraPanels in a splitter grid with draggable dividers.

    Supports three layout modes: grid (rows × cols), horizontal (single row),
    and vertical (single column).  User-adjusted splitter sizes are preserved
    across view-mode toggles.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._v_splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(self._v_splitter)

        self._panels: dict[str, CameraPanel] = {}
        self._camera_order: list[str] = []
        self._row_splitters: list[QSplitter] = []
        self._frame_provider: FrameProvider | None = None
        self._solo_camera: str | None = None
        self._layout_mode: LayoutMode = LayoutMode.GRID
        self._last_global_frame: int = 0

        # Saved splitter state for restore after single-camera toggle
        self._saved_row_sizes: list[list[int]] | None = None
        self._saved_v_sizes: list[int] | None = None

    @property
    def layout_mode(self) -> LayoutMode:
        return self._layout_mode

    def set_layout_mode(self, mode: LayoutMode):
        """Rearrange panels into the given layout mode."""
        if mode == self._layout_mode and self._panels:
            return
        self._layout_mode = mode
        # Splitter structure changes, so saved sizes are no longer valid
        self._saved_row_sizes = None
        self._saved_v_sizes = None
        if self._camera_order:
            self._rebuild_splitters()
            self._set_equal_sizes()
            QApplication.processEvents()
            for panel in self._panels.values():
                if panel.isVisible():
                    panel.fit_view()
            self._update_visible_frames()

    def setup_cameras(self, camera_names: list[str]):
        """Create camera panels arranged according to the current layout mode."""
        # Tear down existing
        for panel in self._panels.values():
            panel.setParent(None)
            panel.deleteLater()
        self._panels.clear()
        self._clear_splitters()

        self._camera_order = list(camera_names)
        n = len(camera_names)
        if n == 0:
            return

        # Create panels
        for cam_name in camera_names:
            self._panels[cam_name] = CameraPanel(cam_name)

        self._rebuild_splitters()
        self._set_equal_sizes()

    def _clear_splitters(self):
        for sp in self._row_splitters:
            sp.setParent(None)
            sp.deleteLater()
        self._row_splitters.clear()

    def _rebuild_splitters(self):
        """Reparent existing panels into splitters for the current layout mode."""
        # Detach panels from old splitters (without deleting panels)
        for panel in self._panels.values():
            panel.setParent(None)
        self._clear_splitters()

        names = self._camera_order
        n = len(names)
        if n == 0:
            return

        if self._layout_mode == LayoutMode.HORIZONTAL:
            row_splitter = QSplitter(Qt.Orientation.Horizontal)
            for cam_name in names:
                row_splitter.addWidget(self._panels[cam_name])
            self._row_splitters.append(row_splitter)
            self._v_splitter.addWidget(row_splitter)
        elif self._layout_mode == LayoutMode.VERTICAL:
            for cam_name in names:
                row_splitter = QSplitter(Qt.Orientation.Horizontal)
                row_splitter.addWidget(self._panels[cam_name])
                self._row_splitters.append(row_splitter)
                self._v_splitter.addWidget(row_splitter)
        else:
            # Grid
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            for r in range(rows):
                row_splitter = QSplitter(Qt.Orientation.Horizontal)
                start = r * cols
                end = min(start + cols, n)
                for i in range(start, end):
                    row_splitter.addWidget(self._panels[names[i]])
                self._row_splitters.append(row_splitter)
                self._v_splitter.addWidget(row_splitter)

    @property
    def camera_names(self) -> list[str]:
        return list(self._camera_order)

    def set_frame_provider(self, provider: FrameProvider):
        self._frame_provider = provider

    # -- View mode switching -------------------------------------------------

    def show_single_camera(self, camera: str | None):
        """Show only one camera, or all if camera is None.

        On the first transition from all→solo: saves every panel's
        view state and the splitter sizes.  Switching between solo
        cameras does NOT re-save — the grid-mode snapshot is kept
        until we return to all-camera view.

        After changing visibility, processEvents() flushes the layout
        so that fit/restore and frame updates operate on correct sizes.
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

            # Flush layout so the visible panel has its final size
            QApplication.processEvents()

            # Fit view and draw frame
            for panel in self._panels.values():
                if panel.isVisible():
                    panel.fit_view()
            self._update_visible_frames()
        else:
            # Solo → all: restore the grid-mode snapshot
            self._solo_camera = None
            for sp in self._row_splitters:
                sp.setVisible(True)
            for panel in self._panels.values():
                panel.setVisible(True)

            # Flush layout so splitter sizes can be restored correctly
            QApplication.processEvents()

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

    def _zoom_targets(self) -> list[CameraPanel]:
        """Panels the zoom actions apply to: the solo camera, else all panels."""
        if self._solo_camera is not None:
            panel = self._panels.get(self._solo_camera)
            return [panel] if panel is not None else []
        return list(self._panels.values())

    def fit_targets(self):
        """Fit the whole image in the target panels."""
        for panel in self._zoom_targets():
            panel.fit_view()

    def zoom_to_instance(self, instance: int):
        """Zoom the target panels to the visible points of one instance."""
        for panel in self._zoom_targets():
            panel.zoom_to_points(panel.skeleton_overlay.visible_points(instance))

    def zoom_to_point(self, instance: int, bodypart: str):
        """Zoom the target panels to a small box around one point."""
        for panel in self._zoom_targets():
            item = panel.skeleton_overlay.get_point_item(instance, bodypart)
            if item is not None and item.isVisible():
                panel.zoom_to_point(item.pos())

    def get_panel(self, camera: str) -> CameraPanel | None:
        return self._panels.get(camera)

    def get_grid_splitter_sizes(self) -> tuple[list[int], list[list[int]]] | None:
        """Return all-cameras splitter sizes.

        In solo mode the live splitter reflects only the visible row, so we
        return the snapshot taken when solo mode was entered instead.
        Returns None if no meaningful sizes are available yet.
        """
        if self._solo_camera is not None:
            if self._saved_v_sizes and self._saved_row_sizes:
                return list(self._saved_v_sizes), [list(s) for s in self._saved_row_sizes]
            return None
        v = list(self._v_splitter.sizes())
        rows = [list(sp.sizes()) for sp in self._row_splitters]
        return v, rows

    def set_grid_splitter_sizes(self, v_sizes: list[int], row_sizes: list[list[int]]) -> None:
        """Restore splitter sizes. Must be called while all cameras are visible."""
        if len(v_sizes) == self._v_splitter.count():
            self._v_splitter.setSizes(v_sizes)
        for sp, sizes in zip(self._row_splitters, row_sizes):
            if len(sizes) == sp.count():
                sp.setSizes(sizes)

    def get_all_scene_rects(self) -> dict[str, QRectF]:
        """Return per-camera visible scene rects (excludes cameras with no saved rect)."""
        return {
            cam: r
            for cam, panel in self._panels.items()
            if (r := panel.get_scene_rect()) is not None
        }

    def set_all_scene_rects(self, rects: dict[str, QRectF]) -> None:
        """Restore per-camera scene rects. Hidden panels store rect for later apply."""
        for cam, rect in rects.items():
            if (panel := self._panels.get(cam)) is not None:
                panel.set_scene_rect(rect)

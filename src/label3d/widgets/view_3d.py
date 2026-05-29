"""3D view panel: triangulated points, skeleton edges, camera frustums.

Uses pyqtgraph.opengl.GLViewWidget. Falls back gracefully if OpenGL is
unavailable (shows a placeholder message instead of crashing).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from label3d.geometry import CameraParams
    from label3d.points import PointsStore
    from label3d.skeleton import Skeleton


# Color thresholds (px) matching skeleton_overlay and DESIGN.md
_ERROR_GREEN = 1.0
_ERROR_YELLOW = 3.0

# Error-based colors (RGBA float 0–1)
_COLOR_GREEN = (0.0, 0.9, 0.2, 1.0)
_COLOR_YELLOW = (1.0, 0.85, 0.0, 1.0)
_COLOR_RED = (1.0, 0.15, 0.15, 1.0)
_COLOR_GREY = (0.55, 0.55, 0.55, 1.0)
_COLOR_SELECTED = (1.0, 1.0, 1.0, 1.0)
_COLOR_EDGE = (0.55, 0.55, 0.55, 0.7)

# Camera frustum depth (world units — cm per DESIGN.md convention)
_FRUSTUM_DEPTH = 20.0


def _error_color(error: float) -> tuple[float, float, float, float]:
    if math.isnan(error):
        return _COLOR_GREY
    if error < _ERROR_GREEN:
        return _COLOR_GREEN
    if error < _ERROR_YELLOW:
        return _COLOR_YELLOW
    return _COLOR_RED


def _frustum_lines(cam: "CameraParams") -> list[np.ndarray]:
    """Compute frustum line segments for one camera (world-space).

    Returns list of (2, 3) arrays — each is one line segment.
    """
    K = cam.matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    W, H = cam.size

    # Camera centre in world space
    C = (-cam.R.T @ cam.tvec.ravel())

    # Four image corners → unit rays in camera space
    corners_img = np.array([
        [0.0, 0.0],
        [W, 0.0],
        [W, H],
        [0.0, H],
    ])
    rays_cam = np.column_stack([
        (corners_img[:, 0] - cx) / fx,
        (corners_img[:, 1] - cy) / fy,
        np.ones(4),
    ])
    # Convert to world space
    rays_world = rays_cam @ cam.R  # (4,3) — each row is a direction

    # Normalize and scale to frustum depth
    norms = np.linalg.norm(rays_world, axis=1, keepdims=True)
    far_pts = C + rays_world / norms * _FRUSTUM_DEPTH

    segments = []
    # Edges from apex to corners
    for p in far_pts:
        segments.append(np.array([C, p]))
    # Base rectangle
    for i in range(4):
        segments.append(np.array([far_pts[i], far_pts[(i + 1) % 4]]))

    return segments


def _try_import_gl():
    try:
        import pyqtgraph.opengl as gl
        return gl
    except Exception:
        return None


class View3D(QWidget):
    """3D scatter/skeleton viewer using pyqtgraph GLViewWidget.

    Shows triangulated body points (coloured by reprojection error),
    skeleton edges, optional camera frustums, and world axes.

    Falls back to a message label if OpenGL is not available.
    """

    def __init__(self, cameras: dict[str, "CameraParams"] | None = None, parent=None):
        super().__init__(parent)
        self._cameras = cameras or {}
        self._gl_available = False
        self._gl = None
        self._view = None

        # GL scene items (created on first use)
        self._scatter = None
        self._edge_items: list = []
        self._frustum_items: list = []
        self._axis_item = None

        self._frustums_visible = False  # start hidden

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        gl = _try_import_gl()
        if gl is None:
            lbl = QLabel("3D view unavailable\n(OpenGL not accessible)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888; font-size: 12px;")
            outer.addWidget(lbl)
            return

        self._gl = gl
        self._gl_available = True

        # Controls bar
        ctrl_bar = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(4, 2, 4, 2)
        self._frustum_cb = QCheckBox("Camera frustums")
        self._frustum_cb.setChecked(False)
        self._frustum_cb.toggled.connect(self._toggle_frustums)
        ctrl_layout.addWidget(self._frustum_cb)
        ctrl_layout.addStretch()
        outer.addWidget(ctrl_bar)

        # GL view
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor((30, 30, 30, 255))
        outer.addWidget(self._view, stretch=1)

        # World axes
        self._axis_item = gl.GLAxisItem()
        self._axis_item.setSize(x=10, y=10, z=10)
        self._view.addItem(self._axis_item)

        # Grid on XY plane
        grid = gl.GLGridItem()
        grid.setSize(x=100, y=100, z=1)
        grid.setSpacing(x=10, y=10, z=10)
        self._view.addItem(grid)

        # Scatter item (initialised empty)
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=np.zeros((1, 4)),
            size=8.0,
            pxMode=True,
        )
        self._view.addItem(self._scatter)

        # Initial camera frustums (hidden by default)
        if self._cameras:
            self._build_frustums()

    # -- Public API ------------------------------------------------------------

    def set_cameras(self, cameras: dict[str, "CameraParams"]) -> None:
        """Update camera parameters (called when calibration changes)."""
        self._cameras = cameras
        if self._gl_available:
            self._build_frustums()

    def update_frame(
        self,
        points_store: "PointsStore | None",
        skeleton: "Skeleton | None",
        trial: str,
        frame: int,
        selected_instance: int,
        selected_bodypart: str,
    ) -> None:
        """Refresh 3D scatter and skeleton edges for the current frame."""
        if not self._gl_available:
            return

        if points_store is None or not trial:
            self._clear_points()
            return

        try:
            inst_df = points_store.df.loc[(points_store.experiment_name, trial, frame)]
        except KeyError:
            self._clear_points()
            return

        # Collect 3D positions and colors
        positions = []
        colors = []
        # Map (instance, bodypart) → index in positions list
        pos_index: dict[tuple[int, str], int] = {}

        for (instance, bodypart), row in inst_df.iterrows():
            x3 = row.get(("3d", "x"), np.nan)
            y3 = row.get(("3d", "y"), np.nan)
            z3 = row.get(("3d", "z"), np.nan)
            if any(np.isnan(v) for v in [x3, y3, z3]):
                continue

            # Compute mean reprojection error
            errs = []
            for cam in points_store.cameras:
                pd = points_store.get_point(trial, frame, instance, bodypart, cam)
                if pd.state in ("set", "predicted") and not np.isnan(pd.x):
                    if not np.isnan(pd.xr):
                        errs.append(math.hypot(pd.x - pd.xr, pd.y - pd.yr))
            error = float(np.mean(errs)) if errs else np.nan

            is_selected = (
                instance == selected_instance and bodypart == selected_bodypart
            )
            color = _COLOR_SELECTED if is_selected else _error_color(error)

            pos_index[(instance, bodypart)] = len(positions)
            positions.append([float(x3), float(y3), float(z3)])
            colors.append(color)

        if not positions:
            self._clear_points()
        else:
            pos_arr = np.array(positions, dtype=np.float32)
            col_arr = np.array(colors, dtype=np.float32)
            sizes = np.where(
                [
                    (inst == selected_instance and bp == selected_bodypart)
                    for inst, bp in pos_index.keys()
                ],
                14.0,
                8.0,
            )
            self._scatter.setData(pos=pos_arr, color=col_arr, size=sizes)

        # Skeleton edges
        self._update_edges(skeleton, pos_index, positions, selected_instance)

    # -- Internal --------------------------------------------------------------

    def _clear_points(self):
        self._scatter.setData(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.zeros((1, 4), dtype=np.float32),
            size=1.0,
        )
        for item in self._edge_items:
            self._view.removeItem(item)
        self._edge_items.clear()

    def _update_edges(self, skeleton, pos_index, positions, selected_instance):
        # Remove old edge items
        for item in self._edge_items:
            self._view.removeItem(item)
        self._edge_items.clear()

        if skeleton is None or not positions:
            return

        for src_bp, dst_bp in skeleton.edges:
            # Find the positions — use selected_instance first, else any
            def find_pos(bp):
                if selected_instance >= 0:
                    key = (selected_instance, bp)
                    if key in pos_index:
                        return positions[pos_index[key]]
                # Fall back to first instance with this bodypart
                for (inst, b), idx in pos_index.items():
                    if b == bp:
                        return positions[idx]
                return None

            p1 = find_pos(src_bp)
            p2 = find_pos(dst_bp)
            if p1 is None or p2 is None:
                continue

            pts = np.array([p1, p2], dtype=np.float32)
            color = np.array([_COLOR_EDGE] * 2, dtype=np.float32)
            line = self._gl.GLLinePlotItem(pos=pts, color=color, width=1.5, antialias=True)
            self._view.addItem(line)
            self._edge_items.append(line)

    def _build_frustums(self):
        for item in self._frustum_items:
            self._view.removeItem(item)
        self._frustum_items.clear()

        cam_colors = [
            (0.9, 0.4, 0.1, 0.8),
            (0.1, 0.6, 0.9, 0.8),
            (0.3, 0.9, 0.4, 0.8),
            (0.9, 0.1, 0.7, 0.8),
            (0.9, 0.9, 0.1, 0.8),
        ]
        for ci, (cam_name, cam) in enumerate(sorted(self._cameras.items())):
            col = cam_colors[ci % len(cam_colors)]
            for seg in _frustum_lines(cam):
                pts = seg.astype(np.float32)
                colors = np.array([col, col], dtype=np.float32)
                line = self._gl.GLLinePlotItem(
                    pos=pts, color=colors, width=1.0, antialias=True
                )
                line.setVisible(self._frustums_visible)
                self._view.addItem(line)
                self._frustum_items.append(line)

    def _toggle_frustums(self, visible: bool):
        self._frustums_visible = visible
        for item in self._frustum_items:
            item.setVisible(visible)

"""CharucoOverlay: renders detected Charuco corners in one camera panel.

Analogous to SkeletonOverlay but for calibration data.  Each corner is
displayed as a PointItem (reusing the same visual/hover logic), colored by
reprojection error when calibration data is available.

The corner index is used as the 'bodypart' string in point_clicked signals,
so CalibrationView can decode it as int(bodypart) to look up coordinates.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import QGraphicsScene

from label3d.widgets.point_item import PointItem


class CharucoOverlay:
    """Manages PointItems for detected Charuco corners in one camera panel.

    Corners are shown as PointItems with:
    - state="set"    when the corner was detected
    - Colored by reprojection error (same thresholds as SkeletonOverlay)
    - Hover tooltip showing corner index and error
    - Selected corner gets a white selection ring

    No skeleton edges are drawn (Charuco corners are independent).
    """

    def __init__(self, scene: QGraphicsScene, camera_name: str):
        self._scene = scene
        self._camera_name = camera_name
        self._point_items: list[PointItem] = []
        self._n_corners = 0
        self._selected_idx: int | None = None

    def update(
        self,
        corners_filled: np.ndarray | None,
        errors: np.ndarray | None = None,
    ):
        """Redraw corners for the current calibration frame.

        Args:
            corners_filled: Shape (N, 2) float array; NaN means undetected.
                            Pass None or empty to hide all corners.
            errors: Shape (N,) float array of per-corner reprojection errors,
                    or None if calibration has not been run yet.
        """
        if corners_filled is None or len(corners_filled) == 0:
            self._hide_all()
            self._n_corners = 0
            return

        pts = np.asarray(corners_filled, dtype=float).reshape(-1, 2)
        n = len(pts)
        self._n_corners = n

        for i, (x, y) in enumerate(pts):
            item = self._get_or_create(i)
            detected = not (math.isnan(x) or math.isnan(y))

            if detected:
                err = float(errors[i]) if errors is not None and not math.isnan(errors[i]) else None
                item.configure(
                    bodypart=str(i),
                    instance=0,
                    state="set",
                    x=x,
                    y=y,
                    error=err,
                    selected=(i == self._selected_idx),
                )
                item.setVisible(True)
            else:
                item.setVisible(False)

        # Hide any items beyond the current corner count
        for j in range(n, len(self._point_items)):
            self._point_items[j].setVisible(False)

    def set_selected(self, corner_idx: int | None):
        """Highlight the selected corner; deselect all others."""
        prev = self._selected_idx
        self._selected_idx = corner_idx

        for i, item in enumerate(self._point_items):
            if not item.isVisible():
                continue
            # Only update items whose selection state changed
            is_sel = i == corner_idx
            was_sel = i == prev
            if is_sel != was_sel:
                item.set_selected(is_sel)

    def clear(self):
        """Hide all corners and reset state."""
        self._hide_all()
        self._n_corners = 0
        self._selected_idx = None

    def hit_test(self, scene_pos, threshold: float = 15.0) -> str | None:
        """Find the nearest visible corner within threshold pixels.

        Returns:
            str(corner_index) if a corner is close enough, else None.
        """
        best: int | None = None
        best_dist = threshold

        for i, item in enumerate(self._point_items):
            if not item.isVisible():
                continue
            dx = scene_pos.x() - item.pos().x()
            dy = scene_pos.y() - item.pos().y()
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best = i

        return str(best) if best is not None else None

    # ── Internals ─────────────────────────────────────────────────────────

    def _get_or_create(self, index: int) -> PointItem:
        while index >= len(self._point_items):
            item = PointItem()
            self._scene.addItem(item)
            self._point_items.append(item)
        return self._point_items[index]

    def _hide_all(self):
        for item in self._point_items:
            item.setVisible(False)

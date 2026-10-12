"""Zoom to instance / point: cameras without the instance or point show everything."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF

from label3d.points import PointsStore
from label3d.widgets.camera_grid import CameraGrid

CAMERAS = ["cam0", "cam1", "cam2"]
W, H = 640, 480


@pytest.fixture
def grid(qapp, simple_skeleton):
    store = PointsStore.create_empty(cameras=CAMERAS, bodyparts=[], experiment_name="exp")
    # cam0: snout and left_eye placed; cam1: snout not visible; cam2: nothing
    store.set_point("t1", 0, 0, "snout", "cam0", 100.0, 100.0, state="set")
    store.set_point("t1", 0, 0, "left_eye", "cam0", 140.0, 120.0, state="set")
    store.set_point("t1", 0, 0, "snout", "cam1", 300.0, 200.0, state="set")
    store.set_not_visible("t1", 0, 0, "snout", "cam1")

    grid = CameraGrid()
    grid.setup_cameras(CAMERAS)
    grid.resize(1200, 400)
    grid.show()
    qapp.processEvents()
    for cam in CAMERAS:
        panel = grid.get_panel(cam)
        panel.set_frame(np.zeros((H, W, 3), dtype=np.uint8))
        panel.skeleton_overlay.update(store, simple_skeleton, "t1", 0, 0, "snout")
    qapp.processEvents()
    yield grid
    grid.close()


def _visible(grid, cam) -> QRectF:
    return grid.get_panel(cam).get_scene_rect()


def _shows_whole_image(rect: QRectF) -> bool:
    return rect.contains(QRectF(1, 1, W - 2, H - 2))


def test_zoom_to_instance_fits_cameras_without_it(grid):
    grid.zoom_to_instance(0)
    assert not _shows_whole_image(_visible(grid, "cam0"))
    assert _shows_whole_image(_visible(grid, "cam1"))   # only a not-visible point
    assert _shows_whole_image(_visible(grid, "cam2"))   # nothing placed


def test_zoom_to_point_fits_cameras_where_it_is_not_placed(grid):
    grid.zoom_to_point(0, "snout")
    cam0 = _visible(grid, "cam0")
    assert not _shows_whole_image(cam0)
    assert cam0.contains(100.0, 100.0)
    assert _shows_whole_image(_visible(grid, "cam1"))
    assert _shows_whole_image(_visible(grid, "cam2"))


def test_zoomed_camera_without_point_is_reset(grid):
    """A camera left zoomed in from before goes back to the whole image."""
    grid.get_panel("cam2").zoom_to_point(QPointF(50.0, 50.0))
    assert not _shows_whole_image(_visible(grid, "cam2"))
    grid.zoom_to_point(0, "snout")
    assert _shows_whole_image(_visible(grid, "cam2"))


def test_zoom_to_instance_includes_reprojected_estimates(grid, simple_skeleton):
    """A camera with only reprojected (unset) points still zooms to the instance."""
    panel = grid.get_panel("cam2")
    est = PointsStore.create_empty(cameras=CAMERAS, bodyparts=[], experiment_name="exp")
    est.set_point("t1", 0, 0, "snout", "cam0", 100.0, 100.0, state="set")
    est.set_point("t1", 0, 0, "left_eye", "cam0", 140.0, 120.0, state="set")
    est.set_reprojected("t1", 0, 0, "snout", "cam2", 400.0, 300.0)
    est.set_reprojected("t1", 0, 0, "left_eye", "cam2", 450.0, 330.0)
    panel.skeleton_overlay.update(est, simple_skeleton, "t1", 0, 0, "snout")

    grid.zoom_to_instance(0)
    cam2 = _visible(grid, "cam2")
    assert not _shows_whole_image(cam2)
    assert cam2.contains(400.0, 300.0) and cam2.contains(450.0, 330.0)


def test_zoom_in_and_out_steps(grid):
    panel = grid.get_panel("cam0")
    whole = _visible(grid, "cam0")
    panel.zoom_in()
    assert _visible(grid, "cam0").width() < whole.width()
    for _ in range(5):
        panel.zoom_out()
    assert _shows_whole_image(_visible(grid, "cam0"))

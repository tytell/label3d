"""Average instance placement: first node at the centre, the rest at mean offsets."""

from __future__ import annotations

import pytest

from label3d.instance_init import init_average
from label3d.points import PointsStore

CAMERAS = ["cam0", "cam1"]
FRAME_SIZE = (640, 480)


@pytest.fixture
def store():
    store = PointsStore.create_empty(cameras=CAMERAS, bodyparts=[], experiment_name="exp")
    # Two labelled instances in different trials of the same experiment, in cam0.
    # Offsets from snout: left_eye (10, 0) and (20, 0); dorsal (0, 30) and (0, 50)
    for trial, (sx, sy), eye_dx, dorsal_dy in [
        ("t1", (100.0, 100.0), 10.0, 30.0),
        ("t2", (200.0, 300.0), 20.0, 50.0),
    ]:
        store.set_point(trial, 0, 0, "snout", "cam0", sx, sy, state="set")
        store.set_point(trial, 0, 0, "left_eye", "cam0", sx + eye_dx, sy, state="set")
        store.set_point(trial, 0, 0, "dorsal", "cam0", sx, sy + dorsal_dy, state="set")
    return store


def test_first_node_at_center_others_at_mean_offset(store, simple_skeleton):
    coords, fallback = init_average(
        store, "exp", "cam0", simple_skeleton, (300.0, 200.0), FRAME_SIZE
    )
    assert not fallback
    assert coords["snout"] == pytest.approx((300.0, 200.0))
    assert coords["left_eye"] == pytest.approx((315.0, 200.0))
    assert coords["dorsal"] == pytest.approx((300.0, 240.0))


def test_without_center_uses_mean_first_node_position(store, simple_skeleton):
    coords, fallback = init_average(store, "exp", "cam0", simple_skeleton, None, FRAME_SIZE)
    assert not fallback
    assert coords["snout"] == pytest.approx((150.0, 200.0))
    assert coords["left_eye"] == pytest.approx((165.0, 200.0))


def test_camera_without_labels_falls_back_to_random(store, simple_skeleton):
    coords, fallback = init_average(store, "exp", "cam1", simple_skeleton, None, FRAME_SIZE)
    assert fallback
    assert set(coords) == set(simple_skeleton.nodes)
    # Not all stacked on one spot
    assert len(set(coords.values())) == len(simple_skeleton.nodes)

"""Tests for predicted instances: locking, user instances made from predictions,
hidden predictions through SLP sync, and coloring by prediction score."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio

from label3d.adapters.sleap_adapter import SleapAdapter
from label3d.points import SOURCE_LABEL3D, SOURCE_SLEAP_PRED, PointsStore
from label3d.slp_sync import SlpBlock, SlpFileSpec, SlpMapping, compute_merge
from label3d.widgets.point_item import COLOR_BAD, COLOR_GOOD, COLOR_NONE, COLOR_OK, score_color
from tests.test_sleap_adapter import CAMERAS, EXP, _df, _one_file_per_camera, _save, _trials_df

T = "t1"


def _predicted_store(qapp) -> PointsStore:
    """Instance 0 on frame 0: predicted in cam1 (head, tail), set in cam2 (head)."""
    store = PointsStore(_df({
        (T, 0, 0, "head"): {
            "cam1": (10.0, 20.0, "predicted", SOURCE_SLEAP_PRED, 0.9),
            "cam2": (5.0, 6.0, "set"),
        },
        (T, 0, 0, "tail"): {"cam1": (30.0, 40.0, "predicted", SOURCE_SLEAP_PRED, 0.4)},
    }))
    store._experiment_name = EXP
    return store


class TestStore:
    def test_is_predicted_instance(self, qapp):
        store = _predicted_store(qapp)
        assert store.is_predicted_instance(T, 0, 0, "cam1")
        assert not store.is_predicted_instance(T, 0, 0, "cam2")
        assert not store.is_predicted_instance(T, 0, 7, "cam1")

    def test_make_user_instance_copies_and_hides_prediction(self, qapp):
        store = _predicted_store(qapp)
        assert store.make_user_instance(T, 0, 0, "cam1")
        pt = store.get_point(T, 0, 0, "tail", "cam1")
        assert (pt.x, pt.y, pt.state, pt.source) == (30.0, 40.0, "set", SOURCE_LABEL3D)
        assert math.isnan(pt.score)
        assert (pt.px, pt.py, pt.pscore) == (30.0, 40.0, 0.4)
        assert not store.is_predicted_instance(T, 0, 0, "cam1")
        # other cameras untouched
        assert math.isnan(store.get_point(T, 0, 0, "head", "cam2").px)

    def test_make_user_instance_refuses_user_instance(self, qapp):
        store = _predicted_store(qapp)
        assert not store.make_user_instance(T, 0, 0, "cam2")

    def test_delete_instance_in_camera_restores_prediction(self, qapp):
        store = _predicted_store(qapp)
        store.make_user_instance(T, 0, 0, "cam1")
        store.set_point(T, 0, 0, "head", "cam1", 12.0, 22.0)
        store.delete_instance_in_camera(T, 0, 0, "cam1")
        pt = store.get_point(T, 0, 0, "head", "cam1")
        assert (pt.x, pt.y, pt.state, pt.source, pt.score) == (
            10.0, 20.0, "predicted", SOURCE_SLEAP_PRED, 0.9
        )
        assert math.isnan(pt.px)
        assert store.is_predicted_instance(T, 0, 0, "cam1")
        assert store.get_point(T, 0, 0, "head", "cam2").state == "set"

    def test_delete_instance_in_camera_without_prediction_clears_it(self, qapp):
        store = _predicted_store(qapp)
        store.delete_instance_in_camera(T, 0, 0, "cam2")
        assert store.get_point(T, 0, 0, "head", "cam2").state == "not_set"
        assert store.get_instances(T, 0) == [0]  # still predicted in cam1

    def test_deleting_every_camera_drops_the_instance(self, qapp):
        store = _predicted_store(qapp)
        store.delete_instance_in_camera(T, 0, 0, "cam1")
        store.delete_instance_in_camera(T, 0, 0, "cam2")
        assert store.get_instances(T, 0) == []

    def test_deleting_last_user_point_restores_prediction(self, qapp):
        store = _predicted_store(qapp)
        store.make_user_instance(T, 0, 0, "cam1")
        store.delete_point(T, 0, 0, "head", "cam1")
        assert store.get_point(T, 0, 0, "head", "cam1").state == "not_set"
        assert store.get_point(T, 0, 0, "tail", "cam1").state == "set"
        store.delete_point(T, 0, 0, "tail", "cam1")
        assert store.is_predicted_instance(T, 0, 0, "cam1")
        assert store.get_point(T, 0, 0, "head", "cam1").x == 10.0

    def test_hidden_prediction_survives_parquet(self, qapp, tmp_path):
        store = _predicted_store(qapp)
        store.make_user_instance(T, 0, 0, "cam1")
        store.to_parquet(tmp_path / "points.parquet")
        reloaded = PointsStore.from_parquet(tmp_path / "points.parquet")
        assert reloaded.get_point(T, 0, 0, "tail", "cam1").pscore == pytest.approx(0.4)


class TestSync:
    def _user_from_prediction(self) -> object:
        return _df({
            (T, 0, 0, "head"): {"cam1": (11.0, 21.0, "set", SOURCE_LABEL3D, np.nan, np.nan,
                                         10.0, 20.0, 0.9)},
            (T, 0, 0, "tail"): {"cam1": (31.0, 41.0, "set", SOURCE_LABEL3D, np.nan, np.nan,
                                         30.0, 40.0, 0.4)},
        })

    def test_save_writes_linked_prediction(self, tmp_path):
        _save(tmp_path, self._user_from_prediction(), _one_file_per_camera())
        labels = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        (lf,) = labels.labeled_frames
        pred, user = lf.instances
        assert type(pred) is sio.PredictedInstance and type(user) is sio.Instance
        assert user.from_predicted is pred
        np.testing.assert_allclose(pred.numpy(), [[10, 20], [30, 40]])
        np.testing.assert_allclose(pred.numpy(scores=True)[:, 2], [0.9, 0.4], rtol=1e-6)
        np.testing.assert_allclose(user.numpy(), [[11, 21], [31, 41]])

    def test_save_load_merge_is_identity(self, tmp_path):
        store = self._user_from_prediction()
        _save(tmp_path, store, _one_file_per_camera())
        loaded = SleapAdapter().load(
            None, _trials_df(), CAMERAS, mapping=_one_file_per_camera(), root=tmp_path
        )
        assert loaded.loc[(EXP, T, 0, 0, "tail"), ("cam1", "px")] == 30.0
        r = compute_merge(store, loaded)
        assert r.conflicts == [] and r.changed_frames == set()

    def test_user_instance_made_in_sleap_hides_store_prediction(self, tmp_path):
        """SLEAP user instance from a prediction, with one node deleted."""
        sk = sio.Skeleton(["head", "tail"])
        pred = sio.PredictedInstance.from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]]), skeleton=sk, point_scores=np.array([0.9, 0.4]),
            score=0.65,
        )
        user = sio.Instance.from_numpy(np.array([[11.0, 21.0], [np.nan, np.nan]]), skeleton=sk)
        user.from_predicted = pred
        frame = sio.LabeledFrame(
            video=sio.Video("cam1_t1.mp4", open_backend=False), frame_idx=0,
            instances=[pred, user],
        )
        path = tmp_path / "x.slp"
        sio.save_file(sio.Labels(labeled_frames=[frame]), str(path))

        store = _df({
            (T, 0, 0, "head"): {"cam1": (10.0, 20.0, "predicted", SOURCE_SLEAP_PRED, 0.9)},
            (T, 0, 0, "tail"): {"cam1": (30.0, 40.0, "predicted", SOURCE_SLEAP_PRED, 0.4)},
        })
        loaded = SleapAdapter().load([path], _trials_df(), CAMERAS)
        df = compute_merge(store, loaded).df
        head = df.loc[(EXP, T, 0, 0, "head"), "cam1"]
        tail = df.loc[(EXP, T, 0, 0, "tail"), "cam1"]
        assert (head["x"], head["state"], head["px"]) == (11.0, "set", 10.0)
        assert tail["state"] == "not_set"
        assert (tail["px"], tail["py"]) == (30.0, 40.0)
        assert tail["pscore"] == pytest.approx(0.4)


class TestScoreColor:
    def test_ends_and_middle(self):
        assert score_color(1.0) == COLOR_GOOD
        assert score_color(0.5) == COLOR_OK
        assert score_color(0.0) == COLOR_BAD

    def test_clamped(self):
        assert score_color(1.03) == COLOR_GOOD
        assert score_color(-0.2) == COLOR_BAD

    def test_no_score_is_grey(self):
        assert score_color(None) == COLOR_NONE
        assert score_color(float("nan")) == COLOR_NONE


class TestOverlayLock:
    def test_predicted_instance_is_locked_per_camera(self, qapp):
        from PySide6.QtWidgets import QGraphicsScene

        from label3d.skeleton import Skeleton
        from label3d.widgets.skeleton_overlay import SkeletonOverlay

        store = _predicted_store(qapp)
        skeleton = Skeleton(nodes=["head", "tail"], edges=[], symmetries=[])
        scene = QGraphicsScene()
        overlays = {cam: SkeletonOverlay(scene, cam) for cam in CAMERAS}
        for overlay in overlays.values():
            overlay.update(store, skeleton, T, 0, 0, "head")
        assert overlays["cam1"].is_locked(0)
        assert not overlays["cam2"].is_locked(0)

        store.make_user_instance(T, 0, 0, "cam1")
        overlays["cam1"].update(store, skeleton, T, 0, 0, "head")
        assert not overlays["cam1"].is_locked(0)

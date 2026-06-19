"""Tests for Phase 7: SLP import and PointsStore.merge_from_df."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from label3d.points import CAM_COORDS, INDEX_NAMES, THREE_D_COORDS, PointsStore

sleap_io = pytest.importorskip("sleap_io", reason="sleap_io not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAMERAS = ["cam1", "cam2"]
BODYPARTS = ["head", "tail"]
EXP = "default"
TRIAL = "trial1"


def _make_adapter_row(
    experiment=EXP,
    trial=TRIAL,
    frame=0,
    instance=0,
    bodypart="head",
    cam1_x=10.0,
    cam1_y=20.0,
    cam1_state="predicted",
    cam2_x=float("nan"),
    cam2_y=float("nan"),
    cam2_state="not_set",
) -> pd.DataFrame:
    """Build a minimal adapter DataFrame with one row."""
    col_tuples = [(c, coord) for c in CAMERAS for coord in CAM_COORDS]
    col_tuples += [("3d", coord) for coord in THREE_D_COORDS]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["view", "coord"])
    idx = pd.MultiIndex.from_tuples(
        [(experiment, trial, frame, instance, bodypart)], names=INDEX_NAMES
    )
    vals = [
        cam1_x, cam1_y, float("nan"), float("nan"), cam1_state,
        cam2_x, cam2_y, float("nan"), float("nan"), cam2_state,
        float("nan"), float("nan"), float("nan"),
    ]
    return pd.DataFrame([vals], index=idx, columns=columns)


def _empty_store() -> PointsStore:
    return PointsStore.create_empty(CAMERAS, BODYPARTS, experiment_name=EXP)


# ---------------------------------------------------------------------------
# merge_from_df — basic insertion
# ---------------------------------------------------------------------------

class TestMergeFromDf:
    def test_inserts_absent_row(self, qapp):
        store = _empty_store()
        df = _make_adapter_row()
        added = store.merge_from_df(df)
        assert added == 1
        pt = store.get_point(TRIAL, 0, 0, "head", "cam1")
        assert pt.x == pytest.approx(10.0)
        assert pt.state == "predicted"

    def test_returns_zero_for_empty_df(self, qapp):
        store = _empty_store()
        empty = _make_adapter_row()
        empty = empty.iloc[0:0]  # empty but right schema
        added = store.merge_from_df(empty)
        assert added == 0

    def test_marks_dirty(self, qapp):
        store = _empty_store()
        assert not store.dirty
        store.merge_from_df(_make_adapter_row())
        assert store.dirty

    def test_emits_changed_once(self, qapp):
        store = _empty_store()
        emit_count = []
        store.changed.connect(lambda: emit_count.append(1))
        # Two bodypart rows
        df1 = _make_adapter_row(bodypart="head")
        df2 = _make_adapter_row(bodypart="tail")
        combined = pd.concat([df1, df2])
        store.merge_from_df(combined)
        assert len(emit_count) == 1  # one bulk emit at end

    # -- Merge rules: user data wins -----------------------------------------

    def test_skips_row_with_set_in_any_camera(self, qapp):
        store = _empty_store()
        store.set_point(TRIAL, 0, 0, "head", "cam1", 99.0, 99.0, state="set")

        df = _make_adapter_row(cam1_x=10.0, cam1_y=20.0, cam1_state="predicted")
        store.merge_from_df(df)

        # cam1 coords must still be the user value (99, 99)
        pt = store.get_point(TRIAL, 0, 0, "head", "cam1")
        assert pt.x == pytest.approx(99.0)
        assert pt.state == "set"

    def test_skips_row_with_not_visible_in_any_camera(self, qapp):
        store = _empty_store()
        store.set_point(TRIAL, 0, 0, "head", "cam1", 50.0, 60.0)
        store.set_not_visible(TRIAL, 0, 0, "head", "cam1")

        df = _make_adapter_row(cam1_x=10.0, cam1_y=20.0, cam1_state="predicted")
        store.merge_from_df(df)

        pt = store.get_point(TRIAL, 0, 0, "head", "cam1")
        assert pt.state == "not_visible"

    # -- Merge rules: predicted replaced by newer prediction -----------------

    def test_replaces_all_predicted_row(self, qapp):
        store = _empty_store()
        # Seed with old prediction
        old = _make_adapter_row(cam1_x=1.0, cam1_y=2.0, cam1_state="predicted")
        store.merge_from_df(old)

        # Newer prediction should replace
        new = _make_adapter_row(cam1_x=5.0, cam1_y=6.0, cam1_state="predicted")
        added = store.merge_from_df(new)
        assert added == 1
        pt = store.get_point(TRIAL, 0, 0, "head", "cam1")
        assert pt.x == pytest.approx(5.0)

    # -- Merge rules: not_set rows accept predictions -----------------------

    def test_inserts_into_not_set_row(self, qapp):
        store = _empty_store()
        # Create a not_set row
        store._ensure_row((EXP, TRIAL, 0, 0, "head"))

        df = _make_adapter_row(cam1_x=7.0, cam1_y=8.0, cam1_state="predicted")
        added = store.merge_from_df(df)
        assert added == 1
        pt = store.get_point(TRIAL, 0, 0, "head", "cam1")
        assert pt.x == pytest.approx(7.0)
        assert pt.state == "predicted"


# ---------------------------------------------------------------------------
# merge_from_df — both instance types from SleapAdapter
# ---------------------------------------------------------------------------

class TestMergeInstanceTypes:
    def _make_slp_with_both_types(self, tmp_path: "Path", video_name: str):
        """Create an SLP with one PredictedInstance and one regular Instance."""
        import sleap_io as sio

        sk = sio.Skeleton(name="test")
        sk.add_node("head")
        sk.add_node("tail")
        video = sio.Video(filename=video_name)

        # PredictedInstance (has score attribute → state=predicted)
        pred_pts = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
        pred_inst = sio.PredictedInstance.from_numpy(pred_pts, skeleton=sk, score=0.9)

        # Regular Instance (state=set for non-NaN coords)
        user_pts = np.array([[110.0, 210.0], [float("nan"), float("nan")]], dtype=np.float32)
        user_inst = sio.Instance.from_numpy(user_pts, skeleton=sk)

        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst, user_inst])
        labels = sio.Labels(labeled_frames=[lf])
        slp_path = tmp_path / "both_types.slp"
        sio.save_file(labels, str(slp_path))
        return slp_path

    def test_predicted_instance_has_predicted_state(self, qapp, tmp_path):
        from label3d.adapters.sleap_adapter import SleapAdapter

        video_name = "cam1_trial1.mp4"
        slp_path = self._make_slp_with_both_types(tmp_path, video_name)

        trials = pd.DataFrame({
            "experiment": [EXP],
            "trial": [TRIAL],
            "cam1": [video_name],
            "cam1_start": [0],
        })
        result = SleapAdapter().load([slp_path], trials, ["cam1"])
        # Instance 0 is predicted
        state_0 = result.loc[(EXP, TRIAL, 0, 0, "head"), ("cam1", "state")]
        assert state_0 == "predicted"

    def test_user_instance_has_set_state(self, qapp, tmp_path):
        from label3d.adapters.sleap_adapter import SleapAdapter

        video_name = "cam1_trial1.mp4"
        slp_path = self._make_slp_with_both_types(tmp_path, video_name)

        trials = pd.DataFrame({
            "experiment": [EXP],
            "trial": [TRIAL],
            "cam1": [video_name],
            "cam1_start": [0],
        })
        result = SleapAdapter().load([slp_path], trials, ["cam1"])
        # Instance 1 is a regular user Instance with coords → state=set
        state_1 = result.loc[(EXP, TRIAL, 0, 1, "head"), ("cam1", "state")]
        assert state_1 == "set"

    def test_user_instance_nan_coords_have_not_set_state(self, qapp, tmp_path):
        from label3d.adapters.sleap_adapter import SleapAdapter

        video_name = "cam1_trial1.mp4"
        slp_path = self._make_slp_with_both_types(tmp_path, video_name)

        trials = pd.DataFrame({
            "experiment": [EXP],
            "trial": [TRIAL],
            "cam1": [video_name],
            "cam1_start": [0],
        })
        result = SleapAdapter().load([slp_path], trials, ["cam1"])
        # Instance 1 / tail → NaN coords → not_set
        state_tail = result.loc[(EXP, TRIAL, 0, 1, "tail"), ("cam1", "state")]
        assert state_tail == "not_set"

    def test_both_types_merge_into_store(self, qapp, tmp_path):
        from label3d.adapters.sleap_adapter import SleapAdapter

        video_name = "cam1_trial1.mp4"
        slp_path = self._make_slp_with_both_types(tmp_path, video_name)

        trials = pd.DataFrame({
            "experiment": [EXP],
            "trial": [TRIAL],
            "cam1": [video_name],
            "cam1_start": [0],
        })
        adapter_df = SleapAdapter().load([slp_path], trials, ["cam1", "cam2"])
        store = PointsStore.create_empty(["cam1", "cam2"], ["head", "tail"], experiment_name=EXP)
        added = store.merge_from_df(adapter_df)
        assert added > 0
        # Both instances should be present
        instances = store.get_instances(TRIAL, 0)
        assert 0 in instances
        assert 1 in instances


# ---------------------------------------------------------------------------
# Project.slp_import_available
# ---------------------------------------------------------------------------

class TestSlpImportAvailable:
    def test_false_when_no_slp_dir(self, tmp_path):
        from label3d.project import Project
        p = Project(
            root=tmp_path,
            cameras=["cam1"],
            active_experiment=EXP,
        )
        assert not p.slp_import_available

    def test_false_when_slp_dir_empty(self, tmp_path):
        from label3d.project import Project
        slp_dir = tmp_path / "slp"
        slp_dir.mkdir()
        p = Project(root=tmp_path, cameras=["cam1"], active_experiment=EXP)
        assert not p.slp_import_available

    def test_false_when_points_not_empty(self, qapp, tmp_path):
        from label3d.project import Project
        slp_dir = tmp_path / "slp"
        slp_dir.mkdir()
        (slp_dir / "dummy.slp").touch()
        store = PointsStore.create_empty(["cam1"], ["head"], experiment_name=EXP)
        store.set_point(TRIAL, 0, 0, "head", "cam1", 1.0, 2.0)
        p = Project(root=tmp_path, cameras=["cam1"], active_experiment=EXP, points=store)
        assert not p.slp_import_available

    def test_true_when_slp_present_and_points_empty(self, qapp, tmp_path):
        from label3d.project import Project
        slp_dir = tmp_path / "slp"
        slp_dir.mkdir()
        (slp_dir / "dummy.slp").touch()
        store = PointsStore.create_empty(["cam1"], ["head"], experiment_name=EXP)
        p = Project(root=tmp_path, cameras=["cam1"], active_experiment=EXP, points=store)
        assert p.slp_import_available

    def test_true_when_points_is_none(self, tmp_path):
        from label3d.project import Project
        slp_dir = tmp_path / "slp"
        slp_dir.mkdir()
        (slp_dir / "dummy.slp").touch()
        p = Project(root=tmp_path, cameras=["cam1"], active_experiment=EXP, points=None)
        assert p.slp_import_available

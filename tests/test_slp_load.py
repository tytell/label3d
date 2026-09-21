"""Tests for SleapAdapter.load(): instance types and states read from SLP files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from label3d.adapters.sleap_adapter import SleapAdapter
from label3d.points import PointsStore
from label3d.slp_sync import compute_merge

sleap_io = pytest.importorskip("sleap_io", reason="sleap_io not installed")

CAMERAS = ["cam1", "cam2"]
EXP = "default"
TRIAL = "trial1"
VIDEO = "cam1_trial1.mp4"


def _trials_df() -> pd.DataFrame:
    return pd.DataFrame({
        "experiment": [EXP],
        "trial": [TRIAL],
        "cam1": [VIDEO],
        "cam1_start": [0],
    })


def _slp_with_both_types(tmp_path: Path) -> Path:
    """An SLP file with one PredictedInstance and one user Instance in frame 0."""
    import sleap_io as sio

    sk = sio.Skeleton(name="test")
    sk.add_node("head")
    sk.add_node("tail")
    video = sio.Video(filename=VIDEO)

    pred_pts = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
    pred_inst = sio.PredictedInstance.from_numpy(pred_pts, skeleton=sk, score=0.9)

    user_pts = np.array([[110.0, 210.0], [float("nan"), float("nan")]], dtype=np.float32)
    user_inst = sio.Instance.from_numpy(user_pts, skeleton=sk)

    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst, user_inst])
    slp_path = tmp_path / "both_types.slp"
    sio.save_file(sio.Labels(labeled_frames=[lf]), str(slp_path))
    return slp_path


class TestLoadInstanceTypes:
    def test_predicted_instance_has_predicted_state(self, tmp_path):
        result = SleapAdapter().load([_slp_with_both_types(tmp_path)], _trials_df(), ["cam1"])
        assert result.loc[(EXP, TRIAL, 0, 0, "head"), ("cam1", "state")] == "predicted"

    def test_user_instance_has_set_state(self, tmp_path):
        result = SleapAdapter().load([_slp_with_both_types(tmp_path)], _trials_df(), ["cam1"])
        assert result.loc[(EXP, TRIAL, 0, 1, "head"), ("cam1", "state")] == "set"

    def test_user_instance_nan_coords_have_not_set_state(self, tmp_path):
        result = SleapAdapter().load([_slp_with_both_types(tmp_path)], _trials_df(), ["cam1"])
        assert result.loc[(EXP, TRIAL, 0, 1, "tail"), ("cam1", "state")] == "not_set"

    def test_user_instance_wins_over_prediction_in_same_frame(self, tmp_path):
        """A frame with a user instance contributes only that instance (8.8 frame rule)."""
        slp_df = SleapAdapter().load([_slp_with_both_types(tmp_path)], _trials_df(), CAMERAS)
        result = compute_merge(PointsStore.empty_df(CAMERAS), slp_df)
        assert result.n_new > 0
        assert result.n_predictions == 0
        states = result.df.xs("head", level="bodypart")[("cam1", "state")]
        assert set(states) == {"set"}
        assert result.df.loc[(EXP, TRIAL, 0, 1, "head"), ("cam1", "x")] == pytest.approx(110.0)

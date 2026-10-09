"""Tests for calibration refinement with labelled points (Phase 12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from label3d.calibration_refine import (
    PointSelection,
    count_points,
    frames_by_trial,
    gather_points,
    holdout_frames,
    holdout_mask,
    load_points_record,
    save_points_record,
    usable_mask,
)
from label3d.points import INDEX_NAMES, PointsStore

CAMS = ["cam1", "cam2", "cam3"]


def _points_df(rows: list[dict]) -> pd.DataFrame:
    """A points DataFrame from {trial, frame, bodypart, cam: (x, y, state, source, score)}."""
    df = PointsStore.empty_df(CAMS)
    records = {}
    for r in rows:
        key = ("exp1", r["trial"], r["frame"], 0, r["bodypart"])
        rec = {col: np.nan for col in df.columns}
        for cam in CAMS:
            rec[(cam, "state")] = "not_set"
            rec[(cam, "source")] = ""
            if cam in r:
                x, y, state, source, score = r[cam]
                rec.update({
                    (cam, "x"): x, (cam, "y"): y, (cam, "state"): state,
                    (cam, "source"): source, (cam, "score"): score,
                })
        records[key] = rec
    out = pd.DataFrame.from_dict(records, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=INDEX_NAMES)
    out.columns = df.columns
    return out


def _selection_df() -> pd.DataFrame:
    user = lambda x: (x, x + 1, "set", "label3d", np.nan)          # noqa: E731
    sleap_user = lambda x: (x, x + 1, "set", "sleap_user", np.nan)  # noqa: E731
    pred = lambda x, s: (x, x + 1, "predicted", "sleap_pred", s)    # noqa: E731
    guess = lambda x: (x, x + 1, "predicted", "label3d", np.nan)    # noqa: E731
    return _points_df([
        # three label3d user points
        {"trial": "t1", "frame": 0, "bodypart": "a", "cam1": user(1), "cam2": user(2), "cam3": user(3)},
        # one user point plus a label3d reprojection guess: only 1 camera counts
        {"trial": "t1", "frame": 0, "bodypart": "b", "cam1": user(4), "cam2": guess(5)},
        # SLEAP predictions, one above and one below the threshold
        {"trial": "t1", "frame": 1, "bodypart": "a", "cam1": pred(6, 0.9), "cam2": pred(7, 0.5)},
        # SLEAP user labels in two cameras, another trial
        {"trial": "t2", "frame": 3, "bodypart": "a", "cam2": sleap_user(8), "cam3": sleap_user(9)},
    ])


class TestPointSelection:
    def test_sources_and_min_cameras(self):
        df = _selection_df()
        sel = PointSelection(sources={"label3d", "sleap_user"})
        p2ds, index = gather_points(df, CAMS, sel)
        assert list(index) == [("t1", 0, 0, "a"), ("t2", 3, 0, "a")]
        assert p2ds.shape == (3, 2, 2)
        np.testing.assert_allclose(p2ds[:, 0, 0], [1, 2, 3])
        assert np.isnan(p2ds[0, 1, 0]) and p2ds[1, 1, 0] == 8

    def test_reprojection_guesses_left_out(self):
        df = _selection_df()
        mask = usable_mask(df, CAMS, PointSelection(sources={"label3d"}, min_cameras=1))
        assert mask.loc[("t1", 0, 0, "b"), "cam1"]
        assert not mask.loc[("t1", 0, 0, "b"), "cam2"]

    def test_score_threshold(self):
        df = _selection_df()
        sel = PointSelection(sources={"sleap_pred"}, score_min=0.8, min_cameras=1)
        p2ds, index = gather_points(df, CAMS, sel)
        assert list(index) == [("t1", 1, 0, "a")]
        assert p2ds[0, 0, 0] == 6 and np.isnan(p2ds[1, 0, 0])
        sel.score_min = 0.4
        p2ds, _ = gather_points(df, CAMS, sel)
        assert p2ds[1, 0, 0] == 7

    def test_min_cameras_three(self):
        df = _selection_df()
        sel = PointSelection(sources={"label3d", "sleap_user"}, min_cameras=3)
        _, index = gather_points(df, CAMS, sel)
        assert list(index) == [("t1", 0, 0, "a")]

    def test_trials_filter(self):
        df = _selection_df()
        sel = PointSelection(sources={"label3d", "sleap_user"}, trials=["t2"])
        _, index = gather_points(df, CAMS, sel)
        assert list(index) == [("t2", 3, 0, "a")]

    def test_count_points(self):
        df = _selection_df()
        counts = count_points(df, CAMS, PointSelection(sources={"label3d", "sleap_user"}))
        assert counts == {("cam1", "cam2"): 1, ("cam1", "cam3"): 1, ("cam2", "cam3"): 2}


class TestHoldout:
    def _index(self):
        keys = [("t1", f, 0, bp) for f in range(20) for bp in ("a", "b")]
        keys += [("t2", f, 0, "a") for f in range(10)]
        return pd.MultiIndex.from_tuples(keys)

    def test_split_by_frame(self):
        index = self._index()
        frames = holdout_frames(index, 0.2, seed=7)
        assert len(frames) == 6   # 20% of 30 frames
        mask = holdout_mask(index, frames)
        # both bodyparts of a held-out frame are held out together
        for t, f, _, bp in index:
            if t == "t1":
                assert mask[list(index).index((t, f, 0, "a"))] == mask[list(index).index((t, f, 0, "b"))]

    def test_reproducible(self):
        index = self._index()
        assert holdout_frames(index, 0.2, seed=7) == holdout_frames(index, 0.2, seed=7)
        assert holdout_frames(index, 0.2, seed=7) != holdout_frames(index, 0.2, seed=8)

    def test_edge_cases(self):
        index = self._index()
        assert holdout_frames(index, 0.0, seed=1) == set()
        one = pd.MultiIndex.from_tuples([("t1", 0, 0, "a")])
        assert holdout_frames(one, 0.5, seed=1) == set()
        two = pd.MultiIndex.from_tuples([("t1", 0, 0, "a"), ("t1", 1, 0, "a")])
        assert len(holdout_frames(two, 0.9, seed=1)) == 1

    def test_frames_by_trial(self):
        assert frames_by_trial(self._index())["t2"] == list(range(10))


class TestPointsRecord:
    def test_round_trip(self, tmp_path):
        df = _selection_df()
        sel = PointSelection(sources={"label3d", "sleap_user", "sleap_pred"}, score_min=0.8, min_cameras=1)
        p2ds, index = gather_points(df, CAMS, sel)
        holdout = np.array([False, True, False, True][: len(index)])
        path = tmp_path / "x_points.csv"
        save_points_record(path, p2ds, index, CAMS, df, holdout)

        record = pd.read_csv(path)
        assert list(record.columns) == [
            "trial", "frame", "instance", "bodypart", "camera", "x", "y", "source", "score", "holdout",
        ]
        assert set(record["source"]) == {"label3d", "sleap_user", "sleap_pred"}

        p2ds_back, index_back, holdout_back = load_points_record(path, CAMS)
        assert list(index_back) == list(index)
        np.testing.assert_array_equal(holdout_back, holdout)
        np.testing.assert_allclose(p2ds_back, p2ds, equal_nan=True)

"""Batch triangulation (Phase 13.1): triangulate_trial and triangulate_frame."""

from __future__ import annotations

import numpy as np
import pytest

from label3d.geometry import (
    build_camera_group,
    load_cameras,
    triangulate_frame,
    triangulate_trial,
)
from label3d.points import PointsStore
from tests.conftest import TEST_PROJECT, requires_test_project

EXP_DIR = TEST_PROJECT / "experiments" / "exp1"
OLD = 999.0  # marks values left from an earlier triangulation


def _reference(cameras, store):
    """Per-point triangulation through aniposelib, one sub-CameraGroup per point.

    This is how triangulate_frame worked before Phase 13. Returns (p3d, reproj)
    aligned with store.df rows.
    """
    df = store.df
    cams = sorted(cameras)
    p3d = np.full((len(df), 3), np.nan)
    reproj = {cam: np.full((len(df), 2), np.nan) for cam in cams}
    for i, (_idx, row) in enumerate(df.iterrows()):
        used = [
            cam for cam in cams
            if row[(cam, "state")] in ("set", "predicted") and not np.isnan(row[(cam, "x")])
        ]
        if len(used) < 2:
            continue
        pts = np.array([[row[(c, "x")], row[(c, "y")]] for c in used])[:, np.newaxis, :]
        cgroup = build_camera_group({c: cameras[c] for c in used})
        xyz = cgroup.triangulate(pts, undistort=True)
        p3d[i] = xyz[0]
        for cam in cams:
            reproj[cam][i] = cameras[cam].project(xyz)[0]
    return p3d, reproj


@pytest.fixture
def cameras():
    return load_cameras(next((EXP_DIR / "calibrations").glob("*.toml")))


@pytest.fixture
def store(qapp):
    """exp1 points with old 3D values everywhere and some bodyparts left with 1 camera."""
    store = PointsStore.from_csv(EXP_DIR / "points.csv")
    df = store.df
    df[("3d", "x")] = OLD
    df[("3d", "y")] = OLD
    df[("3d", "z")] = OLD
    for cam in store.cameras:
        df[(cam, "xr")] = OLD
        df[(cam, "yr")] = OLD
    # Every third row: only the first camera still counts
    for cam in store.cameras[1:]:
        df.iloc[::3, df.columns.get_loc((cam, "state"))] = "not_visible"
    return store


def _result(store):
    cams = store.cameras
    p3d = store.df[[("3d", c) for c in "xyz"]].to_numpy(float)
    reproj = {cam: store.df[[(cam, "xr"), (cam, "yr")]].to_numpy(float) for cam in cams}
    return p3d, reproj


def _assert_matches(store, expected):
    p3d, reproj = _result(store)
    exp_p3d, exp_reproj = expected
    np.testing.assert_allclose(p3d, exp_p3d, atol=1e-6, equal_nan=True)
    for cam in exp_reproj:
        np.testing.assert_allclose(reproj[cam], exp_reproj[cam], atol=1e-6, equal_nan=True)


def _trials(store):
    return store.df.index.get_level_values("trial").unique()


@requires_test_project
class TestTriangulateTrial:
    def test_matches_per_point_aniposelib(self, cameras, store):
        expected = _reference(cameras, store)
        assert np.isnan(expected[0][::3]).all()
        assert not np.isnan(expected[0][1::3]).any()
        for trial in _trials(store):
            triangulate_trial(cameras, store, trial)
        _assert_matches(store, expected)

    def test_too_few_cameras_clears_old_values(self, cameras, store):
        for trial in _trials(store):
            triangulate_trial(cameras, store, trial)
        p3d, reproj = _result(store)
        assert np.isnan(p3d[::3]).all()
        for xy in reproj.values():
            assert np.isnan(xy[::3]).all()
        assert not (p3d == OLD).any()

    def test_frames_subset_leaves_other_frames(self, cameras, store):
        trial = _trials(store)[0]
        frames = sorted(store.trial_rows(trial).index.get_level_values("frame").unique())
        triangulate_trial(cameras, store, trial, frames[:1])
        rows = store.trial_rows(trial)
        in_first = rows.index.get_level_values("frame") == frames[0]
        assert not (rows.loc[in_first, ("3d", "x")] == OLD).any()
        assert (rows.loc[~in_first, ("3d", "x")] == OLD).all()

    def test_returns_count_of_3d_points(self, cameras, store):
        n = sum(triangulate_trial(cameras, store, t) for t in _trials(store))
        assert n == store.df[("3d", "x")].notna().sum()


@requires_test_project
class TestTriangulateFrame:
    def test_matches_triangulate_trial(self, cameras, store):
        expected = _reference(cameras, store)
        keys = store.df.index.droplevel("bodypart").unique()
        for _exp, trial, frame, inst in keys:
            errors = triangulate_frame(cameras, store, trial, frame, inst)
            assert set(errors) == set(store.instance_rows(trial, frame, inst)
                                      .index.get_level_values("bodypart"))
        _assert_matches(store, expected)

    def test_errors_nan_with_one_camera(self, cameras, store):
        _exp, trial, frame, inst, bp = store.df.index[0]  # row 0 has one camera
        errors = triangulate_frame(cameras, store, trial, frame, inst)
        assert np.isnan(errors[bp])
        assert all(np.isfinite(e) for b, e in errors.items() if b != bp)

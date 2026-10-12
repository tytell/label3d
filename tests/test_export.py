"""Long-format export and its pre-flight checks (Phase 13.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from label3d.export import (
    EXPORT_COLUMNS,
    ExportOptions,
    export_points,
    pending_slp_changes,
    stale_experiments,
    uncalibrated_experiments,
)
from label3d.geometry import load_cameras, triangulate_trial
from tests.test_project_sync import (  # noqa: F401  (project is a fixture)
    BODYPARTS,
    CAMERAS,
    EXP,
    TRIALS,
    _activate_calibration,
    project,
)


def _triangulate(project):
    cameras = load_cameras(project.calibration_path(EXP))
    for trial in TRIALS:
        triangulate_trial(cameras, project.points, trial)
    project.set_triangulated_with(EXP, project.active_calibration_id(EXP))


@pytest.fixture
def calibrated(project):
    _activate_calibration(project)
    _triangulate(project)
    return project


def _export(project, tmp_path, **kwargs):
    result = export_points(project, ExportOptions(output=tmp_path / "out.csv", **kwargs))
    return result, pd.read_csv(result.path, dtype={"experiment": str, "trial": str})


class TestFormat:
    def test_columns_in_order(self, calibrated, tmp_path):
        _result, out = _export(calibrated, tmp_path)
        assert list(out.columns) == EXPORT_COLUMNS

    def test_one_row_per_point_and_view(self, calibrated, tmp_path):
        result, out = _export(calibrated, tmp_path)
        assert result.n_rows == len(out) == len(TRIALS) * len(BODYPARTS) * (len(CAMERAS) + 1)
        first = out.iloc[: len(CAMERAS) + 1]
        assert list(first["view"]) == [*CAMERAS, "3d"]
        assert first["bodypart"].nunique() == 1

    def test_rows_round_trip_to_store(self, calibrated, tmp_path):
        _result, out = _export(calibrated, tmp_path)
        df = calibrated.points.df
        out = out.set_index(["experiment", "trial", "frame", "instance", "bodypart", "view"])
        for idx, row in df.iterrows():
            for cam in CAMERAS:
                got = out.loc[(*idx, cam)]
                for coord in ("x", "y", "xr", "yr"):
                    assert got[coord] == pytest.approx(row[(cam, coord)])
                assert got["state"] == row[(cam, "state")]
                assert got["source"] == row[(cam, "source")]
                assert got["error"] == pytest.approx(
                    np.hypot(row[(cam, "x")] - row[(cam, "xr")], row[(cam, "y")] - row[(cam, "yr")])
                )
            got = out.loc[(*idx, "3d")]
            for coord in "xyz":
                assert got[coord] == pytest.approx(row[("3d", coord)])
            assert got["n_cameras"] == len(CAMERAS)
            assert got["calibration"] == calibrated.active_calibration_id(EXP)
            assert pd.isna(got["state"]) and pd.isna(got["video_frame"])

    def test_video_frame_adds_cam_start(self, calibrated, tmp_path):
        trials_df = calibrated.trials_df
        trials_df.loc[trials_df["trial"] == "t2", "cam1_start"] = 5
        _result, out = _export(calibrated, tmp_path)
        cam1 = out[out["view"] == "cam1"]
        expected = cam1["frame"] + np.where(cam1["trial"] == "t2", 5, 0)
        assert (cam1["video_frame"] == expected).all()

    def test_default_path_is_in_exports(self, calibrated):
        result = export_points(calibrated, ExportOptions(write_slp=False))
        assert result.path.parent == calibrated.root / "exports"
        assert result.path.name.startswith("points_")


class TestFilters:
    def _mark_prediction(self, project, score):
        df = project.points.df
        idx = (EXP, "t1", 0, 0, "head")
        df.loc[idx, ("cam0", "source")] = "sleap_pred"
        df.loc[idx, ("cam0", "state")] = "predicted"
        df.loc[idx, ("cam0", "score")] = score

    def test_low_scores_are_kept(self, calibrated, tmp_path):
        self._mark_prediction(calibrated, 0.01)
        _result, out = _export(calibrated, tmp_path)
        pred = out[out["source"] == "sleap_pred"]
        assert len(pred) == 1 and pred["score"].iloc[0] == pytest.approx(0.01)

    def test_no_predictions(self, calibrated, tmp_path):
        self._mark_prediction(calibrated, 0.9)
        result, out = _export(calibrated, tmp_path, include_predictions=False)
        assert not (out["source"] == "sleap_pred").any()
        assert result.counts[EXP] == {("set", "label3d"): len(TRIALS) * len(BODYPARTS) * 2 - 1}

    def test_only_3d(self, calibrated, tmp_path):
        calibrated.points.delete_point("t1", 0, 0, "head", "cam1")
        _triangulate(calibrated)
        _result, all_rows = _export(calibrated, tmp_path)
        _result, only = _export(calibrated, tmp_path, only_3d=True)
        head_t1 = (all_rows["trial"] == "t1") & (all_rows["bodypart"] == "head")
        assert list(all_rows.loc[head_t1, "view"]) == ["cam0"]
        assert not ((only["trial"] == "t1") & (only["bodypart"] == "head")).any()
        assert len(only) == len(all_rows) - 1

    def test_experiments(self, calibrated, tmp_path):
        result, out = _export(calibrated, tmp_path, experiments=[])
        assert result.n_rows == 0 and list(out.columns) == EXPORT_COLUMNS
        with pytest.raises(KeyError):
            _export(calibrated, tmp_path, experiments=["nope"])


class TestChecks:
    def test_uncalibrated_exports_2d_only(self, project, tmp_path):
        assert uncalibrated_experiments(project) == [EXP]
        assert stale_experiments(project) == []
        result, out = _export(project, tmp_path)
        assert result.two_d_only == [EXP]
        assert set(out["view"]) == set(CAMERAS)
        assert out[["xr", "yr", "error"]].isna().all().all()

    def test_stale_refuses(self, project, tmp_path):
        _activate_calibration(project)
        assert stale_experiments(project) == [EXP]
        with pytest.raises(ValueError, match=EXP):
            _export(project, tmp_path)

    def test_pending_slp_changes(self, calibrated):
        calibrated.write_slp_files()
        assert pending_slp_changes(calibrated) == {}
        calibrated.points.delete_point("t1", 0, 0, "head", "cam0")
        assert list(pending_slp_changes(calibrated)) == [EXP]

    def test_export_writes_slp_files(self, calibrated, tmp_path):
        result, _out = _export(calibrated, tmp_path)
        assert result.slp_files_written == len(CAMERAS)
        assert pending_slp_changes(calibrated) == {}

    def test_no_slp_write(self, calibrated, tmp_path):
        result, _out = _export(calibrated, tmp_path, write_slp=False)
        assert result.slp_files_written is None
        assert not list(calibrated.slp_dir.glob("*.slp"))

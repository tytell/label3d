"""Wide-format export and its pre-flight checks (Phase 13.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from label3d.export import (
    ExportOptions,
    export_columns,
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
        assert list(out.columns) == export_columns(CAMERAS)
        assert list(out.columns[5:9]) == ["cam0_video_frame", "cam0_x.px", "cam0_y.px", "cam0_state"]
        assert list(out.columns[-6:-3]) == ["x.cm", "y.cm", "z.cm"]

    def test_one_row_per_point(self, calibrated, tmp_path):
        result, out = _export(calibrated, tmp_path)
        assert result.n_rows == len(out) == len(TRIALS) * len(BODYPARTS)
        assert not out.duplicated(["experiment", "trial", "frame", "instance", "bodypart"]).any()

    def test_rows_round_trip_to_store(self, calibrated, tmp_path):
        _result, out = _export(calibrated, tmp_path)
        df = calibrated.points.df
        out = out.set_index(["experiment", "trial", "frame", "instance", "bodypart"])
        for idx, row in df.iterrows():
            got = out.loc[idx]
            for cam in CAMERAS:
                for coord in ("x", "y", "xr", "yr"):
                    assert got[f"{cam}_{coord}.px"] == pytest.approx(row[(cam, coord)])
                assert got[f"{cam}_state"] == row[(cam, "state")]
                assert got[f"{cam}_source"] == row[(cam, "source")]
                assert got[f"{cam}_error.px"] == pytest.approx(
                    np.hypot(row[(cam, "x")] - row[(cam, "xr")], row[(cam, "y")] - row[(cam, "yr")])
                )
            for coord in "xyz":
                assert got[f"{coord}.cm"] == pytest.approx(row[("3d", coord)])
            assert got["n_cameras"] == len(CAMERAS)
            assert got["calibration"] == calibrated.active_calibration_id(EXP)

    def test_video_frame_adds_cam_start(self, calibrated, tmp_path):
        trials_df = calibrated.trials_df
        trials_df.loc[trials_df["trial"] == "t2", "cam1_start"] = 5
        _result, out = _export(calibrated, tmp_path)
        expected = out["frame"] + np.where(out["trial"] == "t2", 5, 0)
        assert (out["cam1_video_frame"] == expected).all()
        assert (out["cam0_video_frame"] == out["frame"]).all()

    def test_state_and_source_per_camera(self, calibrated, tmp_path):
        idx = (EXP, "t1", 0, 0, "head")
        df = calibrated.points.df
        df.loc[idx, ("cam0", "state")] = "predicted"
        df.loc[idx, ("cam0", "source")] = "sleap_pred"
        _result, out = _export(calibrated, tmp_path)
        row = out.set_index(["experiment", "trial", "frame", "instance", "bodypart"]).loc[idx]
        assert (row["cam0_state"], row["cam0_source"]) == ("predicted", "sleap_pred")
        assert (row["cam1_state"], row["cam1_source"]) == ("set", "label3d")

    def test_point_without_3d_keeps_row(self, calibrated, tmp_path):
        calibrated.points.delete_point("t1", 0, 0, "head", "cam1")
        _triangulate(calibrated)
        _result, out = _export(calibrated, tmp_path)
        row = out[(out["trial"] == "t1") & (out["bodypart"] == "head")]
        assert len(row) == 1
        row = row.iloc[0]
        assert row["cam0_state"] == "set"
        assert pd.isna(row["cam1_state"]) and pd.isna(row["cam1_x.px"])
        assert row[["x.cm", "y.cm", "z.cm", "n_cameras", "calibration"]].isna().all()

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
        pred = out[out["cam0_source"] == "sleap_pred"]
        assert len(pred) == 1 and pred["cam0_score"].iloc[0] == pytest.approx(0.01)

    def test_no_predictions(self, calibrated, tmp_path):
        self._mark_prediction(calibrated, 0.9)
        result, out = _export(calibrated, tmp_path, include_predictions=False)
        assert not (out["cam0_source"] == "sleap_pred").any()
        assert len(out) == len(TRIALS) * len(BODYPARTS)
        row = out[(out["trial"] == "t1") & (out["bodypart"] == "head")].iloc[0]
        assert pd.isna(row["cam0_x.px"]) and pd.isna(row["cam0_score"])
        assert row["cam1_state"] == "set" and not pd.isna(row["x.cm"])
        assert result.counts[EXP] == {("set", "label3d"): len(TRIALS) * len(BODYPARTS) * 2 - 1}

    def test_only_3d(self, calibrated, tmp_path):
        calibrated.points.delete_point("t1", 0, 0, "head", "cam1")
        _triangulate(calibrated)
        _result, all_rows = _export(calibrated, tmp_path)
        _result, only = _export(calibrated, tmp_path, only_3d=True)
        assert ((all_rows["trial"] == "t1") & (all_rows["bodypart"] == "head")).sum() == 1
        assert not ((only["trial"] == "t1") & (only["bodypart"] == "head")).any()
        assert len(only) == len(all_rows) - 1

    def test_experiments(self, calibrated, tmp_path):
        result, out = _export(calibrated, tmp_path, experiments=[])
        assert result.n_rows == 0 and list(out.columns) == export_columns(CAMERAS)
        with pytest.raises(KeyError):
            _export(calibrated, tmp_path, experiments=["nope"])


class TestChecks:
    def test_uncalibrated_exports_2d_only(self, project, tmp_path):
        assert uncalibrated_experiments(project) == [EXP]
        assert stale_experiments(project) == []
        result, out = _export(project, tmp_path)
        assert result.two_d_only == [EXP]
        assert len(out) == len(TRIALS) * len(BODYPARTS)
        reproj = [f"{cam}_{c}.px" for cam in CAMERAS for c in ("xr", "yr", "error")]
        assert out[reproj].isna().all().all()
        assert out[["x.cm", "y.cm", "z.cm", "n_cameras", "calibration"]].isna().all().all()
        assert out[[f"{cam}_x.px" for cam in CAMERAS]].notna().all().all()

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

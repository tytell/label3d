"""Saved alignment clicks and computing an alignment from them (Phase 16.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from label3d.alignment import AlignmentSettings, compute_alignment
from label3d.alignment_picks import (
    picked_frames,
    picks_path,
    read_picks,
    targets_for,
    write_picks,
)
from label3d.geometry import AlignmentError
from tests.test_axis_alignment import O_TRUE, R_TRUE, click, look_at


@pytest.fixture
def cameras():
    return {
        "cam0": look_at("cam0", [80.0, 0.0, 30.0]),
        "cam1": look_at("cam1", [0.0, 80.0, 40.0]),
        "cam2": look_at("cam2", [-60.0, -50.0, 50.0]),
    }


# Object-frame points of each target: axis points offset from the axes in 2-point mode
OBJECT_POINTS = {
    "origin": [0, 0, 0],
    "x1": [2.0, 0.5, 0], "x2": [10.0, 0.5, 0],
    "y1": [0.3, 2.0, 0], "y2": [0.3, 9.0, 0],
    "z1": [0, -0.4, 3.0], "z2": [0, -0.4, 8.0],
}
ONE_POINT = {"origin": [0, 0, 0], "x1": [8.0, 0, 0], "y1": [0, 6.0, 0], "z1": [0, 0, 5.0]}


def picks_for(cameras, points):
    return {target: click(cameras, p) for target, p in points.items()}


class TestFile:
    def test_round_trip_and_other_frames_kept(self, tmp_path):
        a = {"origin": {"cam0": (1.5, 2.5), "cam1": (3.0, 4.0)}, "x1": {"cam0": (5.0, 6.0)}}
        b = {"origin": {"cam1": (9.0, 9.5)}}
        write_picks(tmp_path, "Align", 0, a)
        write_picks(tmp_path, "Align", 3, b)
        write_picks(tmp_path, "Other", 0, b)
        assert read_picks(tmp_path, "Align", 0) == a
        assert read_picks(tmp_path, "Align", 3) == b
        assert picked_frames(tmp_path, "Align") == [0, 3]

        write_picks(tmp_path, "Align", 0, {"y1": {"cam2": (1.0, 1.0)}})
        assert read_picks(tmp_path, "Align", 0) == {"y1": {"cam2": (1.0, 1.0)}}
        assert read_picks(tmp_path, "Other", 0) == b

    def test_empty_picks_remove_the_frame(self, tmp_path):
        write_picks(tmp_path, "Align", 0, {"origin": {"cam0": (1.0, 2.0)}})
        write_picks(tmp_path, "Align", 0, {})
        assert read_picks(tmp_path, "Align", 0) == {}
        assert picked_frames(tmp_path, "Align") == []

    def test_no_file_reads_empty_and_empty_write_makes_none(self, tmp_path):
        assert read_picks(tmp_path, "Align", 0) == {}
        write_picks(tmp_path, "Align", 0, {})
        assert not picks_path(tmp_path).exists()

    def test_columns_on_disk(self, tmp_path):
        write_picks(tmp_path, "Align", 2, {"origin": {"cam0": (1.0, 2.0)}})
        df = pd.read_parquet(picks_path(tmp_path))
        assert df.columns.tolist() == ["take", "frame", "target", "camera", "x", "y"]
        assert picks_path(tmp_path).parent.name == "calibrations"

    def test_targets(self):
        assert targets_for(1) == ["origin", "x1", "y1", "z1"]
        assert targets_for(2) == ["origin", "x1", "x2", "y1", "y2", "z1", "z2"]
        assert targets_for(2, "y") == ["origin", "x1", "x2", "z1", "z2"]


class TestCompute:
    def test_seven_points(self, cameras):
        settings = AlignmentSettings(points_per_axis=2)
        computation = compute_alignment(cameras, picks_for(cameras, OBJECT_POINTS), settings)
        np.testing.assert_allclose(computation.result.R_align, R_TRUE, atol=1e-6)
        np.testing.assert_allclose(computation.result.origin, O_TRUE, atol=1e-6)
        assert computation.max_error_px < 1e-6
        assert computation.mean_error("x2") < 1e-6

    def test_four_points(self, cameras):
        computation = compute_alignment(cameras, picks_for(cameras, ONE_POINT), AlignmentSettings())
        np.testing.assert_allclose(computation.result.R_align, R_TRUE, atol=1e-6)

    def test_cross_product_axis_needs_no_clicks(self, cameras):
        points = {k: v for k, v in ONE_POINT.items() if k != "y1"}
        settings = AlignmentSettings(cross_product="y")
        computation = compute_alignment(cameras, picks_for(cameras, points), settings)
        assert computation.result.derived == "y"
        np.testing.assert_allclose(computation.result.R_align, R_TRUE, atol=1e-6)

    def test_flip(self, cameras):
        points = dict(ONE_POINT, z1=[0, 0, -5.0])
        settings = AlignmentSettings(flip=["z"])
        computation = compute_alignment(cameras, picks_for(cameras, points), settings)
        np.testing.assert_allclose(computation.result.R_align, R_TRUE, atol=1e-6)

    def test_missing_clicks_named(self, cameras):
        picks = picks_for(cameras, ONE_POINT)
        picks["y1"] = {"cam0": picks["y1"]["cam0"]}
        del picks["z1"]
        with pytest.raises(AlignmentError, match="y1, z1"):
            compute_alignment(cameras, picks, AlignmentSettings())

    def test_scale_check_and_alignment_record(self, cameras):
        settings = AlignmentSettings(points_per_axis=2, known_length_cm={"x": 8.0, "y": 7.7})
        computation = compute_alignment(cameras, picks_for(cameras, OBJECT_POINTS), settings)
        assert computation.scale["x"]["error_pct"] == pytest.approx(0.0, abs=1e-6)
        assert computation.scale["y"]["error_pct"] == pytest.approx(100 * (7.0 - 7.7) / 7.7)
        alignment = computation.alignment("charuco_a")
        assert alignment.calibration == "charuco_a"
        assert set(alignment.scale_error_pct) == {"x", "y"}
        p_obj = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(alignment.align(p_obj @ R_TRUE.T + O_TRUE), p_obj, atol=1e-6)

    def test_lowest_error_rule(self, cameras):
        rng = np.random.default_rng(1105)
        picks = picks_for(cameras, ONE_POINT)
        # Noise on x only: the y axis has the lowest error and is kept exactly
        picks["x1"] = {c: (x + rng.normal(0, 2), y + rng.normal(0, 2)) for c, (x, y) in picks["x1"].items()}
        computation = compute_alignment(cameras, picks, AlignmentSettings(primary="lowest_error"))
        assert computation.result.primary in ("y", "z")
        assert computation.result.secondary != "x"

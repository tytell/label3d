"""The type column of trials.csv (Phase 16): trial, calibration or axis rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from label3d.project import Project, _extract_cameras
from label3d.slp_sync import video_cells
from label3d.trial_types import row_types, with_type_column

CAMERAS = ["cam1", "cam2"]


class TestRowTypes:
    def test_any_case_and_spaces(self):
        df = pd.DataFrame({"type": ["Trial", "CALIBRATION", " axis ", "trial"]})
        assert row_types(df).tolist() == ["trial", "calibration", "axis", "trial"]

    def test_blank_is_trial(self):
        df = pd.DataFrame({"type": ["", np.nan, "axis"]})
        assert row_types(df).tolist() == ["trial", "trial", "axis"]

    def test_all_blank_column(self):
        df = pd.DataFrame({"type": [np.nan, np.nan]})
        assert row_types(df).tolist() == ["trial", "trial"]

    def test_unknown_type_raises(self):
        df = pd.DataFrame({"type": ["trial", "charuco"]})
        with pytest.raises(ValueError, match="charuco"):
            row_types(df)

    def test_no_type_column_is_all_trials(self):
        df = pd.DataFrame({"trial": ["a", "b"]})
        assert row_types(df).tolist() == ["trial", "trial"]

    def test_legacy_column_is_read(self):
        df = pd.DataFrame({"is_calibration": [False, True, "True", "false"]})
        assert row_types(df).tolist() == ["trial", "calibration", "calibration", "trial"]

    def test_type_column_wins_over_legacy(self):
        df = pd.DataFrame({"type": ["axis"], "is_calibration": [True]})
        assert row_types(df).tolist() == ["axis"]


class TestWithTypeColumn:
    def test_replaces_legacy_column_in_place(self):
        df = pd.DataFrame({
            "experiment": ["e", "e"], "trial": ["t1", "PreCal"],
            "is_calibration": [False, True], "cam1": ["a.mp4", "b.mp4"],
        })
        out = with_type_column(df)
        assert out.columns.tolist() == ["experiment", "trial", "type", "cam1"]
        assert out["type"].tolist() == ["trial", "calibration"]
        assert "is_calibration" in df.columns  # the input is left alone

    def test_no_legacy_column_unchanged(self):
        df = pd.DataFrame({"trial": ["t1"], "type": ["trial"]})
        assert with_type_column(df) is df


def test_type_is_not_a_camera():
    columns = ["experiment", "trial", "type", "cam1", "cam1_start", "is_calibration"]
    assert _extract_cameras(columns) == ["cam1"]


def test_video_cells_skip_takes():
    df = pd.DataFrame({
        "experiment": ["e", "e", "e"], "trial": ["t1", "PreCal", "Align"],
        "type": ["trial", "Calibration", "AXIS"],
        "cam1": ["t1.mp4", "pre.mp4", "align.png"], "cam1_start": [0, 0, 0],
    })
    assert {cell[1] for cell in video_cells(df, ["cam1"])} == {"t1"}


def test_project_sorts_rows_by_type(tmp_path: Path, qapp):
    header = ["experiment", "trial", "type"]
    for cam in CAMERAS:
        header += [cam, f"{cam}_start", f"{cam}_end"]
    rows = [
        "exp1,trial1,,cam1_t1.mp4,1,,cam2_t1.mp4,1,",
        "exp1,trial2,Trial,cam1_t2.mp4,1,,cam2_t2.mp4,1,",
        "exp1,PreCal,calibration,pre1.mp4,1,,pre2.mp4,1,",
        "exp1,Align,axis,align1.png,1,,align2.png,1,",
    ]
    (tmp_path / "trials.csv").write_text(",".join(header) + "\n" + "\n".join(rows) + "\n")
    exp = tmp_path / "experiments" / "exp1"
    (exp / "videos").mkdir(parents=True)
    for cam in CAMERAS:
        (exp / "calibrations" / cam).mkdir(parents=True)
        (exp / "calibrations" / cam / f"align{cam[-1]}.png").write_bytes(b"\x00")

    project = Project.load(tmp_path)
    assert project.trial_names() == ["trial1", "trial2"]
    assert [t.name for t in project.calibration_takes] == ["PreCal"]
    assert [t.name for t in project.axis_takes] == ["Align"]
    assert project.get_take("Align").name == "Align"
    assert project.video_path("Align", "cam1") == exp / "calibrations" / "cam1" / "align1.png"

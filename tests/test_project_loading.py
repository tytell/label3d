"""Integration tests for Project.load() — skeleton, points, and error handling."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from label3d.project import Project
from tests.conftest import TEST_PROJECT


# ---------------------------------------------------------------------------
# Successful loads from the synthetic test project
# ---------------------------------------------------------------------------

class TestProjectLoad:
    def test_loads_cameras(self):
        proj = Project.load(TEST_PROJECT)
        assert proj.cameras == ["cam1", "cam2", "cam3"]

    def test_loads_trials(self):
        proj = Project.load(TEST_PROJECT)
        assert "trial1" in proj.trial_names()

    def test_loads_skeleton(self):
        proj = Project.load(TEST_PROJECT)
        assert proj.skeleton is not None
        assert proj.skeleton.nodes == ["snout", "left_eye", "dorsal"]

    def test_loads_points(self):
        proj = Project.load(TEST_PROJECT)
        assert proj.points is not None
        assert len(proj.points.df) > 0

    def test_loads_trials_df(self):
        proj = Project.load(TEST_PROJECT)
        assert proj.trials_df is not None
        assert "trial" in proj.trials_df.columns

    def test_has_calibration(self):
        proj = Project.load(TEST_PROJECT)
        assert proj.has_calibration is True


# ---------------------------------------------------------------------------
# Missing optional files — graceful degradation
# ---------------------------------------------------------------------------

class TestMissingOptionalFiles:
    def test_missing_skeleton_json_is_none(self, tmp_path):
        """No skeleton.json → project.skeleton is None, no error raised."""
        _copy_project_without(TEST_PROJECT, tmp_path, exclude={"skeleton.json"})
        proj = Project.load(tmp_path)
        assert proj.skeleton is None

    def test_missing_points_csv_is_none(self, tmp_path):
        """No points.csv → project.points is None, no error raised."""
        _copy_project_without(TEST_PROJECT, tmp_path,
                               exclude={"experiments/exp1/points.csv"})
        proj = Project.load(tmp_path)
        assert proj.points is None

    def test_missing_calibration_toml_flagged(self, tmp_path):
        """No calibration.toml → has_calibration=False, no error."""
        _copy_project_without(TEST_PROJECT, tmp_path,
                               exclude={"experiments/exp1/calibration.toml"})
        proj = Project.load(tmp_path)
        assert proj.has_calibration is False


# ---------------------------------------------------------------------------
# Error cases — informative messages
# ---------------------------------------------------------------------------

class TestLoadErrors:
    def test_nonexistent_dir_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "no_such_project"
        with pytest.raises(FileNotFoundError) as exc_info:
            Project.load(missing)
        assert str(missing) in str(exc_info.value) or "no_such_project" in str(exc_info.value)

    def test_missing_trials_csv_raises_file_not_found(self, tmp_path):
        """Directory exists but has no trials.csv."""
        with pytest.raises(FileNotFoundError) as exc_info:
            Project.load(tmp_path)
        msg = str(exc_info.value).lower()
        assert "trials" in msg or "csv" in msg

    def test_file_path_instead_of_dir_raises(self, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        with pytest.raises(FileNotFoundError):
            Project.load(f)


# ---------------------------------------------------------------------------
# Path edge cases
# ---------------------------------------------------------------------------

class TestPathEdgeCases:
    def test_load_project_path_with_spaces(self, tmp_path):
        spaced = tmp_path / "my project 2024"
        shutil.copytree(TEST_PROJECT, spaced)
        proj = Project.load(spaced)
        assert proj.cameras == ["cam1", "cam2", "cam3"]

    def test_load_project_nested_path_with_spaces(self, tmp_path):
        nested = tmp_path / "lab data" / "subject 01" / "session 1"
        shutil.copytree(TEST_PROJECT, nested)
        proj = Project.load(nested)
        assert proj.skeleton is not None

    def test_video_path_with_spaces(self, tmp_path):
        """video_path() resolves correctly even when the project dir has spaces."""
        spaced = tmp_path / "project dir"
        shutil.copytree(TEST_PROJECT, spaced)
        proj = Project.load(spaced)
        vp = proj.video_path("trial1", "cam1")
        assert vp.exists()

    def test_trial_length_resolved(self):
        """Trial length must be set (not None/0) after loading with real video files."""
        proj = Project.load(TEST_PROJECT)
        length = proj.trial_length("trial1")
        assert length > 0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _copy_project_without(src: Path, dst: Path, exclude: set[str]) -> None:
    """Copy TEST_PROJECT to dst, omitting the relative paths listed in exclude."""
    shutil.copytree(src, dst, dirs_exist_ok=True)
    for rel in exclude:
        target = dst / rel
        if target.exists():
            target.unlink()

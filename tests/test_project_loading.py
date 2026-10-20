"""Integration tests for Project.load() — skeleton, points, and error handling."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import toml

from label3d import project as project_module
from label3d.project import LoadCancelled, Project
from tests.conftest import TEST_PROJECT, requires_test_project


# ---------------------------------------------------------------------------
# Successful loads from the synthetic test project
# ---------------------------------------------------------------------------

@requires_test_project
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

@requires_test_project
class TestMissingOptionalFiles:
    def test_no_skeleton_anywhere_is_none(self, tmp_path):
        """No .slp files and no [skeleton] in project.toml → skeleton is None."""
        _copy_project_without(TEST_PROJECT, tmp_path, exclude={"skeleton.json"})
        shutil.rmtree(tmp_path / "slp")
        config = toml.load(tmp_path / "project.toml")
        config.pop("skeleton", None)
        config.pop("slp", None)
        (tmp_path / "project.toml").write_text(toml.dumps(config))
        proj = Project.load(tmp_path)
        assert proj.skeleton is None
        assert proj.skeletons is None

    def test_skeleton_from_slp_files_alone(self, tmp_path):
        """Without skeleton.json or [skeleton], the .slp files give the skeleton."""
        _copy_project_without(TEST_PROJECT, tmp_path, exclude={"skeleton.json"})
        config = toml.load(tmp_path / "project.toml")
        config.pop("skeleton", None)
        (tmp_path / "project.toml").write_text(toml.dumps(config))
        proj = Project.load(tmp_path)
        assert proj.skeleton.nodes == ["snout", "left_eye", "dorsal"]
        assert proj.skeletons.by_camera == {}
        assert proj.skeletons.skeleton_for(("exp1", "trial1", "cam2")).nodes == proj.skeleton.nodes

    def test_missing_points_csv_gives_empty_store(self, tmp_path):
        """No points file → an empty store ready to label into, no error raised."""
        _copy_project_without(TEST_PROJECT, tmp_path,
                               exclude={"experiments/exp1/points.parquet"})
        proj = Project.load(tmp_path)
        assert proj.points is not None
        assert proj.points.df.empty

    def test_missing_points_csv_store_knows_where_to_save(self, tmp_path):
        """The empty store carries the path its first save should write to."""
        _copy_project_without(TEST_PROJECT, tmp_path,
                               exclude={"experiments/exp1/points.parquet"})
        proj = Project.load(tmp_path)
        assert proj.points._save_path == tmp_path / "experiments/exp1/points.parquet"

    def test_missing_points_csv_store_has_the_experiment_cameras(self, tmp_path):
        """Columns come from the experiment, so points can be set immediately."""
        _copy_project_without(TEST_PROJECT, tmp_path,
                               exclude={"experiments/exp1/points.parquet"})
        proj = Project.load(tmp_path)
        views = set(proj.points.df.columns.get_level_values("view"))
        assert views == set(proj.cameras) | {"3d"}

    def test_no_active_calibration_flagged(self, tmp_path):
        """No calibration pointer → has_calibration=False, no error."""
        _copy_project_without(TEST_PROJECT, tmp_path, exclude=set())
        shutil.rmtree(tmp_path / "experiments/exp1/calibrations", ignore_errors=True)
        config = toml.load(tmp_path / "project.toml")
        config.get("calibration", {}).pop("active", None)
        (tmp_path / "project.toml").write_text(toml.dumps(config))
        proj = Project.load(tmp_path)
        assert proj.has_calibration is False
        assert proj.calibration_path() is None

    def test_pointer_to_missing_file_flagged(self, tmp_path):
        """A pointer whose registry file is gone counts as no calibration."""
        _copy_project_without(TEST_PROJECT, tmp_path, exclude=set())
        proj = Project.load(tmp_path)
        proj.calibration_path().unlink()
        proj = Project.load(tmp_path)
        assert proj.has_calibration is False
        assert proj.calibration_path() is None


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

    @requires_test_project
    @pytest.mark.parametrize("legacy", [
        "sync_history.csv",
        "experiments/exp1/calibrations/cam1/corners.csv",
    ])
    def test_other_csv_working_files_refused(self, tmp_path, legacy):
        _copy_project_without(TEST_PROJECT, tmp_path, exclude=set())
        (tmp_path / legacy).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / legacy).write_text("x\n")
        with pytest.raises(ValueError, match=legacy.rsplit("/", 1)[-1]):
            Project.load(tmp_path)

    @requires_test_project
    def test_points_csv_refused(self, tmp_path):
        """A project that still has points.csv and no points.parquet is set up again."""
        _copy_project_without(TEST_PROJECT, tmp_path,
                               exclude={"experiments/exp1/points.parquet"})
        (tmp_path / "experiments/exp1/points.csv").write_text("x\n")
        with pytest.raises(ValueError, match="points.csv") as exc_info:
            Project.load(tmp_path)
        assert "set the project up again" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Path edge cases
# ---------------------------------------------------------------------------

@requires_test_project
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


@requires_test_project
class TestVideoCheck:
    """Blank end frames and video mtimes are filled in once and written back."""

    def _project(self, tmp_path):
        _copy_project_without(TEST_PROJECT, tmp_path, exclude=set())
        df = pd.read_csv(tmp_path / "trials.csv")
        df = df[[c for c in df.columns if not c.endswith(("_end", "_mtime"))]]
        df.to_csv(tmp_path / "trials.csv", index=False)
        (tmp_path / "trials.csv.bak").unlink(missing_ok=True)
        return tmp_path

    def test_fills_ends_and_mtimes(self, tmp_path):
        root = self._project(tmp_path)
        proj = Project.load(root)
        df = pd.read_csv(root / "trials.csv")
        assert df["cam1_end"].tolist() == [9, 9]
        assert df["cam1_mtime"].str.endswith("Z").all()
        assert list(df.columns[:6]) == [
            "experiment", "trial", "cam1", "cam1_start", "cam1_end", "cam1_mtime",
        ]
        assert proj.cameras == ["cam1", "cam2", "cam3"]   # _mtime is not a camera
        assert proj.trial_length("trial1") == 10
        assert (root / "trials.csv.bak").exists()
        assert proj.video_warnings == []

    def test_second_load_does_not_open_videos(self, tmp_path, monkeypatch):
        root = self._project(tmp_path)
        Project.load(root)
        before = (root / "trials.csv").read_text()

        def fail(path):
            raise AssertionError(f"opened {path}")

        monkeypatch.setattr(project_module, "_frame_count", fail)
        proj = Project.load(root)
        assert proj.trial_length("trial1") == 10
        assert (root / "trials.csv").read_text() == before

    def test_changed_video_is_a_warning(self, tmp_path):
        root = self._project(tmp_path)
        Project.load(root)
        video = root / "experiments/exp1/videos/cam2_exp1_trial1.mp4"
        os.utime(video, (0, 1_000_000))
        proj = Project.load(root)
        assert len(proj.video_warnings) == 1
        assert "cam2_exp1_trial1.mp4" in proj.video_warnings[0]

    def test_progress_and_cancel(self, tmp_path):
        root = self._project(tmp_path)
        calls = []

        def progress(done, total, text):
            calls.append((done, total, text))
            return True

        Project.load(root, progress=progress)
        assert calls[0][:2] == (0, 6)
        assert calls[0][2].startswith("Measuring videos: exp1/trial1")
        assert calls[-1][:2] == (6, 6)

        with pytest.raises(LoadCancelled):
            Project.load(self._project(tmp_path / "again"), progress=lambda *a: False)

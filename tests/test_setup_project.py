"""Tests for the project setup utility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import toml

from label3d.setup_project import (
    FileAssignment,
    SetupConfig,
    classify_files,
    execute_setup,
    preview_assignment,
    write_project_toml,
    _identify_camera,
    _extract_trial_key_number,
    _extract_trial_key_datestamp,
)


# --- Fixtures ---


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """Create a flat source directory with test video files."""
    src = tmp_path / "source"
    src.mkdir()
    # Data videos: 2 cameras, 2 trials
    for cam in ["left", "right"]:
        for trial in ["001", "002"]:
            (src / f"{cam}_trial{trial}.mp4").write_bytes(b"\x00" * 100)
    # Calibration videos
    (src / "calib_left.mp4").write_bytes(b"\x00" * 100)
    (src / "calib_right.mp4").write_bytes(b"\x00" * 100)
    # SLP files
    (src / "left_trial001.slp").write_bytes(b"\x00" * 100)
    (src / "right_trial001.slp").write_bytes(b"\x00" * 100)
    return src


@pytest.fixture
def basic_config(source_dir: Path, tmp_path: Path) -> SetupConfig:
    return SetupConfig(
        source_dir=source_dir,
        output_dir=tmp_path / "project",
        camera_names=["left", "right"],
        calib_keyword="calib",
        match_strategy="number",
        project_name="test_project",
        charuco={
            "squares_x": 6,
            "squares_y": 8,
            "square_length": 30.0,
            "marker_length": 22.0,
            "marker_bits": 4,
            "dict_size": 50,
        },
    )


# --- Unit tests: camera identification ---


class TestIdentifyCamera:
    def test_single_match(self):
        assert _identify_camera("left_trial001.mp4", ["left", "right"]) == "left"

    def test_no_match(self):
        assert _identify_camera("unknown_trial001.mp4", ["left", "right"]) is None

    def test_case_insensitive(self):
        assert _identify_camera("LEFT_trial001.mp4", ["left", "right"]) == "left"

    def test_ambiguous_substring(self):
        # "cam1" matches both "cam1" and "cam10" in "cam10_trial.mp4"
        # but "cam10" is longer and should win
        assert _identify_camera("cam10_trial.mp4", ["cam1", "cam10"]) == "cam10"

    def test_truly_ambiguous(self):
        # Both "ab" and "cd" appear in the filename, neither is substring of the other
        assert _identify_camera("ab_cd_trial.mp4", ["ab", "cd"]) is None


# --- Unit tests: trial key extraction ---


class TestExtractTrialKeyNumber:
    def test_basic(self):
        assert _extract_trial_key_number("left_trial001.mp4", ["left"]) == "001"

    def test_no_number(self):
        assert _extract_trial_key_number("left_trial.mp4", ["left"]) is None

    def test_multiple_numbers(self):
        # Takes the first number after removing camera name
        assert _extract_trial_key_number("left_2_trial_003.mp4", ["left"]) == "2"

    def test_camera_name_removed(self):
        # Camera name "cam1" contains "1", should be removed before number extraction
        key = _extract_trial_key_number("cam1_trial002.mp4", ["cam1"])
        assert key == "002"


class TestExtractTrialKeyDatestamp:
    def test_yyyymmdd_hhmmss(self):
        result = _extract_trial_key_datestamp(
            "left_20260315_143000.mp4", ["left"]
        )
        assert result is not None
        assert result.label == "20260315_143000"
        assert result.date_str == "20260315"
        assert result.timestamp == pytest.approx(14 * 3600 + 30 * 60)

    def test_yyyymmdd(self):
        result = _extract_trial_key_datestamp("left_20260315.mp4", ["left"])
        assert result is not None
        assert result.label == "20260315"
        assert result.timestamp == 0.0

    def test_iso_format(self):
        result = _extract_trial_key_datestamp(
            "left_2026-03-15_14-30-00.mp4", ["left"]
        )
        assert result is not None
        assert result.label == "20260315_143000"

    def test_no_datestamp(self):
        assert _extract_trial_key_datestamp("left_trial001.mp4", ["left"]) is None

    def test_yh_format(self):
        """YYYYYMMDDHHHMMSSS.frac format from .cine files."""
        result = _extract_trial_key_datestamp(
            "Lateral03Y20260116H142107.519576000.cine", ["Lateral03"]
        )
        assert result is not None
        assert result.label == "20260116_142107"
        assert result.date_str == "20260116"
        expected_ts = 14 * 3600 + 21 * 60 + 7 + 0.519576
        assert result.timestamp == pytest.approx(expected_ts)

    def test_yh_format_no_fractional(self):
        result = _extract_trial_key_datestamp(
            "Lateral03Y20260116H142107.cine", ["Lateral03"]
        )
        assert result is not None
        assert result.label == "20260116_142107"
        assert result.timestamp == pytest.approx(14 * 3600 + 21 * 60 + 7)

    def test_yyyymmdd_hhmmss_with_fractional(self):
        result = _extract_trial_key_datestamp(
            "left_20260315_143000.123456.mp4", ["left"]
        )
        assert result is not None
        assert result.label == "20260315_143000"
        assert result.timestamp == pytest.approx(14 * 3600 + 30 * 60 + 0.123456)


# --- Integration tests: classify_files ---


class TestClassifyFiles:
    def test_basic_classification(self, basic_config):
        assignment = classify_files(basic_config)

        # Should have 2 trials
        assert len(assignment.trials) == 2
        assert "trial_001" in assignment.trials
        assert "trial_002" in assignment.trials

        # Each trial should have both cameras
        for trial_name in ["trial_001", "trial_002"]:
            assert "left" in assignment.trials[trial_name]
            assert "right" in assignment.trials[trial_name]

        # Should have calibration files for both cameras
        assert "left" in assignment.calibration
        assert "right" in assignment.calibration
        assert len(assignment.calibration["left"]) == 1
        assert len(assignment.calibration["right"]) == 1

        # Should have SLP files
        assert "trial_001" in assignment.slp_files

        # No unmatched files
        assert len(assignment.unmatched) == 0

    def test_unmatched_files(self, basic_config, source_dir):
        # Add a file that matches no camera
        (source_dir / "mystery_video.mp4").write_bytes(b"\x00")
        assignment = classify_files(basic_config)
        assert any(f.name == "mystery_video.mp4" for f in assignment.unmatched)
        assert any("mystery_video" in w for w in assignment.warnings)

    def test_missing_camera_warning(self, tmp_path):
        """Trial with only one camera should produce a warning."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        # right_001.mp4 missing

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        assert len(assignment.trials) == 1
        assert any("missing cameras" in w for w in assignment.warnings)

    def test_source_not_found(self, tmp_path):
        config = SetupConfig(
            source_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "out",
            camera_names=["left"],
            calib_keyword="calib",
            match_strategy="number",
        )
        with pytest.raises(FileNotFoundError):
            classify_files(config)

    def test_datestamp_strategy(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_20260315_143000.mp4").write_bytes(b"\x00")
        (src / "right_20260315_143000.mp4").write_bytes(b"\x00")

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="datestamp",
        )
        assignment = classify_files(config)
        assert len(assignment.trials) == 1
        trial_name = list(assignment.trials.keys())[0]
        assert "20260315_143000" in trial_name

    def test_mtime_strategy(self, tmp_path):
        import os
        import time

        src = tmp_path / "src"
        src.mkdir()

        # Create files with similar mtimes (within window)
        f1 = src / "left_data.mp4"
        f2 = src / "right_data.mp4"
        f1.write_bytes(b"\x00")
        f2.write_bytes(b"\x00")

        # They were just created, so mtimes should be within 1 second
        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="mtime",
            mtime_window_sec=60.0,
        )
        assignment = classify_files(config)
        assert len(assignment.trials) == 1
        trial = list(assignment.trials.values())[0]
        assert "left" in trial
        assert "right" in trial

    def test_cine_yh_format(self, tmp_path):
        """Match .cine files with Y...H... timestamp format."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "Lateral03Y20260116H142107.519576000.cine").write_bytes(b"\x00")
        (src / "Ventral03Y20260116H142107.519576000.cine").write_bytes(b"\x00")

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["Lateral03", "Ventral03"],
            calib_keyword="calib",
            match_strategy="datestamp",
        )
        assignment = classify_files(config)
        assert len(assignment.trials) == 1
        trial_name = list(assignment.trials.keys())[0]
        assert "20260116_142107" in trial_name
        trial = list(assignment.trials.values())[0]
        assert "Lateral03" in trial
        assert "Ventral03" in trial

    def test_datestamp_windowed_matching(self, tmp_path):
        """Cameras with slightly different sub-second timestamps should match."""
        src = tmp_path / "src"
        src.mkdir()
        # Timestamps differ by ~0.003 seconds
        (src / "Lateral03Y20260116H142107.519576000.cine").write_bytes(b"\x00")
        (src / "Ventral03Y20260116H142107.522123000.cine").write_bytes(b"\x00")

        # Without window — these are different trials (different exact labels
        # won't help since labels truncate fractional, but let's test with
        # files that have different integer seconds)
        src2 = tmp_path / "src2"
        src2.mkdir()
        (src2 / "Lateral03Y20260116H142107.519576000.cine").write_bytes(b"\x00")
        (src2 / "Ventral03Y20260116H142109.100000000.cine").write_bytes(b"\x00")

        # Without window — 2 second difference, different labels → 2 trials
        config_no_window = SetupConfig(
            source_dir=src2,
            output_dir=tmp_path / "out1",
            camera_names=["Lateral03", "Ventral03"],
            calib_keyword="calib",
            match_strategy="datestamp",
            datestamp_window_sec=0.0,
        )
        assignment = classify_files(config_no_window)
        assert len(assignment.trials) == 2

        # With 5-second window — should merge into 1 trial
        config_windowed = SetupConfig(
            source_dir=src2,
            output_dir=tmp_path / "out2",
            camera_names=["Lateral03", "Ventral03"],
            calib_keyword="calib",
            match_strategy="datestamp",
            datestamp_window_sec=5.0,
        )
        assignment = classify_files(config_windowed)
        assert len(assignment.trials) == 1
        trial = list(assignment.trials.values())[0]
        assert "Lateral03" in trial
        assert "Ventral03" in trial

    def test_datestamp_window_different_dates_not_merged(self, tmp_path):
        """Files on different dates should never be merged even within window."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "Lateral03Y20260116H235959.000000000.cine").write_bytes(b"\x00")
        (src / "Ventral03Y20260117H000001.000000000.cine").write_bytes(b"\x00")

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["Lateral03", "Ventral03"],
            calib_keyword="calib",
            match_strategy="datestamp",
            datestamp_window_sec=10.0,
        )
        assignment = classify_files(config)
        # Different dates → 2 separate trials despite 2-second gap
        assert len(assignment.trials) == 2


# --- Integration tests: execute_setup ---


class TestExecuteSetup:
    def test_creates_project_structure(self, basic_config):
        assignment = classify_files(basic_config)
        result = execute_setup(basic_config, assignment)

        assert result.is_dir()
        assert (result / "videos").is_dir()
        assert (result / "videos" / "trials.csv").is_file()
        assert (result / "calibration" / "left").is_dir()
        assert (result / "calibration" / "right").is_dir()
        assert (result / "project.toml").is_file()

    def test_trials_csv_content(self, basic_config):
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        df = pd.read_csv(basic_config.output_dir / "videos" / "trials.csv")
        assert "trial" in df.columns
        assert "left" in df.columns
        assert "right" in df.columns
        assert "left_start" in df.columns
        assert "right_end" in df.columns
        assert len(df) == 2  # 2 trials

    def test_files_moved(self, basic_config, source_dir):
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        # Source files should be gone (moved)
        remaining = list(source_dir.iterdir())
        # Only unmatched files should remain
        assert not any(f.name == "left_trial001.mp4" for f in remaining)

        # Files should exist in project
        videos = list((basic_config.output_dir / "videos").iterdir())
        video_names = [f.name for f in videos]
        assert "left_trial001.mp4" in video_names
        assert "right_trial002.mp4" in video_names

    def test_calibration_files_moved(self, basic_config):
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        calib_left = list((basic_config.output_dir / "calibration" / "left").iterdir())
        assert len(calib_left) == 1
        assert calib_left[0].name == "calib_left.mp4"

    def test_slp_files_moved(self, basic_config):
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        slp_dir = basic_config.output_dir / "slp"
        assert slp_dir.is_dir()
        slp_files = list(slp_dir.iterdir())
        assert len(slp_files) == 2

    def test_project_toml_written(self, basic_config):
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        config_data = toml.load(basic_config.output_dir / "project.toml")
        assert config_data["project"]["name"] == "test_project"
        assert config_data["charuco"]["squares_x"] == 6
        assert config_data["charuco"]["marker_bits"] == 4

    def test_output_exists_error(self, basic_config):
        basic_config.output_dir.mkdir(parents=True)
        assignment = classify_files(basic_config)
        with pytest.raises(FileExistsError):
            execute_setup(basic_config, assignment)

    def test_project_loadable(self, basic_config, qapp):
        """The created project should be loadable by Project.load()."""
        from label3d.project import Project

        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        project = Project.load(basic_config.output_dir)
        assert set(project.cameras) == {"left", "right"}
        assert len(project.trials) == 2
        assert project.config["charuco"]["squares_x"] == 6


# --- Unit tests: preview ---


class TestPreview:
    def test_preview_output(self, basic_config):
        assignment = classify_files(basic_config)
        text = preview_assignment(basic_config, assignment)

        assert "test_project" in text
        assert "trial_001" in text
        assert "trial_002" in text
        assert "left" in text
        assert "right" in text
        assert "Calibration" in text


# --- Unit tests: write_project_toml ---


class TestWriteProjectToml:
    def test_basic(self, tmp_path):
        charuco = {"squares_x": 6, "squares_y": 8}
        path = write_project_toml(tmp_path, "test", charuco)

        data = toml.load(path)
        assert data["project"]["name"] == "test"
        assert data["charuco"]["squares_x"] == 6

    def test_no_charuco(self, tmp_path):
        path = write_project_toml(tmp_path, "test", None)
        data = toml.load(path)
        assert "charuco" not in data
        assert data["project"]["name"] == "test"

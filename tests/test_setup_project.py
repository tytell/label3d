"""Tests for the project setup utility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import toml

from label3d.setup_project import (
    FileAssignment,
    SetupConfig,
    classify_files,
    execute_setup,
    parse_camera_aliases,
    preview_assignment,
    skeleton_from_setup,
    split_project_groups,
    write_project_toml,
    _identify_camera,
    _extract_trial_key_number,
    _extract_trial_key_datestamp,
    _extract_calib_prefix,
    _extract_trial_prefix,
)
from label3d.project import Project
from label3d.skeleton import Skeleton, SkeletonConflict


# --- Helpers ---


def _make_slp_file(
    path: Path,
    video_filenames: list[str],
    node_names: list[str] | None = None,
    edges: list[tuple[str, str]] | None = None,
) -> None:
    """Write a minimal valid .slp file referencing the given video filenames."""
    import sleap_io as sio

    if node_names is None:
        node_names = ["nose", "tail"]

    skeleton = sio.Skeleton(name="test")
    for name in node_names:
        skeleton.add_node(name)
    for a, b in edges or []:
        skeleton.add_edge(a, b)

    labeled_frames = []
    for vf in video_filenames:
        video = sio.Video(filename=vf)
        pts = np.full((len(node_names), 2), 100.0, dtype=np.float32)
        inst = sio.Instance.from_numpy(pts, skeleton=skeleton)
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labeled_frames.append(lf)

    labels = sio.Labels(labeled_frames=labeled_frames)
    sio.save_file(labels, str(path))


# --- Fixtures ---


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """Create a flat source directory with test video and SLP files."""
    src = tmp_path / "source"
    src.mkdir()
    # Data videos: 2 cameras, 2 trials
    for cam in ["left", "right"]:
        for trial in ["001", "002"]:
            (src / f"{cam}_trial{trial}.mp4").write_bytes(b"\x00" * 100)
    # Calibration videos
    (src / "calib_left.mp4").write_bytes(b"\x00" * 100)
    (src / "calib_right.mp4").write_bytes(b"\x00" * 100)
    # SLP files: one per camera for trial001, referencing the actual video filenames
    _make_slp_file(src / "left_trial001.slp", ["left_trial001.mp4"])
    _make_slp_file(src / "right_trial001.slp", ["right_trial001.mp4"])
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


class TestCameraAliases:
    CAMS = ["LateralOrtho", "LateralOffAxis", "Ventral"]
    ALIASES = {"LateralOrtho": ["Lateral"], "LateralOffAxis": ["OffAxisLateral"]}

    @pytest.mark.parametrize("name, camera", [
        ("PS20_Trial9_Lateral_H264.mp4", "LateralOrtho"),
        ("PS20_Trial9_OffAxisLateral_H264.mp4", "LateralOffAxis"),  # longest alias wins
        ("PS37_Trial7_LateralOffAxis_H265.mp4", "LateralOffAxis"),  # name beats alias
        ("PS37_Trial7_LateralOrtho_H265.mp4", "LateralOrtho"),
        ("PS37_Trial7_Ventral_H265.mp4", "Ventral"),
    ])
    def test_alias_matching(self, name, camera):
        assert _identify_camera(name, self.CAMS, self.ALIASES) == camera

    def test_without_aliases_old_names_match_nothing(self):
        assert _identify_camera("PS20_Trial9_Lateral_H264.mp4", self.CAMS) is None

    def test_alias_for_unknown_camera_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="camera_aliases"):
            SetupConfig(
                source_dir=tmp_path, output_dir=tmp_path / "p", camera_names=["left"],
                calib_keyword="calib", match_strategy="number",
                camera_aliases={"top": ["T"]},
            )

    def test_parse_camera_aliases(self):
        assert parse_camera_aliases(["a=x,y", "b=z", "a=w"]) == {"a": ["x", "y", "w"], "b": ["z"]}
        with pytest.raises(ValueError):
            parse_camera_aliases(["no_equals"])

    def test_aliases_in_folder_setup(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        # The alias carries a digit, so the number strategy must strip it too
        for name in ("L2_trial001.mp4", "right_trial001.mp4"):
            (src / name).write_bytes(b"\x00" * 100)
        config = SetupConfig(
            source_dir=src, output_dir=tmp_path / "p", camera_names=["left", "right"],
            calib_keyword="calib", match_strategy="number",
            camera_aliases={"left": ["L2"]},
        )
        assignment = classify_files(config)
        assert sorted(assignment.trials["trial_001"]) == ["left", "right"]


class TestUnlistedExperiments:
    def _config(self, tmp_path, keywords):
        src = tmp_path / "src"
        src.mkdir()
        for exp in ("PS1", "PS2"):
            for cam in ("left", "right"):
                (src / f"{exp}_{cam}_trial001.mp4").write_bytes(b"\x00" * 100)
                (src / f"{exp}_{cam}_calib.mp4").write_bytes(b"\x00" * 100)
        return SetupConfig(
            source_dir=src, output_dir=tmp_path / "p", camera_names=["left", "right"],
            calib_keyword="calib", match_strategy="keyword", trial_keyword="trial",
            experiment_keywords=keywords,
        )

    def test_unlisted_experiment_left_out(self, tmp_path):
        config = self._config(tmp_path, ["PS1"])
        assignment = classify_files(config)
        assert set(assignment.trial_experiment.values()) == {"PS1"}
        assert sorted(assignment.exp_calibration) == ["PS1"]
        assert sorted(f.name for f in assignment.excluded) == [
            "PS2_left_calib.mp4", "PS2_left_trial001.mp4",
            "PS2_right_calib.mp4", "PS2_right_trial001.mp4",
        ]
        assert "Left out: 4 video(s)" in preview_assignment(config, assignment)

    def test_no_keywords_means_one_default_experiment(self, tmp_path):
        config = self._config(tmp_path, [])
        for f in config.source_dir.glob("PS2_*_calib.mp4"):
            f.unlink()  # one calibration take per camera
        assignment = classify_files(config)
        assert assignment.excluded == []
        assert sorted(assignment.exp_calibration) == ["default"]
        assert len(assignment.trials) == 2


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

        # SLP files matched to trial_001 via embedded video names
        assert "trial_001" in assignment.slp_files
        assert "left" in assignment.slp_files["trial_001"]
        assert "right" in assignment.slp_files["trial_001"]
        # trial_002 has no SLP files (we only created SLP for trial001)
        assert "trial_002" not in assignment.slp_files

        # No unmatched video/slp files
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
        assert (result / "experiments" / "default" / "videos").is_dir()
        assert (result / "trials.csv").is_file()
        assert (result / "experiments" / "default" / "calibrations" / "left").is_dir()
        assert (result / "experiments" / "default" / "calibrations" / "right").is_dir()
        assert (result / "project.toml").is_file()

    def test_refined_is_not_a_camera_name(self, source_dir, tmp_path):
        with pytest.raises(ValueError, match="refined"):
            SetupConfig(
                source_dir=source_dir, output_dir=tmp_path / "project",
                camera_names=["left", "refined"], calib_keyword="calib",
                match_strategy="number",
            )

    def test_trials_csv_content(self, basic_config):
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        df = pd.read_csv(basic_config.output_dir / "trials.csv")
        assert "experiment" in df.columns
        assert "trial" in df.columns
        assert "is_calibration" in df.columns
        assert "left" in df.columns
        assert "right" in df.columns
        assert "left_start" in df.columns
        assert "right_end" in df.columns

        # 2 trials + 1 calibration take
        assert len(df) == 3
        trials = df[~df["is_calibration"]]
        takes = df[df["is_calibration"]]
        assert len(trials) == 2
        assert len(takes) == 1
        assert takes.iloc[0]["trial"] == "calib"
        # The take's cells name the calibration videos, not the trial videos
        assert takes.iloc[0]["left"] == "calib_left.mp4"

    def test_calibration_takes_are_not_trials(self, basic_config, qapp):
        """Project.load() keeps calibration takes out of the trial list."""
        from label3d.project import Project

        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        project = Project.load(basic_config.output_dir)
        assert project.trial_names() == ["trial_001", "trial_002"]
        assert [t.name for t in project.calibration_takes] == ["calib"]

    def test_calibration_video_path_resolves_to_calibration_dir(
        self, basic_config, qapp
    ):
        from label3d.project import Project

        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        project = Project.load(basic_config.output_dir)
        path = project.video_path("calib", "left")
        assert path.parent.name == "left"
        assert path.parent.parent.name == "calibrations"
        assert path.is_file()

    def test_files_moved(self, basic_config, source_dir):
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        # Source files should be gone (moved)
        remaining = list(source_dir.iterdir())
        # Only unmatched files should remain
        assert not any(f.name == "left_trial001.mp4" for f in remaining)

        # Files should exist in project
        videos = list((basic_config.output_dir / "experiments" / "default" / "videos").iterdir())
        video_names = [f.name for f in videos]
        assert "left_trial001.mp4" in video_names
        assert "right_trial002.mp4" in video_names

    def test_calibration_files_moved(self, basic_config):
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)

        calib_left = list(
            (basic_config.output_dir / "experiments" / "default" / "calibrations" / "left").iterdir()
        )
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

    def test_no_skeleton_files_written(self, basic_config):
        """The .slp files are the skeleton's source: no skeleton.json, no [skeleton]."""
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)
        assert not (basic_config.output_dir / "skeleton.json").exists()
        assert "skeleton" not in toml.load(basic_config.output_dir / "project.toml")
        project = Project.load(basic_config.output_dir)
        assert project.skeleton.nodes == ["nose", "tail"]

    def test_no_skeleton_without_slp(self, tmp_path):
        """No SLP files and no setup skeleton: nothing is written."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        execute_setup(config, assignment)
        assert not (tmp_path / "out" / "skeleton.json").exists()
        assert "skeleton" not in toml.load(tmp_path / "out" / "project.toml")

    def test_setup_skeleton_written_without_slp(self, tmp_path):
        """A setup file skeleton is assigned to every camera in project.toml."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        config = SetupConfig(
            source_dir=src, output_dir=tmp_path / "out",
            camera_names=["left", "right"], calib_keyword="calib",
            match_strategy="number",
            skeleton=Skeleton(["head", "tail"], edges=[("head", "tail")], name="fish"),
        )
        assignment = classify_files(config)
        execute_setup(config, assignment)
        [entry] = toml.load(tmp_path / "out" / "project.toml")["skeleton"]["assigned"]
        assert entry["name"] == "fish"
        assert entry["cameras"] == ["left", "right"]
        assert entry["nodes"] == ["head", "tail"]
        assert entry["edges"] == [["head", "tail"]]

    def test_setup_skeleton_for_some_cameras(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        config = SetupConfig(
            source_dir=src, output_dir=tmp_path / "out",
            camera_names=["left", "right"], calib_keyword="calib",
            match_strategy="number",
            skeleton=Skeleton(["head", "tail"], name="fish"),
            skeleton_cameras=["right"],
        )
        execute_setup(config, classify_files(config))
        [entry] = toml.load(tmp_path / "out" / "project.toml")["skeleton"]["assigned"]
        assert entry["cameras"] == ["right"]

    def test_setup_skeleton_unknown_camera(self, tmp_path):
        with pytest.raises(ValueError, match="not cameras of the project: top"):
            SetupConfig(
                source_dir=tmp_path, output_dir=tmp_path / "out",
                camera_names=["left", "right"], calib_keyword="calib",
                match_strategy="number",
                skeleton=Skeleton(["head"]), skeleton_cameras=["top"],
            )

    def test_setup_skeleton_matching_slp_is_used(self, basic_config):
        """A matching setup skeleton is assigned, so its offsets are kept."""
        basic_config.skeleton = Skeleton(
            ["tail", "nose"], node_offsets_cm={"nose": [1.0, 0.0, 0.0]},
        )
        assignment = classify_files(basic_config)
        execute_setup(basic_config, assignment)
        section = toml.load(basic_config.output_dir / "project.toml")["skeleton"]
        assert section["assigned"][0]["nodes"] == ["tail", "nose"]
        assert section["node_offsets_cm"] == {"nose": [1.0, 0.0, 0.0]}
        project = Project.load(basic_config.output_dir)
        assert project.skeleton.node_offsets_cm == {"nose": [1.0, 0.0, 0.0]}

    def test_setup_skeleton_with_more_nodes_is_merged(self, basic_config):
        basic_config.skeleton = Skeleton(["nose", "tail", "fin"])
        assignment = classify_files(basic_config)
        assert assignment.skeleton["nodes"] == ["nose", "tail", "fin"]

    def test_setup_skeleton_conflicting_with_slp_raises(self, basic_config):
        basic_config.skeleton = Skeleton(["nose", "tail"], symmetries=[("nose", "tail")])
        with pytest.raises(SkeletonConflict, match="in the setup file but not in"):
            classify_files(basic_config)

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


# --- Unit tests: prefix extraction ---


class TestExtractCalibPrefix:
    def test_keyword_and_camera_removed(self):
        assert _extract_calib_prefix("PS18_calib_left.mp4", ["left", "right"], "calib") == "PS18"

    def test_no_prefix(self):
        # file like "calib_left.mp4" — nothing remains after removing cam + keyword
        assert _extract_calib_prefix("calib_left.mp4", ["left", "right"], "calib") == ""

    def test_multi_part_name(self):
        assert _extract_calib_prefix("Exp03_charuco_cam0.mp4", ["cam0", "cam1"], "charuco") == "Exp03"

    def test_no_calib_keyword(self):
        assert _extract_calib_prefix("PS18_left.mp4", ["left", "right"], "") == "PS18"


class TestExtractTrialPrefix:
    def test_keyword_strategy(self):
        assert (
            _extract_trial_prefix("PS18_Trial1_left.mp4", ["left", "right"], "keyword", trial_keyword="Trial")
            == "PS18"
        )

    def test_number_strategy(self):
        # Prefix must not contain digits; "18" in "PS18" would be found first
        assert (
            _extract_trial_prefix("PS_001_left.mp4", ["left", "right"], "number")
            == "PS"
        )

    def test_datestamp_strategy(self):
        assert (
            _extract_trial_prefix("PS18_20260316_142000_left.mp4", ["left", "right"], "datestamp")
            == "PS18"
        )

    def test_no_prefix(self):
        # Number at start → nothing before it
        result = _extract_trial_prefix("001_left.mp4", ["left", "right"], "number")
        assert result == ""

    def test_fallback_first_token(self):
        # mtime strategy falls back to first token
        result = _extract_trial_prefix("PS19_randomname_left.mp4", ["left", "right"], "mtime")
        assert result == "PS19"


# --- Integration tests: split detection ---


@pytest.fixture
def takes_source_dir(tmp_path: Path) -> Path:
    """Flat source dir: two experiments, each with a PreCal and PostCal take."""
    src = tmp_path / "source"
    src.mkdir()
    for cam in ["left", "right"]:
        for exp in ["PS18", "PS19"]:
            for take in ["PreCal", "PostCal"]:
                (src / f"{exp}_{take}_{cam}.mp4").write_bytes(b"\x00" * 100)
            (src / f"{exp}_Trial1_{cam}.mp4").write_bytes(b"\x00" * 100)
    return src


def _takes_config(src: Path, out: Path, **kw) -> SetupConfig:
    params = dict(
        source_dir=src,
        output_dir=out,
        camera_names=["left", "right"],
        calib_keyword="",
        calib_keywords=["PreCal", "PostCal"],
        match_strategy="keyword",
        trial_keyword="Trial",
        experiment_keywords=["PS18", "PS19"],
    )
    params.update(kw)
    return SetupConfig(**params)


class TestCalibrationTakes:
    def test_takes_grouped_per_experiment(self, takes_source_dir, tmp_path):
        config = _takes_config(takes_source_dir, tmp_path / "project")
        assignment = classify_files(config)

        assert set(assignment.calibration_takes) == {"PS18", "PS19"}
        for exp in ("PS18", "PS19"):
            takes = assignment.calibration_takes[exp]
            assert set(takes) == {"PreCal", "PostCal"}
            for take, cam_files in takes.items():
                assert set(cam_files) == {"left", "right"}
                for cam, fp in cam_files.items():
                    assert exp in fp.name and take in fp.name and cam in fp.name

    def test_multiple_takes_do_not_trigger_split(self, takes_source_dir, tmp_path):
        config = _takes_config(takes_source_dir, tmp_path / "project")
        assignment = classify_files(config)
        assert not assignment.needs_split
        assert assignment.calibration_groups == {}

    def test_calibration_videos_are_not_trials(self, takes_source_dir, tmp_path):
        config = _takes_config(takes_source_dir, tmp_path / "project")
        assignment = classify_files(config)
        for cam_files in assignment.trials.values():
            for fp in cam_files.values():
                assert "Cal" not in fp.name

    def test_undetermined_take_raises(self, tmp_path):
        """Two takes per camera that share one generic keyword cannot be split."""
        src = tmp_path / "source"
        src.mkdir()
        for cam in ["left", "right"]:
            for i in (1, 2):
                (src / f"PS18_calib{i}_{cam}.mp4").write_bytes(b"\x00" * 100)
            (src / f"PS18_Trial1_{cam}.mp4").write_bytes(b"\x00" * 100)

        config = _takes_config(
            src, tmp_path / "project", calib_keywords=["calib"],
            experiment_keywords=["PS18"],
        )
        with pytest.raises(ValueError, match="calibration take"):
            classify_files(config)

    def test_regex_take_group(self, takes_source_dir, tmp_path):
        config = _takes_config(
            takes_source_dir, tmp_path / "project",
            calib_keywords=[], calib_regex=r"_(?P<take>(?:Pre|Post)Cal)_",
        )
        assignment = classify_files(config)
        assert set(assignment.calibration_takes["PS18"]) == {"PreCal", "PostCal"}

    def test_missing_camera_in_take_warns(self, takes_source_dir, tmp_path):
        (takes_source_dir / "PS18_PreCal_right.mp4").unlink()
        config = _takes_config(takes_source_dir, tmp_path / "project")
        assignment = classify_files(config)
        assert any(
            "PreCal" in w and "right" in w for w in assignment.warnings
        )

    def test_single_take_still_works(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        for cam in ["left", "right"]:
            (src / f"PS18_calib_{cam}.mp4").write_bytes(b"\x00" * 100)
            (src / f"PS18_Trial1_{cam}.mp4").write_bytes(b"\x00" * 100)
        config = _takes_config(
            src, tmp_path / "project", calib_keywords=["calib"],
            experiment_keywords=["PS18"],
        )
        assignment = classify_files(config)
        assert set(assignment.calibration_takes["PS18"]) == {"calib"}


# --- Unit tests: SLP matching ---


class TestSlpMatching:
    def test_slp_matched_via_video_content(self, tmp_path):
        """SLP file is matched to trial by the video it references, not by filename."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        # SLP file has an arbitrary name but references left_001.mp4
        _make_slp_file(src / "tracking_results.slp", ["left_001.mp4"])

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        assert "trial_001" in assignment.slp_files
        assert "left" in assignment.slp_files["trial_001"]
        assert "right" not in assignment.slp_files["trial_001"]

    def test_slp_unmatched_video_warns(self, tmp_path):
        """SLP referencing a video not in the folder produces a warning."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        _make_slp_file(src / "tracking.slp", ["ghost_camera_001.mp4"])

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        assert any("ghost_camera_001.mp4" in w for w in assignment.warnings)

    def test_uncovered_video_warns(self, tmp_path):
        """Videos with no corresponding SLP file produce a warning."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        # SLP only covers left camera
        _make_slp_file(src / "tracking.slp", ["left_001.mp4"])

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        assert any("right_001.mp4" in w for w in assignment.warnings)

    def test_skeleton_extracted(self, tmp_path):
        """Skeleton is extracted from SLP files and stored in assignment."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        _make_slp_file(src / "tracking.slp", ["left_001.mp4"], node_names=["head", "spine", "tail"])

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        assert assignment.skeleton is not None
        assert assignment.skeleton["nodes"] == ["head", "spine", "tail"]

    def test_consistent_skeletons_ok(self, tmp_path):
        """Two SLP files with identical skeletons should not raise."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        _make_slp_file(src / "left.slp", ["left_001.mp4"], node_names=["nose", "tail"])
        _make_slp_file(src / "right.slp", ["right_001.mp4"], node_names=["nose", "tail"])

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        assert assignment.skeleton is not None
        assert assignment.skeleton["nodes"] == ["nose", "tail"]

    def _two_slp_config(self, tmp_path, left_nodes, right_nodes, left_edges=None):
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        _make_slp_file(src / "left.slp", ["left_001.mp4"], left_nodes, left_edges)
        _make_slp_file(src / "right.slp", ["right_001.mp4"], right_nodes)
        return SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )

    def test_different_skeletons_are_merged(self, tmp_path):
        """SLP files may have different skeletons: the project gets their union."""
        config = self._two_slp_config(tmp_path, ["nose", "tail"], ["nose", "fin"])
        assignment = classify_files(config)
        assert assignment.skeleton["nodes"] == ["nose", "tail", "fin"]
        summary = preview_assignment(config, assignment)
        assert "left.slp (test) has no fin" in summary
        assert "right.slp (test) has no tail" in summary

    def test_skeletons_with_different_edges_are_merged(self, tmp_path):
        """Star skeletons around different centres (as SLEAP makes them) merge."""
        nodes = ["nose", "eye", "tail"]
        config = self._two_slp_config(
            tmp_path, nodes, nodes, left_edges=[("nose", "eye"), ("nose", "tail")]
        )
        assignment = classify_files(config)
        assert assignment.skeleton["nodes"] == nodes
        assert {frozenset(e) for e in assignment.skeleton["edges"]} == {
            frozenset(("nose", "eye")), frozenset(("nose", "tail"))
        }

    def test_single_slp_covers_multiple_cameras(self, tmp_path):
        """One SLP file referencing videos for two cameras maps to both."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        _make_slp_file(src / "both.slp", ["left_001.mp4", "right_001.mp4"])

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        assert "trial_001" in assignment.slp_files
        assert "left" in assignment.slp_files["trial_001"]
        assert "right" in assignment.slp_files["trial_001"]
        # The same slp file is referenced for both cameras
        assert (
            assignment.slp_files["trial_001"]["left"]
            == assignment.slp_files["trial_001"]["right"]
        )

    def test_single_slp_deduped_on_move(self, tmp_path):
        """A single SLP covering both cameras should only be moved once."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        _make_slp_file(src / "both.slp", ["left_001.mp4", "right_001.mp4"])

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        execute_setup(config, assignment)

        slp_files = list((tmp_path / "out" / "slp").iterdir())
        assert len(slp_files) == 1
        assert slp_files[0].name == "both.slp"

    def test_corrupt_slp_warns(self, tmp_path):
        """A file with .slp extension that can't be parsed produces a warning."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "left_001.mp4").write_bytes(b"\x00" * 100)
        (src / "right_001.mp4").write_bytes(b"\x00" * 100)
        (src / "corrupt.slp").write_bytes(b"\x00" * 50)

        config = SetupConfig(
            source_dir=src,
            output_dir=tmp_path / "out",
            camera_names=["left", "right"],
            calib_keyword="calib",
            match_strategy="number",
        )
        assignment = classify_files(config)
        assert any("corrupt.slp" in w for w in assignment.warnings)


class TestSkeletonFromSetup:
    def test_file(self, tmp_path):
        (tmp_path / "sk.json").write_text(
            json.dumps(Skeleton(["a", "b"], edges=[("a", "b")]).to_dict())
        )
        sk = skeleton_from_setup({"file": "sk.json"}, base_dir=tmp_path)
        assert sk.nodes == ["a", "b"]

    def test_inline(self):
        sk = skeleton_from_setup({
            "nodes": ["a", "b"], "symmetries": [["a", "b"]],
            "node_offsets_cm": {"a": [1, 0, 0]},
        })
        assert sk.symmetries == [("a", "b")]
        assert sk.node_offsets_cm == {"a": [1.0, 0.0, 0.0]}

    def test_file_with_cameras_key(self, tmp_path):
        (tmp_path / "sk.json").write_text(json.dumps(Skeleton(["a"]).to_dict()))
        sk = skeleton_from_setup({"file": "sk.json", "cameras": ["c1"]}, base_dir=tmp_path)
        assert sk.nodes == ["a"]

    def test_file_and_inline_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="either"):
            skeleton_from_setup({"file": "sk.json", "nodes": ["a"]}, base_dir=tmp_path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            skeleton_from_setup({"file": "nope.json"}, base_dir=tmp_path)

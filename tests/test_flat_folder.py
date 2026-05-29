"""Tests for the flat folder generator and the full setup_project workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import toml

from synthetic.writers import _make_flat_filename, write_flat_folder
from label3d.setup_project import SetupConfig, classify_files, execute_setup


# --- Unit tests: filename generation ---


class TestMakeFlatFilename:
    def test_number_strategy(self):
        name = _make_flat_filename("cam1", 0, "number", "mp4")
        assert name == "cam1_trial001.mp4"

    def test_number_strategy_second_trial(self):
        name = _make_flat_filename("cam1", 1, "number", "mp4")
        assert name == "cam1_trial002.mp4"

    def test_number_strategy_slp(self):
        name = _make_flat_filename("cam1", 0, "number", "slp")
        assert name == "cam1_trial001.slp"

    def test_datestamp_strategy_contains_yh(self):
        name = _make_flat_filename("Lateral", 0, "datestamp", "cine",
                                   datestamp="20260116", base_time_sec=51667.519576)
        assert "Y20260116H" in name
        assert name.startswith("Lateral")
        assert name.endswith(".cine")

    def test_datestamp_strategy_cam_offset(self):
        name0 = _make_flat_filename("cam1", 0, "datestamp", "mp4",
                                    base_time_sec=51667.0, time_offset_per_cam=0.003, cam_index=0)
        name1 = _make_flat_filename("cam2", 0, "datestamp", "mp4",
                                    base_time_sec=51667.0, time_offset_per_cam=0.003, cam_index=1)
        # Different cameras should have different timestamps
        assert name0 != name1

    def test_datestamp_simple_strategy(self):
        name = _make_flat_filename("cam1", 0, "datestamp_simple", "mp4",
                                   datestamp="20260116", base_time_sec=51667.0)
        assert "20260116" in name
        assert name.startswith("cam1_")
        assert name.endswith(".mp4")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            _make_flat_filename("cam1", 0, "badstrategy", "mp4")


# --- Integration tests: write_flat_folder ---


@pytest.fixture
def fake_project(tmp_path: Path) -> tuple[Path, dict, str]:
    """Build a minimal fake structured project directory."""
    cam_names = ["cam1", "cam2"]
    trial = "trial1"

    cameras = {
        cam: {
            "resolution": [320, 240],
            "K": None,
            "dist": None,
            "R": None,
            "t": None,
            "rvec": None,
            "position": None,
        }
        for cam in cam_names
    }

    proj = tmp_path / "project"
    exp_dir = proj / "experiments" / "default"

    # experiments/default/videos/
    videos = exp_dir / "videos"
    videos.mkdir(parents=True)
    for cam in cam_names:
        (videos / f"{cam}_{trial}.mp4").write_bytes(b"\x00" * 64)

    # slp/ (at project root)
    slp = proj / "slp"
    slp.mkdir()
    for cam in cam_names:
        (slp / f"{cam}_{trial}.slp").write_bytes(b"\x00" * 32)

    # experiments/default/calibration/
    calib = exp_dir / "calibration"
    for cam in cam_names:
        cam_dir = calib / cam
        cam_dir.mkdir(parents=True)
        for i in range(3):
            (cam_dir / f"board_{i:03d}.png").write_bytes(b"\x00" * 16)

    return proj, cameras, trial


class TestWriteFlatFolder:
    def test_number_creates_expected_files(self, fake_project, tmp_path):
        proj, cameras, trial = fake_project
        flat = tmp_path / "flat_number"
        write_flat_folder(proj, flat, cameras, trial, strategy="number")

        names = {f.name for f in flat.iterdir()}
        assert "cam1_trial001.mp4" in names
        assert "cam2_trial001.mp4" in names
        assert "cam1_trial001.slp" in names
        assert "cam2_trial001.slp" in names

    def test_calibration_files_have_calib_keyword(self, fake_project, tmp_path):
        proj, cameras, trial = fake_project
        flat = tmp_path / "flat"
        write_flat_folder(proj, flat, cameras, trial, strategy="number", calib_keyword="calib")

        names = {f.name for f in flat.iterdir()}
        calib_files = [n for n in names if n.startswith("calib_")]
        # 2 cameras × 3 board images = 6 calibration files
        assert len(calib_files) == 6
        assert all("calib_" in n for n in calib_files)

    def test_datestamp_creates_yh_names(self, fake_project, tmp_path):
        proj, cameras, trial = fake_project
        flat = tmp_path / "flat_ds"
        write_flat_folder(proj, flat, cameras, trial, strategy="datestamp")

        names = {f.name for f in flat.iterdir()}
        mp4s = [n for n in names if n.endswith(".mp4")]
        assert len(mp4s) == 2
        assert all("Y" in n and "H" in n for n in mp4s)

    def test_datestamp_simple_creates_underscore_names(self, fake_project, tmp_path):
        proj, cameras, trial = fake_project
        flat = tmp_path / "flat_ds_simple"
        write_flat_folder(proj, flat, cameras, trial, strategy="datestamp_simple")

        names = {f.name for f in flat.iterdir()}
        mp4s = [n for n in names if n.endswith(".mp4")]
        assert len(mp4s) == 2
        # Should NOT have Y...H... format
        assert all("Y" not in n for n in mp4s)

    def test_setup_config_written(self, fake_project, tmp_path):
        proj, cameras, trial = fake_project
        flat = tmp_path / "flat"
        write_flat_folder(proj, flat, cameras, trial, strategy="number")

        config_path = flat / "setup_config.toml"
        assert config_path.exists()
        cfg = toml.load(config_path)
        assert cfg["setup"]["match_strategy"] == "number"
        assert cfg["setup"]["camera_names"] == ["cam1", "cam2"]
        assert cfg["setup"]["calib_keyword"] == "calib"

    def test_setup_config_datestamp_has_window(self, fake_project, tmp_path):
        proj, cameras, trial = fake_project
        flat = tmp_path / "flat"
        write_flat_folder(proj, flat, cameras, trial, strategy="datestamp")

        cfg = toml.load(flat / "setup_config.toml")
        assert cfg["setup"]["match_strategy"] == "datestamp"
        assert cfg["setup"]["datestamp_window_sec"] > 0

    def test_charuco_config_written(self, fake_project, tmp_path):
        proj, cameras, trial = fake_project
        flat = tmp_path / "flat"
        charuco = {"squares_x": 6, "squares_y": 8, "square_length": 30.0,
                   "marker_length": 22.0, "marker_bits": 4, "dict_size": 50}
        write_flat_folder(proj, flat, cameras, trial, strategy="number",
                          charuco_config=charuco)

        cfg = toml.load(flat / "setup_config.toml")
        assert cfg["charuco"]["squares_x"] == 6
        assert cfg["charuco"]["marker_bits"] == 4

    def test_no_slp_dir_is_ok(self, fake_project, tmp_path):
        """Projects without slp/ should still produce a valid flat folder."""
        proj, cameras, trial = fake_project
        shutil.rmtree(proj / "slp")
        flat = tmp_path / "flat"
        write_flat_folder(proj, flat, cameras, trial, strategy="number")

        names = {f.name for f in flat.iterdir()}
        assert "cam1_trial001.mp4" in names
        slps = [n for n in names if n.endswith(".slp")]
        assert len(slps) == 0

    def test_no_calibration_dir_is_ok(self, fake_project, tmp_path):
        """Projects without calibration/ should still produce a valid flat folder."""
        proj, cameras, trial = fake_project
        shutil.rmtree(proj / "experiments" / "default" / "calibration")
        flat = tmp_path / "flat"
        write_flat_folder(proj, flat, cameras, trial, strategy="number")

        names = {f.name for f in flat.iterdir()}
        calib_files = [n for n in names if n.startswith("calib_")]
        assert len(calib_files) == 0
        assert "cam1_trial001.mp4" in names


# --- End-to-end: flat folder → setup_project → loadable project ---


class TestEndToEndWorkflow:
    def test_number_strategy_roundtrip(self, fake_project, tmp_path):
        """write_flat_folder → classify_files → execute_setup → Project.load()."""
        from label3d.project import Project

        proj, cameras, trial = fake_project
        flat = tmp_path / "flat"
        write_flat_folder(proj, flat, cameras, trial, strategy="number")

        # Read the generated setup_config.toml
        cfg = toml.load(flat / "setup_config.toml")
        setup = cfg["setup"]

        config = SetupConfig(
            source_dir=flat,
            output_dir=tmp_path / "output",
            camera_names=setup["camera_names"],
            calib_keyword=setup["calib_keyword"],
            match_strategy=setup["match_strategy"],
        )
        assignment = classify_files(config)

        assert len(assignment.trials) == 1
        trial_data = list(assignment.trials.values())[0]
        assert set(trial_data.keys()) == {"cam1", "cam2"}

        execute_setup(config, assignment)

        # Project should be loadable
        loaded = Project.load(tmp_path / "output")
        assert set(loaded.cameras) == {"cam1", "cam2"}
        assert len(loaded.trials) == 1

    def test_datestamp_strategy_roundtrip(self, fake_project, tmp_path, qapp):
        """Datestamp flat folder correctly matched and loaded."""
        from label3d.project import Project

        proj, cameras, trial = fake_project
        flat = tmp_path / "flat"
        write_flat_folder(proj, flat, cameras, trial, strategy="datestamp")

        cfg = toml.load(flat / "setup_config.toml")
        setup = cfg["setup"]

        config = SetupConfig(
            source_dir=flat,
            output_dir=tmp_path / "output",
            camera_names=setup["camera_names"],
            calib_keyword=setup["calib_keyword"],
            match_strategy=setup["match_strategy"],
            datestamp_window_sec=setup.get("datestamp_window_sec", 0.0),
        )
        assignment = classify_files(config)

        assert len(assignment.trials) == 1
        trial_data = list(assignment.trials.values())[0]
        assert set(trial_data.keys()) == {"cam1", "cam2"}

        execute_setup(config, assignment)
        loaded = Project.load(tmp_path / "output")
        assert set(loaded.cameras) == {"cam1", "cam2"}

    def test_datestamp_simple_strategy_roundtrip(self, fake_project, tmp_path, qapp):
        from label3d.project import Project

        proj, cameras, trial = fake_project
        flat = tmp_path / "flat"
        write_flat_folder(proj, flat, cameras, trial, strategy="datestamp_simple")

        cfg = toml.load(flat / "setup_config.toml")
        setup = cfg["setup"]

        config = SetupConfig(
            source_dir=flat,
            output_dir=tmp_path / "output",
            camera_names=setup["camera_names"],
            calib_keyword=setup["calib_keyword"],
            match_strategy=setup["match_strategy"],
            datestamp_window_sec=setup.get("datestamp_window_sec", 0.0),
        )
        assignment = classify_files(config)

        assert len(assignment.trials) == 1
        execute_setup(config, assignment)
        loaded = Project.load(tmp_path / "output")
        assert set(loaded.cameras) == {"cam1", "cam2"}

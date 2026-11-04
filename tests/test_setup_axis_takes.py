"""Axis takes at project setup (Phase 16.3): folder setup and setup from .slp files."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import toml

from label3d.project import Project
from label3d.setup_project import (
    SetupConfig,
    classify_files,
    classify_slp_files,
    config_from_toml,
    execute_setup,
    preview_assignment,
)
from label3d.video import extract_frame
from tests.test_setup_from_slp import CAMERAS, _make_slp, _make_video

sys.path.insert(0, str(Path(__file__).parent.parent))
from setup_project_cli import save_config_toml  # noqa: E402


def write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((16, 16, 3), dtype=np.uint8))
    return path


# -- Folder setup ------------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """Trials, a calibration take and an axis take of still images, in one folder."""
    src = tmp_path / "src"
    for cam in CAMERAS:
        for trial in ("Trial1", "Trial2"):
            _make_video(src / f"{cam}_{trial}.mp4")
        _make_video(src / f"{cam}_calib.mp4")
        write_image(src / f"Align_{cam}.png")
    return src


def folder_config(src: Path, **overrides) -> SetupConfig:
    defaults = dict(
        source_dir=src, output_dir=src.parent / "proj", camera_names=CAMERAS,
        calib_keyword="calib", match_strategy="keyword", trial_keyword="Trial",
        align_keywords=["Align"],
    )
    defaults.update(overrides)
    return SetupConfig(**defaults)


class TestFolderSetup:
    def test_images_become_an_axis_take(self, folder):
        assignment = classify_files(folder_config(folder))
        take = assignment.axis_takes["default"]["Align"]
        assert sorted(take) == CAMERAS
        assert take["cam1"].name == "Align_cam1.png"
        assert sorted(assignment.trials) == ["Trial1", "Trial2"]
        assert sorted(assignment.calibration_takes["default"]) == ["calib"]
        assert not assignment.errors

    def test_axis_videos_too(self, folder):
        for cam in CAMERAS:
            (folder / f"Align_{cam}.png").unlink()
            _make_video(folder / f"Align_{cam}.mp4")
        assignment = classify_files(folder_config(folder))
        assert sorted(assignment.axis_takes["default"]["Align"]) == CAMERAS
        assert sorted(assignment.trials) == ["Trial1", "Trial2"]

    def test_no_keywords_no_axis_takes(self, folder):
        assignment = classify_files(folder_config(folder, align_keywords=[]))
        assert assignment.axis_takes == {}
        # The images stay unmatched, as before
        assert {p.name for p in assignment.unmatched} >= {"Align_cam1.png", "Align_cam2.png"}

    def test_regex_names_takes(self, folder):
        for cam in CAMERAS:
            write_image(folder / f"Axes02_{cam}.png")
        config = folder_config(folder, align_keywords=[], align_regex=r"(Align|Axes\d+)")
        assert sorted(classify_files(config).axis_takes["default"]) == ["Align", "Axes02"]

    def test_both_keywords_is_an_error(self, folder):
        write_image(folder / "Align_calib_cam1.png")
        assignment = classify_files(folder_config(folder))
        assert any("Align_calib_cam1.png" in e and "both" in e for e in assignment.errors)
        assert all(p.name != "Align_calib_cam1.png"
                   for take in assignment.axis_takes["default"].values() for p in take.values())

    def test_calibration_images_still_raise(self, folder):
        write_image(folder / "cam1_calib_01.png")
        with pytest.raises(ValueError, match="deprecated"):
            classify_files(folder_config(folder))

    def test_name_clash_with_a_trial_is_an_error(self, folder):
        # The regex names the take Trial1 without matching the trial videos
        for cam in CAMERAS:
            write_image(folder / f"Trial1_Ax_{cam}.png")
        config = folder_config(folder, align_keywords=[], align_regex=r"(Trial1)_Ax")
        assignment = classify_files(config)
        assert any("same name" in e for e in assignment.errors)

    def test_one_camera_warns(self, folder):
        (folder / "Align_cam2.png").unlink()
        assignment = classify_files(folder_config(folder))
        assert any("at least two" in w for w in assignment.warnings)

    def test_experiment_keywords(self, tmp_path):
        src = tmp_path / "src"
        for exp in ("PS1", "PS2"):
            for cam in CAMERAS:
                _make_video(src / f"{exp}_{cam}_Trial1.mp4")
                write_image(src / f"{exp}_Align_{cam}.png")
        write_image(src / "BG3_Align_cam1.png")   # an experiment that isn't listed
        config = folder_config(src, experiment_keywords=["PS1", "PS2"], match_strategy="regex",
                               trial_regex=r"(PS\d_Trial\d)")
        assignment = classify_files(config)
        assert sorted(assignment.axis_takes) == ["PS1", "PS2"]
        assert assignment.axis_takes["PS2"]["Align"]["cam2"].name == "PS2_Align_cam2.png"
        assert any(p.name == "BG3_Align_cam1.png" for p in assignment.excluded)

    def test_execute_writes_axis_rows(self, folder, qapp):
        config = folder_config(folder)
        out = execute_setup(config, classify_files(config))
        df = pd.read_csv(out / "trials.csv")
        axis = df[df["type"] == "axis"]
        assert axis["trial"].tolist() == ["Align"]
        assert axis.iloc[0]["cam1"] == "Align_cam1.png"
        assert axis.iloc[0]["cam1_end"] == 1
        for cam in CAMERAS:
            assert (out / "experiments/default/calibrations" / cam / f"Align_{cam}.png").is_file()
        project = Project.load(out)
        assert [t.name for t in project.axis_takes] == ["Align"]
        assert project.trial_names() == ["Trial1", "Trial2"]

    def test_preview_lists_axis_takes(self, folder):
        config = folder_config(folder)
        text = preview_assignment(config, classify_files(config))
        assert "Axis takes (1):" in text
        assert "Align — cam1: image  cam2: image" in text
        short = preview_assignment(config, classify_files(config), verbosity=1)
        assert "[default]  Align" in short

    def test_preview_without_keywords_says_nothing(self, folder):
        config = folder_config(folder, align_keywords=[])
        assert "Axis takes" not in preview_assignment(config, classify_files(config))

    def test_preview_none_found(self, folder):
        config = folder_config(folder, align_keywords=["Nothing"])
        assert "Axis takes: none found" in preview_assignment(config, classify_files(config))

    def test_clashing_file_names_stop_setup(self, folder, tmp_path):
        # An axis file named like a calibration file of the same camera
        config = folder_config(folder)
        assignment = classify_files(config)
        other = _make_video(tmp_path / "elsewhere" / "cam1_calib.mp4")
        assignment.axis_takes["default"]["Align"]["cam1"] = other
        with pytest.raises(ValueError, match="same name"):
            execute_setup(config, assignment)
        assert not config.output_dir.exists()


# -- Setup from .slp files ---------------------------------------------------


@pytest.fixture
def slp_source(tmp_path: Path) -> dict:
    rig = tmp_path / "rig"
    videos = {(cam, t): _make_video(rig / f"{cam}_{t}.mp4") for cam in CAMERAS for t in ("Trial1", "Trial2")}
    slp_dir = tmp_path / "labels"
    slps = [_make_slp(slp_dir / f"{cam}.slp", [videos[(cam, t)] for t in ("Trial1", "Trial2")])
            for cam in CAMERAS]
    for cam in CAMERAS:
        write_image(slp_dir / f"Align_{cam}.png")
    return {"tmp": tmp_path, "slps": slps, "slp_dir": slp_dir, "rig": rig}


def slp_config(source: dict, **overrides) -> SetupConfig:
    defaults = dict(
        source_dir=source["tmp"], output_dir=source["tmp"] / "proj", camera_names=CAMERAS,
        calib_keyword="calib", match_strategy="keyword", trial_keyword="Trial",
        slp_paths=source["slps"], video_mode="reference", align_keywords=["Align"],
    )
    defaults.update(overrides)
    return SetupConfig(**defaults)


class TestSlpSetup:
    def test_found_beside_the_slp_files(self, slp_source):
        assignment = classify_slp_files(slp_config(slp_source))
        assert sorted(assignment.axis_takes["default"]["Align"]) == CAMERAS
        assert sorted(assignment.trials) == ["Trial1", "Trial2"]

    def test_found_with_a_calibration_toml(self, slp_source):
        calib = slp_source["tmp"] / "calibration.toml"
        calib.write_text("")
        assignment = classify_slp_files(slp_config(slp_source, calibration_toml=calib))
        assert sorted(assignment.axis_takes["default"]) == ["Align"]

    def test_calibration_dir_is_searched(self, slp_source):
        cal = slp_source["tmp"] / "cal"
        for cam in CAMERAS:
            write_image(cal / f"Axes_{cam}.png")
        assignment = classify_slp_files(slp_config(slp_source, calibration_dir=cal,
                                                   align_keywords=["Axes"]))
        assert sorted(assignment.axis_takes["default"]) == ["Axes"]

    def test_not_calibration_videos(self, slp_source):
        for cam in CAMERAS:
            _make_video(slp_source["slp_dir"] / f"{cam}_calib.mp4")
            _make_video(slp_source["slp_dir"] / f"Align_{cam}_calib.mp4")
        assignment = classify_slp_files(slp_config(slp_source))
        assert sorted(assignment.calibration_takes["default"]) == ["calib"]
        names = {p.name for take in assignment.calibration_takes["default"].values()
                 for p in take.values()}
        assert not any(n.startswith("Align") for n in names)
        assert any("both" in e for e in assignment.errors)

    def test_unreferenced_source_videos_skip_axis_files(self, slp_source):
        for cam in CAMERAS:
            _make_video(slp_source["tmp"] / f"Align_{cam}.mp4")
        config = slp_config(slp_source, include_source_videos=True, align_keywords=["Align"])
        assignment = classify_slp_files(config)
        assert sorted(assignment.trials) == ["Trial1", "Trial2"]

    def test_reference_mode_copies_axis_files(self, slp_source, qapp):
        config = slp_config(slp_source)
        out = execute_setup(config, classify_slp_files(config))
        for cam in CAMERAS:
            assert (out / "experiments/default/calibrations" / cam / f"Align_{cam}.png").is_file()
            assert (slp_source["slp_dir"] / f"Align_{cam}.png").is_file()   # copied, not moved
        df = pd.read_csv(out / "trials.csv")
        assert df.loc[df["type"] == "axis", "trial"].tolist() == ["Align"]


# -- Setup files ---------------------------------------------------------------


def test_setup_file_round_trip(tmp_path):
    path = tmp_path / "setup.toml"
    path.write_text(toml.dumps({"setup": {
        "source_dir": str(tmp_path), "output_dir": str(tmp_path / "proj"),
        "camera_names": CAMERAS, "align_keywords": ["Align"], "align_regex": "Ax\\d+",
    }}))
    config = config_from_toml(path)
    assert config.align_keywords == ["Align"]
    assert config.align_regex == "Ax\\d+"
    saved = tmp_path / "saved.toml"
    save_config_toml(config, saved)
    again = config_from_toml(saved)
    assert again.align_keywords == ["Align"]
    assert again.align_regex == "Ax\\d+"


# -- Axis takes from a frame of a take or trial ----------------------------------


def numbered_video(path: Path, n_frames: int = 5) -> Path:
    """A video whose frame i has every pixel near 40 * i, so the frame can be told."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), 40 * i, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def frames_folder(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    for cam in CAMERAS:
        for trial in ("Trial1", "Trial2"):
            numbered_video(src / f"{cam}_{trial}.mp4")
        numbered_video(src / f"{cam}_calib.mp4")
    return src


def frame_value(path: Path) -> float:
    return float(cv2.imread(str(path)).mean())


class TestAxisFrames:
    def test_frame_of_a_calibration_take(self, frames_folder, qapp):
        config = folder_config(frames_folder, align_keywords=[], axis_frames={"Align": ("calib", 2)})
        assignment = classify_files(config)
        axis_frame = assignment.axis_frames["default"]["Align"]
        assert (axis_frame.source, axis_frame.source_type, axis_frame.frame) == ("calib", "calibration", 2)
        assert not assignment.errors

        out = execute_setup(config, assignment)
        for cam in CAMERAS:
            still = out / "experiments/default/calibrations" / cam / f"Align_{cam}.png"
            assert frame_value(still) == pytest.approx(80, abs=4)
        df = pd.read_csv(out / "trials.csv")
        row = df[df["trial"] == "Align"].iloc[0]
        assert row["type"] == "axis"
        assert row["cam1"] == "Align_cam1.png"
        assert row["cam1_end"] == 1

        extracted = toml.load(out / "project.toml")["alignment"]["default"]["extracted"]["Align"]
        assert extracted == {"take": "calib", "type": "calibration", "frame": 3,
                             "videos": {"cam1": {"file": "cam1_calib.mp4", "frame": 3},
                                        "cam2": {"file": "cam2_calib.mp4", "frame": 3}}}
        project = Project.load(out)
        assert [t.name for t in project.axis_takes] == ["Align"]
        assert project.extracted_axis_takes()["Align"].frame == 2

    def test_frame_of_a_trial_before_it_is_moved(self, frames_folder, qapp):
        config = folder_config(frames_folder, align_keywords=[], axis_frames={"Align": ("Trial2", 0)})
        out = execute_setup(config, classify_files(config))
        assert not (frames_folder / "cam1_Trial2.mp4").exists()     # moved
        still = out / "experiments/default/calibrations/cam1/Align_cam1.png"
        assert frame_value(still) == pytest.approx(0, abs=4)
        project = Project.load(out)
        assert project.trial_names() == ["Trial1", "Trial2"]
        assert project.extracted_axis_takes()["Align"].type == "trial"

    def test_unknown_source_is_an_error(self, frames_folder):
        config = folder_config(frames_folder, align_keywords=[], axis_frames={"Align": ("Nope", 0)})
        assert any("no trial or calibration take is named 'Nope'" in e
                   for e in classify_files(config).errors)

    def test_missing_frame_is_an_error(self, frames_folder):
        config = folder_config(frames_folder, align_keywords=[], axis_frames={"Align": ("calib", 5)})
        errors = classify_files(config).errors
        assert any("no frame 6 for cam1, cam2" in e for e in errors)

    def test_name_used_is_an_error(self, frames_folder):
        config = folder_config(frames_folder, align_keywords=[], axis_frames={"Trial1": ("calib", 0)})
        assert any("same name" in e for e in classify_files(config).errors)

    def test_preview(self, frames_folder):
        config = folder_config(frames_folder, align_keywords=[], axis_frames={"Align": ("calib", 0)})
        text = preview_assignment(config, classify_files(config))
        assert "Align — frame 1 of calibration 'calib'  cam1: ok  cam2: ok" in text

    def test_bad_frame_number(self, frames_folder):
        with pytest.raises(ValueError, match="numbered from 1"):
            folder_config(frames_folder, axis_frames={"Align": ("calib", -1)})

    def test_from_a_trial_in_slp_setup(self, tmp_path, qapp):
        rig = tmp_path / "rig"
        videos = {(cam, t): numbered_video(rig / f"{cam}_{t}.mp4") for cam in CAMERAS
                  for t in ("Trial1", "Trial2")}
        slps = [_make_slp(tmp_path / "labels" / f"{cam}.slp",
                          [videos[(cam, t)] for t in ("Trial1", "Trial2")]) for cam in CAMERAS]
        config = SetupConfig(
            source_dir=tmp_path, output_dir=tmp_path / "proj", camera_names=CAMERAS,
            calib_keyword="calib", match_strategy="keyword", trial_keyword="Trial",
            slp_paths=slps, video_mode="reference", axis_frames={"Align": ("Trial1", 1)},
        )
        out = execute_setup(config, classify_slp_files(config))
        still = out / "experiments/default/calibrations/cam2/Align_cam2.png"
        assert frame_value(still) == pytest.approx(40, abs=4)
        assert videos[("cam2", "Trial1")].is_file()   # referenced in place, untouched


class TestAxisFramesSetupFile:
    def write(self, tmp_path, axis_frames) -> Path:
        path = tmp_path / "setup.toml"
        path.write_text(toml.dumps({"setup": {
            "source_dir": str(tmp_path), "output_dir": str(tmp_path / "proj"),
            "camera_names": CAMERAS, "axis_frames": axis_frames,
        }}))
        return path

    def test_frames_from_one_and_default(self, tmp_path):
        config = config_from_toml(self.write(tmp_path, {
            "Align": {"from": "PreCal", "frame": 3}, "Align2": {"from": "Trial1"},
        }))
        assert config.axis_frames == {"Align": ("PreCal", 2), "Align2": ("Trial1", 0)}

    def test_round_trip(self, tmp_path):
        config = config_from_toml(self.write(tmp_path, {"Align": {"from": "PreCal", "frame": 3}}))
        saved = tmp_path / "saved.toml"
        save_config_toml(config, saved)
        assert toml.load(saved)["setup"]["axis_frames"] == {"Align": {"from": "PreCal", "frame": 3}}
        assert config_from_toml(saved).axis_frames == {"Align": ("PreCal", 2)}

    def test_from_is_needed(self, tmp_path):
        with pytest.raises(ValueError, match="from"):
            config_from_toml(self.write(tmp_path, {"Align": {"frame": 2}}))


def test_extract_frame_errors(tmp_path):
    video = numbered_video(tmp_path / "v.mp4", 3)
    with pytest.raises(IOError, match="no frame 3"):
        extract_frame(video, 3, tmp_path / "out.png")
    out = extract_frame(video, 1, tmp_path / "sub" / "out.png")
    assert frame_value(out) == pytest.approx(40, abs=4)

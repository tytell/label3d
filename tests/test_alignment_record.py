"""The [alignment] record in project.toml, still-image takes and camera aliases (Phase 16.2)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import toml

from label3d import alignment as record
from label3d.alignment import Alignment, AlignmentSettings, AlignmentSource, alignment_status
from label3d.calibration_registry import CalibrationEntry
from label3d.geometry import compute_axis_alignment
from label3d.project import Project
from label3d.setup_project import SetupConfig, classify_files, execute_setup
from label3d.video import VideoReader, frame_count
from tests.helpers import make_current

CAMERAS = ["cam1", "cam2"]
WHEN = datetime(2026, 11, 3, 14, 15, 0)


def make_alignment(calibration: str = "charuco_a", angle: float = 0.3) -> Alignment:
    axes = {"x": np.array([np.cos(angle), np.sin(angle), 0.0]), "y": np.array([-np.sin(angle), np.cos(angle), 0.0])}
    result = compute_axis_alignment(np.array([1.0, 2.0, 3.0]), axes)
    return Alignment.from_result(result, calibration, max_error_px=0.8,
                                 scale_error_pct={"x": 0.4}, created=WHEN)


def write_image(path: Path, value: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((48, 64, 3), value, dtype=np.uint8))
    return path


def write_video(path: Path, n_frames: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


# -- Still images ------------------------------------------------------------


class TestStillImages:
    def test_reader_gives_one_frame(self, tmp_path):
        path = write_image(tmp_path / "a.png", 77)
        reader = VideoReader(path)
        assert reader.frame_count == 1
        assert reader.read_frame(0)[0, 0, 0] == 77
        assert reader.read_frame(1) is None
        reader.close()

    def test_unreadable_image_raises(self, tmp_path):
        (tmp_path / "bad.png").write_bytes(b"not an image")
        with pytest.raises(IOError):
            VideoReader(tmp_path / "bad.png")

    def test_frame_count(self, tmp_path):
        assert frame_count(write_image(tmp_path / "a.jpg")) == 1
        (tmp_path / "bad.png").write_bytes(b"")
        assert frame_count(tmp_path / "bad.png") == 0
        assert frame_count(write_video(tmp_path / "v.mp4", 5)) == 5


def _project_with_axis_take(root: Path) -> Path:
    """A project with one trial (offset camera 2) and an axis take of still images."""
    header = ["experiment", "trial", "type"]
    for cam in CAMERAS:
        header += [cam, f"{cam}_start", f"{cam}_end"]
    (root / "trials.csv").write_text(
        ",".join(header) + "\n"
        "exp1,trial1,trial,cam1_t1.mp4,,,cam2_t1.mp4,,\n"
        "exp1,Align,axis,Align_cam1.png,,,Align_cam2.png,,\n"
    )
    exp = root / "experiments" / "exp1"
    for cam in CAMERAS:
        write_video(exp / "videos" / f"{cam}_t1.mp4", 6)
        write_image(exp / "calibrations" / cam / f"Align_{cam}.png")
    return make_current(root)


class TestAxisTakeOnLoad:
    def test_image_take_measured_as_one_frame(self, tmp_path, qapp):
        root = _project_with_axis_take(tmp_path)
        Project.load(root)
        raw = pd.read_csv(root / "trials.csv")
        take = raw[raw["trial"] == "Align"].iloc[0]
        assert take["cam1_end"] == 1  # 1-based in the file
        assert take["cam2_end"] == 1

    def test_axis_take_is_not_a_charuco_take(self, tmp_path, qapp):
        project = Project.load(_project_with_axis_take(tmp_path))
        assert [t.name for t in project.axis_takes] == ["Align"]
        assert project.calibration_takes == []


# -- The record ------------------------------------------------------------


class TestRecord:
    def test_alignment_round_trip(self):
        a = make_alignment()
        b = Alignment.from_config(toml.loads(toml.dumps({"t": a.to_config()}))["t"])
        np.testing.assert_allclose(b.R_align, a.R_align)
        np.testing.assert_allclose(b.origin, a.origin)
        assert b.calibration == "charuco_a"
        assert b.created == WHEN
        assert b.primary == a.primary and b.derived == "z"
        assert b.max_error_px == pytest.approx(0.8)
        assert b.scale_error_pct == {"x": pytest.approx(0.4)}

    def test_R_is_stored_by_rows(self):
        a = make_alignment()
        np.testing.assert_allclose(a.to_config()["R"][0], a.R_align[0])

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            Alignment.from_config({"R": [[1.0, 0.0]], "origin": [0.0, 0.0, 0.0]})

    def test_source_frames_are_one_based_on_disk(self):
        source = AlignmentSource(take="trial1", type="trial", frame=0,
                                 videos={"cam1": ("c1.mp4", 0), "cam2": ("c2.mp4", 2)})
        table = source.to_config()
        assert table["frame"] == 1
        assert table["videos"]["cam2"] == {"file": "c2.mp4", "frame": 3}
        assert AlignmentSource.from_config(table) == source

    def test_settings_round_trip_and_defaults(self):
        assert AlignmentSettings.from_config({}) == AlignmentSettings()
        settings = AlignmentSettings(points_per_axis=2, cross_product="z", primary="x",
                                     flip=["y"], known_length_cm={"x": 10.0})
        assert AlignmentSettings.from_config(settings.to_config()) == settings

    @pytest.mark.parametrize("kwargs", [
        {"points_per_axis": 3}, {"cross_product": "w"}, {"primary": "shortest"},
        {"flip": ["q"]}, {"known_length_cm": {"w": 1.0}},
    ])
    def test_bad_settings_raise(self, kwargs):
        with pytest.raises(ValueError):
            AlignmentSettings(**kwargs)

    def test_new_alignment_moves_old_to_history(self):
        source = AlignmentSource(take="Align", type="axis", frame=0)
        config: dict = {}
        config["alignment"] = record.with_alignment(config, "exp1", make_alignment("a"), source)
        config["alignment"] = record.with_alignment(config, "exp1", make_alignment("b", 0.5), source)
        assert record.read_alignment(config, "exp1").calibration == "b"
        assert [h.calibration for h in record.read_history(config, "exp1")] == ["a"]
        assert record.read_alignment(config, "exp2") is None

    def test_settings_keep_the_transform(self):
        config: dict = {}
        source = AlignmentSource(take="Align", type="axis", frame=0)
        config["alignment"] = record.with_alignment(config, "exp1", make_alignment(), source)
        config["alignment"] = record.with_settings(config, "exp1", AlignmentSettings(primary="y"))
        assert record.read_alignment(config, "exp1") is not None
        assert record.read_settings(config, "exp1").primary == "y"


class TestStatus:
    ENTRIES = {
        "charuco_a": ("charuco", ""),
        "refine_1": ("refine", "charuco_a"),
        "refine_2": ("refine", "refine_1"),
        "charuco_b": ("charuco", ""),
        "refine_b": ("refine", "charuco_b"),
    }

    def find(self, entry_id):
        if entry_id not in self.ENTRIES:
            return None
        method, parent = self.ENTRIES[entry_id]
        return CalibrationEntry(id=entry_id, label=entry_id, created=WHEN, method=method,
                                parent=parent, path=Path(f"{entry_id}.toml"), metadata={})

    @pytest.mark.parametrize("points_from, expected", [
        ("charuco_a", record.CURRENT),
        ("refine_1", record.CURRENT),
        ("refine_2", record.CURRENT),
        ("charuco_b", record.STALE),
        ("refine_b", record.STALE),
        ("deleted", record.STALE),
        (None, record.STALE),
    ])
    def test_status(self, points_from, expected):
        assert alignment_status(make_alignment("charuco_a"), points_from, self.find) == expected

    def test_none_without_alignment(self):
        assert alignment_status(None, "charuco_a", self.find) == record.NONE


# -- Through Project -------------------------------------------------------


class TestProject:
    def test_save_and_read(self, tmp_path, qapp):
        root = _project_with_axis_take(tmp_path)
        project = Project.load(root)
        source = project.alignment_source_for("Align", 0)
        project.save_alignment(make_alignment("charuco_a"), source)
        project.save_alignment_settings(AlignmentSettings(points_per_axis=2))

        reloaded = Project.load(root)
        assert reloaded.alignment().calibration == "charuco_a"
        assert reloaded.alignment_source() == source
        assert reloaded.alignment_settings().points_per_axis == 2
        assert reloaded.alignment_history() == []
        on_disk = toml.load(root / "project.toml")["alignment"]["exp1"]["source"]
        assert on_disk == {"take": "Align", "type": "axis", "frame": 1,
                           "videos": {"cam1": {"file": "Align_cam1.png", "frame": 1},
                                      "cam2": {"file": "Align_cam2.png", "frame": 1}}}

    def test_source_for_trial_frame_adds_offsets(self, tmp_path, qapp):
        root = _project_with_axis_take(tmp_path)
        df = pd.read_csv(root / "trials.csv")
        df.loc[df["trial"] == "trial1", "cam2_start"] = 3   # 1-based: video frame 2
        df.to_csv(root / "trials.csv", index=False)
        project = Project.load(root)
        source = project.alignment_source_for("trial1", 1)
        assert source.type == "trial"
        assert source.videos == {"cam1": ("cam1_t1.mp4", 1), "cam2": ("cam2_t1.mp4", 3)}

    def test_status_follows_triangulated_with(self, tmp_path, qapp):
        project = Project.load(_project_with_axis_take(tmp_path))
        assert project.alignment_status() == record.NONE
        project.save_alignment(make_alignment("charuco_a"), project.alignment_source_for("Align", 0))
        project.set_triangulated_with(None, "charuco_a")
        assert project.alignment_status() == record.CURRENT
        project.set_triangulated_with(None, "charuco_b")
        assert project.alignment_status() == record.STALE

    def test_camera_aliases_from_setup(self, tmp_path, qapp):
        src = tmp_path / "src"
        src.mkdir()
        for name in ("L2_trial001.mp4", "right_trial001.mp4"):
            write_video(src / name, 3)
        config = SetupConfig(
            source_dir=src, output_dir=tmp_path / "p", camera_names=["left", "right"],
            calib_keyword="calib", match_strategy="number",
            camera_aliases={"left": ["L2", "Lft"]},
        )
        execute_setup(config, classify_files(config))
        assert toml.load(tmp_path / "p" / "project.toml")["camera_aliases"] == {"left": ["L2", "Lft"]}
        assert Project.load(tmp_path / "p").camera_aliases == {"left": ["L2", "Lft"]}

    def test_no_aliases_no_table(self, tmp_path, qapp):
        project = Project.load(_project_with_axis_take(tmp_path))
        assert project.camera_aliases == {}
        assert "camera_aliases" not in toml.load(tmp_path / "project.toml")

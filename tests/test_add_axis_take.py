"""Adding an axis take after setup (Phase 16.4): files per camera, or a frame of a take or trial."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import toml

from label3d import axis_takes
from label3d.axis_takes import AxisTakeError
from label3d.project import Project
from label3d.widgets.add_axis_take_dialog import FRAME_TAB, AddAxisTakeDialog
from tests.helpers import make_current

CAMERAS = ["cam1", "cam2"]


def numbered_video(path: Path, n_frames: int = 6) -> Path:
    """Frame i has every pixel near 40 * i."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), 40 * i, dtype=np.uint8))
    writer.release()
    return path


def write_image(path: Path, value: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((32, 32, 3), value, dtype=np.uint8))
    return path


def mean(path: Path) -> float:
    return float(cv2.imread(str(path)).mean())


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """One trial (cam2 starts 2 frames in) and a calibration take, all 0-based here."""
    root = tmp_path / "proj"
    exp = root / "experiments" / "exp1"
    header = ["experiment", "trial", "type"]
    for cam in CAMERAS:
        header += [cam, f"{cam}_start", f"{cam}_end"]
    root.mkdir()
    (root / "trials.csv").write_text(
        ",".join(header) + "\n"
        "exp1,trial1,trial,cam1_t1.mp4,0,,cam2_t1.mp4,2,\n"
        "exp1,PreCal,calibration,cam1_pre.mp4,0,,cam2_pre.mp4,0,\n"
    )
    for cam in CAMERAS:
        numbered_video(exp / "videos" / f"{cam}_t1.mp4")
        numbered_video(exp / "calibrations" / cam / f"{cam}_pre.mp4")
    return make_current(root)


@pytest.fixture
def project(project_root, qapp) -> Project:
    return Project.load(project_root)


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    folder = tmp_path / "outside"
    for cam in CAMERAS:
        write_image(folder / f"PS18_Align_{cam}.png")
    return folder


class TestMatching:
    def test_by_camera_with_take_name(self, outside):
        match = axis_takes.match_files(axis_takes.media_files(outside), CAMERAS)
        assert {cam: f.name for cam, f in match.by_camera.items()} == {
            "cam1": "PS18_Align_cam1.png", "cam2": "PS18_Align_cam2.png"}
        assert match.take == "PS18_Align"

    def test_aliases_and_extra_files(self, tmp_path):
        files = [tmp_path / n for n in ("Ax_left.png", "Ax_cam2.png", "Ax_cam2_b.png", "notes.png")]
        match = axis_takes.match_files(files, CAMERAS, {"cam1": ["left"]})
        assert match.by_camera["cam1"].name == "Ax_left.png"
        assert {f.name for f in match.unmatched} == {"Ax_cam2_b.png", "notes.png"}
        assert match.take == "Ax"

    def test_nothing_shared_gives_default(self, tmp_path):
        match = axis_takes.match_files([tmp_path / "a_cam1.png", tmp_path / "b_cam2.png"], CAMERAS)
        assert match.take == "Align"

    def test_unique_name(self, project):
        assert axis_takes.unique_take_name(project, "PreCal") == "PreCal2"
        assert axis_takes.unique_take_name(project) == "Align"

    @pytest.mark.parametrize("name", ["", " Align", "a/b", "trial1", "PreCal"])
    def test_bad_names(self, project, name):
        with pytest.raises(AxisTakeError):
            axis_takes.check_take_name(project, name)


class TestFromFiles:
    def test_copies_and_adds_a_row(self, project, outside, project_root):
        files = axis_takes.match_files(axis_takes.media_files(outside), CAMERAS).by_camera
        axis_takes.add_from_files(project, "Align", files)
        for cam in CAMERAS:
            copied = project_root / "experiments/exp1/calibrations" / cam / f"PS18_Align_{cam}.png"
            assert copied.is_file()
            assert (outside / f"PS18_Align_{cam}.png").is_file()   # copied, not moved
        raw = pd.read_csv(project_root / "trials.csv")
        row = raw[raw["trial"] == "Align"].iloc[0]
        assert (row["type"], row["cam1"], row["cam1_start"], row["cam1_end"]) == (
            "axis", "PS18_Align_cam1.png", 1, 1)
        assert (project_root / "trials.csv.bak").is_file()
        assert [t.name for t in project.axis_takes] == ["Align"]
        assert [t.name for t in Project.load(project_root).axis_takes] == ["Align"]

    def test_same_file_already_there_is_fine(self, project, outside, project_root):
        dst = project_root / "experiments/exp1/calibrations/cam1/PS18_Align_cam1.png"
        dst.write_bytes((outside / "PS18_Align_cam1.png").read_bytes())
        files = {cam: outside / f"PS18_Align_{cam}.png" for cam in CAMERAS}
        axis_takes.add_from_files(project, "Align", files)
        assert [t.name for t in project.axis_takes] == ["Align"]

    def test_different_file_of_that_name_changes_nothing(self, project, outside, project_root):
        write_image(project_root / "experiments/exp1/calibrations/cam2/PS18_Align_cam2.png", 7)
        before = (project_root / "trials.csv").read_text()
        files = {cam: outside / f"PS18_Align_{cam}.png" for cam in CAMERAS}
        with pytest.raises(AxisTakeError, match="different"):
            axis_takes.add_from_files(project, "Align", files)
        assert (project_root / "trials.csv").read_text() == before
        assert not (project_root / "experiments/exp1/calibrations/cam1/PS18_Align_cam1.png").exists()

    def test_one_camera_is_refused(self, project, outside):
        with pytest.raises(AxisTakeError, match="two cameras"):
            axis_takes.add_from_files(project, "Align", {"cam1": outside / "PS18_Align_cam1.png"})

    def test_trials_csv_without_type_column(self, project, outside, project_root):
        raw = pd.read_csv(project_root / "trials.csv")
        raw[raw["type"] == "trial"].drop(columns="type").to_csv(project_root / "trials.csv", index=False)
        project = Project.load(project_root)
        axis_takes.add_from_files(project, "Align", {cam: outside / f"PS18_Align_{cam}.png" for cam in CAMERAS})
        raw = pd.read_csv(project_root / "trials.csv")
        assert raw.columns.tolist()[:3] == ["experiment", "trial", "type"]
        assert raw["type"].tolist() == ["trial", "axis"]


class TestFromFrame:
    def test_frame_of_a_trial_uses_offsets(self, project, project_root):
        record = axis_takes.add_from_frame(project, "Align", "trial1", 1)
        calib = project_root / "experiments/exp1/calibrations"
        assert mean(calib / "cam1/Align_cam1.png") == pytest.approx(40, abs=4)
        assert mean(calib / "cam2/Align_cam2.png") == pytest.approx(120, abs=4)
        assert record.videos == {"cam1": ("cam1_t1.mp4", 1), "cam2": ("cam2_t1.mp4", 3)}

        on_disk = toml.load(project_root / "project.toml")["alignment"]["exp1"]["extracted"]["Align"]
        assert on_disk["frame"] == 2 and on_disk["type"] == "trial"
        assert on_disk["videos"]["cam2"] == {"file": "cam2_t1.mp4", "frame": 4}
        reloaded = Project.load(project_root)
        assert [t.name for t in reloaded.axis_takes] == ["Align"]
        assert reloaded.extracted_axis_takes()["Align"] == record

    def test_frame_of_a_calibration_take(self, project, project_root):
        axis_takes.add_from_frame(project, "Align", "PreCal", 0)
        assert mean(project_root / "experiments/exp1/calibrations/cam2/Align_cam2.png") == pytest.approx(0, abs=4)
        assert project.extracted_axis_takes()["Align"].type == "calibration"

    def test_missing_frame_writes_nothing(self, project, project_root):
        before = (project_root / "trials.csv").read_text()
        with pytest.raises(AxisTakeError, match="no frame 5 for cam2"):
            axis_takes.add_from_frame(project, "Align", "trial1", 4)   # cam2 runs out at 4 + 2
        assert (project_root / "trials.csv").read_text() == before
        assert not list((project_root / "experiments/exp1/calibrations").rglob("Align_*.png"))

    def test_unknown_source(self, project):
        with pytest.raises(AxisTakeError, match="no calibration take or trial 'Nope'"):
            axis_takes.add_from_frame(project, "Align", "Nope", 0)

    def test_sources_and_lengths(self, project):
        assert axis_takes.frame_sources(project) == [("PreCal", "calibration"), ("trial1", "trial")]
        assert axis_takes.source_length(project, "PreCal") == 6
        assert axis_takes.source_length(project, "trial1") == 4


class TestDialog:
    def test_files(self, project, outside):
        dialog = AddAxisTakeDialog(project)
        assert not dialog._add_button.isEnabled()
        dialog.set_files(axis_takes.media_files(outside))
        assert {cam: f.name for cam, f in dialog.chosen_files().items()} == {
            "cam1": "PS18_Align_cam1.png", "cam2": "PS18_Align_cam2.png"}
        assert dialog._name.text() == "PS18_Align"
        assert dialog._add_button.isEnabled()
        dialog._on_add()
        assert dialog.take() == "PS18_Align"
        assert [t.name for t in project.axis_takes] == ["PS18_Align"]

    def test_name_taken_disables_add(self, project, outside):
        dialog = AddAxisTakeDialog(project)
        dialog.set_files(axis_takes.media_files(outside))
        dialog._name.setText("trial1")
        dialog._on_name_edited("trial1")
        assert not dialog._add_button.isEnabled()
        assert "already has" in dialog._message.text()

    def test_frame(self, project, project_root):
        dialog = AddAxisTakeDialog(project)
        dialog._tabs.setCurrentIndex(FRAME_TAB)
        assert dialog._source.currentData() == "PreCal"
        assert dialog._frame.maximum() == 6
        dialog._source.setCurrentIndex(1)
        assert dialog._frame.maximum() == 4
        dialog._frame.setValue(2)
        dialog._on_add()
        assert dialog.take() == "Align"
        assert project.extracted_axis_takes()["Align"].frame == 1

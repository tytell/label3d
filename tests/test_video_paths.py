"""Resolution of trials.csv camera cells to video paths.

Covers the three forms accepted by _resolve_video: absolute cells, cells
resolved against an experiment's [video_roots] entry, and the classic bare
filename under experiments/{exp}/videos/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import toml

from label3d.project import Project, _resolve_video


def _write_project(
    root: Path,
    cells: dict[str, str],
    video_roots: dict[str, str] | None = None,
) -> Path:
    """Write a minimal one-trial project; cells maps camera name -> csv cell."""
    cameras = sorted(cells)
    (root / "experiments" / "default" / "videos").mkdir(parents=True)

    header = ["experiment", "trial", "is_calibration"]
    row = ["default", "trial1", "False"]
    for cam in cameras:
        header += [cam, f"{cam}_start", f"{cam}_end"]
        row += [cells[cam], "0", "10"]
    (root / "trials.csv").write_text(",".join(header) + "\n" + ",".join(row) + "\n")

    config: dict = {"project": {"name": "test"}}
    if video_roots:
        config["video_roots"] = video_roots
    (root / "project.toml").write_text(toml.dumps(config))
    return root


class TestResolveVideo:
    """The helper itself, independent of a project on disk."""

    def test_absolute_cell_used_as_is(self):
        cell = "/data/rig/trial1_cam1.mp4"
        path = _resolve_video(cell, None, Path("/proj/videos"), Path("/proj"))
        assert path == Path(cell)

    def test_absolute_cell_ignores_video_root(self):
        path = _resolve_video(
            "/data/trial1.mp4", Path("/elsewhere"), Path("/proj/videos"), Path("/proj")
        )
        assert path == Path("/data/trial1.mp4")

    def test_bare_filename_with_video_root(self):
        path = _resolve_video(
            "trial1_cam1.mp4", Path("/data/rig"), Path("/proj/videos"), Path("/proj")
        )
        assert path == Path("/data/rig/trial1_cam1.mp4")

    def test_relative_path_with_video_root(self):
        path = _resolve_video(
            "cam1/trial1.mp4", Path("/data/rig"), Path("/proj/videos"), Path("/proj")
        )
        assert path == Path("/data/rig/cam1/trial1.mp4")

    def test_relative_path_without_video_root_uses_project_root(self):
        path = _resolve_video(
            "raw/cam1/trial1.mp4", None, Path("/proj/videos"), Path("/proj")
        )
        assert path == Path("/proj/raw/cam1/trial1.mp4")

    def test_bare_filename_without_video_root_uses_default_dir(self):
        """The classic layout — unchanged for projects written before this."""
        path = _resolve_video("trial1.mp4", None, Path("/proj/videos"), Path("/proj"))
        assert path == Path("/proj/videos/trial1.mp4")


class TestProjectVideoPath:
    """End-to-end through Project.load() and Project.video_path()."""

    def test_classic_layout_unaffected(self, tmp_path):
        root = _write_project(tmp_path, {"cam1": "trial1_cam1.mp4"})
        (root / "experiments/default/videos/trial1_cam1.mp4").write_bytes(b"")
        proj = Project.load(root)
        assert proj.video_path("trial1", "cam1") == (
            root / "experiments/default/videos/trial1_cam1.mp4"
        )
        assert proj.experiments["default"].video_root is None

    def test_video_root_resolution(self, tmp_path):
        external = tmp_path / "external"
        external.mkdir()
        (external / "trial1_cam1.mp4").write_bytes(b"")
        root = _write_project(
            tmp_path / "proj",
            {"cam1": "trial1_cam1.mp4"},
            video_roots={"default": str(external)},
        )
        proj = Project.load(root)
        assert proj.video_path("trial1", "cam1") == external / "trial1_cam1.mp4"
        assert proj.experiments["default"].video_root == external

    def test_absolute_cell_resolution(self, tmp_path):
        external = tmp_path / "external"
        external.mkdir()
        video = external / "trial1_cam1.mp4"
        video.write_bytes(b"")
        root = _write_project(tmp_path / "proj", {"cam1": str(video)})
        proj = Project.load(root)
        assert proj.video_path("trial1", "cam1") == video

    def test_missing_external_video_raises_at_load(self, tmp_path):
        root = _write_project(
            tmp_path / "proj",
            {"cam1": "gone.mp4"},
            video_roots={"default": str(tmp_path / "external")},
        )
        with pytest.raises(FileNotFoundError, match="Referenced videos not found"):
            Project.load(root)

    def test_missing_external_video_names_trial_and_camera(self, tmp_path):
        root = _write_project(
            tmp_path / "proj",
            {"cam1": "gone.mp4"},
            video_roots={"default": str(tmp_path / "external")},
        )
        with pytest.raises(FileNotFoundError) as excinfo:
            Project.load(root)
        message = str(excinfo.value)
        assert "trial1" in message and "cam1" in message and "gone.mp4" in message

    def test_missing_classic_video_still_tolerated(self, tmp_path):
        """A camera that did not record leaves a bare cell with no file."""
        root = _write_project(tmp_path, {"cam1": "never_recorded.mp4"})
        proj = Project.load(root)
        assert proj.trial_names() == ["trial1"]

    def test_blank_cell_is_not_a_missing_reference(self, tmp_path):
        external = tmp_path / "external"
        external.mkdir()
        (external / "trial1_cam1.mp4").write_bytes(b"")
        root = _write_project(
            tmp_path / "proj",
            {"cam1": "trial1_cam1.mp4", "cam2": ""},
            video_roots={"default": str(external)},
        )
        proj = Project.load(root)
        assert proj.video_path("trial1", "cam1") == external / "trial1_cam1.mp4"

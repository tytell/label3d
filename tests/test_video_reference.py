"""Resolution of video paths recorded inside .slp files.

A .slp records the video path as it was when labeling happened, typically on
another machine. _resolve_video_reference re-homes such a path onto a search
directory while keeping as much of the recorded subfolder structure as it can.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from label3d.setup_project import (
    _recorded_parts,
    _resolve_video_reference,
)

RECORDED = Path(
    "R:/NewData/VerticalOMR/3_edited_fish_videos/"
    "20260619_zebrafish_1_synced_trimmed/left/"
    "20260619-trial01-left-GX010451_reencoded.mp4"
)
SUBPATH = Path(
    "20260619_zebrafish_1_synced_trimmed/left/"
    "20260619-trial01-left-GX010451_reencoded.mp4"
)


def _make(root: Path, relative: Path | str) -> Path:
    """Create an empty file at root/relative, parents included."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class TestRecordedParts:
    def test_drive_letter_is_dropped(self):
        assert _recorded_parts("R:/NewData/x.mp4") == ["NewData", "x.mp4"]

    def test_backslashes_split(self):
        assert _recorded_parts(r"R:\NewData\rig\x.mp4") == ["NewData", "rig", "x.mp4"]

    def test_mixed_separators_split(self):
        assert _recorded_parts(r"C:\data/rig\x.mp4") == ["data", "rig", "x.mp4"]

    def test_posix_path_unchanged(self):
        assert _recorded_parts("/data/rig/x.mp4") == ["data", "rig", "x.mp4"]


class TestAnchor:
    def test_search_dir_name_anchors_the_recorded_path(self, tmp_path):
        """The motivating case: same collection folder, different mount."""
        mount = tmp_path / "Volumes/TytellLab$/NewData/VerticalOMR/3_edited_fish_videos"
        video = _make(mount, SUBPATH)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [mount])
        assert outcome.path == video.resolve()
        assert outcome.rule == "anchor"
        assert outcome.old_root == "NewData/VerticalOMR/3_edited_fish_videos"
        assert outcome.new_root == mount

    def test_anchor_is_case_sensitive(self, tmp_path):
        """An exact component match only — a differently cased twin is not it."""
        mount = tmp_path / "3_EDITED_FISH_VIDEOS"
        _make(mount, SUBPATH)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [mount])
        assert outcome.rule == "suffix"  # found, but by structure rather than name

    def test_deepest_occurrence_of_a_repeated_name_wins(self, tmp_path):
        recorded = Path("R:/videos/rig/videos/left/t1.mp4")
        mount = tmp_path / "mnt/videos"
        video = _make(mount, "left/t1.mp4")
        outcome = _resolve_video_reference(recorded, tmp_path / "slp", [mount])
        assert outcome.path == video.resolve()
        assert outcome.rule == "anchor"

    def test_anchor_beats_a_basename_sitting_in_another_dir(self, tmp_path):
        """A loose copy of the file must not win over the structural match."""
        flat = tmp_path / "flat"
        _make(flat, RECORDED.name)
        mount = tmp_path / "3_edited_fish_videos"
        video = _make(mount, SUBPATH)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [flat, mount])
        assert outcome.path == video.resolve()
        assert outcome.rule == "anchor"

    def test_filename_component_cannot_anchor(self, tmp_path):
        """Only directories anchor, so a search dir named like the file does not."""
        mount = tmp_path / RECORDED.name
        mount.mkdir()
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [mount])
        assert outcome.path is None


class TestSuffix:
    def test_renamed_collection_folder_still_matches_by_structure(self, tmp_path):
        mount = tmp_path / "edited_videos"
        video = _make(mount, SUBPATH)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [mount])
        assert outcome.path == video.resolve()
        assert outcome.rule == "suffix"
        assert outcome.old_root == "NewData/VerticalOMR/3_edited_fish_videos"

    def test_deepest_suffix_wins_over_a_shallower_one(self, tmp_path):
        """A same-named file in the wrong camera folder must not be chosen."""
        mount = tmp_path / "mnt"
        _make(mount, Path("right") / RECORDED.name)
        video = _make(mount, SUBPATH)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [mount])
        assert outcome.path == video.resolve()


class TestFallbacks:
    def test_recorded_path_used_when_it_exists(self, tmp_path):
        video = _make(tmp_path, "rig/t1.mp4")
        outcome = _resolve_video_reference(video, tmp_path, [])
        assert outcome.path == video.resolve()
        assert outcome.rule == "recorded"

    def test_basename_beside_the_slp(self, tmp_path):
        slp_dir = tmp_path / "labels"
        video = _make(slp_dir, RECORDED.name)
        outcome = _resolve_video_reference(RECORDED, slp_dir, [])
        assert outcome.path == video.resolve()
        assert outcome.rule == "beside_slp"

    def test_basename_in_a_search_dir(self, tmp_path):
        mount = tmp_path / "flat"
        video = _make(mount, RECORDED.name)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [mount])
        assert outcome.path == video.resolve()
        assert outcome.rule == "basename"

    def test_unfound_reference_returns_none(self, tmp_path):
        outcome = _resolve_video_reference(RECORDED, tmp_path, [tmp_path / "nope"])
        assert outcome.path is None
        assert outcome.rule == "unresolved"


class TestAmbiguity:
    def test_two_anchored_matches_refuse(self, tmp_path):
        first = tmp_path / "a/3_edited_fish_videos"
        second = tmp_path / "b/3_edited_fish_videos"
        _make(first, SUBPATH)
        _make(second, SUBPATH)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [first, second])
        assert outcome.path is None
        assert outcome.rule == "ambiguous"
        assert len(outcome.candidates) == 2

    def test_two_suffix_matches_refuse(self, tmp_path):
        first = tmp_path / "mount_a"
        second = tmp_path / "mount_b"
        _make(first, SUBPATH)
        _make(second, SUBPATH)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [first, second])
        assert outcome.rule == "ambiguous"

    def test_same_file_via_two_dirs_is_not_ambiguous(self, tmp_path):
        """Two search dirs naming one file on disk is a single candidate."""
        mount = tmp_path / "3_edited_fish_videos"
        video = _make(mount, SUBPATH)
        link = tmp_path / "alias"
        link.symlink_to(mount)
        outcome = _resolve_video_reference(RECORDED, tmp_path / "slp", [mount, link])
        assert outcome.path == video.resolve()

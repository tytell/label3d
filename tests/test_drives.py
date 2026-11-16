"""Named drives in project.toml ([drives.{system}]) and {drive} video roots."""

from __future__ import annotations

import pytest

from label3d.drives import (
    DriveError,
    convert_video_roots,
    resolve_root,
    to_drive_form,
)

UNC = "\\\\rstore.it.tufts.edu\\TytellLab$"
CONFIG = {
    "drives": {
        "mac": {"tytelllab": "/Volumes/TytellLab$"},
        "windows": {"tytelllab": UNC},
        "linux": {"tytelllab": "/mnt/tytelllab"},
    }
}


class TestResolveRoot:
    def test_mac(self):
        value = "{tytelllab}/NewData/VerticalOMR"
        assert resolve_root(value, CONFIG, "mac") == "/Volumes/TytellLab$/NewData/VerticalOMR"

    def test_windows_unc(self):
        value = "{tytelllab}/NewData/VerticalOMR"
        assert resolve_root(value, CONFIG, "windows") == UNC + "\\NewData\\VerticalOMR"

    def test_backslash_rest(self):
        value = "{tytelllab}\\NewData\\VerticalOMR"
        assert resolve_root(value, CONFIG, "linux") == "/mnt/tytelllab/NewData/VerticalOMR"

    def test_drive_alone(self):
        assert resolve_root("{tytelllab}", CONFIG, "mac") == "/Volumes/TytellLab$"

    def test_plain_path_unchanged(self):
        assert resolve_root("/data/rig", CONFIG, "mac") == "/data/rig"

    def test_unknown_drive_raises(self):
        with pytest.raises(DriveError, match=r"\[drives\.mac\]"):
            resolve_root("{other}/x", CONFIG, "mac")

    def test_no_drives_table_raises(self):
        with pytest.raises(DriveError):
            resolve_root("{tytelllab}/x", {}, "windows")


class TestToDriveForm:
    def test_mac_path(self):
        path = "/Volumes/TytellLab$/NewData/VerticalOMR"
        assert to_drive_form(path, CONFIG, "mac") == "{tytelllab}/NewData/VerticalOMR"

    def test_unc_path_on_windows_ignores_case(self):
        path = "\\\\RSTORE.it.tufts.edu\\tytelllab$\\NewData\\VerticalOMR"
        assert to_drive_form(path, CONFIG, "windows") == "{tytelllab}/NewData/VerticalOMR"

    def test_unc_path_converted_on_mac(self):
        """A root written on Windows converts when the project opens on a Mac."""
        path = UNC + "\\NewData\\VerticalOMR\\3_edited_fish_videos"
        assert (
            to_drive_form(path, CONFIG, "mac")
            == "{tytelllab}/NewData/VerticalOMR/3_edited_fish_videos"
        )

    def test_linux_is_case_sensitive(self):
        config = {"drives": {"linux": CONFIG["drives"]["linux"]}}
        assert to_drive_form("/mnt/TYTELLLAB/x", config, "linux") is None

    def test_prefix_of_a_name_is_not_a_match(self):
        assert to_drive_form("/Volumes/TytellLab$2/x", CONFIG, "mac") is None

    def test_not_under_a_drive(self):
        assert to_drive_form("/Users/me/videos", CONFIG, "mac") is None

    def test_already_in_drive_form(self):
        assert to_drive_form("{tytelllab}/x", CONFIG, "mac") is None

    def test_longest_root_wins(self):
        config = {"drives": {"mac": {"share": "/Volumes/S", "data": "/Volumes/S/data"}}}
        assert to_drive_form("/Volumes/S/data/rig", config, "mac") == "{data}/rig"


class TestConvertVideoRoots:
    def test_converts_in_place(self):
        config = {
            **CONFIG,
            "video_roots": {
                "a": UNC + "\\NewData\\a",
                "b": "/Users/me/b",
                "c": "{tytelllab}/c",
            },
        }
        assert convert_video_roots(config, "mac")
        assert config["video_roots"] == {
            "a": "{tytelllab}/NewData/a",
            "b": "/Users/me/b",
            "c": "{tytelllab}/c",
        }

    def test_nothing_to_convert(self):
        config = {**CONFIG, "video_roots": {"b": "/Users/me/b"}}
        assert not convert_video_roots(config, "mac")

    def test_no_drives(self):
        config = {"video_roots": {"a": "/Volumes/TytellLab$/a"}}
        assert not convert_video_roots(config, "mac")

"""OS-level path robustness tests for PointsStore.

These tests verify that PointsStore handles:
- Paths with spaces in directory or file names
- Unicode characters in paths
- Forward-slash vs back-slash path separators
- Backup file creation alongside paths that contain spaces
- Informative errors for missing files / directories
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import pytest

from label3d.points import PointsStore
from tests.helpers import BODYPARTS, CAMERAS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store_with_point(cameras=None, bodyparts=None) -> PointsStore:
    cameras = cameras or CAMERAS
    bodyparts = bodyparts or BODYPARTS
    store = PointsStore.create_empty(cameras, bodyparts)
    store.set_point("trial1", 0, 0, "snout", cameras[0], 10.0, 20.0)
    return store


# ---------------------------------------------------------------------------
# Spaces in paths
# ---------------------------------------------------------------------------

class TestPathsWithSpaces:
    def test_save_to_dir_with_spaces(self, qapp, tmp_path):
        spaced_dir = tmp_path / "my project data"
        spaced_dir.mkdir()
        dest = spaced_dir / "points.csv"
        store = make_store_with_point()
        store.to_csv(dest)
        assert dest.exists()

    def test_load_from_dir_with_spaces(self, qapp, tmp_path):
        spaced_dir = tmp_path / "my project data"
        spaced_dir.mkdir()
        dest = spaced_dir / "points.csv"
        store = make_store_with_point()
        store.to_csv(dest)
        reloaded = PointsStore.from_csv(dest)
        pt = reloaded.get_point("trial1", 0, 0, "snout", CAMERAS[0])
        assert abs(pt.x - 10.0) < 1e-6

    def test_save_filename_with_spaces(self, qapp, tmp_path):
        dest = tmp_path / "my points file.csv"
        store = make_store_with_point()
        store.to_csv(dest)
        assert dest.exists()

    def test_load_filename_with_spaces(self, qapp, tmp_path):
        dest = tmp_path / "my points file.csv"
        store = make_store_with_point()
        store.to_csv(dest)
        reloaded = PointsStore.from_csv(dest)
        assert len(reloaded.df) > 0

    def test_backup_path_with_spaces(self, qapp, tmp_path):
        spaced_dir = tmp_path / "project dir with spaces"
        spaced_dir.mkdir()
        dest = spaced_dir / "points.csv"
        store = make_store_with_point()
        store.to_csv(dest)
        # Mutate and trigger backup
        store.set_point("trial1", 0, 0, "snout", CAMERAS[0], 99.0, 99.0)
        store.save_if_dirty()
        bak = spaced_dir / "points.csv.bak"
        assert bak.exists()

    def test_nested_dirs_with_spaces(self, qapp, tmp_path):
        nested = tmp_path / "project dir" / "session 01" / "labels"
        nested.mkdir(parents=True)
        dest = nested / "points.csv"
        store = make_store_with_point()
        store.to_csv(dest)
        reloaded = PointsStore.from_csv(dest)
        assert len(reloaded.df) == len(store.df)


# ---------------------------------------------------------------------------
# Unicode characters in paths
# ---------------------------------------------------------------------------

class TestUnicodePaths:
    def test_save_unicode_dir(self, qapp, tmp_path):
        uni_dir = tmp_path / "données"
        uni_dir.mkdir()
        dest = uni_dir / "points.csv"
        store = make_store_with_point()
        store.to_csv(dest)
        assert dest.exists()

    def test_roundtrip_unicode_dir(self, qapp, tmp_path):
        uni_dir = tmp_path / "données"
        uni_dir.mkdir()
        dest = uni_dir / "points.csv"
        store = make_store_with_point()
        store.to_csv(dest)
        reloaded = PointsStore.from_csv(dest)
        pt = reloaded.get_point("trial1", 0, 0, "snout", CAMERAS[0])
        assert abs(pt.x - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# Path separator handling
# ---------------------------------------------------------------------------

class TestPathSeparators:
    def test_posix_path_object(self, qapp, tmp_path):
        """Path objects (which use the OS separator) work correctly."""
        dest = tmp_path / "sub" / "points.csv"
        dest.parent.mkdir()
        store = make_store_with_point()
        store.to_csv(dest)
        reloaded = PointsStore.from_csv(dest)
        assert len(reloaded.df) > 0

    def test_string_path_with_forward_slashes(self, qapp, tmp_path):
        """Explicit forward slashes in string paths — Path() normalises them."""
        sub = tmp_path / "sub"
        sub.mkdir()
        dest = Path(str(sub) + "/points.csv")
        store = make_store_with_point()
        store.to_csv(dest)
        assert dest.exists()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_string_path_with_backslashes_windows(self, qapp, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        dest = Path(str(sub) + "\\points.csv")
        store = make_store_with_point()
        store.to_csv(dest)
        assert dest.exists()


# ---------------------------------------------------------------------------
# Error handling for missing files / directories
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_from_csv_missing_file_raises_file_not_found(self, qapp, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError) as exc_info:
            PointsStore.from_csv(missing)
        # Error message should mention the path so the user knows what's missing
        assert str(missing) in str(exc_info.value) or "does_not_exist" in str(exc_info.value)

    def test_from_csv_missing_dir_raises_file_not_found(self, qapp, tmp_path):
        missing = tmp_path / "no_such_dir" / "points.csv"
        with pytest.raises((FileNotFoundError, OSError)):
            PointsStore.from_csv(missing)

    def test_to_csv_missing_dir_raises(self, qapp, tmp_path):
        store = make_store_with_point()
        missing_dir = tmp_path / "no_such_dir" / "points.csv"
        with pytest.raises((FileNotFoundError, OSError)):
            store.to_csv(missing_dir)

    def test_to_csv_no_path_raises_value_error(self, qapp):
        store = make_store_with_point()
        with pytest.raises(ValueError) as exc_info:
            store.to_csv()
        assert "path" in str(exc_info.value).lower()

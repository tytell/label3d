"""Tests for label3d.points.PointsStore."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from label3d import points as points_module
from label3d.points import (
    CAM_COORDS,
    INDEX_NAMES,
    THREE_D_COORDS,
    PointData,
    PointsStore,
    read_points_df,
    write_points_file,
)
from tests.helpers import BODYPARTS, CAMERAS


# ---------------------------------------------------------------------------
# Construction and I/O
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_create_empty_column_structure(self, empty_store):
        cols = empty_store.df.columns
        for cam in CAMERAS:
            for coord in CAM_COORDS:
                assert (cam, coord) in cols, f"Missing column ({cam!r}, {coord!r})"
        for coord in THREE_D_COORDS:
            assert ("3d", coord) in cols, f"Missing column ('3d', {coord!r})"

    def test_create_empty_zero_rows(self, empty_store):
        assert len(empty_store.df) == 0

    def test_create_empty_index_names(self, empty_store):
        assert list(empty_store.df.index.names) == INDEX_NAMES

    def test_from_parquet_loads(self, populated_store):
        assert len(populated_store.df) > 0

    def test_from_parquet_index_names(self, populated_store):
        assert list(populated_store.df.index.names) == INDEX_NAMES

    def test_from_parquet_frame_is_int(self, populated_store):
        frame_level = populated_store.df.index.get_level_values("frame")
        assert frame_level.dtype == int or np.issubdtype(frame_level.dtype, np.integer)

    def test_from_parquet_instance_is_int(self, populated_store):
        inst_level = populated_store.df.index.get_level_values("instance")
        assert inst_level.dtype == int or np.issubdtype(inst_level.dtype, np.integer)

    def test_from_parquet_numeric_coords_are_float(self, populated_store):
        for cam in populated_store.cameras:
            for coord in ["x", "y", "xr", "yr"]:
                col = populated_store.df[(cam, coord)]
                assert pd.api.types.is_float_dtype(col), f"({cam}, {coord}) not float"

    def test_to_parquet_roundtrip(self, populated_store, tmp_path):
        dest = tmp_path / "points_out.parquet"
        populated_store.to_parquet(dest)
        reloaded = PointsStore.from_parquet(dest)
        # Shape and index
        assert len(reloaded.df) == len(populated_store.df)
        assert list(reloaded.df.index.names) == INDEX_NAMES
        # Spot-check a numeric value
        orig_val = populated_store.df.iloc[0][("cam1", "x")]
        new_val = reloaded.df.iloc[0][("cam1", "x")]
        assert abs(float(orig_val) - float(new_val)) < 1e-6

    def test_from_parquet_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PointsStore.from_parquet(tmp_path / "no_such_file.parquet")

    def test_to_parquet_no_path_raises(self, empty_store):
        with pytest.raises(ValueError, match="[Nn]o save path"):
            empty_store.to_parquet()



class TestPointsFile:
    """The Parquet points file: exact round-trip, flat layout, safe saves."""

    @staticmethod
    def _varied_store() -> PointsStore:
        """Every state and source, NaNs, a score and a sync id, all-digit names."""
        store = PointsStore.create_empty(CAMERAS, BODYPARTS, experiment_name="20260619")
        df = store.df
        rows = [("20260619", "12", f, 0, bp) for f in (0, 1) for bp in BODYPARTS]
        df = df.reindex(pd.MultiIndex.from_tuples(rows, names=INDEX_NAMES))
        cam = CAMERAS[0]
        df[(cam, "state")] = ["set", "predicted", "not_visible", "not_set", "set", "predicted"]
        df[(cam, "source")] = ["label3d", "sleap_pred", "sleap_user", "", "label3d", "sleap_pred"]
        df[(cam, "x")] = [1.5, 2.5, np.nan, np.nan, 3.0, 4.0]
        df[(cam, "y")] = [5.0, 6.0, np.nan, np.nan, 7.0, 8.0]
        df[(cam, "score")] = [np.nan, 0.9, np.nan, np.nan, np.nan, 0.4]
        df[(cam, "sync")] = [np.nan, 3.0, 2.0, np.nan, np.nan, 3.0]
        df[("3d", "x")] = [0.1, np.nan, np.nan, np.nan, 0.2, np.nan]
        for other in CAMERAS[1:]:
            df[(other, "state")] = "not_set"
            df[(other, "source")] = ""
        store.replace_df(points_module.normalize_columns(df))
        return store

    def test_roundtrip_is_exact(self, qapp, tmp_path):
        store = self._varied_store()
        store.to_parquet(tmp_path / "points.parquet")
        pd.testing.assert_frame_equal(read_points_df(tmp_path / "points.parquet"), store.df)

    def test_empty_store_roundtrips(self, empty_store, tmp_path):
        empty_store.to_parquet(tmp_path / "points.parquet")
        back = read_points_df(tmp_path / "points.parquet")
        assert back.empty
        assert back.columns.equals(empty_store.df.columns)

    def test_all_digit_names_stay_strings(self, qapp, tmp_path):
        self._varied_store().to_parquet(tmp_path / "points.parquet")
        back = read_points_df(tmp_path / "points.parquet")
        assert set(back.index.get_level_values("experiment")) == {"20260619"}
        assert set(back.index.get_level_values("trial")) == {"12"}

    def test_file_is_flat(self, qapp, tmp_path):
        """Readable without pandas: index levels and view/coord are plain columns."""
        self._varied_store().to_parquet(tmp_path / "points.parquet")
        names = pq.read_schema(tmp_path / "points.parquet").names
        assert names[:5] == INDEX_NAMES
        assert f"{CAMERAS[0]}/state" in names
        assert names[-3:] == ["3d/x", "3d/y", "3d/z"]

    def test_read_some_trials(self, qapp, tmp_path):
        store = self._varied_store()
        df = store.df.rename(index={"12": "13"}, level="trial").iloc[:3]
        write_points_file(pd.concat([store.df, df]), tmp_path / "points.parquet")
        back = read_points_df(tmp_path / "points.parquet", trials=["13"])
        assert set(back.index.get_level_values("trial")) == {"13"}
        assert len(back) == 3

    def test_backup_holds_previous_contents(self, qapp, tmp_path):
        dest = tmp_path / "points.parquet"
        store = self._varied_store()
        store.to_parquet(dest)
        before = read_points_df(dest)
        store.set_point("12", 0, 0, BODYPARTS[0], CAMERAS[1], 9.0, 9.0)
        store.save_if_changed()
        pd.testing.assert_frame_equal(read_points_df(tmp_path / "points.parquet.bak"), before)
        assert read_points_df(dest).loc[("20260619", "12", 0, 0, BODYPARTS[0]), (CAMERAS[1], "x")] == 9.0

    def test_failed_write_leaves_file(self, qapp, tmp_path, monkeypatch):
        dest = tmp_path / "points.parquet"
        store = self._varied_store()
        store.to_parquet(dest)
        before = dest.read_bytes()

        def fail(table, where, **kwargs):
            Path(where).write_bytes(b"partial")
            raise OSError("disk full")

        monkeypatch.setattr(points_module.pq, "write_table", fail)
        store.set_point("12", 0, 0, BODYPARTS[0], CAMERAS[1], 9.0, 9.0)
        with pytest.raises(OSError):
            store.save_if_changed()
        assert dest.read_bytes() == before
        assert not (tmp_path / "points.parquet.tmp").exists()
        assert not (tmp_path / "points.parquet.bak").exists()


class TestProperties:
    def test_cameras_excludes_3d(self, populated_store):
        assert "3d" not in populated_store.cameras

    def test_cameras_contains_real_cameras(self, populated_store):
        for cam in CAMERAS:
            assert cam in populated_store.cameras

    def test_bodyparts_not_empty(self, populated_store):
        assert len(populated_store.bodyparts) > 0

    def test_bodyparts_are_strings(self, populated_store):
        for bp in populated_store.bodyparts:
            assert isinstance(bp, str)


# ---------------------------------------------------------------------------
# Point-level API
# ---------------------------------------------------------------------------

class TestGetPoint:
    def test_existing_point(self, store_with_one_point):
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert abs(pt.x - 100.0) < 1e-9
        assert abs(pt.y - 200.0) < 1e-9
        assert pt.state == "set"

    def test_missing_row_returns_nan_not_set(self, empty_store):
        pt = empty_store.get_point("trial1", 99, 0, "snout", "cam1")
        assert math.isnan(pt.x)
        assert math.isnan(pt.y)
        assert pt.state == "not_set"

    def test_missing_camera_returns_nan(self, store_with_one_point):
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam2")
        assert math.isnan(pt.x)
        assert pt.state == "not_set"

    def test_returns_pointdata_namedtuple(self, store_with_one_point):
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert isinstance(pt, PointData)


class TestSetPoint:
    def test_creates_new_row(self, empty_store):
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 10.0, 20.0)
        assert len(empty_store.df) == 1

    def test_set_values_retrievable(self, empty_store):
        empty_store.set_point("t", 5, 2, "dorsal", "cam2", 42.5, 99.1, state="set")
        pt = empty_store.get_point("t", 5, 2, "dorsal", "cam2")
        assert abs(pt.x - 42.5) < 1e-9
        assert abs(pt.y - 99.1) < 1e-9
        assert pt.state == "set"

    def test_update_existing_row(self, store_with_one_point):
        store_with_one_point.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert abs(pt.x - 1.0) < 1e-9
        assert abs(pt.y - 2.0) < 1e-9

    def test_marks_dirty(self, empty_store):
        assert not empty_store.changed
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        assert empty_store.changed

    def test_emits_changed_signal(self, empty_store):
        received = []
        empty_store.data_changed.connect(lambda: received.append(1))
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        assert len(received) == 1

    def test_bodypart_with_spaces(self, empty_store):
        empty_store.set_point("trial1", 0, 0, "left front paw", "cam1", 5.0, 6.0)
        pt = empty_store.get_point("trial1", 0, 0, "left front paw", "cam1")
        assert abs(pt.x - 5.0) < 1e-9

    def test_camera_with_spaces(self, qapp):
        store = PointsStore.create_empty(["cam left", "cam right"], ["snout"])
        store.set_point("trial1", 0, 0, "snout", "cam left", 7.0, 8.0)
        pt = store.get_point("trial1", 0, 0, "snout", "cam left")
        assert abs(pt.x - 7.0) < 1e-9

    def test_trial_name_with_spaces(self, empty_store):
        empty_store.set_point("my trial 1", 0, 0, "snout", "cam1", 3.0, 4.0)
        pt = empty_store.get_point("my trial 1", 0, 0, "snout", "cam1")
        assert abs(pt.x - 3.0) < 1e-9

    def test_default_state_is_set(self, empty_store):
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        pt = empty_store.get_point("trial1", 0, 0, "snout", "cam1")
        assert pt.state == "set"


class TestSetNotVisible:
    def test_state_becomes_not_visible(self, store_with_one_point):
        store_with_one_point.set_not_visible("trial1", 0, 0, "snout", "cam1")
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert pt.state == "not_visible"

    def test_coords_preserved(self, store_with_one_point):
        store_with_one_point.set_not_visible("trial1", 0, 0, "snout", "cam1")
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert pt.x == 100.0
        assert pt.y == 200.0

    def test_creates_row_if_missing(self, empty_store):
        empty_store.set_not_visible("trial1", 0, 0, "snout", "cam1")
        assert len(empty_store.df) == 1

    def test_marks_dirty(self, empty_store):
        empty_store.set_not_visible("trial1", 0, 0, "snout", "cam1")
        assert empty_store.changed

    def test_emits_changed_signal(self, empty_store):
        received = []
        empty_store.data_changed.connect(lambda: received.append(1))
        empty_store.set_not_visible("trial1", 0, 0, "snout", "cam1")
        assert len(received) == 1


class TestDeletePoint:
    def test_state_becomes_not_set(self, store_with_one_point):
        store_with_one_point.delete_point("trial1", 0, 0, "snout", "cam1")
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert pt.state == "not_set"

    def test_coords_become_nan(self, store_with_one_point):
        store_with_one_point.delete_point("trial1", 0, 0, "snout", "cam1")
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert math.isnan(pt.x)
        assert math.isnan(pt.y)

    def test_reprojected_coords_become_nan(self, qapp):
        store = PointsStore.create_empty(CAMERAS, BODYPARTS)
        store.set_point("t", 0, 0, "snout", "cam1", 1.0, 2.0)
        store.set_reprojected("t", 0, 0, "snout", "cam1", 1.1, 2.1)
        store.delete_point("t", 0, 0, "snout", "cam1")
        row = store.df.loc[("default", "t", 0, 0, "snout")]
        assert pd.isna(row[("cam1", "xr")])
        assert pd.isna(row[("cam1", "yr")])

    def test_row_still_exists(self, store_with_one_point):
        store_with_one_point.delete_point("trial1", 0, 0, "snout", "cam1")
        assert ("default", "trial1", 0, 0, "snout") in store_with_one_point.df.index

    def test_nonexistent_is_noop(self, empty_store):
        empty_store.delete_point("trial1", 99, 0, "snout", "cam1")  # no error

    def test_marks_dirty(self, store_with_one_point):
        store_with_one_point._changed = False
        store_with_one_point.delete_point("trial1", 0, 0, "snout", "cam1")
        assert store_with_one_point.changed

    def test_emits_changed_signal(self, store_with_one_point):
        received = []
        store_with_one_point.data_changed.connect(lambda: received.append(1))
        store_with_one_point.delete_point("trial1", 0, 0, "snout", "cam1")
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Frame-level API
# ---------------------------------------------------------------------------

class TestFrameLevel:
    def test_get_frame_data_returns_dataframe(self, populated_store):
        result = populated_store.get_frame_data("trial1", 0)
        assert isinstance(result, pd.DataFrame)

    def test_get_frame_data_correct_bodyparts(self, populated_store):
        result = populated_store.get_frame_data("trial1", 0)
        bps = result.index.get_level_values("bodypart").unique().tolist()
        for bp in BODYPARTS:
            assert bp in bps

    def test_get_frame_data_nonexistent_frame_returns_empty(self, empty_store):
        result = empty_store.get_frame_data("trial1", 9999)
        assert len(result) == 0

    def test_get_instances_returns_sorted_list(self, populated_store):
        instances = populated_store.get_instances("trial1", 0)
        assert instances == sorted(instances)
        assert len(instances) >= 1

    def test_get_instances_nonexistent_returns_empty(self, empty_store):
        assert empty_store.get_instances("trial1", 9999) == []

    def test_get_instances_type_is_int(self, populated_store):
        instances = populated_store.get_instances("trial1", 0)
        for i in instances:
            assert isinstance(i, (int, np.integer))


# ---------------------------------------------------------------------------
# Instance-level API
# ---------------------------------------------------------------------------

class TestInstanceLevel:
    def test_add_instance_first_returns_zero(self, empty_store):
        idx = empty_store.add_instance("trial1", 0, BODYPARTS)
        assert idx == 0

    def test_add_instance_second_returns_one(self, empty_store):
        empty_store.add_instance("trial1", 0, BODYPARTS)
        idx = empty_store.add_instance("trial1", 0, BODYPARTS)
        assert idx == 1

    def test_add_instance_creates_rows(self, empty_store):
        empty_store.add_instance("trial1", 0, BODYPARTS)
        assert len(empty_store.df) == len(BODYPARTS)

    def test_add_instance_rows_are_not_set(self, empty_store):
        empty_store.add_instance("trial1", 0, BODYPARTS)
        pt = empty_store.get_point("trial1", 0, 0, "snout", "cam1")
        assert pt.state == "not_set"

    def test_add_instance_marks_dirty(self, empty_store):
        empty_store.add_instance("trial1", 0, BODYPARTS)
        assert empty_store.changed

    def test_delete_instance_removes_rows(self, empty_store):
        empty_store.add_instance("trial1", 0, BODYPARTS)
        assert len(empty_store.df) == len(BODYPARTS)
        empty_store.delete_instance("trial1", 0, 0)
        assert len(empty_store.df) == 0

    def test_delete_instance_nonexistent_is_noop(self, empty_store):
        empty_store.delete_instance("trial1", 0, 99)  # no error

    def test_copy_instance_to_new_frame(self, store_with_one_point):
        new_idx = store_with_one_point.copy_instance("trial1", 0, 0, 5)
        pt = store_with_one_point.get_point("trial1", 5, new_idx, "snout", "cam1")
        assert abs(pt.x - 100.0) < 1e-9
        assert abs(pt.y - 200.0) < 1e-9

    def test_copy_instance_returns_correct_index(self, store_with_one_point):
        idx = store_with_one_point.copy_instance("trial1", 0, 0, 0)
        assert idx == 1  # frame 0 already has instance 0

    def test_copy_instance_source_unchanged(self, store_with_one_point):
        store_with_one_point.copy_instance("trial1", 0, 0, 5)
        pt = store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert abs(pt.x - 100.0) < 1e-9

    def test_copy_nonexistent_instance_returns_new_idx(self, empty_store):
        idx = empty_store.copy_instance("trial1", 0, 99, 1)
        assert idx == 0  # no existing instances in frame 1


# ---------------------------------------------------------------------------
# 3D API
# ---------------------------------------------------------------------------

class TestThreeDApi:
    def test_set_3d_stores_values(self, store_with_one_point):
        store_with_one_point.set_3d("trial1", 0, 0, "snout", 1.0, 2.0, 3.0)
        row = store_with_one_point.df.loc[("default", "trial1", 0, 0, "snout")]
        assert abs(float(row[("3d", "x")]) - 1.0) < 1e-9
        assert abs(float(row[("3d", "y")]) - 2.0) < 1e-9
        assert abs(float(row[("3d", "z")]) - 3.0) < 1e-9

    def test_set_3d_marks_dirty(self, store_with_one_point):
        store_with_one_point._changed = False
        store_with_one_point.set_3d("trial1", 0, 0, "snout", 1.0, 2.0, 3.0)
        assert store_with_one_point.changed

    def test_set_3d_nonexistent_row_is_noop(self, empty_store):
        empty_store.set_3d("trial1", 0, 0, "snout", 1.0, 2.0, 3.0)
        assert len(empty_store.df) == 0

    def test_set_reprojected_stores_values(self, store_with_one_point):
        store_with_one_point.set_reprojected("trial1", 0, 0, "snout", "cam1", 101.0, 201.0)
        row = store_with_one_point.df.loc[("default", "trial1", 0, 0, "snout")]
        assert abs(float(row[("cam1", "xr")]) - 101.0) < 1e-9
        assert abs(float(row[("cam1", "yr")]) - 201.0) < 1e-9

    def test_set_reprojected_nonexistent_row_is_noop(self, empty_store):
        empty_store.set_reprojected("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        assert len(empty_store.df) == 0


# ---------------------------------------------------------------------------
# Frame cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_populated_after_get(self, store_with_one_point):
        store_with_one_point._cache_key = None
        store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert store_with_one_point._cache_key == ("trial1", 0)

    def test_cache_hit_reused(self, store_with_one_point):
        store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        key_after_first = store_with_one_point._cache_key
        store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert store_with_one_point._cache_key == key_after_first

    def test_set_point_invalidates_cache(self, store_with_one_point):
        store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        assert store_with_one_point._cache_key is not None
        store_with_one_point.set_point("trial1", 0, 0, "snout", "cam1", 9.0, 9.0)
        assert store_with_one_point._cache_key is None

    def test_delete_point_invalidates_cache(self, store_with_one_point):
        store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        store_with_one_point.delete_point("trial1", 0, 0, "snout", "cam1")
        assert store_with_one_point._cache_key is None

    def test_set_not_visible_invalidates_cache(self, store_with_one_point):
        store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        store_with_one_point.set_not_visible("trial1", 0, 0, "snout", "cam1")
        assert store_with_one_point._cache_key is None

    def test_different_frame_misses_cache(self, store_with_one_point):
        store_with_one_point.get_point("trial1", 0, 0, "snout", "cam1")
        store_with_one_point.get_point("trial1", 1, 0, "snout", "cam1")
        assert store_with_one_point._cache_key == ("trial1", 1)


# ---------------------------------------------------------------------------
# Dirty tracking and save
# ---------------------------------------------------------------------------

class TestDirtyAndSave:
    def test_starts_clean(self, empty_store):
        assert not empty_store.changed

    def test_dirty_after_set_point(self, empty_store):
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        assert empty_store.changed

    def test_clean_after_to_parquet(self, empty_store, tmp_path):
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        assert empty_store.changed
        empty_store.to_parquet(tmp_path / "p.parquet")
        assert not empty_store.changed

    def test_save_if_changed_writes_file(self, empty_store, tmp_path):
        dest = tmp_path / "save_test.parquet"
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        empty_store._save_path = dest
        empty_store.save_if_changed()
        assert dest.exists()

    def test_save_if_changed_creates_backup(self, empty_store, tmp_path):
        dest = tmp_path / "points.parquet"
        # Write initial file so backup is possible
        empty_store.to_parquet(dest)
        # Mutate and save again
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        empty_store.save_if_changed()
        assert (tmp_path / "points.parquet.bak").exists()

    def test_save_if_changed_noop_when_clean(self, empty_store, tmp_path):
        dest = tmp_path / "points.parquet"
        empty_store._save_path = dest
        empty_store.save_if_changed()  # dirty=False, save_path set — should not write
        assert not dest.exists()

    def test_save_if_changed_noop_without_path(self, empty_store):
        empty_store.set_point("trial1", 0, 0, "snout", "cam1", 1.0, 2.0)
        assert empty_store._save_path is None
        empty_store.save_if_changed()  # no error — just silently skips


# ---------------------------------------------------------------------------
# Special characters in names (CSV round-trip)
# ---------------------------------------------------------------------------

class TestSpecialCharNames:
    def test_bodypart_with_spaces_csv_roundtrip(self, qapp, tmp_path):
        store = PointsStore.create_empty(["cam1"], ["left front paw", "dorsal fin"])
        store.set_point("trial1", 0, 0, "left front paw", "cam1", 10.0, 20.0)
        dest = tmp_path / "p.parquet"
        store.to_parquet(dest)
        reloaded = PointsStore.from_parquet(dest)
        pt = reloaded.get_point("trial1", 0, 0, "left front paw", "cam1")
        assert abs(pt.x - 10.0) < 1e-6

    def test_bodypart_with_colon_csv_roundtrip(self, qapp, tmp_path):
        store = PointsStore.create_empty(["cam1"], ["joint:elbow"])
        store.set_point("trial1", 0, 0, "joint:elbow", "cam1", 5.0, 6.0)
        dest = tmp_path / "p.parquet"
        store.to_parquet(dest)
        reloaded = PointsStore.from_parquet(dest)
        pt = reloaded.get_point("trial1", 0, 0, "joint:elbow", "cam1")
        assert abs(pt.x - 5.0) < 1e-6

    def test_trial_name_with_spaces_csv_roundtrip(self, qapp, tmp_path):
        store = PointsStore.create_empty(["cam1"], ["snout"])
        store.set_point("my trial session 1", 3, 0, "snout", "cam1", 7.0, 8.0)
        dest = tmp_path / "p.parquet"
        store.to_parquet(dest)
        reloaded = PointsStore.from_parquet(dest)
        pt = reloaded.get_point("my trial session 1", 3, 0, "snout", "cam1")
        assert abs(pt.x - 7.0) < 1e-6

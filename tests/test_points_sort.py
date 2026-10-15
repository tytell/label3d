"""The points index stays sorted, so partial-key lookups don't warn."""

from __future__ import annotations

import warnings

from pandas.errors import PerformanceWarning

from label3d.points import PointsStore

BODYPARTS = ["snout", "left_eye", "dorsal"]


def _store_with_rows_out_of_order() -> PointsStore:
    store = PointsStore.create_empty(cameras=["cam0"], bodyparts=[], experiment_name="exp")
    store.add_instance("t2", 5, BODYPARTS)
    store.add_instance("t1", 9, BODYPARTS)  # earlier trial, added later
    store.set_point("t2", 0, 0, "snout", "cam0", 1.0, 2.0, state="set")  # earlier frame
    return store


def test_out_of_order_inserts_keep_index_sorted():
    store = _store_with_rows_out_of_order()
    assert store.df.index._lexsort_depth >= 4
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        store.get_labeled_frames_by_camera("t1")
        store.get_instances("t2", 5)


def test_bodyparts_keep_skeleton_order():
    store = _store_with_rows_out_of_order()
    bps = store.df.loc[("exp", "t1", 9, 0)].index.tolist()
    assert bps == BODYPARTS


def test_unsorted_csv_is_sorted_on_load(tmp_path):
    store = _store_with_rows_out_of_order()
    path = tmp_path / "points.csv"
    df = store.df
    df.iloc[::-1].to_csv(path)  # write rows in reverse order
    loaded = PointsStore.from_csv(path)
    assert loaded.df.index._lexsort_depth >= 4

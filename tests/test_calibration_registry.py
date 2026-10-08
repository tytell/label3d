"""Tests for the calibration registry and the active-calibration pointer (Phase 11)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from aniposelib.cameras import Camera, CameraGroup

from label3d.calibration_registry import (
    REFINED_DIR,
    clean_label,
    delete_calibration,
    entry_error,
    entry_summary,
    entry_tree,
    find_calibration,
    list_calibrations,
    load_calibration,
    registry_dir,
    rename_calibration,
    save_calibration,
)
from label3d.geometry import cameras_from_cgroup, save_aligned_calibration

CAMERAS = ["cam0", "cam1"]
WHEN = datetime(2026, 10, 5, 14, 22, 10)


def make_cgroup(shift: float = 0.0) -> CameraGroup:
    cams = []
    for i, name in enumerate(CAMERAS):
        cams.append(Camera(
            name=name,
            size=(640, 480),
            matrix=[[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]],
            dist=[0.0] * 5,
            rvec=[0.0, 0.1 * i, 0.0],
            tvec=[10.0 * i + shift, 0.0, 0.0],
        ))
    return CameraGroup(cams)


class TestLabels:
    def test_clean_label(self):
        assert clean_label("All takes (joint)") == "All_takes_joint"
        assert clean_label("Pre Cal!") == "Pre_Cal"
        assert clean_label("refine-a_2") == "refine-a_2"
        assert clean_label("()") == "calibration"


class TestRegistry:
    def test_round_trip_keeps_metadata(self, tmp_path):
        entry = save_calibration(
            tmp_path, make_cgroup(), "charuco-all", "charuco", created=WHEN,
            charuco_takes=["PreCal", "PostCal"], errors={"charuco": 0.41},
        )
        assert entry.id == "charuco-all_20261005-142210"
        assert entry.path == registry_dir(tmp_path) / f"{entry.id}.toml"

        [listed] = list_calibrations(tmp_path)
        assert listed.id == entry.id
        assert listed.created == WHEN
        assert listed.method == "charuco"
        assert listed.charuco_takes == ["PreCal", "PostCal"]
        assert listed.metadata["errors"]["charuco"] == pytest.approx(0.41)

        cgroup = load_calibration(listed)
        assert cgroup.get_names() == CAMERAS
        assert cgroup.metadata["id"] == entry.id

    def test_camera_group_load_reads_saved_file(self, tmp_path):
        entry = save_calibration(tmp_path, make_cgroup(), "x", "charuco")
        cgroup = CameraGroup.load(str(entry.path))
        assert len(cgroup.cameras) == 2
        np.testing.assert_allclose(cgroup.cameras[1].get_translation(), [10.0, 0, 0])

    def test_refine_goes_to_refined_folder(self, tmp_path):
        parent = save_calibration(tmp_path, make_cgroup(), "charuco-all", "charuco")
        child = save_calibration(tmp_path, make_cgroup(1.0), "refine", "refine",
                                 parent=parent.id)
        assert child.path.parent == registry_dir(tmp_path) / REFINED_DIR
        assert {e.id for e in list_calibrations(tmp_path)} == {parent.id, child.id}
        assert find_calibration(tmp_path, child.id).parent == parent.id

    def test_newest_first(self, tmp_path):
        old = save_calibration(tmp_path, make_cgroup(), "a", "charuco",
                               created=datetime(2026, 1, 1))
        new = save_calibration(tmp_path, make_cgroup(), "b", "charuco",
                               created=datetime(2026, 2, 1))
        assert [e.id for e in list_calibrations(tmp_path)] == [new.id, old.id]

    def test_ids_unique_within_one_second(self, tmp_path):
        a = save_calibration(tmp_path, make_cgroup(), "same", "charuco", created=WHEN)
        b = save_calibration(tmp_path, make_cgroup(), "same", "charuco", created=WHEN)
        assert a.id != b.id
        assert b.id == f"{a.id}-2"
        assert len(list_calibrations(tmp_path)) == 2

    def test_rename_keeps_id(self, tmp_path):
        entry = save_calibration(tmp_path, make_cgroup(), "old", "charuco", created=WHEN)
        renamed = rename_calibration(entry, "new name")
        assert renamed.label == "new_name"
        assert renamed.id == entry.id
        assert renamed.path == entry.path

    def test_delete_removes_descendants(self, tmp_path):
        root = save_calibration(tmp_path, make_cgroup(), "root", "charuco")
        other = save_calibration(tmp_path, make_cgroup(), "other", "charuco")
        child = save_calibration(
            tmp_path, make_cgroup(), "child", "refine", parent=root.id,
            points={"points_file": "child_points.csv"},
        )
        (child.path.parent / "child_points.csv").write_text("x\n")
        grandchild = save_calibration(tmp_path, make_cgroup(), "gc", "align",
                                      parent=child.id)

        removed = delete_calibration(tmp_path, root)
        assert {e.id for e in removed} == {root.id, child.id, grandchild.id}
        assert [e.id for e in list_calibrations(tmp_path)] == [other.id]
        assert not (child.path.parent / "child_points.csv").exists()

    def test_saving_does_not_touch_callers_metadata(self, tmp_path):
        a, b = make_cgroup(), make_cgroup()
        save_calibration(tmp_path, a, "a", "charuco")
        assert a.metadata == {}
        assert b.metadata == {}



class TestAlignment:
    def test_aligned_entry_keeps_alignment_in_metadata(self, tmp_path):
        parent = save_calibration(tmp_path, make_cgroup(), "charuco-all", "charuco")
        entry = save_aligned_calibration(
            cameras_from_cgroup(make_cgroup()), tmp_path, np.eye(3), np.zeros(3),
            parent=parent.id,
        )
        assert entry.method == "align"
        assert entry.parent == parent.id
        np.testing.assert_allclose(entry.metadata["alignment"]["R_align"], np.eye(3))
        assert load_calibration(entry).get_names() == CAMERAS


class TestDisplay:
    def test_tree_puts_children_under_parent(self, tmp_path):
        a = save_calibration(tmp_path, make_cgroup(), "a", "charuco",
                             created=datetime(2026, 1, 1))
        b = save_calibration(tmp_path, make_cgroup(), "b", "charuco",
                             created=datetime(2026, 3, 1))
        a1 = save_calibration(tmp_path, make_cgroup(), "a1", "refine", parent=a.id,
                              created=datetime(2026, 4, 1))
        a2 = save_calibration(tmp_path, make_cgroup(), "a2", "refine", parent=a.id,
                              created=datetime(2026, 5, 1))
        tree = entry_tree(list_calibrations(tmp_path))
        assert [(e.id, d) for e, d in tree] == [
            (b.id, 0), (a.id, 0), (a2.id, 1), (a1.id, 1),
        ]

    def test_error_and_summary(self, tmp_path):
        entry = save_calibration(
            tmp_path, make_cgroup(), "r", "refine", parent="p",
            errors={"charuco": 0.4, "holdout": 2.6},
            points={"n_points": 10, "sources": ["label3d"]},
        )
        assert entry_error(entry) == pytest.approx(2.6)
        text = entry_summary(entry)
        assert "Parent: p" in text and "Points: 10 from label3d" in text

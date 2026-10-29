"""Tests for Set up Sleap files (Phase 8.9): mapping editor, project apply steps, dialog."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio
import toml

from label3d.points import PointsStore, read_points_df
from label3d.project import Project
from label3d.slp_sync import SlpBlock, SlpFileSpec, SlpMapping, SlpMappingEditor
from tests.helpers import make_current

EXP = "exp1"
CAMERAS = ["cam0", "cam1", "cam2"]
TRIALS = ["t1", "t2", "t3", "t4", "t5"]


def _trials_df() -> pd.DataFrame:
    rows = []
    for t in TRIALS:
        row = {"experiment": EXP, "trial": t}
        for c in CAMERAS:
            row[c] = f"{c}_{t}.mp4"
            row[f"{c}_start"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def _editor(mapping: SlpMapping | None = None) -> SlpMappingEditor:
    return SlpMappingEditor(mapping or SlpMapping(), _trials_df(), CAMERAS)


def _blocks(editor: SlpMappingEditor) -> dict[str, list[SlpBlock]]:
    return {spec.path.name: spec.blocks for spec in editor.mapping().files}


# ---------------------------------------------------------------------------
# SlpMappingEditor
# ---------------------------------------------------------------------------

class TestEditor:
    def test_new_editor_all_unassigned(self):
        editor = _editor()
        assert editor.files == []
        assert editor.errors() == ["15 video(s) unassigned"]

    @pytest.mark.parametrize("pattern, n_files", [
        ("{camera}.slp", 3),
        ("{experiment}_{camera}.slp", 3),
        ("{experiment}_{trial}_{camera}.slp", 15),
        ("labels.slp", 1),
    ])
    def test_presets(self, pattern, n_files):
        editor = _editor()
        editor.apply_preset(pattern)
        assert len(editor.files) == n_files
        assert editor.errors() == []

    def test_preset_adds_suffix_and_slp_dir(self):
        editor = _editor()
        editor.apply_preset("{camera}")
        assert editor.files == [Path("slp/cam0/cam0.slp"), Path("slp/cam1/cam1.slp"), Path("slp/cam2/cam2.slp")]

    def test_invalid_pattern_raises(self):
        with pytest.raises(KeyError):
            _editor().apply_preset("{animal}.slp")

    def test_example_per_camera_join_split_by_trial(self):
        """phase_8.md example: per camera → join cam0 + cam1 → split trials 4–5."""
        editor = _editor()
        editor.apply_preset("{camera}.slp")
        joined = editor.join([Path("slp/cam0/cam0.slp"), Path("slp/cam1/cam1.slp")])
        assert joined == Path("slp/cam0/cam0.slp")
        late = editor.split_by_trial(joined, {(EXP, "t4"), (EXP, "t5")})
        assert late == Path("slp/cam0_2/cam0_2.slp")

        assert _blocks(editor) == {
            "cam0.slp": [SlpBlock(EXP, ["t1", "t2", "t3"], ["cam0", "cam1"])],
            "cam2.slp": [SlpBlock(EXP, TRIALS, ["cam2"])],
            "cam0_2.slp": [SlpBlock(EXP, ["t4", "t5"], ["cam0", "cam1"])],
        }
        assert editor.errors() == []

    def test_split_by_camera(self):
        editor = _editor()
        editor.apply_preset("labels.slp")
        new = editor.split_by_camera(Path("slp/labels/labels.slp"))
        assert [p.name for p in new] == ["labels_cam0.slp", "labels_cam1.slp", "labels_cam2.slp"]
        assert editor.files == new

    def test_split_by_trial_all_or_none_is_refused(self):
        editor = _editor()
        editor.apply_preset("{camera}.slp")
        path = Path("slp/cam0/cam0.slp")
        assert editor.split_by_trial(path, set()) is None
        assert editor.split_by_trial(path, {(EXP, t) for t in TRIALS}) is None

    def test_new_file_from_cells_and_empty_file_dropped(self):
        editor = _editor()
        editor.apply_preset("{experiment}_{trial}_{camera}.slp")
        path = editor.new_file([(EXP, "t1", "cam0"), (EXP, "t1", "cam1")])
        assert path == Path("slp/new/new.slp")
        assert Path("slp/exp1_t1_cam0/exp1_t1_cam0.slp") not in editor.files
        assert editor.cells_of(path) == [(EXP, "t1", "cam0"), (EXP, "t1", "cam1")]

    def test_move_cells_to_existing_file(self):
        editor = _editor()
        editor.apply_preset("{camera}.slp")
        editor.move_cells([(EXP, "t3", "cam2")], Path("slp/cam0/cam0.slp"))
        assert (EXP, "t3", "cam2") in editor.cells_of(Path("slp/cam0/cam0.slp"))
        assert editor.files_of((EXP, "t3", "cam2")) == [Path("slp/cam0/cam0.slp")]
        assert _blocks(editor)["cam0.slp"] == [
            SlpBlock(EXP, ["t1", "t2", "t4", "t5"], ["cam0"]),
            SlpBlock(EXP, ["t3"], ["cam0", "cam2"]),
        ]

    def test_rename(self):
        editor = _editor()
        editor.apply_preset("{camera}.slp")
        new = editor.rename(Path("slp/cam1/cam1.slp"), Path("slp/cam1/side"))
        assert new == Path("slp/cam1/side.slp")
        assert editor.files == [Path("slp/cam0/cam0.slp"), Path("slp/cam1/side.slp"), Path("slp/cam2/cam2.slp")]
        with pytest.raises(ValueError):
            editor.rename(new, Path("slp/cam1/cam0.slp"))

    def test_unique_path_is_case_insensitive(self):
        editor = _editor()
        editor.apply_preset("New.slp")
        assert editor.unique_path("new") == Path("slp/new_2/new_2.slp")

    def test_loaded_duplicates_and_stale_cells(self):
        mapping = SlpMapping(files=[
            SlpFileSpec(Path("slp/a/a.slp"), [SlpBlock(EXP, TRIALS + ["gone"], ["cam0", "cam1"])]),
            SlpFileSpec(Path("slp/b/b.slp"), [SlpBlock(EXP, TRIALS, ["cam1", "cam2"])]),
        ])
        editor = _editor(mapping)
        assert (EXP, "gone", "cam0") not in editor.cells_of(Path("slp/a/a.slp"))
        assert editor.errors() == ["5 video(s) assigned to more than one file"]
        editor.move_cells([(EXP, t, "cam1") for t in TRIALS], Path("slp/b/b.slp"))
        assert editor.errors() == []


# ---------------------------------------------------------------------------
# Project apply steps
# ---------------------------------------------------------------------------

PCAMS = ["cam1", "cam2"]


def _make_project(tmp_path: Path) -> Project:
    rows = [
        {"experiment": e, "trial": t,
         "cam1": f"cam1_{e}_{t}.mp4", "cam1_start": 0,
         "cam2": f"cam2_{e}_{t}.mp4", "cam2_start": 0}
        for e in ["exp1", "exp2"] for t in ["t1", "t2"]
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "trials.csv", index=False)
    for e in ["exp1", "exp2"]:
        (tmp_path / "experiments" / e / "videos").mkdir(parents=True)
    (tmp_path / "skeleton.json").write_text(json.dumps(
        {"nodes": ["head", "tail"], "edges": [["head", "tail"]], "symmetries": []}
    ))
    (tmp_path / "project.toml").write_text('[project]\nname = "demo"\n')
    return Project.load(make_current(tmp_path))


def _labeled_project(tmp_path: Path) -> Project:
    project = _make_project(tmp_path)
    store = PointsStore.create_empty(PCAMS, [], experiment_name="exp1")
    store._save_path = project.experiments["exp1"].root / "points.parquet"
    store.set_point("t1", 0, 0, "head", "cam1", 10.0, 20.0)
    store.set_point("t1", 0, 0, "tail", "cam1", 30.0, 40.0)
    project.points = store
    return project


def _per_camera(project: Project) -> SlpMapping:
    editor = SlpMappingEditor(SlpMapping(), project.trials_df, project.cameras)
    editor.apply_preset("{camera}.slp")
    return editor.mapping()


def _add_user_frame(path: Path, basename: str, frame_idx: int, pts) -> None:
    labels = sio.load_file(str(path))
    (video,) = [v for v in labels.videos if Path(v.filename).name == basename]
    inst = sio.Instance.from_numpy(np.array(pts, dtype=float), skeleton=labels.skeleton)
    labels.labeled_frames.append(sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst]))
    sio.save_file(labels, str(path))


class TestProjectApply:
    def test_write_then_read_is_unchanged(self, qapp, tmp_path):
        project = _labeled_project(tmp_path)
        mapping = _per_camera(project)
        assert project.write_slp_files(mapping) == 2
        results = project.read_slp_for_sync(mapping)
        assert all(not r.changed_frames and not r.conflicts for r in results.values())

    def test_new_sleap_labels_merged_into_active_store(self, qapp, tmp_path):
        project = _labeled_project(tmp_path)
        mapping = _per_camera(project)
        project.write_slp_files(mapping)
        _add_user_frame(tmp_path / "slp/cam1/cam1.slp", "cam1_exp1_t2.mp4", 5, [[1, 2], [3, 4]])

        results = project.read_slp_for_sync(mapping)
        assert results["exp1"].n_new == 2
        project.apply_merge(results, overwrite=False)

        pt = project.points.get_point("t2", 5, 0, "tail", "cam1")
        assert (pt.x, pt.y, pt.state) == (3.0, 4.0, "set")
        saved = read_points_df(project.experiments["exp1"].root / "points.parquet")
        assert ("exp1", "t2", 5, 0, "tail") in saved.index

    def test_other_experiment_written_to_its_file(self, qapp, tmp_path):
        project = _labeled_project(tmp_path)
        mapping = _per_camera(project)
        project.write_slp_files(mapping)
        _add_user_frame(tmp_path / "slp/cam2/cam2.slp", "cam2_exp2_t1.mp4", 3, [[5, 6], [7, 8]])

        project.apply_merge(project.read_slp_for_sync(mapping), overwrite=False)
        df = read_points_df(project.experiments["exp2"].root / "points.parquet")
        assert df.loc[("exp2", "t1", 3, 0, "head"), ("cam2", "state")] == "set"

    def test_conflict_keep_vs_overwrite(self, qapp, tmp_path):
        project = _labeled_project(tmp_path)
        mapping = _per_camera(project)
        project.write_slp_files(mapping)
        path = tmp_path / "slp/cam1/cam1.slp"
        labels = sio.load_file(str(path))
        labels.labeled_frames[0].instances[0].points["xy"][0] = [99.0, 99.0]
        sio.save_file(labels, str(path))

        results = project.read_slp_for_sync(mapping)
        assert len(results["exp1"].conflicts) == 1
        project.apply_merge(results, overwrite=False)
        assert project.points.get_point("t1", 0, 0, "head", "cam1").x == 10.0
        project.apply_merge(results, overwrite=True)
        assert project.points.get_point("t1", 0, 0, "head", "cam1").x == 99.0

    def test_read_without_mapping_uses_all_files(self, qapp, tmp_path):
        project = _labeled_project(tmp_path)
        project.write_slp_files(_per_camera(project))
        _add_user_frame(tmp_path / "slp/cam1/cam1.slp", "cam1_exp1_t2.mp4", 5, [[1, 2], [3, 4]])
        assert project.read_slp_for_sync(SlpMapping())["exp1"].n_new == 2

    def test_restructure_backup_and_remove(self, qapp, tmp_path):
        project = _labeled_project(tmp_path)
        project.write_slp_files(_per_camera(project))

        editor = SlpMappingEditor(SlpMapping(), project.trials_df, project.cameras)
        editor.apply_preset("labels.slp")
        new_mapping = editor.mapping()
        project.save_slp_mapping(new_mapping)

        backup = project.backup_slp_files()
        assert sorted(p.relative_to(backup).as_posix() for p in backup.rglob("*.slp")) == [
            "cam1/cam1.slp", "cam2/cam2.slp"
        ]
        assert project.write_slp_files() == 1
        removed = project.remove_unmapped_slp_files(new_mapping)
        assert sorted(p.name for p in removed) == ["cam1.slp", "cam2.slp"]
        assert [p.name for p in project.slp_files()] == ["labels.slp"]

        labels = sio.load_file(str(tmp_path / "slp/labels/labels.slp"))
        assert len(labels.videos) == 8
        assert len(labels.labeled_frames) == 1
        assert toml.load(tmp_path / "project.toml")["project"]["name"] == "demo"


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class TestDialog:
    def _dialog(self, project):
        from label3d.widgets.slp_setup_dialog import SlpSetupDialog
        return SlpSetupDialog(project)

    def _select_cells(self, dialog, cells):
        dialog._grid.clearSelection()
        for row in range(dialog._grid.rowCount()):
            for col in range(dialog._grid.columnCount()):
                item = dialog._grid.item(row, col)
                if item.data(0x0100) in cells:
                    item.setSelected(True)

    def _select_files(self, dialog, names):
        for i in range(dialog._file_list.count()):
            item = dialog._file_list.item(i)
            item.setSelected(item.text() in names)

    def test_starts_unassigned_with_apply_disabled(self, qapp, tmp_path):
        dialog = self._dialog(_make_project(tmp_path))
        assert not dialog._apply_btn.isEnabled()
        assert dialog._grid.rowCount() == 2 and dialog._grid.columnCount() == 2
        assert dialog._grid.item(0, 0).text() == "unassigned"

    def test_preset_join_split_flow(self, qapp, tmp_path):
        dialog = self._dialog(_make_project(tmp_path))
        dialog._pattern_edit.setText("{camera}.slp")
        dialog._on_apply_preset()
        assert dialog._apply_btn.isEnabled()
        assert dialog._grid.item(0, 1).text() == "cam2.slp"

        self._select_files(dialog, {"cam1.slp", "cam2.slp"})
        assert dialog._join_btn.isEnabled()
        dialog._on_join()
        assert dialog._editor.files == [Path("slp/cam1/cam1.slp")]

        self._select_cells(dialog, {("exp1", "t2", "cam1")})
        self._select_files(dialog, {"cam1.slp"})
        assert dialog._split_trial_btn.isEnabled()
        dialog._on_split_trial()

        # Only the selected experiment's trial moves; exp2 rows stay in cam1.slp
        files = {spec.path.name: spec.blocks for spec in dialog.mapping().files}
        assert files["cam1_2.slp"] == [SlpBlock("exp1", ["t2"], ["cam1", "cam2"])]
        assert files["cam1.slp"] == [
            SlpBlock("exp1", ["t1"], ["cam1", "cam2"]),
            SlpBlock("exp2", ["t1", "t2"], ["cam1", "cam2"]),
        ]
        assert dialog._apply_btn.isEnabled()

    def test_rename_invalid_shows_message(self, qapp, tmp_path):
        dialog = self._dialog(_make_project(tmp_path))
        dialog._pattern_edit.setText("{camera}.slp")
        dialog._on_apply_preset()
        dialog._file_list.item(1).setText("cam1.slp")
        assert "already exists" in dialog._status.text()
        assert dialog._editor.files == [Path("slp/cam1/cam1.slp"), Path("slp/cam2/cam2.slp")]

    def test_adopts_existing_files_when_no_mapping(self, qapp, tmp_path):
        project = _labeled_project(tmp_path)
        project.write_slp_files(_per_camera(project))
        dialog = self._dialog(project)
        assert "Loaded assignments from 2" in dialog._status.text()
        assert dialog.mapping() == _per_camera(project)

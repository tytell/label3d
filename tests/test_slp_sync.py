"""Tests for SLP file mapping (8.6), merge rules (8.8) and mapping edits (8.9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import toml

from label3d.points import CAM_COORDS, INDEX_NAMES, THREE_D_COORDS
from label3d.project import Project
from label3d.slp_sync import (
    SlpBlock,
    SlpFileSpec,
    SlpMapping,
    SlpMappingEditor,
    compact_blocks,
    compute_merge,
    find_slp_files,
    video_cells,
)

NAN = float("nan")
EXP = "exp1"
CAMERAS = ["cam0", "cam1", "cam2"]
TRIALS = ["t1", "t2", "t3", "t4", "t5"]


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _trials_df() -> pd.DataFrame:
    rows = []
    for t in TRIALS:
        row = {"experiment": EXP, "trial": t}
        for c in CAMERAS:
            row[c] = f"{c}_{t}.mp4"
            row[f"{c}_start"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def _example_mapping() -> SlpMapping:
    """The 3-camera example from phase_8.md."""
    return SlpMapping(files=[
        SlpFileSpec(Path("slp/cam0_cam1_early/cam0_cam1_early.slp"),
                    [SlpBlock(EXP, ["t1", "t2", "t3"], ["cam0", "cam1"])]),
        SlpFileSpec(Path("slp/cam0_cam1_late/cam0_cam1_late.slp"),
                    [SlpBlock(EXP, ["t4", "t5"], ["cam0", "cam1"])]),
        SlpFileSpec(Path("slp/cam2/cam2.slp"), [SlpBlock(EXP, list(TRIALS), ["cam2"])]),
    ])


# ---------------------------------------------------------------------------
# SlpMapping
# ---------------------------------------------------------------------------

class TestSlpMapping:
    def test_toml_round_trip(self):
        mapping = _example_mapping()
        text = toml.dumps({"slp": mapping.to_config()})
        assert "[[slp.files]]" in text
        assert SlpMapping.from_config(toml.loads(text)) == mapping

    def test_from_config_without_slp_section(self):
        assert SlpMapping.from_config({"project": {"name": "x"}}).files == []

    def test_cell_map(self):
        cell_map = _example_mapping().cell_map()
        assert len(cell_map) == 15
        assert cell_map[(EXP, "t2", "cam1")] == Path("slp/cam0_cam1_early/cam0_cam1_early.slp")
        assert cell_map[(EXP, "t5", "cam0")] == Path("slp/cam0_cam1_late/cam0_cam1_late.slp")
        assert cell_map[(EXP, "t4", "cam2")] == Path("slp/cam2/cam2.slp")

    def test_complete(self):
        mapping = _example_mapping()
        assert mapping.missing_cells(_trials_df(), CAMERAS) == []
        assert mapping.duplicate_cells() == []
        assert mapping.is_complete(_trials_df(), CAMERAS)

    def test_missing_cells(self):
        mapping = _example_mapping()
        mapping.files[1].blocks[0].trials = ["t4"]
        assert mapping.missing_cells(_trials_df(), CAMERAS) == [
            (EXP, "t5", "cam0"), (EXP, "t5", "cam1"),
        ]
        assert not mapping.is_complete(_trials_df(), CAMERAS)

    def test_duplicate_cells(self):
        mapping = _example_mapping()
        mapping.files[0].blocks.append(SlpBlock(EXP, ["t1"], ["cam2"]))
        assert mapping.duplicate_cells() == [(EXP, "t1", "cam2")]
        assert not mapping.is_complete(_trials_df(), CAMERAS)

    def test_empty_mapping_not_complete(self):
        assert not SlpMapping().is_complete(_trials_df(), CAMERAS)

    def test_camera_without_video_is_not_a_cell(self):
        trials = _trials_df()
        trials.loc[trials["trial"] == "t1", "cam2"] = np.nan
        mapping = _example_mapping()
        mapping.files[2].blocks[0].trials = ["t2", "t3", "t4", "t5"]
        assert (EXP, "t1", "cam2") not in video_cells(trials, CAMERAS)
        assert mapping.is_complete(trials, CAMERAS)

    def test_video_cells_start_frames_per_experiment(self):
        trials = pd.DataFrame({
            "experiment": ["a", "b"],
            "trial": ["t1", "t1"],
            "cam0": ["a.mp4", "b.mp4"],
            "cam0_start": [3, 7],
        })
        cells = video_cells(trials, ["cam0"])
        assert cells[("a", "t1", "cam0")] == ("a.mp4", 3)
        assert cells[("b", "t1", "cam0")] == ("b.mp4", 7)


class TestFindSlpFiles:
    def test_recursive_skipping_models_backups_and_exclude(self, tmp_path):
        for rel in [
            "a.slp", "one/one.slp", "one/models/m1/labels_gt.train.slp",
            "two/deep/two.slp", "two/two.tmp.slp", "backup/x/one.slp", "out/slp/c.slp",
        ]:
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).touch()
        found = find_slp_files(tmp_path, skip=("models", "backup"), exclude=tmp_path / "out")
        assert [p.relative_to(tmp_path).as_posix() for p in found] == [
            "a.slp", "one/one.slp", "two/deep/two.slp"
        ]


class TestCompactBlocks:
    def test_example_compacts_to_one_block_per_file(self):
        expected = _example_mapping()
        order = list(video_cells(_trials_df(), CAMERAS))
        rebuilt = SlpMapping.from_cell_map(expected.cell_map(), order)
        assert {s.path: s.blocks for s in rebuilt.files} == {
            s.path: s.blocks for s in expected.files
        }

    def test_non_rectangular_needs_two_blocks(self):
        blocks = compact_blocks([(EXP, "t1", "cam0"), (EXP, "t2", "cam1")])
        assert blocks == [SlpBlock(EXP, ["t1"], ["cam0"]), SlpBlock(EXP, ["t2"], ["cam1"])]

    def test_camera_order_follows_first_appearance(self):
        blocks = compact_blocks([
            (EXP, "t1", "cam0"), (EXP, "t1", "cam1"),
            (EXP, "t2", "cam1"), (EXP, "t2", "cam0"),
        ])
        assert blocks == [SlpBlock(EXP, ["t1", "t2"], ["cam0", "cam1"])]

    def test_experiments_kept_separate(self):
        blocks = compact_blocks([("a", "t1", "cam0"), ("b", "t1", "cam0")])
        assert [b.experiment for b in blocks] == ["a", "b"]


class TestProjectMapping:
    def test_save_preserves_other_sections(self, tmp_path):
        config_path = tmp_path / "project.toml"
        config_path.write_text('[project]\nname = "demo"\n\n[charuco]\nsquares_x = 6\n')
        project = Project(root=tmp_path, cameras=CAMERAS, config=toml.load(config_path))

        mapping = _example_mapping()
        project.save_slp_mapping(mapping)

        saved = toml.load(config_path)
        assert saved["project"]["name"] == "demo"
        assert saved["charuco"]["squares_x"] == 6
        assert SlpMapping.from_config(saved) == mapping
        assert project.slp_mapping == mapping

    def test_save_creates_project_toml(self, tmp_path):
        project = Project(root=tmp_path, cameras=CAMERAS)
        project.save_slp_mapping(_example_mapping())
        assert (tmp_path / "project.toml").exists()


# ---------------------------------------------------------------------------
# compute_merge
# ---------------------------------------------------------------------------

MCAMS = ["cam1", "cam2"]


def _cam_values(
    x, y, state, source=None, score=NAN, sync=NAN, px=NAN, py=NAN, pscore=NAN
) -> list:
    """One camera's CAM_COORDS values; source defaults to label3d for labeled states."""
    if source is None:
        source = "label3d" if state in ("set", "predicted", "not_visible") else ""
    return [x, y, NAN, NAN, state, source, score, sync, px, py, pscore]


def _df(cells: dict) -> pd.DataFrame:
    """Build a points DataFrame.

    cells: {(frame, instance, bodypart): {camera: (x, y, state)}}, trial "t1".
    """
    col_tuples = [(c, coord) for c in MCAMS for coord in CAM_COORDS]
    col_tuples += [("3d", coord) for coord in THREE_D_COORDS]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["view", "coord"])
    idx, data = [], []
    for (frame, inst, bp), cams in cells.items():
        row = []
        for c in MCAMS:
            x, y, s, *extra = cams.get(c, (NAN, NAN, "not_set"))
            row += _cam_values(x, y, s, *extra)
        row += [NAN, NAN, NAN]
        idx.append((EXP, "t1", frame, inst, bp))
        data.append(row)
    return pd.DataFrame(
        data, index=pd.MultiIndex.from_tuples(idx, names=INDEX_NAMES), columns=columns
    )


def _get(df, frame, inst, bp, cam):
    row = df.loc[(EXP, "t1", frame, inst, bp)]
    return row[(cam, "x")], row[(cam, "y")], row[(cam, "state")]


def _assert_cell(df, frame, inst, bp, cam, x, y, state):
    cx, cy, cstate = _get(df, frame, inst, bp, cam)
    assert cstate == state
    if np.isnan(x):
        assert np.isnan(cx) and np.isnan(cy)
    else:
        assert cx == pytest.approx(x) and cy == pytest.approx(y)


class TestMergeTable:
    def test_missing_takes_slp_set(self):
        store = _df({})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "set")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 10, 20, "set")
        _assert_cell(r.df, 0, 0, "head", "cam2", NAN, NAN, "not_set")
        assert r.n_new == 1 and not r.conflicts
        assert r.changed_frames == {(EXP, "t1", 0)}

    def test_not_set_takes_slp_prediction(self):
        store = _df({(0, 0, "head"): {}})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 10, 20, "predicted")
        assert r.n_predictions == 1

    def test_prediction_replaces_prediction(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "predicted")}})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 10, 20, "predicted")
        assert r.n_predictions == 1 and not r.conflicts

    def test_user_beats_store_prediction(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "predicted")}})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "set")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 10, 20, "set")
        assert r.n_new == 1

    def test_prediction_in_user_instance_same_coords_unchanged(self):
        """save() writes a mixed instance as a user Instance, so predicted points reload as set."""
        store = _df({
            (0, 0, "head"): {"cam1": (1, 2, "set")},
            (0, 0, "tail"): {"cam1": (3, 4, "predicted")},
        })
        slp = _df({
            (0, 0, "head"): {"cam1": (1, 2, "set")},
            (0, 0, "tail"): {"cam1": (3, 4, "set")},
        })
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "tail", "cam1", 3, 4, "predicted")
        assert r.n_new == 0 and not r.changed_frames

    def test_store_user_beats_slp_prediction(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "set")}})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 1, 2, "set")
        assert r.n_new == 0 and r.n_predictions == 0 and not r.changed_frames

    def test_slp_missing_point_keeps_store_set(self):
        store = _df({
            (0, 0, "head"): {"cam1": (1, 2, "set")},
            (0, 0, "tail"): {"cam1": (3, 4, "set")},
        })
        slp = _df({
            (0, 0, "head"): {"cam1": (NAN, NAN, "not_set")},
            (0, 0, "tail"): {"cam1": (3, 4, "set")},
        })
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 1, 2, "set")
        assert not r.changed_frames and not r.conflicts

    def test_slp_missing_point_keeps_store_not_visible(self):
        store = _df({
            (0, 0, "head"): {"cam1": (1, 2, "not_visible")},
            (0, 0, "tail"): {"cam1": (3, 4, "set")},
        })
        slp = _df({(0, 0, "tail"): {"cam1": (3, 4, "set")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 1, 2, "not_visible")
        assert not r.changed_frames and not r.conflicts

    def test_same_coords_no_change(self):
        store = _df({(0, 0, "head"): {"cam1": (1.0, 2.0, "set")}})
        slp = _df({(0, 0, "head"): {"cam1": (1.005, 2.0, "set")}})
        r = compute_merge(store, slp)
        assert r.n_new == 0 and not r.conflicts and not r.changed_frames
        assert r.df_overwrite is r.df

    def test_different_coords_conflict(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "set")}})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "set")}})
        r = compute_merge(store, slp)
        assert r.conflicts == [(EXP, "t1", 0, 0, "head", "cam1")]
        assert r.conflict_frames == {(EXP, "t1", 0)}
        _assert_cell(r.df, 0, 0, "head", "cam1", 1, 2, "set")
        _assert_cell(r.df_overwrite, 0, 0, "head", "cam1", 10, 20, "set")

    def test_not_visible_vs_set_conflict(self):
        store = _df({
            (0, 0, "head"): {"cam1": (1, 2, "not_visible")},
            (0, 0, "tail"): {"cam1": (3, 4, "set")},
        })
        slp = _df({
            (0, 0, "head"): {"cam1": (10, 20, "set")},
            (0, 0, "tail"): {"cam1": (3, 4, "set")},
        })
        r = compute_merge(store, slp)
        assert r.conflicts == [(EXP, "t1", 0, 0, "head", "cam1")]
        _assert_cell(r.df, 0, 0, "head", "cam1", 1, 2, "not_visible")
        _assert_cell(r.df_overwrite, 0, 0, "head", "cam1", 10, 20, "set")

    def test_empty_slp_returns_store(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "set")}})
        r = compute_merge(store, _df({}))
        assert r.df.equals(store) and not r.changed_frames


class TestMergeFrameRules:
    def test_slp_predictions_ignored_when_store_has_user(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "set")}})
        slp = _df({
            (0, 0, "head"): {"cam1": (1, 2, "set")},
            (0, 1, "head"): {"cam1": (50, 60, "predicted")},
        })
        r = compute_merge(store, slp)
        assert (EXP, "t1", 0, 1, "head") not in r.df.index
        assert r.n_predictions == 0

    def test_slp_predictions_ignored_when_slp_has_user(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "predicted")}})
        slp = _df({
            (0, 0, "head"): {"cam1": (10, 20, "set")},
            (0, 1, "head"): {"cam1": (50, 60, "predicted")},
        })
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 10, 20, "set")
        assert (EXP, "t1", 0, 1, "head") not in r.df.index

    def test_fewer_predictions_clear_old_ones(self):
        store = _df({
            (0, 0, "head"): {"cam1": (1, 2, "predicted")},
            (0, 1, "head"): {"cam1": (5, 6, "predicted")},
        })
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam1", 10, 20, "predicted")
        _assert_cell(r.df, 0, 1, "head", "cam1", NAN, NAN, "not_set")

    def test_predictions_only_replaced_in_cameras_present_in_slp(self):
        store = _df({(0, 0, "head"): {
            "cam1": (1, 2, "predicted"), "cam2": (3, 4, "predicted"),
        }})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam2", 3, 4, "predicted")

    def test_frame_absent_from_slp_unchanged(self):
        store = _df({(5, 0, "head"): {"cam1": (1, 2, "predicted")}})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 5, 0, "head", "cam1", 1, 2, "predicted")
        assert r.changed_frames == {(EXP, "t1", 0)}


class TestMergeInstanceMatching:
    def test_instance_gap_round_trip(self):
        """Instance 0 has no cam1 data, so cam1's first SLP instance is store instance 1."""
        store = _df({
            (0, 0, "head"): {"cam2": (1, 2, "set")},
            (0, 1, "head"): {"cam1": (30, 40, "set")},
        })
        # What SleapAdapter.load() returns after save(): instances numbered per camera
        slp = _df({(0, 0, "head"): {"cam1": (30, 40, "set"), "cam2": (1, 2, "set")}})
        r = compute_merge(store, slp)
        assert not r.conflicts and r.n_new == 0 and not r.changed_frames

    def test_new_slp_instance_joins_instance_without_camera_data(self):
        """Single animal labeled in cam1 in label3d, then in cam2 in SLEAP."""
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "set")}})
        slp = _df({(0, 0, "head"): {"cam1": (1, 2, "set"), "cam2": (7, 8, "set")}})
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 0, "head", "cam2", 7, 8, "set")
        assert r.n_new == 1

    def test_new_instances_align_across_cameras(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "set"), "cam2": (3, 4, "set")}})
        slp = _df({
            (0, 0, "head"): {"cam1": (1, 2, "set"), "cam2": (3, 4, "set")},
            (0, 1, "head"): {"cam1": (50, 60, "set"), "cam2": (70, 80, "set")},
        })
        r = compute_merge(store, slp)
        _assert_cell(r.df, 0, 1, "head", "cam1", 50, 60, "set")
        _assert_cell(r.df, 0, 1, "head", "cam2", 70, 80, "set")
        assert not r.conflicts

    def test_numeric_trial_names_match(self):
        """points.csv may parse trial "1" as int; the SLP side is always str."""
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "set")}})
        store.index = store.index.set_levels([1], level="trial")
        slp = _df({(0, 0, "head"): {"cam1": (1, 2, "set")}})
        slp.index = slp.index.set_levels(["1"], level="trial")
        r = compute_merge(store, slp)
        assert not r.changed_frames and not r.conflicts


# ---------------------------------------------------------------------------
# SlpMappingEditor (Phase 8.9) — the worked example from phase_8.md
# ---------------------------------------------------------------------------

class TestSlpMappingEditor:
    def _editor(self) -> SlpMappingEditor:
        return SlpMappingEditor(SlpMapping(files=[]), _trials_df(), CAMERAS)

    def test_preset_per_camera_gives_one_file_per_camera(self):
        editor = self._editor()
        editor.apply_preset("{camera}.slp")
        assert sorted(p.name for p in editor.files) == ["cam0.slp", "cam1.slp", "cam2.slp"]
        assert all(len(editor.cells_of(p)) == len(TRIALS) for p in editor.files)

    def test_preset_join_split_reproduces_example(self):
        """Preset per camera → join cam0+cam1 → split trials 4–5 into their own file."""
        editor = self._editor()
        editor.apply_preset("{camera}.slp")
        joined = editor.join([Path("slp/cam0/cam0.slp"), Path("slp/cam1/cam1.slp")])
        assert joined == Path("slp/cam0/cam0.slp")
        assert len(editor.files) == 2

        late = editor.split_by_trial(joined, {(EXP, "t4"), (EXP, "t5")})
        assert late is not None
        assert len(editor.files) == 3

        editor.rename(joined, Path("slp/cam0_cam1_early/cam0_cam1_early.slp"))
        editor.rename(late, Path("slp/cam0_cam1_late/cam0_cam1_late.slp"))

        assert editor.errors() == []
        assert editor.mapping().cell_map() == _example_mapping().cell_map()

    def test_split_by_camera_undoes_a_join(self):
        editor = self._editor()
        editor.apply_preset("{camera}.slp")
        joined = editor.join([Path("slp/cam0/cam0.slp"), Path("slp/cam1/cam1.slp")])
        new_paths = editor.split_by_camera(joined)
        assert len(new_paths) == 2
        assert {c[2] for c in editor.cells_of(new_paths[0])} == {"cam0"}
        assert {c[2] for c in editor.cells_of(new_paths[1])} == {"cam1"}

    def test_split_by_trial_that_moves_everything_is_refused(self):
        editor = self._editor()
        editor.apply_preset("{camera}.slp")
        assert editor.split_by_trial(Path("slp/cam0/cam0.slp"), {(EXP, t) for t in TRIALS}) is None

    def test_errors_report_unassigned_cells(self):
        editor = SlpMappingEditor(SlpMapping(files=[]), _trials_df(), CAMERAS)
        assert editor.errors() == [f"{len(TRIALS) * len(CAMERAS)} video(s) unassigned"]

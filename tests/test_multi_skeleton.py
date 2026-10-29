"""Tests for per-camera skeletons: from_slp, merge_skeletons, ProjectSkeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio

from label3d.skeleton import (
    ProjectSkeleton,
    Skeleton,
    SkeletonConflict,
    merge_skeletons,
)

CAMS = ["cam0", "cam1", "cam2"]

FULL = Skeleton(
    nodes=["snout", "left_eye", "right_eye", "tail"],
    edges=[("snout", "left_eye"), ("snout", "right_eye"), ("snout", "tail")],
    symmetries=[("left_eye", "right_eye")],
    name="full",
)
NO_TAIL = Skeleton(
    nodes=["snout", "left_eye", "right_eye"],
    edges=[("snout", "left_eye"), ("snout", "right_eye")],
    symmetries=[("left_eye", "right_eye")],
    name="no_tail",
)


def write_slp(
    path: Path, skeleton: Skeleton, with_frame: bool = True, video_name: str = "vid.mp4"
) -> Path:
    """Write a .slp file with this skeleton and (optionally) one labelled frame."""
    sk = sio.Skeleton(list(skeleton.nodes), name=skeleton.name)
    for a, b in skeleton.edges:
        sk.add_edge(a, b)
    for a, b in skeleton.symmetries:
        sk.add_symmetry(a, b)
    video = sio.Video(filename=video_name, open_backend=False)
    frames = []
    if with_frame:
        pts = np.arange(len(skeleton.nodes) * 2, dtype=float).reshape(-1, 2)
        frames = [sio.LabeledFrame(
            video=video, frame_idx=0,
            instances=[sio.Instance.from_numpy(pts, skeleton=sk)],
        )]
    sio.save_file(
        sio.Labels(labeled_frames=frames, videos=[video], skeletons=[sk]), str(path)
    )
    return path


class TestFromSlp:
    def test_matches_full_load(self, tmp_path):
        path = write_slp(tmp_path / "a.slp", FULL)
        sk = Skeleton.from_slp(path)
        assert sk.same_as(FULL)
        assert sk.name == "full"
        loaded = sio.load_file(str(path)).skeleton
        assert sk.nodes == loaded.node_names

    def test_symmetry_in_node_order(self, tmp_path):
        reversed_sym = Skeleton(
            nodes=FULL.nodes, edges=FULL.edges, symmetries=[("right_eye", "left_eye")]
        )
        sk = Skeleton.from_slp(write_slp(tmp_path / "a.slp", reversed_sym))
        assert sk.symmetries == [("left_eye", "right_eye")]

    def test_file_without_frames(self, tmp_path):
        sk = Skeleton.from_slp(write_slp(tmp_path / "a.slp", NO_TAIL, with_frame=False))
        assert sk.nodes == NO_TAIL.nodes


class TestMergeSkeletons:
    def test_union_order_edges_symmetries(self):
        extra = Skeleton(
            nodes=["fin", "snout", "left_eye", "right_eye"],
            edges=[("left_eye", "snout"), ("right_eye", "snout"), ("snout", "fin")],
            symmetries=[("right_eye", "left_eye")],
        )
        union = merge_skeletons({"a": NO_TAIL, "b": FULL, "c": extra})
        assert union.nodes == ["snout", "left_eye", "right_eye", "tail", "fin"]
        assert {frozenset(e) for e in union.edges} == {
            frozenset(e) for e in FULL.edges + [("snout", "fin")]
        }
        assert len(union.edges) == 4
        assert union.symmetries == [("left_eye", "right_eye")]

    def test_identical_skeletons(self):
        union = merge_skeletons({"a": FULL, "b": FULL})
        assert union.same_as(FULL)

    def test_different_edges_are_merged(self):
        """Edges only affect drawing, so skeletons may disagree on them."""
        eye_star = Skeleton(
            nodes=FULL.nodes,
            edges=[("left_eye", "snout"), ("left_eye", "right_eye"), ("left_eye", "tail")],
            symmetries=FULL.symmetries,
        )
        union = merge_skeletons({"a.slp": FULL, "b.slp": eye_star})
        assert {frozenset(e) for e in union.edges} == {
            frozenset(e) for e in FULL.edges + eye_star.edges
        }
        assert len(union.edges) == 5  # snout - left_eye is in both

    def test_edge_to_node_missing_elsewhere_is_fine(self):
        # NO_TAIL lacks tail, so it can't disagree about snout - tail
        merge_skeletons({"a": FULL, "b": NO_TAIL})

    def test_symmetry_conflict(self):
        no_sym = Skeleton(nodes=FULL.nodes, edges=FULL.edges)
        with pytest.raises(SkeletonConflict) as exc:
            merge_skeletons({"a": FULL, "b": no_sym})
        assert exc.value.conflicts == [
            "symmetry left_eye - right_eye is in a but not in b"
        ]

    def test_two_symmetry_partners(self):
        other = Skeleton(
            nodes=["left_eye", "left_eye_2"], symmetries=[("left_eye", "left_eye_2")]
        )
        with pytest.raises(SkeletonConflict) as exc:
            merge_skeletons({"a": NO_TAIL, "b": other})
        assert exc.value.conflicts == [
            "left_eye has more than one symmetry partner: right_eye, left_eye_2"
        ]

    def test_lists_every_conflict(self):
        bare = Skeleton(nodes=FULL.nodes)
        with pytest.raises(SkeletonConflict) as exc:
            merge_skeletons({"a": FULL, "b": bare})
        assert len(exc.value.conflicts) == 1  # the symmetry; edges don't conflict

    def test_many_identical_labels_shortened(self):
        bare = Skeleton(nodes=["left_eye", "right_eye"])
        skeletons = {f"f{i}": FULL for i in range(5)} | {"g": bare}
        with pytest.raises(SkeletonConflict) as exc:
            merge_skeletons(skeletons)
        assert "f0, f1, f2 (+2 more)" in exc.value.conflicts[0]


def project_skeleton(**kwargs) -> ProjectSkeleton:
    return ProjectSkeleton.build(CAMS, **kwargs)


class TestSkeletonFor:
    def test_file_wins(self):
        ps = project_skeleton(
            by_file={Path("slp/c1.slp"): NO_TAIL},
            by_camera={"cam1": FULL},
            cell_files={("e", "t1", "cam1"): Path("slp/c1.slp")},
        )
        assert ps.skeleton_for(("e", "t1", "cam1")) is NO_TAIL
        # A cell whose file does not exist gets the assigned skeleton
        assert ps.skeleton_for(("e", "t2", "cam1")) is FULL

    def test_inferred_from_camera_files(self):
        ps = project_skeleton(
            by_file={Path("slp/a.slp"): FULL, Path("slp/b.slp"): NO_TAIL},
            cell_files={
                ("e", "t1", "cam0"): Path("slp/a.slp"),
                ("e", "t1", "cam1"): Path("slp/b.slp"),
            },
        )
        assert ps.skeleton_for(("e", "t2", "cam1")) is NO_TAIL
        assert ps.skeleton_for(("e", "t2", "cam0")) is FULL
        # No files for cam2: the union
        assert ps.skeleton_for(("e", "t2", "cam2")).same_as(FULL)

    def test_disagreeing_files_fall_back_to_union(self):
        other = Skeleton(nodes=["snout", "fin"], name="other")
        ps = project_skeleton(
            by_file={Path("slp/a.slp"): NO_TAIL, Path("slp/b.slp"): other},
            cell_files={
                ("e", "t1", "cam1"): Path("slp/a.slp"),
                ("e", "t2", "cam1"): Path("slp/b.slp"),
            },
        )
        assert ps.nodes_for(("e", "t3", "cam1")) == set(ps.union.nodes)

    def test_partial_nodes(self):
        ps = project_skeleton(by_camera={"cam0": FULL, "cam1": NO_TAIL, "cam2": FULL})
        assert ps.partial_nodes("e", "t") == {"tail": ["cam0", "cam2"]}
        assert ps.cameras_for("e", "t", "snout") == CAMS
        assert ps.available("e", "t")["cam1"] == set(NO_TAIL.nodes)

    def test_build_reports_conflicts(self):
        no_sym = Skeleton(nodes=FULL.nodes, edges=FULL.edges)
        with pytest.raises(SkeletonConflict) as exc:
            project_skeleton(by_file={Path("slp/a.slp"): FULL}, by_camera={"cam1": no_sym})
        assert "slp/a.slp" in exc.value.conflicts[0]
        assert "assigned to cam1" in exc.value.conflicts[0]


class TestMappingProblems:
    def setup_method(self):
        self.ps = project_skeleton(by_camera={"cam0": FULL, "cam1": NO_TAIL, "cam2": FULL})

    def test_one_file_per_camera_is_fine(self):
        cells = {("e", "t", c): Path(f"slp/{c}.slp") for c in CAMS}
        check = self.ps.mapping_problems(cells, {})
        assert check.ok and not check.warnings

    def test_new_file_mixing_skeletons(self):
        cells = {("e", "t", c): Path("slp/t.slp") for c in CAMS}
        check = self.ps.mapping_problems(cells, {})
        assert check.errors == [
            "slp/t.slp: cameras with different skeletons: "
            "cam0, cam2 (full); cam1 (no_tail)"
        ]

    def test_new_file_same_skeletons(self):
        cells = {("e", "t", c): Path("slp/t.slp") for c in ("cam0", "cam2")}
        assert self.ps.mapping_problems(cells, {}).ok

    def test_existing_file_against_assignment(self):
        cells = {("e", "t", "cam1"): Path("slp/a.slp")}
        check = self.ps.mapping_problems(cells, {Path("slp/a.slp"): FULL})
        assert check.errors == [
            "slp/a.slp: cam1 is assigned skeleton no_tail, but the file has full"
        ]

    def test_cell_moved_into_smaller_file_warns(self):
        ps = project_skeleton(
            by_file={Path("slp/a.slp"): FULL, Path("slp/b.slp"): NO_TAIL},
            cell_files={
                ("e", "t1", "cam0"): Path("slp/a.slp"),
                ("e", "t1", "cam1"): Path("slp/b.slp"),
            },
        )
        moved = {
            ("e", "t1", "cam0"): Path("slp/b.slp"),
            ("e", "t1", "cam1"): Path("slp/b.slp"),
        }
        check = ps.mapping_problems(
            moved, {Path("slp/a.slp"): FULL, Path("slp/b.slp"): NO_TAIL}
        )
        assert check.ok
        assert check.lost == {("e", "t1", "cam0"): ["tail"]}
        assert check.warnings == [
            "slp/b.slp: 1 video(s) would lose node(s) not in the file's skeleton: tail"
        ]


# ---------------------------------------------------------------------------
# Writing .slp files and filtering imports (phase 14.3)
# ---------------------------------------------------------------------------

from label3d.adapters.sleap_adapter import SleapAdapter  # noqa: E402
from label3d.project import drop_unavailable  # noqa: E402
from tests.test_sleap_adapter import (  # noqa: E402
    CAMERAS as ADAPTER_CAMS,
    EXP,
    _df,
    _one_file_per_camera,
    _trials_df,
)

HEAD_TAIL = Skeleton(nodes=["head", "tail"], edges=[("head", "tail")], name="head_tail")
HEAD = Skeleton(nodes=["head"], name="head_only")


def _points():
    return _df({
        ("t1", 0, 0, "head"): {"cam1": (1.0, 2.0, "set"), "cam2": (3.0, 4.0, "set")},
        ("t1", 0, 0, "tail"): {"cam1": (5.0, 6.0, "set"), "cam2": (7.0, 8.0, "set")},
    })


def _write_existing(tmp_path, rel: str, skeleton: Skeleton) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    write_slp(path, skeleton, with_frame=False)


class TestWriteKeepsFileSkeleton:
    def test_existing_file_keeps_smaller_skeleton(self, tmp_path):
        _write_existing(tmp_path, "slp/cam2.slp", HEAD)
        skeletons = ProjectSkeleton.build(ADAPTER_CAMS, by_camera={"cam1": HEAD_TAIL})
        SleapAdapter().save(
            _points(), cameras=ADAPTER_CAMS, trials_df=_trials_df(),
            mapping=_one_file_per_camera(), root=tmp_path, skeletons=skeletons,
        )
        cam2 = sio.load_file(str(tmp_path / "slp/cam2.slp"))
        assert cam2.skeleton.node_names == ["head"]
        assert cam2.skeleton.name == "head_only"
        [lf] = cam2.labeled_frames
        assert lf.instances[0].numpy().tolist() == [[3.0, 4.0]]
        # cam1's new file is created with cam1's skeleton
        cam1 = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        assert cam1.skeleton.node_names == ["head", "tail"]

    def test_existing_file_kept_even_with_plain_skeleton(self, tmp_path):
        _write_existing(tmp_path, "slp/cam1.slp", HEAD)
        SleapAdapter().save(
            _points(), cameras=ADAPTER_CAMS, skeleton=HEAD_TAIL, trials_df=_trials_df(),
            mapping=_one_file_per_camera(), root=tmp_path,
        )
        cam1 = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        assert cam1.skeleton.node_names == ["head"]

    def test_new_files_get_camera_skeletons_with_symmetries(self, tmp_path):
        sym = Skeleton(
            nodes=["head", "left", "right"], symmetries=[("left", "right")], name="sym"
        )
        skeletons = ProjectSkeleton.build(
            ADAPTER_CAMS, by_camera={"cam1": sym, "cam2": HEAD}
        )
        SleapAdapter().save(
            _points(), cameras=ADAPTER_CAMS, trials_df=_trials_df(),
            mapping=_one_file_per_camera(), root=tmp_path, skeletons=skeletons,
        )
        cam1 = Skeleton.from_slp(tmp_path / "slp/cam1.slp")
        assert cam1.same_as(sym) and cam1.name == "sym"
        assert Skeleton.from_slp(tmp_path / "slp/cam2.slp").nodes == ["head"]

    def test_new_file_mixing_skeletons_raises(self, tmp_path):
        from label3d.slp_sync import SlpBlock, SlpFileSpec, SlpMapping
        mapping = SlpMapping(files=[
            SlpFileSpec(Path("slp/all.slp"), [SlpBlock(EXP, ["t1", "t2"], ADAPTER_CAMS)]),
        ])
        skeletons = ProjectSkeleton.build(
            ADAPTER_CAMS, by_camera={"cam1": HEAD_TAIL, "cam2": HEAD}
        )
        with pytest.raises(ValueError, match="different skeletons"):
            SleapAdapter().save(
                _points(), cameras=ADAPTER_CAMS, trials_df=_trials_df(),
                mapping=mapping, root=tmp_path, skeletons=skeletons,
            )


class TestDropUnavailable:
    def test_clears_and_counts(self):
        skeletons = ProjectSkeleton.build(
            ADAPTER_CAMS, by_camera={"cam1": HEAD_TAIL, "cam2": HEAD}
        )
        df, dropped = drop_unavailable(_points(), skeletons)
        assert dropped == {EXP: {("cam2", "tail"): 1}}
        tail = df.xs("tail", level="bodypart").iloc[0]
        assert tail[("cam2", "state")] == "not_set"
        assert np.isnan(tail[("cam2", "x")])
        assert tail[("cam1", "state")] == "set"
        head = df.xs("head", level="bodypart").iloc[0]
        assert head[("cam2", "x")] == 3.0

    def test_nothing_to_drop_returns_same_df(self):
        skeletons = ProjectSkeleton.build(ADAPTER_CAMS, by_camera={"cam1": HEAD_TAIL})
        points = _points()
        df, dropped = drop_unavailable(points, skeletons)
        assert df is points and dropped == {}


# ---------------------------------------------------------------------------
# Synthetic data with missing_nodes (phase 14.9)
# ---------------------------------------------------------------------------

from synthetic import camera_skeleton, write_slp_files  # noqa: E402


class TestSyntheticMissingNodes:
    NODES = ["snout", "left_eye", "dorsal"]
    EDGES = [("snout", "left_eye"), ("snout", "dorsal"), ("left_eye", "dorsal")]

    def test_camera_skeleton_drops_node_and_its_edges(self):
        skel = camera_skeleton(self.NODES, self.EDGES, {"dorsal"})
        assert skel["name"] == "synthetic_without_dorsal"
        assert skel["nodes"] == ["snout", "left_eye"]
        assert skel["edges"] == [["snout", "left_eye"]]
        assert camera_skeleton(self.NODES, self.EDGES, set())["name"] == "synthetic"

    def test_slp_file_without_missing_node(self, tmp_path):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data = {cam: [(tmp_path / f"{cam}.mp4", [pts])] for cam in ("cam1", "cam3")}
        write_slp_files(data, self.NODES, self.EDGES, tmp_path, {"cam3": {"left_eye"}})
        assert Skeleton.from_slp(tmp_path / "cam1.slp").nodes == self.NODES
        cam3 = sio.load_file(str(tmp_path / "cam3.slp"))
        assert cam3.skeleton.node_names == ["snout", "dorsal"]
        assert cam3.labeled_frames[0].instances[0].numpy().tolist() == [[1, 2], [5, 6]]


# ---------------------------------------------------------------------------
# Project loading (phase 14.1)
# ---------------------------------------------------------------------------

import toml  # noqa: E402

from label3d.project import Project  # noqa: E402
from tests.test_project_sync import (  # noqa: E402
    BODYPARTS as SYNC_NODES,
    CAMERAS as SYNC_CAMS,
    EXP as SYNC_EXP,
    project,  # noqa: F401  (fixture)
)

HEAD_ONLY = Skeleton(nodes=["head"], name="head_only")
# Conflicts with the sync fixture's skeleton, which has no symmetries
SYMMETRIC = Skeleton(
    nodes=SYNC_NODES, edges=[("head", "tail")], symmetries=[("head", "tail")],
    name="symmetric",
)


def _set_config(project: Project, **sections) -> None:
    config = toml.load(project.root / "project.toml")
    for key, value in sections.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    (project.root / "project.toml").write_text(toml.dumps(config))
    project.config = config


class TestProjectSkeletons:
    def test_assigned_skeleton_gives_every_camera_its_nodes(self, project):
        section = toml.load(project.root / "project.toml")["skeleton"]
        [entry] = section["assigned"]
        assert entry["nodes"] == SYNC_NODES
        assert entry["cameras"] == SYNC_CAMS
        assert project.skeletons.by_camera["cam0"].nodes == SYNC_NODES

    def test_file_skeleton_per_camera(self, project):
        write_slp(project.root / "slp/cam1.slp", HEAD_ONLY, with_frame=False)
        _set_config(project, skeleton=None)
        # The only skeleton is cam1's file, so the union loses "tail"
        assert project.load_skeletons() is True
        assert project.skeleton.nodes == ["head"]
        assert project.skeletons.nodes_for((SYNC_EXP, "t1", "cam0")) == {"head"}

    def test_file_wins_over_assignment(self, project):
        write_slp(project.root / "slp/cam1.slp", HEAD_ONLY, with_frame=False)
        project.load_skeletons()
        assert project.skeletons.nodes_for((SYNC_EXP, "t1", "cam1")) == {"head"}
        assert project.skeletons.nodes_for((SYNC_EXP, "t1", "cam0")) == set(SYNC_NODES)
        assert project.skeletons.partial_nodes(SYNC_EXP, "t1") == {"tail": ["cam0"]}

    def test_no_mapping_matches_videos_by_basename(self, project):
        _set_config(project, slp=None)
        write_slp(
            project.root / "slp/some.slp", HEAD_ONLY, with_frame=False,
            video_name="/elsewhere/cam1_t1.mp4",
        )
        project.load_skeletons()
        assert project.skeletons.nodes_for((SYNC_EXP, "t1", "cam1")) == {"head"}
        # No file holds t2's cam1 video: the assigned skeleton
        assert project.skeletons.nodes_for((SYNC_EXP, "t2", "cam1")) == set(SYNC_NODES)

    def test_assign_skeleton_moves_cameras(self, project):
        assert project.assign_skeleton(HEAD_ONLY, ["cam1"]) is False
        entries = toml.load(project.root / "project.toml")["skeleton"]["assigned"]
        assert [(e["name"], e["cameras"]) for e in entries] == [
            ("skeleton", ["cam0"]), ("head_only", ["cam1"])
        ]
        assert project.skeletons.nodes_for((SYNC_EXP, "t1", "cam1")) == {"head"}
        # Assigning the first topology back reuses its entry; head_only is dropped
        project.assign_skeleton(Skeleton(nodes=["tail", "head"], edges=[("head", "tail")]),
                                ["cam1"])
        entries = toml.load(project.root / "project.toml")["skeleton"]["assigned"]
        assert [(e["name"], e["cameras"]) for e in entries] == [("skeleton", SYNC_CAMS)]

    def test_assign_skeleton_adds_offsets(self, project):
        sk = Skeleton(nodes=SYNC_NODES, edges=[("head", "tail")],
                      node_offsets_cm={"head": [1.0, 2.0, 3.0]})
        project.assign_skeleton(sk, SYNC_CAMS)
        assert project.skeleton.node_offsets_cm == {"head": [1.0, 2.0, 3.0]}

    def test_conflict_is_recorded_not_raised(self, project):
        write_slp(project.root / "slp/cam1.slp", SYMMETRIC, with_frame=False)
        loaded = Project.load(project.root)
        assert loaded.skeleton_conflicts == [
            "symmetry head - tail is in slp/cam1.slp but not in "
            "the skeleton assigned to cam0, the skeleton assigned to cam1"
        ]
        assert loaded.skeleton.nodes == SYNC_NODES
        # A later reload with a conflict keeps the skeletons it had
        before = loaded.skeletons
        assert loaded.load_skeletons() is False
        assert loaded.skeletons is before


# ---------------------------------------------------------------------------
# One skeleton per .slp file (phase 14.6)
# ---------------------------------------------------------------------------

from label3d.export import ExportOptions, export_points  # noqa: E402
from label3d.slp_sync import SlpBlock, SlpFileSpec, SlpMapping  # noqa: E402
from tests.test_project_sync import TRIALS as SYNC_TRIALS  # noqa: E402


def _one_file(name: str = "slp/all.slp") -> SlpMapping:
    return SlpMapping(files=[
        SlpFileSpec(Path(name), [SlpBlock(SYNC_EXP, list(SYNC_TRIALS), list(SYNC_CAMS))])
    ])


class TestMappingCheckInProject:
    def test_per_camera_files_are_fine(self, project):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        assert project.check_slp_mapping().ok
        assert project.slp_write_problems() == []

    def test_one_file_mixing_skeletons(self, project):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        problems = project.slp_write_problems(_one_file())
        assert problems == [
            "slp/all.slp: cameras with different skeletons: cam0 (skeleton); cam1 (head_only)"
        ]

    def test_conflicts_are_problems(self, project):
        unnamed = Skeleton(SYNC_NODES, edges=SYMMETRIC.edges, symmetries=SYMMETRIC.symmetries)
        write_slp(project.root / "slp/cam1.slp", unnamed, with_frame=False)
        project.load_skeletons()
        assert project.skeleton_conflicts
        problems = project.slp_write_problems()
        assert problems[:len(project.skeleton_conflicts)] == project.skeleton_conflicts
        # The file also differs from the skeleton assigned to its camera
        assert problems[-1] == (
            "slp/cam1.slp: cam1 is assigned skeleton skeleton, but the file has unnamed"
        )

    def test_export_skips_slp_files_with_problems(self, project, tmp_path):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        project.save_slp_mapping(_one_file())
        result = export_points(project, ExportOptions(output=tmp_path / "out.csv"))
        assert result.slp_files_written is None
        assert result.slp_problems and "different skeletons" in result.slp_problems[0]
        assert not (project.root / "slp/all.slp").exists()


class TestSetupDialogSkeletons:
    def _dialog(self, project):
        from label3d.widgets.slp_setup_dialog import SlpSetupDialog
        return SlpSetupDialog(project)

    def _file_texts(self, dialog):
        return [dialog._file_list.item(i).text() for i in range(dialog._file_list.count())]

    def test_single_skeleton_shows_no_tags(self, project):
        dialog = self._dialog(project)
        assert self._file_texts(dialog) == ["cam0.slp", "cam1.slp"]
        assert dialog._preset_combo.currentData() == "per_camera"

    def test_mixed_file_disables_apply(self, project):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        dialog = self._dialog(project)
        assert dialog._preset_combo.currentData() == "per_experiment_camera"
        assert self._file_texts(dialog) == ["cam0.slp  [skeleton]", "cam1.slp  [head_only]"]
        assert dialog._apply_btn.isEnabled()

        dialog._pattern_edit.setText("labels.slp")
        dialog._on_apply_preset()
        assert self._file_texts(dialog) == ["labels.slp  [mixed]"]
        assert not dialog._apply_btn.isEnabled()
        assert "different skeletons" in dialog._status.text()

    def test_rename_ignores_skeleton_tag(self, project):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        dialog = self._dialog(project)
        item = dialog._file_list.item(1)
        item.setText("renamed.slp  [head_only]")
        assert Path("slp/renamed.slp") in dialog._editor.files


# ---------------------------------------------------------------------------
# Load skeleton for chosen cameras (phase 14.2)
# ---------------------------------------------------------------------------

from label3d.points import PointsStore, read_points_df  # noqa: E402
from label3d.widgets.load_skeleton_dialog import LoadSkeletonDialog, read_skeleton  # noqa: E402

def _write_json(path: Path, skeleton: Skeleton) -> Path:
    import json
    path.write_text(json.dumps(skeleton.to_dict()))
    return path


class TestPreviewAndRemove:
    def test_preview_counts_points_in_chosen_cameras(self, project):
        preview = project.preview_assignment(HEAD_ONLY, ["cam1"])
        assert preview.conflicts == [] and preview.mapping_errors == []
        assert preview.outside == {"tail": 2}   # one per trial in cam1
        assert preview.skeletons.nodes_for((SYNC_EXP, "t1", "cam1")) == {"head"}
        # Nothing changed
        assert "cam1" in project.skeletons.by_camera
        assert project.skeletons.by_camera["cam1"].nodes == SYNC_NODES

    def test_preview_conflict(self, project):
        preview = project.preview_assignment(SYMMETRIC, ["cam1"])
        assert preview.conflicts and preview.skeletons is None

    def test_preview_mapping_error(self, project):
        project.save_slp_mapping(_one_file())
        preview = project.preview_assignment(HEAD_ONLY, ["cam1"])
        assert preview.mapping_errors == [
            "slp/all.slp: cameras with different skeletons: cam0 (skeleton); cam1 (head_only)"
        ]

    def test_remove_clears_only_the_camera(self, project):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        assert project.remove_points_outside(cameras=["cam1"]) == 2
        df = project.points.df
        tail = df.xs("tail", level="bodypart")
        assert (tail[("cam1", "state")] == "not_set").all()
        assert (tail[("cam0", "state")] == "set").all()
        on_disk = read_points_df(project.experiments[SYNC_EXP].root / "points.parquet")
        assert (on_disk.xs("tail", level="bodypart")[("cam1", "state")] == "not_set").all()

    def test_remove_drops_nodes_in_no_skeleton(self, project):
        project.assign_skeleton(HEAD_ONLY, SYNC_CAMS)
        assert project.skeleton.nodes == ["head"]
        assert project.remove_points_outside() == 4
        assert set(project.points.df.index.get_level_values("bodypart")) == {"head"}


class TestLoadSkeletonDialog:
    def test_read_json_and_slp(self, tmp_path):
        assert read_skeleton(_write_json(tmp_path / "fish.json", HEAD_ONLY)).nodes == ["head"]
        slp = write_slp(tmp_path / "model.slp", Skeleton(nodes=["a", "b"]), with_frame=False)
        sk = read_skeleton(slp)
        assert sk.nodes == ["a", "b"] and sk.name == "model"

    def test_all_cameras_by_default(self, project, tmp_path):
        dialog = LoadSkeletonDialog(project, _write_json(tmp_path / "h.json", HEAD_ONLY))
        assert dialog.cameras() == SYNC_CAMS
        assert dialog._ok.isEnabled()
        assert dialog.preview().outside == {"tail": 4}
        assert dialog._remove_box.isVisibleTo(dialog)
        assert not dialog.remove_points()   # off by default

    def test_chosen_cameras(self, project, tmp_path):
        dialog = LoadSkeletonDialog(project, _write_json(tmp_path / "h.json", HEAD_ONLY))
        dialog._camera_boxes["cam0"].setChecked(False)
        assert not dialog._all_box.isChecked()
        assert dialog.cameras() == ["cam1"]
        assert dialog.preview().outside == {"tail": 2}
        dialog._camera_boxes["cam1"].setChecked(False)
        assert not dialog._ok.isEnabled()
        dialog._all_box.setChecked(True)
        assert dialog.cameras() == SYNC_CAMS

    def test_conflict_disables_ok(self, project, tmp_path):
        dialog = LoadSkeletonDialog(
            project, _write_json(tmp_path / "x.json", SYMMETRIC)
        )
        dialog._camera_boxes["cam0"].setChecked(False)
        assert not dialog._ok.isEnabled()
        assert "conflict" in dialog._checks.text()

    def test_mapping_problem_is_a_warning(self, project, tmp_path):
        project.save_slp_mapping(_one_file())
        dialog = LoadSkeletonDialog(project, _write_json(tmp_path / "h.json", HEAD_ONLY))
        dialog._camera_boxes["cam0"].setChecked(False)
        assert dialog._ok.isEnabled()
        assert "Set up Sleap files" in dialog._checks.text()

    def test_existing_files_note(self, project, tmp_path):
        project.write_slp_files()
        project.load_skeletons()
        dialog = LoadSkeletonDialog(project, _write_json(tmp_path / "h.json", HEAD_ONLY))
        assert "keep their skeleton" in dialog._camera_notes["cam1"].text()
        assert "(from its .slp files)" not in dialog._camera_notes["cam1"].text()
        assert "(assigned)" in dialog._camera_notes["cam1"].text()

    def test_unreadable_file(self, project, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{}")
        dialog = LoadSkeletonDialog(project, bad)
        assert dialog.skeleton() is None
        assert not dialog._ok.isEnabled()
        assert "Could not read" in dialog._skeleton_label.text()


# ---------------------------------------------------------------------------
# Triangulation and instance init (phase 14.4)
# ---------------------------------------------------------------------------

from label3d.geometry import (  # noqa: E402
    load_cameras,
    mean_reprojection_errors,
    triangulate_frame,
    triangulate_rows,
    triangulate_trial,
)
from label3d.instance_init import init_average, init_random, init_reprojected  # noqa: E402
from tests.conftest import TEST_PROJECT, requires_test_project  # noqa: E402

SYN_EXP = TEST_PROJECT / "experiments" / "exp1"
SYN_CAMS = ["cam1", "cam2", "cam3"]
SYN_FULL = Skeleton(nodes=["snout", "left_eye", "dorsal"], name="full")
SYN_NO_DORSAL = Skeleton(nodes=["snout", "left_eye"], name="no_dorsal")


@pytest.fixture
def syn_cameras():
    return load_cameras(next((SYN_EXP / "calibrations").glob("*.toml")))


@pytest.fixture
def syn_store(qapp):
    return PointsStore.from_parquet(SYN_EXP / "points.parquet")


def _syn_skeletons(**by_camera) -> ProjectSkeleton:
    return ProjectSkeleton.build(SYN_CAMS, by_camera={
        cam: by_camera.get(cam, SYN_FULL) for cam in SYN_CAMS
    })


@requires_test_project
class TestTriangulationWithSkeletons:
    def test_unavailable_point_is_ignored(self, syn_cameras, syn_store):
        """cam3's dorsal points stay set but don't count: same as clearing them."""
        skeletons = _syn_skeletons(cam3=SYN_NO_DORSAL)
        df = syn_store.df
        p3d, reproj = triangulate_rows(syn_cameras, df, skeletons)

        cleared = df.copy()
        dorsal = cleared.index.get_level_values("bodypart") == "dorsal"
        cleared.loc[dorsal, ("cam3", "state")] = "not_set"
        expected, _ = triangulate_rows(syn_cameras, cleared)
        np.testing.assert_allclose(p3d, expected, equal_nan=True)
        # The mean error ignores cam3 for dorsal but still has cam1 and cam2
        errors = mean_reprojection_errors(df, reproj, skeletons)
        assert np.isfinite(errors[dorsal]).all()

    def test_node_in_one_camera_is_not_triangulated(self, syn_cameras, syn_store):
        skeletons = _syn_skeletons(cam1=SYN_NO_DORSAL, cam2=SYN_NO_DORSAL)
        trial = syn_store.df.index.get_level_values("trial")[0]
        triangulate_trial(syn_cameras, syn_store, trial, skeletons=skeletons)
        df = syn_store.df
        dorsal = df.index.get_level_values("bodypart") == "dorsal"
        assert df.loc[dorsal, ("3d", "x")].isna().all()
        assert df.loc[~dorsal, ("3d", "x")].notna().all()

    def test_triangulate_frame_uses_skeletons(self, syn_cameras, syn_store):
        skeletons = _syn_skeletons(cam1=SYN_NO_DORSAL, cam2=SYN_NO_DORSAL)
        trial = syn_store.df.index.get_level_values("trial")[0]
        errors = triangulate_frame(syn_cameras, syn_store, trial, 0, 0, skeletons)
        assert np.isnan(errors["dorsal"]) and np.isfinite(errors["snout"])


class TestInitWithNodes:
    def test_random_only_available_nodes(self, simple_skeleton):
        coords = init_random(
            simple_skeleton, "cam0", None, (100.0, 100.0), (640, 480),
            nodes={"snout", "dorsal"},
        )
        assert set(coords) == {"snout", "dorsal"}

    def test_random_with_offsets_only_available_nodes(self, simple_skeleton, syn_cameras):
        simple_skeleton.node_offsets_cm = {"snout": [1.0, 0, 0], "left_eye": [0, 1.0, 0]}
        coords = init_random(
            simple_skeleton, "cam1", syn_cameras, (600.0, 500.0), (1280, 1024),
            nodes={"snout", "left_eye"},
        )
        assert set(coords) == {"snout", "left_eye"}

    def test_average_anchor_is_first_available_node(self, simple_skeleton):
        store = PointsStore.create_empty(cameras=["cam0"], bodyparts=[], experiment_name="exp")
        for trial, (sx, sy) in [("t1", (100.0, 100.0)), ("t2", (200.0, 300.0))]:
            store.set_point(trial, 0, 0, "left_eye", "cam0", sx, sy, state="set")
            store.set_point(trial, 0, 0, "dorsal", "cam0", sx, sy + 40, state="set")
        coords, fallback = init_average(
            store, "exp", "cam0", simple_skeleton, (50.0, 60.0), (640, 480),
            nodes={"left_eye", "dorsal"},
        )
        assert not fallback
        assert coords == {"left_eye": (50.0, 60.0), "dorsal": (50.0, 100.0)}

    @requires_test_project
    def test_reprojected_respects_available(self, syn_cameras, syn_store):
        trial = syn_store.df.index.get_level_values("trial")[0]
        all_coords = init_reprojected(syn_store, "exp1", trial, 0, 0, "cam3", syn_cameras)
        assert set(all_coords) == {"snout", "left_eye", "dorsal"}
        no_dorsal = {"cam1": {"snout", "left_eye", "dorsal"}, "cam2": {"snout", "left_eye"},
                     "cam3": {"snout", "left_eye"}}
        coords = init_reprojected(
            syn_store, "exp1", trial, 0, 0, "cam3", syn_cameras, available=no_dorsal
        )
        assert set(coords) == {"snout", "left_eye"}
        # Only cam1 has dorsal as a source, so it can't be placed in cam2 either
        coords = init_reprojected(
            syn_store, "exp1", trial, 0, 0, "cam2", syn_cameras,
            available={**no_dorsal, "cam2": {"snout", "left_eye", "dorsal"}},
        )
        assert "dorsal" not in coords


# ---------------------------------------------------------------------------
# Label view (phase 14.5)
# ---------------------------------------------------------------------------

from PySide6.QtWidgets import QGraphicsScene  # noqa: E402

from label3d.state import ProjectState  # noqa: E402
from label3d.widgets.bodypart_list import BodypartList  # noqa: E402
from label3d.widgets.camera_panel import CameraPanel  # noqa: E402
from label3d.widgets.epipolar_overlay import EpipolarOverlay  # noqa: E402
from label3d.widgets.skeleton_overlay import SkeletonOverlay  # noqa: E402

THREE_CAMS = ["cam0", "cam1", "cam2"]
SYNC_EDGE_SK = Skeleton(nodes=["head", "tail"], edges=[("head", "tail")], name="skeleton")


def _state(project, trial="t1") -> ProjectState:
    state = ProjectState()
    state.points_store = project.points
    state.set_skeletons(project.skeletons)
    state.current_trial = trial
    return state


class TestStateAvailability:
    def test_available_nodes_and_cameras(self, project):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        state = _state(project)
        assert state.available_nodes("cam1") == {"head"}
        assert state.available_nodes("cam0") == set(SYNC_NODES)
        assert state.node_cameras("tail", SYNC_CAMS) == ["cam0"]
        assert state.skeleton.nodes == SYNC_NODES

    def test_each_camera_gets_its_own_edges(self, project):
        eye_star = Skeleton(nodes=["head", "tail"], edges=[("tail", "head")], name="star")
        project.assign_skeleton(eye_star, ["cam1"])
        state = _state(project)
        # cam1's file exists, so its skeleton wins over the assignment (rule 1)
        assert state.skeleton_for_camera("cam1") == project.skeletons.skeleton_for(
            (project.points.experiment_name, "t1", "cam1")
        )
        unmapped = ProjectSkeleton.build(SYNC_CAMS, by_camera={
            "cam0": SYNC_EDGE_SK, "cam1": eye_star,
        })
        state.set_skeletons(unmapped)
        assert state.skeleton_for_camera("cam1").name == "star"
        assert state.skeleton_for_camera("cam0").name == "skeleton"

    def test_no_skeletons_means_everything(self):
        state = ProjectState()
        assert state.available_nodes("cam0") is None
        assert state.node_cameras("x", SYNC_CAMS) == SYNC_CAMS


class TestBodypartListMarks:
    def _texts(self, widget):
        return [widget._list.item(i).text() for i in range(widget._list.count())]

    def test_one_camera_and_partial_marks(self, project):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        state = _state(project)
        widget = BodypartList(state)
        widget.set_skeleton(state.skeleton)
        # t1 frame 0: head in both cameras; tail only in cam0's skeleton
        assert self._texts(widget) == ["head  [2/2]", "tail 2D  [1/1]"]
        assert widget._list.item(1).toolTip() == "Only in cam0 — not triangulated"
        assert widget._key.isVisibleTo(widget)

    def test_partial_with_three_cameras(self):
        skeletons = ProjectSkeleton.build(THREE_CAMS, by_camera={
            "cam0": SYNC_EDGE_SK, "cam1": HEAD_ONLY, "cam2": SYNC_EDGE_SK,
        })
        state = ProjectState()
        state.points_store = PointsStore.create_empty(
            cameras=THREE_CAMS, bodyparts=[], experiment_name="e"
        )
        state.set_skeletons(skeletons)
        state.current_trial = "t"
        widget = BodypartList(state)
        widget.set_skeleton(state.skeleton)
        assert self._texts(widget) == ["head", "tail ◐"]
        assert widget._list.item(1).toolTip() == "Only in cam0, cam2"

    def test_full_skeleton_no_marks(self, project):
        state = _state(project)
        widget = BodypartList(state)
        widget.set_skeleton(state.skeleton)
        assert self._texts(widget) == ["head  [2/2]", "tail  [2/2]"]
        assert not widget._key.isVisibleTo(widget)


class TestOverlays:
    def test_skeleton_overlay_skips_unavailable(self, project, qapp):
        overlay = SkeletonOverlay(QGraphicsScene(), "cam1")
        overlay.update(project.points, project.skeleton, "t1", 0, 0, "head",
                       available={"head"})
        assert overlay._active_points == {(0, "head")}
        assert overlay.hit_test(overlay.get_point_item(0, "head").pos()) == (0, "head")
        assert not any(item.isVisible() for item in overlay._edge_items)
        overlay.update(project.points, project.skeleton, "t1", 0, 0, "head")
        assert overlay._active_points == {(0, "head"), (0, "tail")}

    def test_skeleton_overlay_draws_given_edges(self, project, qapp):
        overlay = SkeletonOverlay(QGraphicsScene(), "cam1")
        no_edges = Skeleton(nodes=SYNC_NODES, name="bare")
        overlay.update(project.points, no_edges, "t1", 0, 0, "head")
        assert not any(item.isVisible() for item in overlay._edge_items)
        overlay.update(project.points, SYNC_EDGE_SK, "t1", 0, 0, "head")
        assert any(item.isVisible() for item in overlay._edge_items)

    def test_epipolar_overlay_respects_available(self, project, qapp):
        class Cache:
            def get_epipolar_line(self, src, dst, xy):
                return (0.0, 0.0), (10.0, 10.0)

        overlay = EpipolarOverlay(QGraphicsScene(), "cam1")
        overlay.update(Cache(), project.points, SYNC_CAMS, "t1", 0, 0, "tail")
        assert len(overlay.active_lines) == 1
        lacking = {"cam0": set(SYNC_NODES), "cam1": {"head"}}
        overlay.update(Cache(), project.points, SYNC_CAMS, "t1", 0, 0, "tail",
                       available=lacking)
        assert overlay.active_lines == []
        # cam0 as target: cam1 lacks tail, so it is no source
        overlay0 = EpipolarOverlay(QGraphicsScene(), "cam0")
        overlay0.update(Cache(), project.points, SYNC_CAMS, "t1", 0, 0, "tail",
                        available=lacking)
        assert overlay0.active_lines == []

    def test_panel_notice(self, qapp):
        panel = CameraPanel("cam0")
        panel.set_notice("<i>tail</i> not in this camera's skeleton")
        assert "tail" in panel.notice
        panel.set_notice("")
        assert panel.notice == ""


class TestLabelViewAvailability:
    def _view(self, project, state):
        from label3d.widgets.camera_grid import CameraGrid
        from label3d.widgets.label_view import LabelView
        grid = CameraGrid()
        grid.setup_cameras(project.cameras)
        return LabelView(project, state, grid), grid

    def test_notice_and_guard(self, project):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        state = _state(project)
        view, grid = self._view(project, state)
        state.select(0, "tail")
        view._update_all_overlays()
        assert "tail" in grid.get_panel("cam1").notice
        assert grid.get_panel("cam0").notice == ""

        before = project.points.get_point("t1", 0, 0, "tail", "cam1")
        view._on_point_dragged("cam1", 0, "tail", 1.0, 2.0)
        view._on_point_right_clicked("cam1", 0, "tail")
        after = project.points.get_point("t1", 0, 0, "tail", "cam1")
        assert (after.x, after.state) == (before.x, before.state)
        view._on_point_dragged("cam0", 0, "tail", 1.0, 2.0)
        assert project.points.get_point("t1", 0, 0, "tail", "cam0").x == 1.0


# ---------------------------------------------------------------------------
# Points that no longer fit (phase 14.7)
# ---------------------------------------------------------------------------

from label3d.export import format_summary  # noqa: E402


class TestPointsOutside:
    def test_counts_by_camera_and_node(self, project):
        assert project.points_outside(SYNC_EXP) == {}
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        assert project.points_outside(SYNC_EXP) == {("cam1", "tail"): 2}

    def test_nodes_in_no_skeleton_count_in_every_camera(self, project):
        project.assign_skeleton(HEAD_ONLY, SYNC_CAMS)
        assert project.points_outside(SYNC_EXP) == {("cam0", "tail"): 2, ("cam1", "tail"): 2}

    def test_export_leaves_them_out(self, project, tmp_path):
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        result = export_points(project, ExportOptions(output=tmp_path / "out.csv", format="csv"))
        assert result.outside == {("cam1", "tail"): 2}
        rows = pd.read_csv(tmp_path / "out.csv")
        tail = rows[rows["bodypart"] == "tail"]
        assert tail["cam1_state"].isna().all()
        assert (tail["cam0_state"] == "set").all()
        assert "Left out 2 point(s)" in format_summary(result)


# ---------------------------------------------------------------------------
# Round trips (phase 14.10)
# ---------------------------------------------------------------------------

class TestRoundTrips:
    def test_load_skeleton_first_then_sync_creates_files(self, project):
        """No .slp files yet: the first Sync creates each with its cameras' skeleton."""
        project.assign_skeleton(HEAD_ONLY, ["cam1"])
        project.remove_points_outside(cameras=["cam1"])
        before = {cam: project.skeletons.nodes_for((SYNC_EXP, "t1", cam)) for cam in SYNC_CAMS}
        assert project.slp_write_problems() == []
        project.write_slp_files()
        assert Skeleton.from_slp(project.root / "slp/cam1.slp").nodes == ["head"]
        assert Skeleton.from_slp(project.root / "slp/cam0.slp").nodes == SYNC_NODES

        reloaded = Project.load(project.root)
        assert reloaded.skeletons.by_file   # now from the files
        after = {cam: reloaded.skeletons.nodes_for((SYNC_EXP, "t1", cam)) for cam in SYNC_CAMS}
        assert after == before

    def test_sync_round_trip_with_smaller_file_skeleton(self, project):
        """cam1's file lacks tail: reading brings no cam1 tail; writing keeps the file's skeleton."""
        write_slp(project.root / "slp/cam1.slp", HEAD_ONLY, with_frame=False)
        _set_config(project, skeleton=None)
        write_slp(project.root / "slp/cam0.slp", SYNC_EDGE_SK, with_frame=False)
        project.load_skeletons()
        project.remove_points_outside()
        project.write_slp_files()
        assert Skeleton.from_slp(project.root / "slp/cam1.slp").nodes == ["head"]

        results = project.read_slp_for_sync()
        assert results[SYNC_EXP].n_new == 0 and results[SYNC_EXP].conflicts == []
        df = results[SYNC_EXP].df
        tail = df.xs("tail", level="bodypart")
        assert (tail[("cam1", "state")] == "not_set").all()
        assert (tail[("cam0", "state")] == "set").all()

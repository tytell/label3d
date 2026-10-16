"""Tests for per-camera skeletons: from_slp, merge_skeletons, ProjectSkeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def write_slp(path: Path, skeleton: Skeleton, with_frame: bool = True) -> Path:
    """Write a .slp file with this skeleton and (optionally) one labelled frame."""
    sk = sio.Skeleton(list(skeleton.nodes), name=skeleton.name)
    for a, b in skeleton.edges:
        sk.add_edge(a, b)
    for a, b in skeleton.symmetries:
        sk.add_symmetry(a, b)
    video = sio.Video(filename="vid.mp4", open_backend=False)
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

    def test_edge_conflict(self):
        no_edge = Skeleton(
            nodes=FULL.nodes, edges=FULL.edges[:2], symmetries=FULL.symmetries
        )
        with pytest.raises(SkeletonConflict) as exc:
            merge_skeletons({"a.slp": FULL, "b.slp": no_edge})
        assert exc.value.conflicts == [
            "edge snout - tail is in a.slp but not in b.slp"
        ]

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
        assert len(exc.value.conflicts) == 4  # three edges, one symmetry

    def test_many_identical_labels_shortened(self):
        bare = Skeleton(nodes=["snout", "tail"])
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

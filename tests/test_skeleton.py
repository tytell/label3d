"""Tests for label3d.skeleton.Skeleton."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from label3d.skeleton import Skeleton
from tests.helpers import make_skeleton_json


# ---------------------------------------------------------------------------
# Loading from JSON
# ---------------------------------------------------------------------------

class TestFromJson:
    def test_loads_nodes(self, skeleton_json_path):
        sk = Skeleton.from_json(skeleton_json_path)
        assert sk.nodes == ["snout", "left_eye", "dorsal"]

    def test_loads_edges(self, skeleton_json_path):
        sk = Skeleton.from_json(skeleton_json_path)
        assert ("snout", "left_eye") in sk.edges
        assert ("snout", "dorsal") in sk.edges

    def test_loads_empty_symmetries(self, skeleton_json_path):
        sk = Skeleton.from_json(skeleton_json_path)
        assert sk.symmetries == []

    def test_loads_symmetries_when_present(self, tmp_path):
        p = make_skeleton_json(
            tmp_path,
            nodes=["left_fin", "right_fin"],
            symmetries=[["left_fin", "right_fin"]],
        )
        sk = Skeleton.from_json(p)
        assert ("left_fin", "right_fin") in sk.symmetries

    def test_minimal_json_no_edges_or_symmetries(self, tmp_path):
        """JSON with only 'nodes' key — edges and symmetries default to []."""
        p = tmp_path / "skeleton.json"
        p.write_text(json.dumps({"nodes": ["head", "tail"]}))
        sk = Skeleton.from_json(p)
        assert sk.nodes == ["head", "tail"]
        assert sk.edges == []
        assert sk.symmetries == []

    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "no_such_file.json"
        with pytest.raises(FileNotFoundError):
            Skeleton.from_json(missing)

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{")
        with pytest.raises(Exception):
            Skeleton.from_json(bad)

    def test_node_names_with_spaces(self, tmp_path):
        p = make_skeleton_json(tmp_path, nodes=["left front paw", "right front paw"])
        sk = Skeleton.from_json(p)
        assert sk.nodes == ["left front paw", "right front paw"]

    def test_node_names_with_special_chars(self, tmp_path):
        """Colons, parens, commas in node names survive the JSON round-trip."""
        nodes = ["eye (left)", "joint:elbow", "tip, distal"]
        p = make_skeleton_json(tmp_path, nodes=nodes)
        sk = Skeleton.from_json(p)
        assert sk.nodes == nodes

    def test_path_with_spaces(self, tmp_path):
        spaced = tmp_path / "my project data" / "skeleton.json"
        spaced.parent.mkdir()
        spaced.write_text(json.dumps({"nodes": ["a", "b"]}))
        sk = Skeleton.from_json(spaced)
        assert sk.nodes == ["a", "b"]


# ---------------------------------------------------------------------------
# Saving to JSON (round-trip)
# ---------------------------------------------------------------------------

class TestToJson:
    def test_roundtrip_nodes(self, simple_skeleton, tmp_path):
        p = tmp_path / "out.json"
        simple_skeleton.to_json(p)
        reloaded = Skeleton.from_json(p)
        assert reloaded.nodes == simple_skeleton.nodes

    def test_roundtrip_edges(self, simple_skeleton, tmp_path):
        p = tmp_path / "out.json"
        simple_skeleton.to_json(p)
        reloaded = Skeleton.from_json(p)
        assert set(reloaded.edges) == set(simple_skeleton.edges)

    def test_roundtrip_symmetries(self, tmp_path):
        sk = Skeleton(
            nodes=["left_fin", "right_fin", "tail"],
            edges=[("left_fin", "tail"), ("right_fin", "tail")],
            symmetries=[("left_fin", "right_fin")],
        )
        p = tmp_path / "out.json"
        sk.to_json(p)
        reloaded = Skeleton.from_json(p)
        assert reloaded.symmetries == sk.symmetries

    def test_roundtrip_special_char_names(self, skeleton_with_spaces, tmp_path):
        p = tmp_path / "special.json"
        skeleton_with_spaces.to_json(p)
        reloaded = Skeleton.from_json(p)
        assert reloaded.nodes == skeleton_with_spaces.nodes
        assert reloaded.edges == skeleton_with_spaces.edges
        assert reloaded.symmetries == skeleton_with_spaces.symmetries

    def test_save_to_path_with_spaces(self, simple_skeleton, tmp_path):
        dest = tmp_path / "output dir" / "skeleton.json"
        dest.parent.mkdir()
        simple_skeleton.to_json(dest)
        assert dest.exists()
        reloaded = Skeleton.from_json(dest)
        assert reloaded.nodes == simple_skeleton.nodes


# ---------------------------------------------------------------------------
# Node navigation
# ---------------------------------------------------------------------------

class TestNodeNavigation:
    def test_node_index(self, simple_skeleton):
        assert simple_skeleton.node_index("snout") == 0
        assert simple_skeleton.node_index("left_eye") == 1
        assert simple_skeleton.node_index("dorsal") == 2

    def test_node_index_missing(self, simple_skeleton):
        with pytest.raises(ValueError):
            simple_skeleton.node_index("nonexistent")

    def test_node_index_with_spaces(self, skeleton_with_spaces):
        assert skeleton_with_spaces.node_index("left front paw") == 0

    def test_next_node(self, simple_skeleton):
        assert simple_skeleton.next_node("snout") == "left_eye"
        assert simple_skeleton.next_node("left_eye") == "dorsal"

    def test_next_node_wraps(self, simple_skeleton):
        assert simple_skeleton.next_node("dorsal") == "snout"

    def test_prev_node(self, simple_skeleton):
        assert simple_skeleton.prev_node("dorsal") == "left_eye"
        assert simple_skeleton.prev_node("left_eye") == "snout"

    def test_prev_node_wraps(self, simple_skeleton):
        assert simple_skeleton.prev_node("snout") == "dorsal"

    def test_next_node_with_spaces(self, skeleton_with_spaces):
        assert skeleton_with_spaces.next_node("left front paw") == "right front paw"

    def test_prev_node_with_spaces(self, skeleton_with_spaces):
        assert skeleton_with_spaces.prev_node("right front paw") == "left front paw"


# ---------------------------------------------------------------------------
# Building from a dict
# ---------------------------------------------------------------------------

class TestFromDict:
    def test_round_trip(self):
        data = {
            "nodes": ["a", "b", "c"],
            "edges": [["a", "b"]],
            "symmetries": [["b", "c"]],
            "node_offsets_cm": {"a": [1.0, 0.0, 0.0]},
        }
        assert Skeleton.from_dict(data).to_dict() == data

    def test_no_nodes_rejected(self):
        with pytest.raises(ValueError, match="no nodes"):
            Skeleton.from_dict({"nodes": []})

    def test_repeated_node_rejected(self):
        with pytest.raises(ValueError, match="repeats"):
            Skeleton.from_dict({"nodes": ["a", "b", "a"]})

    def test_edge_to_unknown_node_rejected(self):
        with pytest.raises(ValueError, match="edge"):
            Skeleton.from_dict({"nodes": ["a", "b"], "edges": [["a", "z"]]})

    def test_symmetry_to_unknown_node_rejected(self):
        with pytest.raises(ValueError, match="symmetry"):
            Skeleton.from_dict({"nodes": ["a", "b"], "symmetries": [["a", "z"]]})

    def test_offset_for_unknown_node_rejected(self):
        with pytest.raises(ValueError, match="offsets"):
            Skeleton.from_dict({"nodes": ["a"], "node_offsets_cm": {"z": [0, 0, 0]}})


class TestSameTopology:
    def test_ignores_order_direction_and_offsets(self):
        a = Skeleton(["a", "b", "c"], edges=[("a", "b")], symmetries=[("b", "c")])
        b = Skeleton(["c", "b", "a"], edges=[("b", "a")], symmetries=[("c", "b")],
                     node_offsets_cm={"a": [1.0, 0.0, 0.0]})
        assert a.same_topology(b)

    def test_different_edges(self):
        a = Skeleton(["a", "b", "c"], edges=[("a", "b")])
        b = Skeleton(["a", "b", "c"], edges=[("a", "c")])
        assert not a.same_topology(b)

    def test_different_nodes(self):
        assert not Skeleton(["a", "b"]).same_topology(Skeleton(["a", "c"]))

"""Skeleton: node names, edges, and symmetry pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skeleton:
    """Defines the bodypart topology for labeling."""

    nodes: list[str]
    edges: list[tuple[str, str]] = field(default_factory=list)
    symmetries: list[tuple[str, str]] = field(default_factory=list)
    # Optional per-node cm offsets from body centre used by init_random.
    # Keys are node names; values are [dx, dy, dz] in cm.
    node_offsets_cm: dict[str, list[float]] = field(default_factory=dict)

    def node_index(self, name: str) -> int:
        """Return the index of a node by name."""
        return self.nodes.index(name)

    def next_node(self, name: str) -> str:
        """Return the next node in order (wraps around)."""
        idx = self.nodes.index(name)
        return self.nodes[(idx + 1) % len(self.nodes)]

    def prev_node(self, name: str) -> str:
        """Return the previous node in order (wraps around)."""
        idx = self.nodes.index(name)
        return self.nodes[(idx - 1) % len(self.nodes)]

    # -- JSON persistence --

    @classmethod
    def from_dict(cls, data: dict) -> Skeleton:
        """Build from a skeleton.json-style dict, checking that it is consistent.

        Raises ValueError for a missing or repeated node, or an edge, symmetry
        or offset that names a node not in the list.
        """
        nodes = [str(n) for n in data.get("nodes", [])]
        if not nodes:
            raise ValueError("Skeleton has no nodes")
        repeated = sorted({n for n in nodes if nodes.count(n) > 1})
        if repeated:
            raise ValueError(f"Skeleton repeats node(s): {', '.join(repeated)}")
        skeleton = cls(
            nodes=nodes,
            edges=[tuple(e) for e in data.get("edges", [])],
            symmetries=[tuple(s) for s in data.get("symmetries", [])],
            node_offsets_cm={
                str(k): [float(x) for x in v]
                for k, v in data.get("node_offsets_cm", {}).items()
            },
        )
        known = set(nodes)
        for kind, pairs in (("edge", skeleton.edges), ("symmetry", skeleton.symmetries)):
            for pair in pairs:
                if len(pair) != 2 or not set(pair) <= known:
                    raise ValueError(f"Skeleton {kind} {list(pair)} names an unknown node")
        unknown = sorted(set(skeleton.node_offsets_cm) - known)
        if unknown:
            raise ValueError(f"Skeleton offsets name unknown node(s): {', '.join(unknown)}")
        return skeleton

    def to_dict(self) -> dict:
        """The skeleton.json contents."""
        data = {
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "symmetries": [list(s) for s in self.symmetries],
        }
        if self.node_offsets_cm:
            data["node_offsets_cm"] = dict(self.node_offsets_cm)
        return data

    def same_topology(self, other: Skeleton) -> bool:
        """True when both have the same nodes, edges and symmetries.

        Node order, edge direction and offsets are ignored.
        """
        def pairs(items):
            return {frozenset(p) for p in items}
        return (
            set(self.nodes) == set(other.nodes)
            and pairs(self.edges) == pairs(other.edges)
            and pairs(self.symmetries) == pairs(other.symmetries)
        )

    @classmethod
    def from_json(cls, path: Path) -> Skeleton:
        """Load from skeleton.json."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: Path) -> None:
        """Save to skeleton.json."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # -- SLEAP import --

    @classmethod
    def from_sleap(cls, slp_path: Path) -> Skeleton:
        """Load skeleton from a .slp file via sleap_io."""
        import sleap_io as sio

        labels = sio.load_file(str(slp_path))
        sk = labels.skeleton
        nodes = [n.name for n in sk.nodes]
        edges = [(e.source.name, e.destination.name) for e in sk.edges]
        symmetries = [(s.nodes[0].name, s.nodes[1].name) for s in sk.symmetries]
        return cls(nodes=nodes, edges=edges, symmetries=symmetries)

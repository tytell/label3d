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
    def from_json(cls, path: Path) -> Skeleton:
        """Load from skeleton.json."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            nodes=data["nodes"],
            edges=[tuple(e) for e in data.get("edges", [])],
            symmetries=[tuple(s) for s in data.get("symmetries", [])],
        )

    def to_json(self, path: Path) -> None:
        """Save to skeleton.json."""
        data = {
            "nodes": self.nodes,
            "edges": [list(e) for e in self.edges],
            "symmetries": [list(s) for s in self.symmetries],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

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

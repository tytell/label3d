"""Skeleton: node names, edges, and symmetry pairs.

Cameras can have different skeletons (docs/phases/phase_14.md). Nodes are
matched across skeletons by name; merge_skeletons() builds their union and
ProjectSkeleton says which skeleton each (experiment, trial, camera) cell has.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sleap_io.io.slp import read_skeletons

from label3d.slp_sync import Cell


@dataclass
class Skeleton:
    """Defines the bodypart topology for labeling."""

    nodes: list[str]
    edges: list[tuple[str, str]] = field(default_factory=list)
    symmetries: list[tuple[str, str]] = field(default_factory=list)
    # Optional per-node cm offsets from body centre used by init_random.
    # Keys are node names; values are [dx, dy, dz] in cm.
    node_offsets_cm: dict[str, list[float]] = field(default_factory=dict)
    # Shown in dialogs only; never used for matching.
    name: str | None = None

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
            name=str(data["name"]) if data.get("name") else None,
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
        if self.name:
            data["name"] = self.name
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

    def same_as(self, other: Skeleton) -> bool:
        """True when both have the same nodes in the same order, edges and symmetries."""
        return self.nodes == other.nodes and self.same_topology(other)

    @classmethod
    def from_json(cls, path: Path) -> Skeleton:
        """Load from skeleton.json; named after the file unless it has a name."""
        with open(path) as f:
            skeleton = cls.from_dict(json.load(f))
        if skeleton.name is None:
            skeleton.name = Path(path).stem
        return skeleton

    # -- SLEAP import --

    @classmethod
    def from_slp(cls, slp_path: Path) -> Skeleton:
        """The first skeleton of a .slp file.

        Reads only the file's metadata, not its labelled frames. Each symmetry
        pair is put in node order (sleap_io keeps it as a set).
        """
        skeletons = read_skeletons(str(slp_path))
        if not skeletons:
            raise ValueError(f"{slp_path} has no skeleton")
        sk = skeletons[0]
        nodes = list(sk.node_names)
        edges = [(e.source.name, e.destination.name) for e in sk.edges]
        symmetries = [
            tuple(sorted((n.name for n in s.nodes), key=nodes.index))
            for s in sk.symmetries
        ]
        return cls(nodes=nodes, edges=edges, symmetries=symmetries, name=sk.name)


# ---------------------------------------------------------------------------
# Several skeletons
# ---------------------------------------------------------------------------

class SkeletonConflict(ValueError):
    """Skeletons that can't be merged; conflicts lists every problem found."""

    def __init__(self, conflicts: list[str]):
        self.conflicts = conflicts
        super().__init__("Skeletons conflict:\n  " + "\n  ".join(conflicts))


def _pair_set(pairs) -> set[frozenset]:
    return {frozenset(p) for p in pairs}


def _describe(labels: list[str]) -> str:
    """Labels of identical skeletons, shortened when there are many."""
    if len(labels) <= 3:
        return ", ".join(labels)
    return f"{', '.join(labels[:3])} (+{len(labels) - 3} more)"


def merge_skeletons(skeletons: dict[str, Skeleton], strict: bool = True) -> Skeleton:
    """The union of skeletons, whose keys label them in conflict messages.

    Nodes keep the order of the first skeleton, then each later skeleton's new
    nodes in its order. Edges and symmetries are the union, ignoring direction.

    Raises SkeletonConflict listing every conflict:
    - two skeletons that both have nodes a and b disagree on edge a-b;
    - ... or on symmetry a-b;
    - a node has more than one symmetry partner in the union.
    With strict=False the union is returned in spite of conflicts.
    """
    if not skeletons:
        raise ValueError("No skeletons to merge")

    # Identical skeletons are compared once, under all their labels.
    groups: list[tuple[Skeleton, list[str]]] = []
    for label, skeleton in skeletons.items():
        for first, labels in groups:
            if first.same_as(skeleton):
                labels.append(label)
                break
        else:
            groups.append((skeleton, [label]))

    conflicts: list[str] = []
    for i, (sk_a, labels_a) in enumerate(groups):
        for sk_b, labels_b in groups[i + 1:]:
            shared = set(sk_a.nodes) & set(sk_b.nodes)
            for kind, attr in (("edge", "edges"), ("symmetry", "symmetries")):
                in_a = {p for p in _pair_set(getattr(sk_a, attr)) if p <= shared}
                in_b = {p for p in _pair_set(getattr(sk_b, attr)) if p <= shared}
                for pair, has, lacks in (
                    *((p, labels_a, labels_b) for p in in_a - in_b),
                    *((p, labels_b, labels_a) for p in in_b - in_a),
                ):
                    a, b = sorted(pair, key=sk_a.nodes.index)
                    conflicts.append(
                        f"{kind} {a} - {b} is in {_describe(has)} "
                        f"but not in {_describe(lacks)}"
                    )

    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    symmetries: list[tuple[str, str]] = []
    for skeleton, _ in groups:
        nodes.extend(n for n in skeleton.nodes if n not in nodes)
        for pairs, union in ((skeleton.edges, edges), (skeleton.symmetries, symmetries)):
            known = _pair_set(union)
            union.extend(tuple(p) for p in pairs if frozenset(p) not in known)

    partners: dict[str, list[str]] = {}
    for a, b in symmetries:
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)
    for node in nodes:
        if len(partners.get(node, [])) > 1:
            conflicts.append(
                f"{node} has more than one symmetry partner: {', '.join(partners[node])}"
            )

    if conflicts and strict:
        raise SkeletonConflict(conflicts)
    return Skeleton(nodes=nodes, edges=edges, symmetries=symmetries)


@dataclass
class MappingCheck:
    """Problems with an SLP mapping: one skeleton per file (phase 14.6)."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Cells that would lose nodes by moving into an existing file (the warnings)
    lost: dict[Cell, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ProjectSkeleton:
    """Every skeleton in a project, and which one each cell has.

    A cell's skeleton is the first of:
    1. its file's, when cell_files maps it to a file in by_file (an existing file);
    2. the one assigned to its camera (by_camera);
    3. the one shared by every existing file that holds its camera's cells;
    4. the union.
    """

    union: Skeleton
    cameras: list[str]
    by_file: dict[Path, Skeleton] = field(default_factory=dict)
    by_camera: dict[str, Skeleton] = field(default_factory=dict)
    cell_files: dict[Cell, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._inferred: dict[str, Skeleton] = {}
        files_of: dict[str, set[Path]] = {}
        for cell, path in self.cell_files.items():
            if path in self.by_file:
                files_of.setdefault(cell[2], set()).add(path)
        for camera, paths in files_of.items():
            found = [self.by_file[p] for p in sorted(paths)]
            if all(found[0].same_topology(s) for s in found[1:]):
                self._inferred[camera] = found[0]
        self._nodes: dict[Cell, frozenset[str]] = {}

    @classmethod
    def build(
        cls,
        cameras: list[str],
        by_file: dict[Path, Skeleton] | None = None,
        by_camera: dict[str, Skeleton] | None = None,
        cell_files: dict[Cell, Path] | None = None,
        strict: bool = True,
    ) -> ProjectSkeleton:
        """Merge the files' then the assigned skeletons.

        Raises SkeletonConflict, unless strict is False.
        """
        by_file = dict(by_file or {})
        by_camera = dict(by_camera or {})
        labelled = {str(path): sk for path, sk in by_file.items()}
        for camera, sk in by_camera.items():
            label = f"the skeleton assigned to {camera}"
            labelled[label] = sk
        return cls(
            union=merge_skeletons(labelled, strict=strict),
            cameras=list(cameras),
            by_file=by_file,
            by_camera=by_camera,
            cell_files=dict(cell_files or {}),
        )

    def inferred(self, camera: str) -> Skeleton | None:
        """The skeleton shared by every existing file of a camera's cells (rule 3)."""
        return self._inferred.get(camera)

    def camera_files(self, camera: str) -> list[Path]:
        """Existing files that hold some of a camera's cells, sorted."""
        return sorted({
            path for cell, path in self.cell_files.items()
            if cell[2] == camera and path in self.by_file
        })

    def camera_skeleton(self, camera: str) -> Skeleton:
        """The skeleton a camera's cells get without an existing file (rules 2-4)."""
        if camera in self.by_camera:
            return self.by_camera[camera]
        return self._inferred.get(camera, self.union)

    def skeleton_for(self, cell: Cell) -> Skeleton:
        path = self.cell_files.get(cell)
        if path is not None and path in self.by_file:
            return self.by_file[path]
        return self.camera_skeleton(cell[2])

    def nodes_for(self, cell: Cell) -> frozenset[str]:
        nodes = self._nodes.get(cell)
        if nodes is None:
            nodes = self._nodes[cell] = frozenset(self.skeleton_for(cell).nodes)
        return nodes

    def available_mask(self, index: pd.MultiIndex, camera: str) -> np.ndarray:
        """Rows of a points index whose node is available in camera.

        The index needs experiment, trial and bodypart levels; each distinct
        (experiment, trial, bodypart) is looked up once.
        """
        keys = pd.MultiIndex.from_arrays([
            index.get_level_values("experiment"),
            index.get_level_values("trial"),
            index.get_level_values("bodypart"),
        ])
        allowed = [
            key for key in keys.unique()
            if key[2] in self.nodes_for((str(key[0]), str(key[1]), camera))
        ]
        return keys.isin(allowed)

    def available(self, experiment: str, trial: str) -> dict[str, frozenset[str]]:
        """Camera -> nodes available in that camera for one trial."""
        return {cam: self.nodes_for((experiment, trial, cam)) for cam in self.cameras}

    def cameras_for(self, experiment: str, trial: str, node: str) -> list[str]:
        """Cameras that have node in this trial, in camera order."""
        return [c for c in self.cameras if node in self.nodes_for((experiment, trial, c))]

    def partial_nodes(self, experiment: str, trial: str) -> dict[str, list[str]]:
        """Union nodes not in every camera of the trial -> the cameras that have them."""
        partial = {}
        for node in self.union.nodes:
            cams = self.cameras_for(experiment, trial, node)
            if len(cams) < len(self.cameras):
                partial[node] = cams
        return partial

    def mapping_problems(
        self, cell_files: dict[Cell, Path], file_skeletons: dict[Path, Skeleton]
    ) -> MappingCheck:
        """Check that a mapping puts one skeleton in each file.

        cell_files is the mapping to check; file_skeletons holds the skeletons
        of its files that already exist. A file not created yet must not mix
        cells with different skeletons (rules 2-4). In an existing file, a cell
        whose camera is assigned a different skeleton is an error, and a cell
        that would lose nodes it has now is a warning.
        """
        check = MappingCheck()
        cells_of: dict[Path, list[Cell]] = {}
        for cell, path in cell_files.items():
            cells_of.setdefault(path, []).append(cell)

        for path, cells in cells_of.items():
            existing = file_skeletons.get(path)
            if existing is None:
                by_skeleton: list[tuple[Skeleton, list[str]]] = []
                for camera in dict.fromkeys(c[2] for c in cells):
                    sk = self.camera_skeleton(camera)
                    for first, cams in by_skeleton:
                        if first.same_topology(sk):
                            cams.append(camera)
                            break
                    else:
                        by_skeleton.append((sk, [camera]))
                if len(by_skeleton) > 1:
                    sides = "; ".join(
                        f"{', '.join(cams)} ({sk.name or 'unnamed'})"
                        for sk, cams in by_skeleton
                    )
                    check.errors.append(
                        f"{path.as_posix()}: cameras with different skeletons: {sides}"
                    )
                continue

            for camera in dict.fromkeys(c[2] for c in cells):
                assigned = self.by_camera.get(camera)
                if assigned is not None and not assigned.same_topology(existing):
                    check.errors.append(
                        f"{path.as_posix()}: {camera} is assigned skeleton "
                        f"{assigned.name or 'unnamed'}, but the file has "
                        f"{existing.name or 'unnamed'}"
                    )
            for cell in cells:
                if self.by_camera.get(cell[2]) is not None:
                    continue
                lost = [n for n in self.skeleton_for(cell).nodes if n not in existing.nodes]
                if lost:
                    check.lost[cell] = lost
            lost_cells = [c for c in cells if c in check.lost]
            if lost_cells:
                nodes = list(dict.fromkeys(n for c in lost_cells for n in check.lost[c]))
                check.warnings.append(
                    f"{path.as_posix()}: {len(lost_cells)} video(s) would lose "
                    f"node(s) not in the file's skeleton: {', '.join(nodes)}"
                )
        return check


def assigned_skeletons(config: dict, cameras: list[str]) -> dict[str, Skeleton]:
    """Camera -> skeleton from [[skeleton.assigned]] in a project.toml dict.

    Raises ValueError for an entry that is not a valid skeleton, or a camera
    in more than one entry. Cameras not in cameras are ignored.
    """
    by_camera: dict[str, Skeleton] = {}
    for entry in config.get("skeleton", {}).get("assigned", []):
        try:
            skeleton = Skeleton.from_dict(entry)
        except ValueError as exc:
            raise ValueError(f"project.toml [[skeleton.assigned]] {entry.get('name')}: {exc}")
        for camera in entry.get("cameras", []):
            if camera in by_camera:
                raise ValueError(
                    f"project.toml assigns more than one skeleton to camera {camera}"
                )
            if camera in cameras:
                by_camera[camera] = skeleton
    return by_camera


def assign_in_config(
    section: dict, skeleton: Skeleton, cameras: list[str], camera_order: list[str]
) -> dict:
    """The [skeleton] section with skeleton assigned to cameras.

    The cameras leave any other entry; an entry with the same topology is
    reused; entries left without cameras are dropped. The skeleton's offsets
    are added to node_offsets_cm.
    """
    section = dict(section)
    entries = []
    for entry in section.get("assigned", []):
        entry = dict(entry)
        entry["cameras"] = [c for c in entry.get("cameras", []) if c not in cameras]
        entries.append(entry)
    for entry in entries:
        if Skeleton.from_dict(entry).same_topology(skeleton):
            entry["cameras"] = [
                c for c in camera_order if c in entry["cameras"] or c in cameras
            ]
            break
    else:
        entries.append({
            "name": skeleton.name or "skeleton",
            "cameras": [c for c in camera_order if c in cameras],
            "nodes": list(skeleton.nodes),
            "edges": [list(e) for e in skeleton.edges],
            "symmetries": [list(e) for e in skeleton.symmetries],
        })
    section["assigned"] = [e for e in entries if e["cameras"]]
    if skeleton.node_offsets_cm:
        offsets = dict(section.get("node_offsets_cm", {}))
        offsets.update(skeleton.node_offsets_cm)
        section["node_offsets_cm"] = offsets
    return section

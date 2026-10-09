"""Calibration registry: every solution an experiment has produced.

Each solution is an aniposelib CameraGroup.dump() file under
experiments/{name}/calibrations/, with its provenance in the [metadata] table
(the only top-level table CameraGroup.load() doesn't read as a camera).
Refinements (Phase 12) go in calibrations/refined/, next to a _points.csv
record of the points that went into them. Which entry is in use is a pointer in
project.toml (see Project.calibration_path), not part of the registry.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import toml
from aniposelib.cameras import CameraGroup

logger = logging.getLogger(__name__)

REGISTRY_DIR = "calibrations"
REFINED_DIR = "refined"

METHODS = ("charuco", "refine", "align", "imported")

_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


@dataclass
class CalibrationEntry:
    """One registry file and its provenance."""

    id: str
    label: str
    created: datetime
    method: str
    parent: str
    path: Path
    metadata: dict

    @property
    def charuco_takes(self) -> list[str]:
        return list(self.metadata.get("charuco_takes", []))


def registry_dir(exp_dir: Path) -> Path:
    return Path(exp_dir) / REGISTRY_DIR


def camera_dir(exp_dir: Path, camera: str) -> Path:
    """Folder of one camera's calibration videos and corners.csv."""
    return registry_dir(exp_dir) / camera


def clean_label(text: str) -> str:
    """A label made of [A-Za-z0-9_-] only, so it is safe in a file name."""
    label = re.sub(r"\s+", "_", str(text).strip())
    label = re.sub(r"[^A-Za-z0-9_-]", "", label)
    return label or "calibration"


def _entry_from_file(path: Path) -> CalibrationEntry | None:
    try:
        metadata = dict(toml.load(path).get("metadata", {}))
    except Exception:
        return None
    try:
        created = datetime.fromisoformat(str(metadata.get("created", "")))
    except ValueError:
        created = datetime.fromtimestamp(path.stat().st_mtime)
    return CalibrationEntry(
        id=str(metadata.get("id", path.stem)),
        label=str(metadata.get("label", path.stem)),
        created=created,
        method=str(metadata.get("method", "imported")),
        parent=str(metadata.get("parent", "")),
        path=path,
        metadata=metadata,
    )


def list_calibrations(exp_dir: Path) -> list[CalibrationEntry]:
    """All registry entries of an experiment, newest first."""
    root = registry_dir(exp_dir)
    paths = sorted(root.glob("*.toml")) + sorted((root / REFINED_DIR).glob("*.toml"))
    entries = [e for e in (_entry_from_file(p) for p in paths) if e is not None]
    entries.sort(key=lambda e: (e.created, e.id), reverse=True)
    return entries


def find_calibration(exp_dir: Path, entry_id: str) -> CalibrationEntry | None:
    """The entry with this id, or None."""
    root = registry_dir(exp_dir)
    for path in (root / f"{entry_id}.toml", root / REFINED_DIR / f"{entry_id}.toml"):
        if path.exists():
            return _entry_from_file(path)
    return None


def load_calibration(entry: CalibrationEntry) -> CameraGroup:
    return CameraGroup.load(str(entry.path))


def save_calibration(
    exp_dir: Path,
    cgroup: CameraGroup,
    label: str,
    method: str,
    parent: str = "",
    created: datetime | None = None,
    **metadata,
) -> CalibrationEntry:
    """Write cgroup as a new registry entry and return it.

    The id is the cleaned label plus the creation time; a counter suffix keeps
    it unique if the same label is saved twice in one second. Extra keyword
    arguments go into [metadata] as they are (charuco_takes, points, errors…).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    label = clean_label(label)
    created = (created or datetime.now()).replace(microsecond=0)

    folder = registry_dir(exp_dir)
    if method == "refine":
        folder = folder / REFINED_DIR
    folder.mkdir(parents=True, exist_ok=True)

    base_id = f"{label}_{created.strftime(_TIMESTAMP_FORMAT)}"
    entry_id, n = base_id, 1
    while find_calibration(exp_dir, entry_id) is not None:
        n += 1
        entry_id = f"{base_id}-{n}"

    full = {
        "id": entry_id,
        "label": label,
        "created": created.isoformat(),
        "method": method,
        "parent": parent,
        **metadata,
    }
    # A shallow copy with its own metadata dict: CameraGroup's metadata default
    # is a shared mutable {}, and the caller's group is left untouched.
    group = copy.copy(cgroup)
    group.metadata = full
    path = folder / f"{entry_id}.toml"
    group.dump(str(path))
    return _entry_from_file(path)


def update_metadata(entry: CalibrationEntry, **fields) -> CalibrationEntry:
    """Set top-level [metadata] fields of an entry. The id and file name stay the same."""
    data = toml.load(entry.path)
    data.setdefault("metadata", {}).update(fields)
    with open(entry.path, "w") as f:
        toml.dump(data, f)
    return _entry_from_file(entry.path)


def rename_calibration(entry: CalibrationEntry, label: str) -> CalibrationEntry:
    """Change an entry's label. The id and file name stay the same."""
    return update_metadata(entry, label=clean_label(label))


def descendants(entries: list[CalibrationEntry], entry_id: str) -> list[CalibrationEntry]:
    """Entries derived from entry_id, directly or through other entries."""
    found: list[CalibrationEntry] = []
    frontier = [entry_id]
    while frontier:
        current = frontier.pop()
        for e in entries:
            if e.parent == current and e not in found:
                found.append(e)
                frontier.append(e.id)
    return found


def delete_calibration(exp_dir: Path, entry: CalibrationEntry) -> list[CalibrationEntry]:
    """Delete an entry and everything derived from it. Returns what was removed."""
    removed = [entry] + descendants(list_calibrations(exp_dir), entry.id)
    for e in removed:
        points_file = e.metadata.get("points", {}).get("points_file")
        if points_file:
            (e.path.parent / points_file).unlink(missing_ok=True)
        e.path.unlink(missing_ok=True)
    return removed


def entry_tree(entries: list[CalibrationEntry]) -> list[tuple[CalibrationEntry, int]]:
    """Entries in display order with their depth: each one under its parent.

    Roots (and entries whose parent is gone) come newest first, and so do the
    children of each entry.
    """
    ids = {e.id for e in entries}
    children: dict[str, list[CalibrationEntry]] = {}
    roots: list[CalibrationEntry] = []
    for e in entries:   # already newest first
        if e.parent and e.parent in ids:
            children.setdefault(e.parent, []).append(e)
        else:
            roots.append(e)

    ordered: list[tuple[CalibrationEntry, int]] = []

    def walk(entry: CalibrationEntry, depth: int) -> None:
        ordered.append((entry, depth))
        for child in children.get(entry.id, []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return ordered


def entry_error(entry: CalibrationEntry) -> float | None:
    """The headline error in px: held-out points for a refinement, else charuco."""
    errors = entry.metadata.get("errors", {})
    for key in ("holdout", "charuco", "points"):
        if key in errors:
            return float(errors[key])
    return None


def entry_summary(entry: CalibrationEntry) -> str:
    """Multi-line description for tooltips: method, parent, takes, points, errors."""
    lines = [
        f"{entry.label}  ({entry.id})",
        f"Method: {entry.method}",
        f"Created: {entry.created:%Y-%m-%d %H:%M:%S}",
    ]
    if entry.parent:
        lines.append(f"Parent: {entry.parent}")
    if entry.charuco_takes:
        lines.append(f"Charuco takes: {', '.join(entry.charuco_takes)}")
    points = entry.metadata.get("points", {})
    if points:
        lines.append(
            f"Points: {points.get('n_points', '?')} from {', '.join(points.get('sources', []))}"
        )
    for key, value in entry.metadata.get("errors", {}).items():
        if isinstance(value, (int, float)):
            lines.append(f"Error ({key}): {value:.3f}")
    if "source_path" in entry.metadata:
        lines.append(f"Imported from: {entry.metadata['source_path']}")
    if entry.metadata.get("notes"):
        lines.append(f"Notes: {entry.metadata['notes']}")
    return "\n".join(lines)

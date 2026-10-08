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

# Legacy files imported by import_legacy, relative to the experiment folder.
LEGACY_CALIBRATION = "calibration.toml"
LEGACY_TAKES_DIR = "calibration_takes"

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


def rename_calibration(entry: CalibrationEntry, label: str) -> CalibrationEntry:
    """Change an entry's label. The id and file name stay the same."""
    data = toml.load(entry.path)
    data.setdefault("metadata", {})["label"] = clean_label(label)
    with open(entry.path, "w") as f:
        toml.dump(data, f)
    return _entry_from_file(entry.path)


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


def import_legacy(exp_dir: Path) -> list[CalibrationEntry]:
    """Copy calibration.toml and calibration_takes/*.toml into the registry.

    Runs only when the registry folder doesn't exist yet, and leaves the old
    files in place. Each entry is method "imported", created at its file's
    modification time, with metadata.source_path set to the file it came from
    (relative to the experiment folder). The entry from calibration.toml, if
    any, comes first.
    """
    exp_dir = Path(exp_dir)
    if registry_dir(exp_dir).exists():
        return []
    sources = [(exp_dir / LEGACY_CALIBRATION, "imported")]
    for path in sorted((exp_dir / LEGACY_TAKES_DIR).glob("*.toml")):
        sources.append((path, f"imported-{path.stem}"))

    entries: list[CalibrationEntry] = []
    for path, label in sources:
        if not path.exists():
            continue
        try:
            cgroup = CameraGroup.load(str(path))
        except Exception as exc:
            logger.warning("Could not import %s: %s", path, exc)
            continue
        entries.append(save_calibration(
            exp_dir, cgroup, label, "imported",
            created=datetime.fromtimestamp(path.stat().st_mtime),
            source_path=path.relative_to(exp_dir).as_posix(),
        ))
    return entries

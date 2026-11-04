"""An experiment's axis alignment, as kept in project.toml (Phase 16).

[alignment.{experiment}] has these parts:

    settings   what the Align tab last used
    transform  the alignment in use (absent until the first Apply): R (rows of
               R_align), origin in cm, the calibration entry the clicks were
               triangulated with, and the checks shown before it was applied
    source     the take or trial and frame it was clicked on, with each
               camera's file and the frame within that file
    history    earlier transforms, newest last
    extracted  for each axis take made from a frame of a calibration take or
               trial, that take or trial, frame and each camera's video

Frame numbers in project.toml count from FRAME_BASE, as in trials.csv; in
memory they are 0-based. The transform itself is geometry.align_points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from label3d.calibration_registry import CalibrationEntry
from label3d.frames import shown_frame, stored_frame
from label3d.geometry import AXES, PRIMARY_RULES, AlignmentResult, align_points
from label3d.trial_types import TRIAL

TABLE = "alignment"

# alignment_status
NONE = "none"
CURRENT = "current"
STALE = "stale"


@dataclass
class AlignmentSettings:
    """The Align tab's choices for one experiment."""

    points_per_axis: int = 1                 # 1 or 2 clicked points on each axis
    cross_product: str = ""                  # the axis derived from the other two, or ""
    primary: str = "longest"                 # PRIMARY_RULES or an axis name
    flip: list[str] = field(default_factory=list)              # axes to reverse
    known_length_cm: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.points_per_axis not in (1, 2):
            raise ValueError(f"points_per_axis must be 1 or 2, not {self.points_per_axis}")
        if self.cross_product not in ("", *AXES):
            raise ValueError(f"cross_product must be an axis or blank, not {self.cross_product!r}")
        if self.primary not in (*PRIMARY_RULES, *AXES):
            raise ValueError(f"Unknown primary axis rule: {self.primary!r}")
        unknown = [k for k in (*self.flip, *self.known_length_cm) if k not in AXES]
        if unknown:
            raise ValueError(f"Unknown axes: {', '.join(unknown)}")

    def to_config(self) -> dict:
        return {
            "points_per_axis": self.points_per_axis,
            "cross_product": self.cross_product,
            "primary": self.primary,
            "flip": sorted(self.flip),
            "known_length_cm": {k: float(v) for k, v in sorted(self.known_length_cm.items())},
        }

    @classmethod
    def from_config(cls, table: dict) -> AlignmentSettings:
        return cls(
            points_per_axis=int(table.get("points_per_axis", 1)),
            cross_product=str(table.get("cross_product", "")),
            primary=str(table.get("primary", "longest")),
            flip=[str(k) for k in table.get("flip", [])],
            known_length_cm={str(k): float(v) for k, v in table.get("known_length_cm", {}).items()},
        )


@dataclass
class Alignment:
    """A rotation and origin applied to the 3D points of one calibration."""

    R_align: np.ndarray                      # (3, 3), columns are the new axes
    origin: np.ndarray                       # (3,) cm, calibration coordinates
    calibration: str                         # entry the clicks were triangulated with
    created: datetime
    primary: str = ""
    derived: str = ""
    axis_angles_deg: dict[str, float] = field(default_factory=dict)
    max_error_px: float | None = None
    scale_error_pct: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        result: AlignmentResult,
        calibration: str,
        max_error_px: float | None = None,
        scale_error_pct: dict[str, float] | None = None,
        created: datetime | None = None,
    ) -> Alignment:
        return cls(
            R_align=np.asarray(result.R_align, dtype=float),
            origin=np.asarray(result.origin, dtype=float),
            calibration=calibration,
            created=(created or datetime.now()).replace(microsecond=0),
            primary=result.primary,
            derived=result.derived,
            axis_angles_deg=dict(result.axis_angles_deg),
            max_error_px=max_error_px,
            scale_error_pct=dict(scale_error_pct or {}),
        )

    def align(self, xyz: np.ndarray) -> np.ndarray:
        """(N, 3) calibration-frame points in the aligned frame."""
        return align_points(xyz, self.R_align, self.origin)

    def to_config(self) -> dict:
        table = {
            "R": [[float(v) for v in row] for row in self.R_align],
            "origin": [float(v) for v in self.origin],
            "calibration": self.calibration,
            "created": self.created,
            "primary": self.primary,
            "derived": self.derived,
            "axis_angles_deg": {k: float(v) for k, v in self.axis_angles_deg.items()},
            "scale_error_pct": {k: float(v) for k, v in self.scale_error_pct.items()},
        }
        if self.max_error_px is not None:
            table["max_error_px"] = float(self.max_error_px)
        return table

    @classmethod
    def from_config(cls, table: dict) -> Alignment:
        R = np.array(table["R"], dtype=float)
        origin = np.array(table["origin"], dtype=float)
        if R.shape != (3, 3) or origin.shape != (3,):
            raise ValueError("An alignment needs a 3x3 R and a 3-element origin.")
        created = table.get("created")
        if not isinstance(created, datetime):
            created = datetime.fromisoformat(str(created)) if created else datetime.min
        max_error = table.get("max_error_px")
        return cls(
            R_align=R,
            origin=origin,
            calibration=str(table.get("calibration", "")),
            created=created,
            primary=str(table.get("primary", "")),
            derived=str(table.get("derived", "")),
            axis_angles_deg={str(k): float(v) for k, v in table.get("axis_angles_deg", {}).items()},
            max_error_px=None if max_error is None else float(max_error),
            scale_error_pct={str(k): float(v) for k, v in table.get("scale_error_pct", {}).items()},
        )


@dataclass
class AlignmentSource:
    """A frame of a take or trial: the one an alignment was clicked on, or the
    one an axis take was extracted from. Frames are 0-based."""

    take: str                                # an axis take, or the take or trial extracted from
    type: str                                # its trials.csv row type
    frame: int                               # synchronized frame
    videos: dict[str, tuple[str, int]] = field(default_factory=dict)  # camera -> (file, frame in file)

    def to_config(self) -> dict:
        return {
            "take": self.take,
            "type": self.type,
            "frame": shown_frame(self.frame),
            "videos": {
                cam: {"file": file, "frame": shown_frame(frame)}
                for cam, (file, frame) in sorted(self.videos.items())
            },
        }

    @classmethod
    def from_config(cls, table: dict) -> AlignmentSource:
        return cls(
            take=str(table["take"]),
            type=str(table.get("type", TRIAL)),
            frame=stored_frame(int(table["frame"])),
            videos={
                str(cam): (str(v["file"]), stored_frame(int(v["frame"])))
                for cam, v in table.get("videos", {}).items()
            },
        )


# -- Reading and writing [alignment.{experiment}] ------------------------------


def _experiment_table(config: dict, experiment: str) -> dict:
    return config.get(TABLE, {}).get(experiment, {})


def read_alignment(config: dict, experiment: str) -> Alignment | None:
    table = _experiment_table(config, experiment).get("transform")
    return Alignment.from_config(table) if table else None


def read_source(config: dict, experiment: str) -> AlignmentSource | None:
    table = _experiment_table(config, experiment).get("source")
    return AlignmentSource.from_config(table) if table else None


def read_settings(config: dict, experiment: str) -> AlignmentSettings:
    return AlignmentSettings.from_config(_experiment_table(config, experiment).get("settings", {}))


def read_extracted(config: dict, experiment: str) -> dict[str, AlignmentSource]:
    """Axis takes extracted from a frame of a take or trial -> where they came from."""
    return {
        str(take): AlignmentSource.from_config(table)
        for take, table in _experiment_table(config, experiment).get("extracted", {}).items()
    }


def read_history(config: dict, experiment: str) -> list[Alignment]:
    """Earlier transforms, oldest first."""
    return [Alignment.from_config(t) for t in _experiment_table(config, experiment).get("history", [])]


def _updated(config: dict, experiment: str, update: Callable[[dict], None]) -> dict:
    """A copy of the [alignment] section with one experiment's table updated."""
    section = {name: dict(table) for name, table in config.get(TABLE, {}).items()}
    table = section.setdefault(experiment, {})
    update(table)
    return section


def with_alignment(
    config: dict, experiment: str, alignment: Alignment, source: AlignmentSource
) -> dict:
    """The [alignment] section with a new transform and source; the old transform goes to history."""
    def update(table: dict) -> None:
        if table.get("transform"):
            table["history"] = [*table.get("history", []), table["transform"]]
        table["transform"] = alignment.to_config()
        table["source"] = source.to_config()
    return _updated(config, experiment, update)


def with_settings(config: dict, experiment: str, settings: AlignmentSettings) -> dict:
    """The [alignment] section with new settings for one experiment."""
    def update(table: dict) -> None:
        table["settings"] = settings.to_config()
    return _updated(config, experiment, update)


def alignment_status(
    alignment: Alignment | None,
    points_from: str | None,
    find_entry: Callable[[str], CalibrationEntry | None],
) -> str:
    """NONE, CURRENT or STALE for the calibration the 3D points come from.

    The alignment is current for the calibration its clicks were triangulated
    with, and for a refinement of it (or a refinement of one): refinement
    aligns its result back onto its start, so the frame is kept.
    """
    if alignment is None:
        return NONE
    entry_id, seen = points_from, set()
    while entry_id and entry_id not in seen:
        if entry_id == alignment.calibration:
            return CURRENT
        seen.add(entry_id)
        entry = find_entry(entry_id)
        if entry is None or entry.method != "refine":
            break
        entry_id = entry.parent
    return STALE

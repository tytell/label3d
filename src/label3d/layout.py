"""Which layout a project folder has, and what upgrading it takes (Phase 20).

Projects made before 0.3.0 don't record a version, so it is worked out from
their files: CSV working files or skeleton.json mean 0.0.9; .slp files directly
in slp/ mean 0.1.0 (0.2.0 looks the same, but its Sync kept what the older ones
dropped, so it can't be told apart and is treated as 0.1.0); anything else is
called 0.2.0. Each step has its own check, so a project between layouts gets
only the steps it needs. upgrade.run_upgrade carries the plan out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import toml

from label3d.calibration_io import LEGACY_CORNERS_FILE
from label3d.calibration_registry import REFINED_DIR, REGISTRY_DIR
from label3d.frames import FRAME_BASE
from label3d.points import LEGACY_POINTS_FILE
from label3d.slp_sync import SLP_DIR, SlpMapping
from label3d.version import LAYOUT_VERSION, parse_version, project_version

LEGACY_SYNC_HISTORY_FILE = "sync_history.csv"
LEGACY_RECORD_SUFFIX = "_points.csv"
SKELETON_JSON = "skeleton.json"
UPGRADE_BACKUP_DIR = "upgrade_backup"
# Found in an experiment folder of a project from before the calibration registry
PRE_REGISTRY = ("calibration.toml", "calibration_takes", "calibration")

# Upgrade steps, in the order they run
POINTS = "points"
CORNERS = "corners"
SYNC_HISTORY = "sync_history"
REFINEMENTS = "refinements"
SKELETON = "skeleton"
SLP_FOLDERS = "slp_folders"
FRAME_NUMBERS = "frame_numbers"
RESTORE_SLP = "restore_slp"

STEP_TEXT = {
    POINTS: "Convert points.csv to points.parquet",
    CORNERS: "Convert corners.csv to corners.parquet",
    SYNC_HISTORY: "Convert sync_history.csv to sync_history.parquet",
    REFINEMENTS: "Convert refinement point records to Parquet",
    SKELETON: "Move skeleton.json into project.toml",
    SLP_FOLDERS: "Give each .slp file in slp/ a folder of its own",
    FRAME_NUMBERS: "Number frames in trials.csv from 1",
    RESTORE_SLP: "Restore the videos and frames that older Syncs dropped from the .slp files",
}


@dataclass
class UpgradePlan:
    """What upgrading one project takes."""

    root: Path
    from_version: str
    steps: list[str] = field(default_factory=list)
    # Mapped .slp files (relative to root, as they are now) that older Syncs
    # may have cut down; RESTORE_SLP replaces them with their originals.
    restore: list[Path] = field(default_factory=list)

    def describe(self) -> list[str]:
        return [STEP_TEXT[step] for step in self.steps]


class ProjectNeedsUpgrade(Exception):
    """Project.load found a project made by an older label3d."""

    def __init__(self, plan: UpgradePlan):
        steps = "\n  ".join(plan.describe())
        super().__init__(
            f"{plan.root} was made by label3d {plan.from_version} or similar and needs "
            f"upgrading to {LAYOUT_VERSION}:\n  {steps}"
        )
        self.plan = plan


def legacy_files(root: Path) -> dict[str, list[Path]]:
    """The files of a 0.0.9 project that later versions keep elsewhere, by step."""
    registries = f"experiments/*/{REGISTRY_DIR}"
    found = {
        POINTS: sorted(root.glob(f"experiments/*/{LEGACY_POINTS_FILE}")),
        CORNERS: sorted(root.glob(f"{registries}/*/{LEGACY_CORNERS_FILE}")),
        SYNC_HISTORY: [root / LEGACY_SYNC_HISTORY_FILE],
        REFINEMENTS: sorted(root.glob(f"{registries}/{REFINED_DIR}/*{LEGACY_RECORD_SUFFIX}")),
        SKELETON: [root / SKELETON_JSON],
    }
    return {step: [p for p in paths if p.is_file()] for step, paths in found.items()}


def flat_slp_files(root: Path) -> list[Path]:
    """.slp files directly in slp/, as projects kept them before 0.3.0."""
    return sorted(p for p in (root / SLP_DIR).glob("*.slp") if not p.name.endswith(".tmp.slp"))


def detect_version(root: Path, config: dict) -> str:
    """The layout version of a project: recorded, or worked out from its files."""
    recorded = project_version(config)
    if recorded is not None:
        return recorded
    old = [p for name in PRE_REGISTRY for p in root.glob(f"experiments/*/{name}")]
    if old:
        raise ValueError(
            "This project is from before the calibration registry, which label3d no "
            "longer reads:\n  " + "\n  ".join(str(p) for p in old)
            + "\nSet the project up again."
        )
    if any(legacy_files(root).values()):
        return "0.0.9"
    if flat_slp_files(root):
        return "0.1.0"
    return "0.2.0"


def plan_upgrade(root: Path, config: dict | None = None) -> UpgradePlan | None:
    """The upgrade an unversioned project needs; None for one that records its version.

    A plan with no steps only needs the version recorded.
    """
    root = Path(root)
    if config is None:
        config = toml.load(root / "project.toml")
    if project_version(config) is not None:
        return None
    version = detect_version(root, config)
    legacy = legacy_files(root)
    steps = [step for step in (POINTS, CORNERS, SYNC_HISTORY, REFINEMENTS, SKELETON)
             if legacy[step]]
    if flat_slp_files(root):
        steps.append(SLP_FOLDERS)
    if config.get("trials", {}).get("frame_base", 0) != FRAME_BASE:
        steps.append(FRAME_NUMBERS)
    restore: list[Path] = []
    if parse_version(version) < (0, 2, 0):
        restore = [spec.path for spec in SlpMapping.from_config(config).files
                   if (root / spec.path).is_file()]
        if restore:
            steps.append(RESTORE_SLP)
    return UpgradePlan(root=root, from_version=version, steps=steps, restore=restore)

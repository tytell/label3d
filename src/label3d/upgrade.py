"""Upgrade a project made by an older label3d to LAYOUT_VERSION (Phase 20).

layout.plan_upgrade says what a project needs; run_upgrade does it:

1. Every file the upgrade may change is copied to
   upgrade_backup/{from_version}_{timestamp}/, keeping its path in the project.
2. The file steps run: CSV working files to Parquet, skeleton.json into
   project.toml, each .slp file into a folder of its own, trials.csv numbered
   from 1. project.toml is written.
3. If the project's mapped .slp files may have been cut down by an older Sync,
   they are restored (restore_slp_files).
4. The layout version is recorded in project.toml.

If anything raises, the files are put back as they were and the error is
raised again; the backup is kept.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import toml

from label3d.adapters.sleap_adapter import SleapAdapter
from label3d.calibration_io import CORNER_COLUMNS, CORNERS_FILE, CORNERS_SCHEMA
from label3d.calibration_refine import RECORD_COLUMNS, RECORD_SCHEMA
from label3d.frames import FRAME_BASE
from label3d.layout import (
    CORNERS,
    FRAME_NUMBERS,
    LEGACY_RECORD_SUFFIX,
    POINTS,
    REFINEMENTS,
    RESTORE_SLP,
    SKELETON,
    SLP_FOLDERS,
    STEP_TEXT,
    SYNC_HISTORY,
    UPGRADE_BACKUP_DIR,
    UpgradePlan,
    flat_slp_files,
    legacy_files,
    plan_upgrade,
)
from label3d.parquet_io import write_table
from label3d.points import (
    INDEX_NAMES,
    POINTS_FILE,
    SOURCE_LABEL3D,
    SOURCE_SLEAP_USER,
    USER_STATES,
    normalize_columns,
    write_points_file,
)
from label3d.project import (
    SYNC_HISTORY_COLUMNS,
    SYNC_HISTORY_FILE,
    SYNC_HISTORY_SCHEMA,
    Project,
    _extract_cameras,
    read_trials_csv,
    write_trials_csv,
)
from label3d.skeleton import Skeleton, assign_in_config
from label3d.slp_sync import BACKUP_DIR, MODELS_DIR, SLP_DIR, find_slp_files, slp_file_path
from label3d.version import write_project_toml

logger = logging.getLogger(__name__)

# Called with a line of text as each part of the upgrade starts
UpgradeProgress = Callable[[str], object]


@dataclass
class UpgradeResult:
    from_version: str
    backup_dir: Path | None
    messages: list[str] = field(default_factory=list)


# -- Finding the original .slp files ------------------------------------------


def find_originals(names: list[str], folder: Path) -> dict[str, list[Path]]:
    """For each .slp file name, the files of that name under folder (not in models/ or backup/)."""
    found: dict[str, list[Path]] = {name: [] for name in names}
    for path in find_slp_files(Path(folder), skip=(MODELS_DIR, BACKUP_DIR)):
        if path.name in found:
            found[path.name].append(path)
    return found


def earliest_backup(root: Path, name: str) -> Path | None:
    """The copy of a .slp file in the earliest Sync backup, slp/backup/{timestamp}/.

    That is the file as it was before the first Sync, the original if setup
    copied it into the project and nothing was changed in SLEAP before then.
    """
    backups = Path(root) / SLP_DIR / BACKUP_DIR
    if not backups.is_dir():
        return None
    for folder in sorted(p for p in backups.iterdir() if p.is_dir()):
        candidates = sorted(folder.rglob(name))
        if candidates:
            return candidates[0]
    return None


# -- Running the upgrade -----------------------------------------------------


def run_upgrade(
    root: Path,
    slp_sources: dict[Path, Path | None] | None = None,
    progress: UpgradeProgress | None = None,
) -> UpgradeResult:
    """Upgrade the project at root.

    slp_sources gives, for each file in plan.restore, the original .slp file to
    restore it from, or None to keep the project's copy; every file in
    plan.restore must have an entry. Must be called from the main thread when
    the .slp files are restored (points are stored through a Project).
    """
    root = Path(root)
    report = progress or (lambda text: None)
    config = toml.load(root / "project.toml")
    plan = plan_upgrade(root, config)
    if plan is None:
        return UpgradeResult(from_version="", backup_dir=None)
    sources = dict(slp_sources or {})
    missing = [str(p) for p in plan.restore if p not in sources]
    if missing:
        raise ValueError("No original chosen for: " + ", ".join(missing))

    result = UpgradeResult(from_version=plan.from_version, backup_dir=None)
    before = _snapshot(root)
    report("Backing up the project files")
    result.backup_dir = _back_up(root, plan, _files_to_back_up(root))
    try:
        _run_steps(root, config, plan, sources, result, report)
    except BaseException:
        logger.exception("Upgrade failed; putting the project back as it was")
        _roll_back(root, before, result.backup_dir)
        raise
    return result


def _run_steps(
    root: Path,
    config: dict,
    plan: UpgradePlan,
    sources: dict[Path, Path | None],
    result: UpgradeResult,
    report: UpgradeProgress,
) -> None:
    legacy = legacy_files(root)
    if POINTS in plan.steps:
        report("Converting points.csv to Parquet")
        for path in legacy[POINTS]:
            _convert_points(path)
    if CORNERS in plan.steps:
        report("Converting corners.csv to Parquet")
        for path in legacy[CORNERS]:
            _convert_corners(path)
    if SYNC_HISTORY in plan.steps:
        _convert_sync_history(root)
    if REFINEMENTS in plan.steps:
        for path in legacy[REFINEMENTS]:
            _convert_record(path)
    if SKELETON in plan.steps:
        _move_skeleton_json(root, config)
    moved: dict[Path, Path] = {}
    if SLP_FOLDERS in plan.steps:
        report("Moving each .slp file into a folder of its own")
        moved = _slp_into_folders(root, config)
    if FRAME_NUMBERS in plan.steps:
        _renumber_frames(root, config)
    write_project_toml(root / "project.toml", config)
    result.messages += [STEP_TEXT[step] for step in plan.steps if step != RESTORE_SLP]

    if RESTORE_SLP in plan.steps:
        restore = {moved.get(rel, rel): sources[rel] for rel in plan.restore}
        result.messages += restore_slp_files(root, restore, report)

    config = toml.load(root / "project.toml")
    write_project_toml(root / "project.toml", config, new_project=True)


# -- Backup and roll back ----------------------------------------------------


def _files_to_back_up(root: Path) -> list[Path]:
    """Every file an upgrade may change, move or remove."""
    files = [root / "project.toml", root / "trials.csv", root / "trials.csv.bak",
             root / SYNC_HISTORY_FILE]
    for paths in legacy_files(root).values():
        for path in paths:
            files += [path, path.with_name(path.name + ".bak")]
    for record in legacy_files(root)[REFINEMENTS]:
        files += sorted(record.parent.glob("*.toml"))
    for exp in sorted((root / "experiments").glob("*/")):
        files += [exp / POINTS_FILE, exp / (POINTS_FILE + ".bak")]
    if (root / SLP_DIR).is_dir():
        files += find_slp_files(root / SLP_DIR, skip=(MODELS_DIR, BACKUP_DIR))
    return sorted({f for f in files if f.is_file()})


def _back_up(root: Path, plan: UpgradePlan, files: list[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = root / UPGRADE_BACKUP_DIR / f"{plan.from_version}_{stamp}"
    for path in files:
        dest = backup_dir / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def _snapshot(root: Path) -> tuple[set[Path], set[Path]]:
    """The project's files and folders, not counting upgrade backups."""
    files, dirs = set(), set()
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if rel.parts[0] == UPGRADE_BACKUP_DIR:
            continue
        (dirs if path.is_dir() else files).add(rel)
    return files, dirs


def _roll_back(root: Path, before: tuple[set[Path], set[Path]], backup_dir: Path) -> None:
    """Remove what the upgrade made and copy the backed-up files back."""
    files_before, dirs_before = before
    files_now, dirs_now = _snapshot(root)
    for rel in files_now - files_before:
        (root / rel).unlink(missing_ok=True)
    for rel in sorted(dirs_now - dirs_before, key=lambda p: len(p.parts), reverse=True):
        shutil.rmtree(root / rel, ignore_errors=True)
    for path in backup_dir.rglob("*"):
        if path.is_file():
            dest = root / path.relative_to(backup_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


# -- 0.0.9: CSV working files --------------------------------------------------


def read_points_csv(path: Path) -> pd.DataFrame:
    """Read a 0.0.9 points.csv, as 0.0.9 did."""
    df = pd.read_csv(path, header=[0, 1], index_col=[0, 1, 2, 3, 4])
    df.index.names = INDEX_NAMES
    # experiment/trial are names, not numbers: all-digit ones are read as int64
    df.index = df.index.set_levels([
        df.index.levels[0].astype(str),
        df.index.levels[1].astype(str),
        df.index.levels[2].astype(int),
        df.index.levels[3].astype(int),
        df.index.levels[4].astype(str),
    ])
    return normalize_columns(df)


def _remove_with_backup(path: Path) -> None:
    path.unlink()
    path.with_name(path.name + ".bak").unlink(missing_ok=True)


def _convert_points(path: Path) -> None:
    parquet = path.with_name(POINTS_FILE)
    if not parquet.exists():
        write_points_file(read_points_csv(path), parquet)
    _remove_with_backup(path)


def _convert_corners(path: Path) -> None:
    parquet = path.with_name(CORNERS_FILE)
    if not parquet.exists():
        df = pd.read_csv(path, dtype={"take": str, "detected": str})
        table = pa.Table.from_pandas(df[CORNER_COLUMNS], schema=CORNERS_SCHEMA,
                                     preserve_index=False)
        write_table(table, parquet)
    _remove_with_backup(path)


def _convert_sync_history(root: Path) -> None:
    path = legacy_files(root)[SYNC_HISTORY][0]
    parquet = root / SYNC_HISTORY_FILE
    if not parquet.exists():
        text = {c: str for c in ("timestamp", "kind", "experiment", "trial")}
        df = pd.read_csv(path, dtype=text)[SYNC_HISTORY_COLUMNS]
        write_table(pa.Table.from_pandas(df, schema=SYNC_HISTORY_SCHEMA, preserve_index=False),
                    parquet)
    _remove_with_backup(path)


def _convert_record(path: Path) -> None:
    """A refinement's points record, and the points_file its registry entry names."""
    parquet_name = path.name.removesuffix(LEGACY_RECORD_SUFFIX) + "_points.parquet"
    parquet = path.with_name(parquet_name)
    if not parquet.exists():
        df = pd.read_csv(path, dtype={c: str for c in ("trial", "bodypart", "camera", "source")})
        df["source"] = df["source"].fillna("")
        df["holdout"] = df["holdout"].astype(bool)
        write_table(pa.Table.from_pandas(df[RECORD_COLUMNS], schema=RECORD_SCHEMA,
                                         preserve_index=False), parquet)
    for entry in sorted(path.parent.glob("*.toml")):
        data = toml.load(entry)
        points = data.get("metadata", {}).get("points", {})
        if points.get("points_file") == path.name:
            points["points_file"] = parquet_name
            with open(entry, "w") as f:
                toml.dump(data, f)
    _remove_with_backup(path)


def _move_skeleton_json(root: Path, config: dict) -> None:
    """skeleton.json goes to [skeleton] in project.toml, assigned to every camera."""
    path = legacy_files(root)[SKELETON][0]
    if "skeleton" not in config:
        cameras = _extract_cameras(pd.read_csv(root / "trials.csv", nrows=0).columns.tolist())
        config["skeleton"] = assign_in_config({}, Skeleton.from_json(path), cameras, cameras)
    path.unlink()


# -- To 0.3.0 ----------------------------------------------------------------


def _slp_into_folders(root: Path, config: dict) -> dict[Path, Path]:
    """slp/{name}.slp -> slp/{stem}/{name}.slp, with [slp] paths rewritten. Returns old -> new."""
    moved: dict[Path, Path] = {}
    for path in flat_slp_files(root):
        rel = path.relative_to(root)
        new = slp_file_path(path.stem)
        if (root / new).exists():
            raise FileExistsError(f"Can't move {rel}: {new} already exists")
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        path.rename(root / new)
        moved[rel] = new
    for spec in config.get("slp", {}).get("files", []):
        new = moved.get(Path(spec["path"]))
        if new is not None:
            spec["path"] = new.as_posix()
    return moved


def _renumber_frames(root: Path, config: dict) -> None:
    """trials.csv from 0-based to FRAME_BASE (the first rewrite keeps trials.csv.bak)."""
    path = root / "trials.csv"
    write_trials_csv(path, read_trials_csv(path, frame_base=0))
    config.setdefault("trials", {})["frame_base"] = FRAME_BASE


# -- Restoring the .slp files --------------------------------------------------


def restore_slp_files(
    root: Path, sources: dict[Path, Path | None], report: UpgradeProgress
) -> list[str]:
    """Put back what older Syncs dropped from the mapped .slp files.

    sources maps each file (relative to root) to its original, or None to keep
    the project's copy. In order:

    1. A Sync read of the project's copies brings into the points anything
       added in SLEAP since the last Sync (points label3d has are kept).
    2. A user label placed over a prediction, before label3d kept predictions
       under user labels, gets that prediction back in px, py, pscore, from
       the project's copy or the original.
    3. Each original replaces the project's copy, with its models folder.
    4. A Sync write rewrites the project's part of each file from the points
       and keeps the rest of the original.
    """
    messages: list[str] = []
    project = Project.load(root, check_version=False)

    report("Reading the project's .slp files")
    results = project.read_slp_for_sync()
    project.apply_merge(results, overwrite=False)
    n_new = sum(r.n_new + r.n_new_pred for r in results.values())
    n_conflicts = sum(len(r.conflicts) for r in results.values())
    if n_new:
        messages.append(f"Read {n_new} new point(s) from the project's .slp files")
    if n_conflicts:
        messages.append(f"Kept label3d's points where {n_conflicts} point(s) differed "
                        "from the project's .slp files")

    report("Recovering predictions hidden under user labels")
    files = [root / rel for rel in sources] + [src for src in sources.values() if src]
    recovered = _recover_hidden_predictions(project, files)
    if recovered:
        messages.append(f"Recovered {recovered} prediction(s) under user labels")

    for rel, original in sources.items():
        dest = root / rel
        if original is None:
            messages.append(f"{rel.as_posix()}: kept the project's copy")
            continue
        report(f"Restoring {rel.as_posix()}")
        shutil.copy2(original, dest)
        messages.append(f"{rel.as_posix()}: restored from {original}")
        models = original.parent / MODELS_DIR
        if not models.is_dir() or (dest.parent / MODELS_DIR).exists():
            continue
        # As in setup, a models folder is only taken from beside a .slp file of its own
        if len(list(original.parent.glob("*.slp"))) > 1:
            messages.append(f"{rel.as_posix()}: {models} is beside several .slp files; "
                            "not copied")
        else:
            shutil.copytree(models, dest.parent / MODELS_DIR)
            messages.append(f"{rel.as_posix()}: copied {models}")

    report("Writing the points to the .slp files")
    project.write_slp_files()
    return messages


def _recover_hidden_predictions(project: Project, files: list[Path]) -> int:
    """Fill px, py, pscore under user labels from the predictions in files. Returns how many."""
    adapter = SleapAdapter()
    predictions: dict[str, pd.DataFrame] = {}
    for path in files:
        slp_df = adapter.load([path], project.trials_df, project.cameras)
        for cam in project.cameras:
            if (cam, "state") not in slp_df.columns:
                continue
            found = _predictions(slp_df, cam)
            # The first file with a prediction for a point gives it
            known = predictions.get(cam)
            predictions[cam] = found if known is None else known.combine_first(found)

    total = 0
    for name in project.experiments:
        df = project.experiment_points_df(name)
        if df.empty:
            continue
        changed = 0
        for cam, pred in predictions.items():
            if (cam, "state") not in df.columns:
                continue
            found = pred.reindex(df.index)
            need = (
                df[(cam, "state")].isin(USER_STATES)
                & df[(cam, "source")].isin((SOURCE_LABEL3D, SOURCE_SLEAP_USER))
                & df[(cam, "px")].isna()
                & found["px"].notna()
            )
            for coord in ("px", "py", "pscore"):
                df.loc[need, (cam, coord)] = found.loc[need, coord]
            changed += int(need.sum())
        if changed:
            project.store_experiment_df(name, df)
            total += changed
    return total


def _predictions(slp_df: pd.DataFrame, cam: str) -> pd.DataFrame:
    """A camera's predictions in a read .slp file: predicted points and hidden ones."""
    predicted = slp_df[(cam, "state")] == "predicted"
    out = pd.DataFrame({
        "px": slp_df[(cam, "x")].where(predicted, slp_df[(cam, "px")]),
        "py": slp_df[(cam, "y")].where(predicted, slp_df[(cam, "py")]),
        "pscore": slp_df[(cam, "score")].where(predicted, slp_df[(cam, "pscore")]),
    })
    return out[out["px"].notna() & out["py"].notna()]

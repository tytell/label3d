"""Project folder I/O: trials.csv, calibration, paths."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import toml

import logging
logger = logging.getLogger(__name__)

from label3d.adapters.sleap_adapter import SleapAdapter
from label3d.calibration_registry import (
    LEGACY_CALIBRATION,
    CalibrationEntry,
    find_calibration,
    import_legacy,
    list_calibrations,
)
from label3d.points import PointsStore
from label3d.skeleton import Skeleton
from label3d.slp_sync import MergeResult, SlpMapping, compute_merge


# triangulated_with value for experiments whose 3D points predate the
# registry: never equal to a real id, so the re-triangulate warning shows.
LEGACY_TRIANGULATION = "legacy"

SYNC_HISTORY_COLUMNS = [
    "id", "timestamp", "kind", "experiment", "trial",
    "n_new_user", "n_new_pred", "n_updated_pred", "n_conflicts", "n_frames",
]


@dataclass
class CameraInfo:
    """Minimal camera info from the active calibration (enough for display; full
    calibration data loaded later when geometry is needed)."""

    name: str
    size: tuple[int, int]  # (width, height)


@dataclass
class TrialInfo:
    """One row of trials.csv, parsed."""

    name: str
    # camera_name -> (video_filename, start_frame, end_frame_or_none)
    camera_videos: dict[str, tuple[str, int, int | None]]
    length: int | None = None  # computed after resolving video lengths


@dataclass
class ExperimentInfo:
    """Per-experiment metadata derived from the project structure."""

    name: str
    root: Path                              # project.root / "experiments" / name
    cameras: list[str]
    trials: list[TrialInfo]
    has_calibration: bool
    camera_info: dict[str, CameraInfo] = field(default_factory=dict)
    # Calibration takes for this experiment, from the is_calibration rows
    calibration_takes: list[TrialInfo] = field(default_factory=list)
    # Base directory for videos referenced in place rather than copied into the
    # project, from the [video_roots] section of project.toml. None for the
    # classic layout, where videos live in videos/ and cells are bare filenames.
    video_root: Path | None = None


@dataclass
class Project:
    """Loaded project folder."""

    root: Path
    cameras: list[str]                   # ordered camera names (active experiment)
    trials: list[TrialInfo] = field(default_factory=list)
    camera_info: dict[str, CameraInfo] = field(default_factory=dict)
    has_calibration: bool = False
    skeleton: Skeleton | None = None
    points: PointsStore | None = None
    trials_df: pd.DataFrame | None = None  # raw global trials.csv
    config: dict = field(default_factory=dict)  # from project.toml
    experiments: dict[str, ExperimentInfo] = field(default_factory=dict)
    active_experiment: str = ""

    def __post_init__(self) -> None:
        self._attach_latest_sync()

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load a project from a directory."""
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"Project directory not found: {root}")

        # --- global trials.csv at project root ---
        trials_path = root / "trials.csv"
        if not trials_path.exists():
            raise FileNotFoundError(f"trials.csv not found at {trials_path}")

        # experiment and trial names are keys everywhere downstream (dict keys,
        # path components, points.csv index levels), so they must stay strings —
        # an all-digit name like "20260619" is otherwise inferred as int64.
        trials_df = pd.read_csv(trials_path, dtype={"experiment": str, "trial": str})
        cameras = _extract_cameras(trials_df.columns.tolist())

        # --- project.toml at project root ---
        # Read before the experiment loop: [video_roots] is needed to resolve
        # video paths while trial lengths are computed below.
        config: dict = {}
        config_path = root / "project.toml"
        if config_path.exists():
            config = toml.load(config_path)
        video_roots = {
            str(name): Path(str(value)).expanduser()
            for name, value in config.get("video_roots", {}).items()
        }

        # --- Build ExperimentInfo for each experiment ---
        if "experiment" in trials_df.columns:
            experiment_names = sorted(trials_df["experiment"].unique().tolist())
        else:
            experiment_names = ["default"]

        experiments: dict[str, ExperimentInfo] = {}
        missing: list[str] = []
        config_changed = False
        for exp_name in experiment_names:
            exp_dir = root / "experiments" / exp_name
            config_changed |= _import_legacy_calibrations(config, exp_name, exp_dir)

            if "experiment" in trials_df.columns:
                exp_rows = trials_df[trials_df["experiment"] == exp_name]
            else:
                exp_rows = trials_df

            exp_trials = _parse_trials(exp_rows, cameras)
            videos_dir = exp_dir / "videos"
            video_root = video_roots.get(exp_name)
            for trial in exp_trials:
                _resolve_trial_length(trial, videos_dir, video_root, root)
            missing.extend(
                _missing_references(exp_name, exp_trials, video_root, videos_dir, root)
            )

            # Calibration take videos live under calibration/{cam}/, not
            # videos/, so their lengths are resolved on demand instead.
            exp_takes = _parse_calibration_takes(exp_rows, cameras)

            has_calibration, camera_info = _read_camera_info(
                _active_calibration_path(config, exp_name, exp_dir)
            )

            experiments[exp_name] = ExperimentInfo(
                name=exp_name,
                root=exp_dir,
                cameras=cameras,
                trials=exp_trials,
                has_calibration=has_calibration,
                camera_info=camera_info,
                calibration_takes=exp_takes,
                video_root=video_root,
            )

        # Videos referenced outside the project must exist — a broken reference
        # is a stale [video_roots] entry or an unmounted drive, and failing here
        # names the trial and path instead of failing later inside OpenCV.
        if missing:
            raise FileNotFoundError(
                "Referenced videos not found:\n  "
                + "\n  ".join(missing)
                + "\nCheck the [video_roots] section of project.toml."
            )

        if config_changed:
            _write_config(root, config)

        active_experiment = experiment_names[0]
        active = experiments[active_experiment]

        # --- skeleton.json at project root (shared) ---
        skeleton: Skeleton | None = None
        skel_path = root / "skeleton.json"
        if skel_path.exists():
            skeleton = Skeleton.from_json(skel_path)

        # --- points.csv inside active experiment folder ---
        points: PointsStore | None = None
        points_path = active.root / "points.csv"
        if points_path.exists():
            try:
                points = PointsStore.from_csv(points_path)
            except Exception:
                pass
        else:
            points = PointsStore.create_empty(
                cameras=active.cameras, bodyparts=[], experiment_name=active_experiment
            )
            points._save_path = points_path

        return cls(
            root=root,
            cameras=active.cameras,
            trials=active.trials,
            camera_info=active.camera_info,
            has_calibration=active.has_calibration,
            skeleton=skeleton,
            points=points,
            trials_df=trials_df,
            config=config,
            experiments=experiments,
            active_experiment=active_experiment,
        )

    def video_path(self, trial_name: str, camera: str) -> Path:
        """Resolve absolute path to a trial or calibration-take video.

        Calibration takes live in calibration/{camera}/, trials in videos/ —
        unless the cell names a path, in which case the video is referenced
        where it already lives (see _resolve_video).
        """
        exp_dir = self.root / "experiments" / self.active_experiment
        video_root = self.experiments[self.active_experiment].video_root
        try:
            trial = self.get_trial(trial_name)
        except KeyError:
            take = self.get_calibration_take(trial_name)
            filename, _, _ = take.camera_videos[camera]
            default_dir = exp_dir / "calibration" / camera
            # Calibration videos are copied into the project even when trial
            # videos are referenced in place, so a cell naming a file in
            # calibration/{camera}/ wins over the experiment's video_root.
            local = default_dir / filename
            if local.exists():
                return local
        else:
            filename, _, _ = trial.camera_videos[camera]
            default_dir = exp_dir / "videos"
        return _resolve_video(filename, video_root, default_dir, self.root)

    def calibration_path(self, experiment: str | None = None) -> Path | None:
        """File of an experiment's active calibration (default experiment: the
        active one). None when no calibration is active or its file is gone."""
        name = experiment or self.active_experiment
        return _active_calibration_path(self.config, name, self.root / "experiments" / name)

    def calibrations(self, experiment: str | None = None) -> list[CalibrationEntry]:
        """An experiment's registry entries, newest first."""
        name = experiment or self.active_experiment
        return list_calibrations(self.root / "experiments" / name)

    def active_calibration_id(self, experiment: str | None = None) -> str | None:
        name = experiment or self.active_experiment
        return _pointer(self.config, "active", name)

    def active_calibration(self, experiment: str | None = None) -> CalibrationEntry | None:
        name = experiment or self.active_experiment
        entry_id = self.active_calibration_id(name)
        if entry_id is None:
            return None
        return find_calibration(self.root / "experiments" / name, entry_id)

    def triangulated_with(self, experiment: str | None = None) -> str | None:
        """Id of the calibration the experiment's 3D points were last computed with."""
        name = experiment or self.active_experiment
        return _pointer(self.config, "triangulated_with", name)

    def set_active_calibration(self, experiment: str | None, entry_id: str | None) -> None:
        """Point an experiment at a registry entry (None clears it) and reload."""
        name = experiment or self.active_experiment
        self._set_pointer("active", name, entry_id)
        self.reload_calibration(name)

    def set_triangulated_with(self, experiment: str | None, entry_id: str | None) -> None:
        name = experiment or self.active_experiment
        self._set_pointer("triangulated_with", name, entry_id)

    def triangulation_is_stale(self, experiment: str | None = None) -> bool:
        """True when 3D points may come from a calibration other than the active one."""
        name = experiment or self.active_experiment
        active = self.active_calibration_id(name)
        return active is not None and self.triangulated_with(name) != active

    def _set_pointer(self, table: str, experiment: str, entry_id: str | None) -> None:
        # save_config_sections replaces whole sections, so copy [calibration]
        # to keep frame_skip, repeats and the drift thresholds.
        calibration = dict(self.config.get("calibration", {}))
        pointers = dict(calibration.get(table, {}))
        if entry_id is None:
            pointers.pop(experiment, None)
        else:
            pointers[experiment] = entry_id
        calibration[table] = pointers
        self.save_config_sections({"calibration": calibration})

    @property
    def slp_dir(self) -> Path:
        return self.root / "slp"

    @property
    def trials_csv_path(self) -> Path:
        return self.root / "trials.csv"

    @property
    def slp_mapping(self) -> SlpMapping:
        """SLP file assignments from the [slp] section of project.toml."""
        return SlpMapping.from_config(self.config)

    def save_config_sections(self, sections: dict[str, dict]) -> None:
        """Merge top-level sections into project.toml, keeping all others."""
        config_path = self.root / "project.toml"
        config = toml.load(config_path) if config_path.exists() else {}
        config.update(sections)
        _write_config(self.root, config)
        self.config = config

    def save_slp_mapping(self, mapping: SlpMapping) -> None:
        """Write the [slp] section to project.toml, keeping all other sections."""
        self.save_config_sections({"slp": mapping.to_config()})

    def experiment_points_df(self, experiment: str) -> pd.DataFrame:
        """Points for an experiment: the live store if active, else its points.csv."""
        if experiment == self.active_experiment and self.points is not None:
            return self.points.df
        csv_path = self.experiments[experiment].root / "points.csv"
        if csv_path.exists():
            return PointsStore.read_csv_df(csv_path)
        return PointsStore.empty_df(self.cameras)

    def read_slp_for_sync(
        self, mapping: SlpMapping | None = None, sync: bool = True
    ) -> dict[str, MergeResult]:
        """Read SLP files and merge them with each experiment's points.

        Uses the mapping (default: project.toml's); with no mapping, reads all
        slp/*.slp files matched by video basename.  Does not modify anything.
        Safe to call from a background thread.

        sync=True gives merged cells the next sync id, so they show as new once
        apply_merge() records the sync.  sync=False (initial import at project
        setup) records sources and scores but marks nothing new.
        """
        mapping = self.slp_mapping if mapping is None else mapping
        adapter = SleapAdapter()
        if mapping.files:
            slp_df = adapter.load(
                None, self.trials_df, self.cameras, mapping=mapping, root=self.root
            )
        else:
            slp_df = adapter.load(sorted(self.slp_dir.glob("*.slp")), self.trials_df, self.cameras)
        return self._merge_by_experiment(slp_df, self.next_sync_id() if sync else None)

    def read_predictions_for_import(
        self, paths: list[Path]
    ) -> tuple[dict[str, MergeResult], list[str]]:
        """Read .slp files from anywhere (e.g. sleap-nn track output) and merge them.

        Videos are matched to (experiment, trial, camera) by basename, so a
        file made on another machine still matches.  The files are only read.
        Returns (results by experiment, basenames of videos that matched no
        trial).  Safe to call from a background thread.
        """
        unmatched: list[str] = []
        slp_df = SleapAdapter().load(
            list(paths), self.trials_df, self.cameras, unmatched=unmatched
        )
        return self._merge_by_experiment(slp_df, self.next_sync_id()), unmatched

    def _merge_by_experiment(
        self, slp_df: pd.DataFrame, sync_id: int | None
    ) -> dict[str, MergeResult]:
        results = {}
        for name in self.experiments:
            exp_slp = slp_df
            if not slp_df.empty:
                exp_slp = slp_df[slp_df.index.get_level_values("experiment") == name]
            results[name] = compute_merge(self.experiment_points_df(name), exp_slp, sync_id)
        return results

    # -- Sync history ----------------------------------------------------------

    @property
    def sync_history_path(self) -> Path:
        return self.root / "sync_history.csv"

    def sync_history(self) -> pd.DataFrame:
        """All recorded syncs, one row per (sync, experiment, trial) that changed."""
        path = self.sync_history_path
        if not path.exists():
            return pd.DataFrame(columns=SYNC_HISTORY_COLUMNS)
        df = pd.read_csv(path, dtype={"experiment": str, "trial": str})
        return df

    def latest_sync_id(self) -> int | None:
        history = self.sync_history()
        if history.empty:
            return None
        return int(history["id"].max())

    def next_sync_id(self) -> int:
        latest = self.latest_sync_id()
        return 1 if latest is None else latest + 1

    def _record_sync(
        self, results: dict[str, MergeResult], overwrite: bool, kind: str
    ) -> None:
        """Append one history row per (experiment, trial) with new points or conflicts taken."""
        rows = []
        timestamp = datetime.now().isoformat(timespec="seconds")
        for name, result in results.items():
            if result.sync_id is None:
                continue
            new_frames = set(result.new_frames)
            if overwrite:
                new_frames |= result.conflict_frames
            for (exp, trial), counts in sorted(result.trial_counts.items()):
                n_conflicts = counts.get("conflicts", 0)
                row = {
                    "id": result.sync_id,
                    "timestamp": timestamp,
                    "kind": kind,
                    "experiment": exp,
                    "trial": trial,
                    "n_new_user": counts.get("new_user", 0)
                    + (n_conflicts if overwrite else 0),
                    "n_new_pred": counts.get("new_pred", 0),
                    "n_updated_pred": counts.get("updated_pred", 0),
                    "n_conflicts": n_conflicts,
                    "n_frames": sum(1 for f in new_frames if f[:2] == (exp, trial)),
                }
                if row["n_new_user"] or row["n_new_pred"] or row["n_updated_pred"]:
                    rows.append(row)
        if not rows:
            return
        path = self.sync_history_path
        pd.DataFrame(rows, columns=SYNC_HISTORY_COLUMNS).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )

    def _attach_latest_sync(self) -> None:
        if self.points is None:
            return
        history = self.sync_history()
        if history.empty:
            self.points.latest_sync = None
            self.points.latest_sync_label = ""
            return
        latest = history[history["id"] == history["id"].max()].iloc[0]
        self.points.latest_sync = int(latest["id"])
        when = str(latest["timestamp"]).replace("T", " ")[:16]
        self.points.latest_sync_label = f"{latest['kind']} {int(latest['id'])} ({when})"

    def apply_merge(
        self, results: dict[str, MergeResult], overwrite: bool, kind: str = "sync"
    ) -> None:
        """Store merge results: the active store in memory + points.csv, others to points.csv.

        overwrite=True takes conflicting points from the SLP files.  Merges that
        brought in new points are recorded in sync_history.csv (kind: sync,
        open, close or import), which makes them the latest sync.
        Must be called from the main thread (may create a PointsStore).
        """
        self._record_sync(results, overwrite, kind)
        for name, result in results.items():
            if not result.has_changes and not (overwrite and result.conflicts):
                continue
            df = result.df_overwrite if overwrite else result.df
            exp = self.experiments[name]
            if name == self.active_experiment:
                if self.points is None:
                    self.points = PointsStore.create_empty(
                        cameras=self.cameras, bodyparts=[], experiment_name=name
                    )
                    self.points._save_path = exp.root / "points.csv"
                self.points.replace_df(df)
                self.points.save_if_changed()
            else:
                exp.root.mkdir(parents=True, exist_ok=True)
                df.to_csv(exp.root / "points.csv")
        self._attach_latest_sync()

    def write_slp_files(self, mapping: SlpMapping | None = None) -> int:
        """Write all experiments' points to the mapped SLP files.  Returns files written.

        Safe to call from a background thread.
        """
        mapping = self.slp_mapping if mapping is None else mapping
        dfs = [self.experiment_points_df(name) for name in self.experiments]
        dfs = [df for df in dfs if not df.empty]
        all_df = pd.concat(dfs) if dfs else PointsStore.empty_df(self.cameras)
        return SleapAdapter().save(
            all_df,
            cameras=self.cameras,
            skeleton=self.skeleton,
            trials_df=self.trials_df,
            mapping=mapping,
            root=self.root,
        )

    def backup_slp_files(self) -> Path | None:
        """Copy slp/*.slp to slp/backup/{timestamp}/.  Returns the backup folder, if any."""
        files = sorted(self.slp_dir.glob("*.slp"))
        if not files:
            return None
        backup_dir = self.slp_dir / "backup" / datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, backup_dir / f.name)
        return backup_dir

    def remove_unmapped_slp_files(self, mapping: SlpMapping) -> list[Path]:
        """Delete slp/*.slp files that are not in the mapping.  Back up first."""
        mapped = {(self.root / spec.path).resolve() for spec in mapping.files}
        removed = []
        for f in sorted(self.slp_dir.glob("*.slp")):
            if f.resolve() not in mapped:
                f.unlink()
                removed.append(f)
        return removed

    def set_experiment(self, name: str) -> None:
        """Switch active experiment, updating cameras/trials/calibration/points."""
        if name not in self.experiments:
            raise KeyError(f"Experiment not found: {name}")
        exp = self.experiments[name]
        self.active_experiment = name
        self.cameras = exp.cameras
        self.trials = exp.trials
        self.camera_info = exp.camera_info
        self.has_calibration = exp.has_calibration
        points_path = exp.root / "points.csv"
        if points_path.exists():
            try:
                self.points = PointsStore.from_csv(points_path)
            except Exception:
                self.points = None
        else:
            self.points = PointsStore.create_empty(
                cameras=exp.cameras, bodyparts=[], experiment_name=name
            )
            self.points._save_path = points_path
        self._attach_latest_sync()

    def reload_calibration(self, experiment: str | None = None) -> None:
        """Re-read an experiment's active calibration (default: the active
        experiment), after the pointer moved or a calibration was written."""
        name = experiment or self.active_experiment
        exp = self.experiments[name]
        exp.has_calibration, exp.camera_info = _read_camera_info(
            self.calibration_path(name)
        )
        if name == self.active_experiment:
            self.has_calibration = exp.has_calibration
            self.camera_info = exp.camera_info

    @property
    def calibration_takes(self) -> list[TrialInfo]:
        """Calibration takes of the active experiment, in trials.csv order."""
        exp = self.experiments.get(self.active_experiment)
        return exp.calibration_takes if exp else []

    def get_trial(self, name: str) -> TrialInfo:
        for t in self.trials:
            if t.name == name:
                return t
        raise KeyError(f"Trial not found: {name}")

    def get_calibration_take(self, name: str) -> TrialInfo:
        for t in self.calibration_takes:
            if t.name == name:
                return t
        raise KeyError(f"Calibration take not found: {name}")

    def trial_names(self) -> list[str]:
        return [t.name for t in self.trials]

    def trial_length(self, name: str) -> int:
        trial = self.get_trial(name)
        return trial.length or 0

    def video_start_frame(self, trial_name: str, camera: str) -> int:
        """Return the start frame offset for a camera in a trial."""
        trial = self.get_trial(trial_name)
        _, start, _ = trial.camera_videos[camera]
        return start


def _write_config(root: Path, config: dict) -> None:
    with open(root / "project.toml", "w") as f:
        toml.dump(config, f)


def _pointer(config: dict, table: str, experiment: str) -> str | None:
    """An experiment's id in [calibration.{table}] of project.toml, or None."""
    value = config.get("calibration", {}).get(table, {}).get(experiment)
    return str(value) if value else None


def _active_calibration_path(config: dict, experiment: str, exp_dir: Path) -> Path | None:
    entry_id = _pointer(config, "active", experiment)
    if entry_id is None:
        return None
    entry = find_calibration(exp_dir, entry_id)
    return entry.path if entry else None


def _import_legacy_calibrations(config: dict, experiment: str, exp_dir: Path) -> bool:
    """Move an experiment's pre-registry calibration files into the registry.

    The entry from calibration.toml becomes active. triangulated_with is set to
    a placeholder, so the Label tab asks for one re-triangulation. Returns True
    when config was changed (the caller writes it).
    """
    entries = import_legacy(exp_dir)
    active = next(
        (e for e in entries if e.metadata.get("source_path") == LEGACY_CALIBRATION), None
    )
    if active is None:
        return False
    calibration = config.setdefault("calibration", {})
    calibration.setdefault("active", {})[experiment] = active.id
    calibration.setdefault("triangulated_with", {})[experiment] = LEGACY_TRIANGULATION
    logger.info("Imported %d legacy calibration(s) for %s", len(entries), experiment)
    return True


def _read_camera_info(calib_path: Path | None) -> tuple[bool, dict[str, CameraInfo]]:
    """(has_calibration, camera name -> CameraInfo) from a calibration file."""
    if calib_path is None or not calib_path.exists():
        return False, {}
    camera_info: dict[str, CameraInfo] = {}
    cal = toml.load(calib_path)
    for key in sorted(cal.keys()):
        if key in ("metadata", "alignment"):
            continue
        section = cal[key]
        name = section.get("name", key)
        size = tuple(section.get("size", [0, 0]))
        camera_info[name] = CameraInfo(name=name, size=size)
    return True, camera_info


# trials.csv columns that are metadata, not camera names.
NON_CAMERA_COLUMNS = ("trial", "experiment", "is_calibration")


def _extract_cameras(columns: list[str]) -> list[str]:
    """Extract ordered camera names from trials.csv columns.

    Skips NON_CAMERA_COLUMNS and any *_start / *_end suffix columns. Anything
    else is taken to be a camera, so every new metadata column must be added to
    NON_CAMERA_COLUMNS or it becomes a phantom camera.
    """
    cameras = []
    for col in columns:
        if col in NON_CAMERA_COLUMNS:
            continue
        if col.endswith("_start") or col.endswith("_end"):
            continue
        cameras.append(col)
    return cameras


def _calibration_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of calibration-take rows."""
    return df["is_calibration"].astype(str).str.lower().isin(("true", "1", "yes"))


def _parse_trials(df: pd.DataFrame, cameras: list[str]) -> list[TrialInfo]:
    """Parse the trial (non-calibration) rows of trials.csv into TrialInfo."""
    if "is_calibration" in df.columns:
        df = df[~_calibration_mask(df)]
    return _parse_rows(df, cameras)


def _parse_calibration_takes(df: pd.DataFrame, cameras: list[str]) -> list[TrialInfo]:
    """Parse the calibration-take rows of trials.csv into TrialInfo.

    A take has the same shape as a trial — one video per camera, with optional
    start/end trimming — so it reuses TrialInfo.
    """
    if "is_calibration" not in df.columns:
        return []
    return _parse_rows(df[_calibration_mask(df)], cameras)


def _parse_rows(df: pd.DataFrame, cameras: list[str]) -> list[TrialInfo]:
    """Parse trials.csv rows into TrialInfo objects."""
    trials = []
    for _, row in df.iterrows():
        camera_videos = {}
        for cam in cameras:
            filename = str(row[cam])
            # A camera may be absent from a row (a calibration take that one
            # camera did not record), leaving every cell blank.
            start_val = row.get(f"{cam}_start", 0)
            start = 0 if pd.isna(start_val) or start_val == "" else int(start_val)
            end_val = row.get(f"{cam}_end")
            end = None if pd.isna(end_val) or end_val == "" else int(end_val)
            camera_videos[cam] = (filename, start, end)
        trials.append(TrialInfo(name=str(row["trial"]), camera_videos=camera_videos))
    return trials


def _is_blank_cell(cell: str) -> bool:
    """True for a camera cell that names no video (blank, or pandas NaN)."""
    return not cell or cell.strip().lower() in ("", "nan")


def _resolve_video(
    cell: str, video_root: Path | None, default_dir: Path, project_root: Path
) -> Path:
    """Resolve one trials.csv camera cell to a video path.

    Three forms are accepted, in order:

    1. an absolute path — used as is;
    2. a path-valued cell, or any cell when the experiment has a ``video_root``
       — resolved against that root, falling back to the project root;
    3. a bare filename with no ``video_root`` — the classic layout, resolved
       against ``default_dir`` (``videos/`` for trials, ``calibration/{cam}/``
       for takes).

    Projects written before external video references existed have no
    ``video_root`` and only bare filenames, so they always take branch 3.
    """
    path = Path(cell)
    if path.is_absolute():
        return path
    if video_root is not None:
        return video_root / path
    if len(path.parts) > 1:
        return project_root / path
    return default_dir / path


def _is_external_cell(cell: str, video_root: Path | None) -> bool:
    """True when a cell references a video outside the project's videos/ dir."""
    if _is_blank_cell(cell):
        return False
    return video_root is not None or len(Path(cell).parts) > 1 or Path(cell).is_absolute()


def _missing_references(
    experiment: str,
    trials: list[TrialInfo],
    video_root: Path | None,
    videos_dir: Path,
    project_root: Path,
) -> list[str]:
    """Describe every external video reference that does not exist on disk.

    Only external references are checked: a bare filename under videos/ may be
    legitimately absent (a trial one camera did not record), and those projects
    predate this check.
    """
    problems = []
    for trial in trials:
        for cam, (filename, _, _) in trial.camera_videos.items():
            if not _is_external_cell(filename, video_root):
                continue
            path = _resolve_video(filename, video_root, videos_dir, project_root)
            if not path.exists():
                problems.append(f"{experiment}/{trial.name} [{cam}]: {path}")
    return problems


def _resolve_trial_length(
    trial: TrialInfo,
    videos_dir: Path,
    video_root: Path | None = None,
    project_root: Path | None = None,
):
    """Compute trial length = min(end - start + 1) across cameras.

    If end is None, read frame count from the video file.
    """
    root = project_root if project_root is not None else videos_dir.parent.parent
    lengths = []
    for cam, (filename, start, end) in trial.camera_videos.items():
        if end is None:
            if _is_blank_cell(filename):
                continue
            video_path = _resolve_video(filename, video_root, videos_dir, root)
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                end = total - 1
                trial.camera_videos[cam] = (filename, start, end)
            else:
                continue
        lengths.append(end - start + 1)

    trial.length = min(lengths) if lengths else 0

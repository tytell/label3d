"""Project folder I/O: trials.csv, calibration, paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import pandas as pd
import toml

import logging
logger = logging.getLogger(__name__)

from label3d.points import PointsStore
from label3d.skeleton import Skeleton


@dataclass
class CameraInfo:
    """Minimal camera info from calibration.toml (enough for display; full
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

        trials_df = pd.read_csv(trials_path)
        cameras = _extract_cameras(trials_df.columns.tolist())

        # --- Build ExperimentInfo for each experiment ---
        if "experiment" in trials_df.columns:
            experiment_names = sorted(trials_df["experiment"].unique().tolist())
        else:
            experiment_names = ["default"]

        experiments: dict[str, ExperimentInfo] = {}
        for exp_name in experiment_names:
            exp_dir = root / "experiments" / exp_name

            if "experiment" in trials_df.columns:
                exp_rows = trials_df[trials_df["experiment"] == exp_name]
            else:
                exp_rows = trials_df

            exp_trials = _parse_trials(exp_rows, cameras)
            videos_dir = exp_dir / "videos"
            for trial in exp_trials:
                _resolve_trial_length(trial, videos_dir)

            calib_path = exp_dir / "calibration.toml"
            has_calibration = False
            camera_info: dict[str, CameraInfo] = {}
            if calib_path.exists():
                has_calibration = True
                cal = toml.load(calib_path)
                for key in sorted(cal.keys()):
                    if key in ("metadata", "alignment"):
                        continue
                    section = cal[key]
                    name = section.get("name", key)
                    size = tuple(section.get("size", [0, 0]))
                    camera_info[name] = CameraInfo(name=name, size=size)

            experiments[exp_name] = ExperimentInfo(
                name=exp_name,
                root=exp_dir,
                cameras=cameras,
                trials=exp_trials,
                has_calibration=has_calibration,
                camera_info=camera_info,
            )

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

        # --- project.toml at project root ---
        config: dict = {}
        config_path = root / "project.toml"
        if config_path.exists():
            config = toml.load(config_path)

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
        """Resolve absolute path to a video file."""
        trial = self.get_trial(trial_name)
        filename, _, _ = trial.camera_videos[camera]
        return self.root / "experiments" / self.active_experiment / "videos" / filename

    def calibration_path(self) -> Path:
        """Path to the active experiment's calibration.toml."""
        return self.root / "experiments" / self.active_experiment / "calibration.toml"

    @property
    def slp_dir(self) -> Path:
        return self.root / "slp"

    @property
    def trials_csv_path(self) -> Path:
        return self.root / "trials.csv"

    @property
    def slp_import_available(self) -> bool:
        """True when points.csv is empty and slp/ contains .slp files."""
        if not self.slp_dir.exists():
            return False
        if not any(self.slp_dir.glob("*.slp")):
            return False
        return self.points is None or len(self.points.df) == 0

    def load_slp_adapter_df(self, experiment: str | None = None) -> pd.DataFrame:
        """Load SLP files and return a filtered adapter DataFrame.

        Pure I/O — does NOT touch PointsStore or any QObject.
        Safe to call from a background thread.
        """
        from label3d.adapters.sleap_adapter import SleapAdapter

        slp_files = list(self.slp_dir.glob("*.slp"))
        if not slp_files or self.trials_df is None:
            return pd.DataFrame()

        target_experiment = experiment or self.active_experiment
        adapter_df = SleapAdapter().load(slp_files, self.trials_df, self.cameras)
        logger.debug(f"SleapAdapter loaded {len(adapter_df)} rows")

        if not adapter_df.empty and "experiment" in adapter_df.index.names:
            logger.debug(
                f"Experiments in SLP import: "
                f"{adapter_df.index.get_level_values('experiment').unique().tolist()} "
                f"- target: {target_experiment}"
            )
            mask = adapter_df.index.get_level_values("experiment") == target_experiment
            adapter_df = adapter_df[mask]
            if adapter_df.empty:
                logger.debug(f"No rows for experiment {target_experiment} in SLP import")

        return adapter_df

    def merge_slp_adapter_df(
        self, adapter_df: pd.DataFrame, experiment: str | None = None
    ) -> int:
        """Merge an already-loaded adapter DataFrame into the active PointsStore.

        Must be called from the main thread (creates PointsStore if needed).
        Returns the number of rows added or updated.
        """
        if adapter_df.empty:
            return 0

        target_experiment = experiment or self.active_experiment

        if self.points is None:
            exp = self.experiments[target_experiment]
            points_path = exp.root / "points.csv"
            self.points = PointsStore.create_empty(
                cameras=self.cameras,
                bodyparts=[],
                experiment_name=target_experiment,
            )
            self.points._save_path = points_path

        added = self.points.merge_from_df(adapter_df)
        logger.debug(f"Merged {added} rows into PointsStore")
        return added

    def import_from_slp(self, experiment: str | None = None) -> int:
        """Load SLP files and merge into the active PointsStore.

        Convenience wrapper — must be called from the main thread.
        Returns the number of rows added or updated.
        """
        adapter_df = self.load_slp_adapter_df(experiment)
        return self.merge_slp_adapter_df(adapter_df, experiment)

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
            self.points = None

    def get_trial(self, name: str) -> TrialInfo:
        for t in self.trials:
            if t.name == name:
                return t
        raise KeyError(f"Trial not found: {name}")

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


def _extract_cameras(columns: list[str]) -> list[str]:
    """Extract ordered camera names from trials.csv columns.

    Skips: experiment, trial, and any *_start / *_end suffix columns.
    """
    cameras = []
    for col in columns:
        if col in ("trial", "experiment"):
            continue
        if col.endswith("_start") or col.endswith("_end"):
            continue
        cameras.append(col)
    return cameras


def _parse_trials(df: pd.DataFrame, cameras: list[str]) -> list[TrialInfo]:
    """Parse trials.csv DataFrame into TrialInfo objects."""
    trials = []
    for _, row in df.iterrows():
        camera_videos = {}
        for cam in cameras:
            filename = str(row[cam])
            start = int(row.get(f"{cam}_start", 0) or 0)
            end_val = row.get(f"{cam}_end")
            end = None if pd.isna(end_val) else int(end_val)
            camera_videos[cam] = (filename, start, end)
        trials.append(TrialInfo(name=str(row["trial"]), camera_videos=camera_videos))
    return trials


def _resolve_trial_length(trial: TrialInfo, videos_dir: Path):
    """Compute trial length = min(end - start + 1) across cameras.

    If end is None, read frame count from the video file.
    """
    lengths = []
    for cam, (filename, start, end) in trial.camera_videos.items():
        if end is None:
            video_path = videos_dir / filename
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

"""Project folder I/O: trials.csv, calibration, paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import pandas as pd
import toml

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
class Project:
    """Loaded project folder."""

    root: Path
    cameras: list[str]                   # ordered camera names
    trials: list[TrialInfo] = field(default_factory=list)
    camera_info: dict[str, CameraInfo] = field(default_factory=dict)
    has_calibration: bool = False
    skeleton: Skeleton | None = None
    points: PointsStore | None = None
    trials_df: pd.DataFrame | None = None  # raw trials.csv for adapter use

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load a project from a directory."""
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"Project directory not found: {root}")

        # --- trials.csv ---
        trials_path = root / "videos" / "trials.csv"
        if not trials_path.exists():
            raise FileNotFoundError(f"trials.csv not found at {trials_path}")

        trials_df = pd.read_csv(trials_path)
        cameras = _extract_cameras(trials_df.columns.tolist())
        trials = _parse_trials(trials_df, cameras)

        # --- Resolve trial lengths ---
        for trial in trials:
            _resolve_trial_length(trial, root / "videos")

        # --- calibration.toml (optional) ---
        calib_path = root / "calibration.toml"
        camera_info: dict[str, CameraInfo] = {}
        has_calibration = False
        if calib_path.exists():
            has_calibration = True
            cal = toml.load(calib_path)
            for key in sorted(cal.keys()):
                if key == "metadata":
                    continue
                section = cal[key]
                name = section.get("name", key)
                size = tuple(section.get("size", [0, 0]))
                camera_info[name] = CameraInfo(name=name, size=size)

        # --- skeleton.json (optional) ---
        skeleton: Skeleton | None = None
        skel_path = root / "skeleton.json"
        if skel_path.exists():
            skeleton = Skeleton.from_json(skel_path)

        # --- points.csv (optional) ---
        points: PointsStore | None = None
        points_path = root / "points.csv"
        if points_path.exists():
            points = PointsStore.from_csv(points_path)

        proj = cls(
            root=root,
            cameras=cameras,
            trials=trials,
            camera_info=camera_info,
            has_calibration=has_calibration,
            skeleton=skeleton,
            points=points,
            trials_df=trials_df,
        )
        return proj

    def video_path(self, trial_name: str, camera: str) -> Path:
        """Resolve absolute path to a video file."""
        trial = self.get_trial(trial_name)
        filename, _, _ = trial.camera_videos[camera]
        return self.root / "videos" / filename

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

    Columns follow the pattern: trial, cam0, cam0_start, cam0_end, cam1, ...
    Camera name columns are those without _start or _end suffix, excluding 'trial'.
    """
    cameras = []
    for col in columns:
        if col == "trial":
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
                # Update the trial info with the resolved end
                trial.camera_videos[cam] = (filename, start, end)
            else:
                continue
        lengths.append(end - start + 1)

    trial.length = min(lengths) if lengths else 0

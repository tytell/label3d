"""PointsStore: DataFrame wrapper for 2D + 3D labeled points.

The DataFrame uses a 5-level row MultiIndex (experiment, trial, frame, instance, bodypart)
and a 2-level column MultiIndex (view, coord).  Real cameras have coords
{x, y, xr, yr, state}; the "3d" pseudo-camera has coords {x, y, z}.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QTimer, Signal


CAM_COORDS = ["x", "y", "xr", "yr", "state"]
THREE_D_COORDS = ["x", "y", "z"]
INDEX_NAMES = ["experiment", "trial", "frame", "instance", "bodypart"]


class PointData(NamedTuple):
    """Data for a single point in one camera."""

    x: float
    y: float
    xr: float
    yr: float
    state: str


class PointsStore(QObject):
    """Wraps a pandas DataFrame for labeled point CRUD with Qt signals.

    The DataFrame is the single source of truth.  For fast frame-level access
    during labeling, a frame cache (dict-of-dicts) is maintained and rebuilt
    on frame change.
    """

    data_changed = Signal()  # emitted on any mutation

    # Auto-save interval in milliseconds
    AUTO_SAVE_MS = 30_000

    def __init__(self, df: pd.DataFrame, parent: QObject | None = None):
        super().__init__(parent)
        self._df = df
        self._changed = False
        self._save_path: Path | None = None
        self._experiment_name: str = ""

        # Frame cache: {(instance, bodypart): {camera: PointData}}
        self._cache: dict[tuple[int, str], dict[str, PointData]] = {}
        self._cache_key: tuple[str, int] | None = None

        # Auto-save timer
        self._save_timer = QTimer(self)
        self._save_timer.setInterval(self.AUTO_SAVE_MS)
        self._save_timer.timeout.connect(self._auto_save)
        self._save_timer.start()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def changed(self) -> bool:
        return self._changed

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def cameras(self) -> list[str]:
        """Return real camera names (excluding '3d')."""
        top_level = self._df.columns.get_level_values(0).unique().tolist()
        return [c for c in top_level if c != "3d"]

    @property
    def bodyparts(self) -> list[str]:
        """Return unique bodypart names in index order."""
        return self._df.index.get_level_values("bodypart").unique().tolist()

    # -- Construction ----------------------------------------------------------

    @classmethod
    def create_empty(
        cls,
        cameras: list[str],
        bodyparts: list[str],
        experiment_name: str = "default",
    ) -> PointsStore:
        """Create an empty PointsStore with the right column structure."""
        store = cls(cls.empty_df(cameras))
        store._experiment_name = experiment_name
        return store

    @staticmethod
    def empty_df(cameras: list[str]) -> pd.DataFrame:
        """Empty points DataFrame with the canonical columns (no QObject created)."""
        col_tuples = []
        for cam in cameras:
            for coord in CAM_COORDS:
                col_tuples.append((cam, coord))
        for coord in THREE_D_COORDS:
            col_tuples.append(("3d", coord))

        columns = pd.MultiIndex.from_tuples(col_tuples, names=["view", "coord"])
        index = pd.MultiIndex.from_tuples(
            [], names=INDEX_NAMES
        )
        return pd.DataFrame(index=index, columns=columns)

    @classmethod
    def from_csv(cls, path: Path) -> PointsStore:
        """Load from points.csv."""
        df = cls.read_csv_df(path)
        store = cls(df)
        store._save_path = path
        # Infer experiment name from first row
        exp_levels = df.index.get_level_values("experiment").unique()
        store._experiment_name = str(exp_levels[0]) if len(exp_levels) > 0 else "default"
        return store

    @staticmethod
    def read_csv_df(path: Path) -> pd.DataFrame:
        """Read points.csv into a DataFrame (no QObject created; thread-safe)."""
        df = pd.read_csv(path, header=[0, 1], index_col=[0, 1, 2, 3, 4])
        df.index.names = INDEX_NAMES
        # Ensure frame and instance are int
        df.index = df.index.set_levels(
            [df.index.levels[0],          # experiment: keep as-is
             df.index.levels[1],          # trial: keep as-is
             df.index.levels[2].astype(int),  # frame
             df.index.levels[3].astype(int),  # instance
             df.index.levels[4]],         # bodypart: keep as-is
        )
        # Ensure numeric columns are float
        for cam in df.columns.get_level_values(0).unique():
            for coord in ["x", "y", "xr", "yr"]:
                if (cam, coord) in df.columns:
                    df[(cam, coord)] = pd.to_numeric(df[(cam, coord)], errors="coerce")
        return df

    @classmethod
    def load_all_experiments(cls, project) -> list[PointsStore]:
        """Load a PointsStore for each experiment that has a saved points.csv.

        Skips experiments whose points.csv does not yet exist.
        """
        stores = []
        for exp_name, exp_info in project.experiments.items():
            csv_path = exp_info.root / "points.csv"
            if csv_path.exists():
                stores.append(cls.from_csv(csv_path))
        return stores

    def to_csv(self, path: Path | None = None) -> None:
        """Save to points.csv."""
        path = path or self._save_path
        if path is None:
            raise ValueError("No save path specified")
        self._df.to_csv(path)
        self._save_path = path
        self._changed = False

    # -- Frame cache -----------------------------------------------------------

    def _ensure_cache(self, trial: str, frame: int) -> None:
        """Rebuild frame cache if needed."""
        key = (trial, frame)
        if self._cache_key == key:
            return
        self._cache.clear()
        self._cache_key = key
        try:
            frame_df = self._df.loc[(self._experiment_name, trial, frame)]
        except KeyError:
            return
        cameras = self.cameras
        for idx_tuple, row in frame_df.iterrows():
            # idx_tuple is (instance, bodypart) when sliced to experiment+trial+frame
            if isinstance(idx_tuple, tuple):
                instance, bodypart = idx_tuple
            else:
                continue
            cam_data = {}
            for cam in cameras:
                try:
                    x = row[(cam, "x")]
                    y = row[(cam, "y")]
                    xr = row[(cam, "xr")]
                    yr = row[(cam, "yr")]
                    state = row[(cam, "state")]
                except KeyError:
                    continue
                cam_data[cam] = PointData(
                    x=float(x) if pd.notna(x) else np.nan,
                    y=float(y) if pd.notna(y) else np.nan,
                    xr=float(xr) if pd.notna(xr) else np.nan,
                    yr=float(yr) if pd.notna(yr) else np.nan,
                    state=str(state) if pd.notna(state) else "not_set",
                )
            self._cache[(int(instance), str(bodypart))] = cam_data

    def _invalidate_cache(self, trial: str, frame: int) -> None:
        """Invalidate cache if it covers the modified frame."""
        if self._cache_key == (trial, frame):
            self._cache_key = None

    # -- Point-level API -------------------------------------------------------

    def get_point(
        self, trial: str, frame: int, instance: int, bodypart: str, camera: str
    ) -> PointData:
        """Get point data for one (instance, bodypart) in one camera.

        Returns PointData with NaN values if the point doesn't exist.
        """
        self._ensure_cache(trial, frame)
        cam_data = self._cache.get((instance, bodypart), {})
        return cam_data.get(
            camera,
            PointData(np.nan, np.nan, np.nan, np.nan, "not_set"),
        )

    def set_point(
        self,
        trial: str,
        frame: int,
        instance: int,
        bodypart: str,
        camera: str,
        x: float,
        y: float,
        state: str = "set",
    ) -> None:
        """Place or move a point.  Creates the row if needed."""
        idx = (self._experiment_name, trial, frame, instance, bodypart)
        if idx not in self._df.index:
            self._ensure_row(idx)
        self._df.at[idx, (camera, "x")] = x
        self._df.at[idx, (camera, "y")] = y
        self._df.at[idx, (camera, "state")] = state
        self._mark_changed()
        self._invalidate_cache(trial, frame)

    def set_not_visible(
        self, trial: str, frame: int, instance: int, bodypart: str, camera: str
    ) -> None:
        """Mark a point as not visible (keeps coords, excluded from epipolar/triangulation)."""
        idx = (self._experiment_name, trial, frame, instance, bodypart)
        if idx not in self._df.index:
            self._ensure_row(idx)
        self._df.at[idx, (camera, "state")] = "not_visible"
        self._mark_changed()
        self._invalidate_cache(trial, frame)

    def delete_point(
        self, trial: str, frame: int, instance: int, bodypart: str, camera: str
    ) -> None:
        """Remove a point (set to not_set with NaN coords)."""
        idx = (self._experiment_name, trial, frame, instance, bodypart)
        if idx not in self._df.index:
            return
        self._df.at[idx, (camera, "x")] = np.nan
        self._df.at[idx, (camera, "y")] = np.nan
        self._df.at[idx, (camera, "xr")] = np.nan
        self._df.at[idx, (camera, "yr")] = np.nan
        self._df.at[idx, (camera, "state")] = "not_set"
        self._mark_changed()
        self._invalidate_cache(trial, frame)

    # -- Frame-level API -------------------------------------------------------

    def get_frame_data(self, trial: str, frame: int) -> pd.DataFrame:
        """Get all data for one frame.  Returns empty DataFrame if none."""
        try:
            return self._df.loc[(self._experiment_name, trial, frame)]
        except KeyError:
            return pd.DataFrame(columns=self._df.columns)

    def get_instances(self, trial: str, frame: int) -> list[int]:
        """Return instance indices present in this frame."""
        try:
            frame_df = self._df.loc[(self._experiment_name, trial, frame)]
            return sorted(frame_df.index.get_level_values("instance").unique().tolist())
        except KeyError:
            return []

    def get_instances_for_camera(
        self, trial: str, frame: int, camera: str
    ) -> list[int]:
        """Return instance indices that have any labeled point in the given camera.

        A point counts as labeled if its state is 'set', 'predicted', or 'not_visible'.
        """
        try:
            frame_df = self._df.loc[(self._experiment_name, trial, frame)]
        except KeyError:
            return []
        if (camera, "state") not in frame_df.columns:
            return []
        has_data = frame_df[(camera, "state")].isin(["set", "predicted", "not_visible"])
        return sorted(
            frame_df[has_data].index.get_level_values("instance").unique().tolist()
        )

    # -- Instance-level API ----------------------------------------------------

    def add_instance(self, trial: str, frame: int, bodyparts: list[str]) -> int:
        """Add a new empty instance.  Returns the new instance index."""
        existing = self.get_instances(trial, frame)
        new_idx = max(existing) + 1 if existing else 0
        for bp in bodyparts:
            self._ensure_row((self._experiment_name, trial, frame, new_idx, bp))
        self._mark_changed()
        self._invalidate_cache(trial, frame)
        return new_idx

    def delete_instance(self, trial: str, frame: int, instance: int) -> None:
        """Delete all rows for an instance in a frame."""
        try:
            to_drop = self._df.loc[(self._experiment_name, trial, frame, instance)].index
            full_idx = pd.MultiIndex.from_tuples(
                [(self._experiment_name, trial, frame, instance, bp) for bp in to_drop],
                names=INDEX_NAMES,
            )
            self._df = self._df.drop(full_idx)
            self._mark_changed()
            self._invalidate_cache(trial, frame)
        except KeyError:
            pass

    def copy_instance(
        self, trial: str, src_frame: int, src_instance: int, dst_frame: int
    ) -> int:
        """Copy an instance to another frame.  Returns new instance index."""
        existing = self.get_instances(trial, dst_frame)
        new_idx = max(existing) + 1 if existing else 0
        try:
            src_data = self._df.loc[(self._experiment_name, trial, src_frame, src_instance)]
        except KeyError:
            return new_idx
        for bodypart in src_data.index:
            row = src_data.loc[bodypart]
            new_key = (self._experiment_name, trial, dst_frame, new_idx, bodypart)
            self._ensure_row(new_key)
            for col in self._df.columns:
                self._df.at[new_key, col] = row[col]
        self._mark_changed()
        self._invalidate_cache(trial, dst_frame)
        return new_idx

    # -- 3D API ----------------------------------------------------------------

    def set_3d(
        self,
        trial: str,
        frame: int,
        instance: int,
        bodypart: str,
        x: float,
        y: float,
        z: float,
    ) -> None:
        """Set triangulated 3D coordinates."""
        idx = (self._experiment_name, trial, frame, instance, bodypart)
        if idx not in self._df.index:
            return
        self._df.at[idx, ("3d", "x")] = x
        self._df.at[idx, ("3d", "y")] = y
        self._df.at[idx, ("3d", "z")] = z
        self._mark_changed()

    def set_reprojected(
        self,
        trial: str,
        frame: int,
        instance: int,
        bodypart: str,
        camera: str,
        xr: float,
        yr: float,
    ) -> None:
        """Set reprojected coordinates for a camera."""
        idx = (self._experiment_name, trial, frame, instance, bodypart)
        if idx not in self._df.index:
            return
        self._df.at[idx, (camera, "xr")] = xr
        self._df.at[idx, (camera, "yr")] = yr
        self._mark_changed()

    # -- Labeled-frame queries -------------------------------------------------

    def get_labeled_frames_by_camera(self, trial: str) -> dict[str, dict[int, str]]:
        """Return which frames have labels, keyed by camera then frame index.

        Each frame maps to 'user' (state 'set' or 'not_visible') or 'predicted'.
        Frames with only 'not_set' states are omitted.
        """
        result: dict[str, dict[int, str]] = {cam: {} for cam in self.cameras}
        try:
            trial_df = self._df.loc[(self._experiment_name, trial)]
        except KeyError:
            return result

        for cam in self.cameras:
            if (cam, "state") not in trial_df.columns:
                continue
            states = trial_df[(cam, "state")]
            for frame_idx, group in states.groupby(level="frame"):
                vals = group.values
                if any(s in ("set", "not_visible") for s in vals):
                    result[cam][int(frame_idx)] = "user"
                elif any(s == "predicted" for s in vals):
                    result[cam][int(frame_idx)] = "predicted"

        return result

    def _all_labeled_frame_indices(self, trial: str) -> list[int]:
        by_cam = self.get_labeled_frames_by_camera(trial)
        frames: set[int] = set()
        for cam_marks in by_cam.values():
            frames.update(cam_marks.keys())
        return sorted(frames)

    def get_next_labeled_frame(self, trial: str, current_frame: int) -> int | None:
        """Return the smallest labeled frame index > current_frame, or None."""
        for f in self._all_labeled_frame_indices(trial):
            if f > current_frame:
                return f
        return None

    def get_prev_labeled_frame(self, trial: str, current_frame: int) -> int | None:
        """Return the largest labeled frame index < current_frame, or None."""
        for f in reversed(self._all_labeled_frame_indices(trial)):
            if f < current_frame:
                return f
        return None

    # -- Internal --------------------------------------------------------------

    def _ensure_row(self, idx: tuple) -> None:
        """Create a row with NaN/not_set if it doesn't exist."""
        if idx in self._df.index:
            return
        row = pd.Series(index=self._df.columns, dtype=object)
        for cam in self.cameras:
            row[(cam, "x")] = np.nan
            row[(cam, "y")] = np.nan
            row[(cam, "xr")] = np.nan
            row[(cam, "yr")] = np.nan
            row[(cam, "state")] = "not_set"
        row[("3d", "x")] = np.nan
        row[("3d", "y")] = np.nan
        row[("3d", "z")] = np.nan
        row.name = idx
        new_idx = pd.MultiIndex.from_tuples([idx], names=INDEX_NAMES)
        new_row = pd.DataFrame([row], index=new_idx, columns=self._df.columns)
        self._df = pd.concat([self._df, new_row])

    def _mark_changed(self) -> None:
        self._changed = True
        self.data_changed.emit()

    def _auto_save(self) -> None:
        if self._changed and self._save_path is not None:
            # Backup before overwriting
            bak = self._save_path.with_suffix(".csv.bak")
            if self._save_path.exists():
                self._save_path.rename(bak)
            self.to_csv(self._save_path)

    def save_if_changed(self) -> None:
        """Save immediately if there are unsaved changes."""
        if self._changed and self._save_path is not None:
            bak = self._save_path.with_suffix(".csv.bak")
            if self._save_path.exists():
                self._save_path.rename(bak)
            self.to_csv(self._save_path)

    # -- Merge -----------------------------------------------------------------

    def replace_df(self, df: pd.DataFrame) -> None:
        """Replace all data (e.g. with a merge result), clear the cache, mark changed."""
        self._df = df
        self._cache.clear()
        self._cache_key = None
        self._mark_changed()

    def merge_from_df(self, adapter_df: pd.DataFrame) -> int:
        """Merge external adapter data into this store without overwriting user work.

        Merge rules per (experiment, trial, frame, instance, bodypart) row:
          - Row absent                          → insert
          - Any camera has set or not_visible   → skip (user data wins)
          - All cameras predicted, incoming predicted → replace
          - All cameras not_set                 → insert/update

        Returns the number of rows added or updated.
        """
        if adapter_df.empty:
            return 0

        cameras = self.cameras
        added = 0

        # Group by (experiment, trial, frame, instance) to batch lookups
        group_levels = ["experiment", "trial", "frame", "instance"]
        for group_key, group_df in adapter_df.groupby(level=group_levels):
            experiment, trial, frame, instance = group_key
            for bodypart, bp_row in group_df.groupby(level="bodypart"):
                idx = (experiment, trial, frame, instance, bodypart)
                bp_row = bp_row.iloc[0]  # one row per (group_key, bodypart)

                if idx in self._df.index:
                    # Skip if any camera already has user data (set/not_visible).
                    # Otherwise allow the incoming row to update predicted/not_set.
                    existing = self._df.loc[idx]
                    skip = False
                    for cam in cameras:
                        try:
                            state = existing[(cam, "state")]
                        except KeyError:
                            continue
                        if pd.isna(state):
                            continue
                        if state in ("set", "not_visible"):
                            skip = True
                            break
                    if skip:
                        continue
                else:
                    self._ensure_row(idx)

                # Write incoming camera values
                for cam in cameras:
                    try:
                        x = bp_row[(cam, "x")]
                        y = bp_row[(cam, "y")]
                        state = bp_row[(cam, "state")]
                    except KeyError:
                        continue
                    self._df.at[idx, (cam, "x")] = x
                    self._df.at[idx, (cam, "y")] = y
                    self._df.at[idx, (cam, "state")] = state

                self._invalidate_cache(str(trial), int(frame))
                added += 1

        if added:
            self._mark_changed()

        return added

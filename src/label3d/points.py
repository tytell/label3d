"""PointsStore: DataFrame wrapper for 2D + 3D labeled points.

The DataFrame uses a 5-level row MultiIndex (experiment, trial, frame, instance, bodypart)
and a 2-level column MultiIndex (view, coord).  Real cameras have coords
{x, y, xr, yr, state, source, score, sync}; the "3d" pseudo-camera has coords {x, y, z}.

source is where a point's x/y came from (label3d, sleap_user, sleap_pred); score
is SLEAP's per-point score for predictions; sync is the id of the sync read
that last brought the point in from SLEAP, cleared when the user corrects or
accepts it.  A point is *new* when its sync equals the project's latest sync id.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QTimer, Signal


CAM_COORDS = ["x", "y", "xr", "yr", "state", "source", "score", "sync"]
THREE_D_COORDS = ["x", "y", "z"]
FLOAT_COORDS = ("x", "y", "xr", "yr", "score", "sync")
TEXT_COORDS = ("state", "source")

SOURCE_LABEL3D = "label3d"
SOURCE_SLEAP_USER = "sleap_user"
SOURCE_SLEAP_PRED = "sleap_pred"
LABELED_STATES = ("set", "predicted", "not_visible")
INDEX_NAMES = ["experiment", "trial", "frame", "instance", "bodypart"]
# Levels the index is kept sorted on.  Bodyparts are left in skeleton order,
# so the lexsort depth is 4: enough for every partial-key .loc lookup.
SORT_LEVELS = INDEX_NAMES[:4]


def sort_points_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by (experiment, trial, frame, instance), keeping bodypart order."""
    return df.sort_index(level=SORT_LEVELS, sort_remaining=False)


class PointData(NamedTuple):
    """Data for a single point in one camera."""

    x: float
    y: float
    xr: float
    yr: float
    state: str
    source: str = ""
    score: float = np.nan
    sync: float = np.nan  # sync id, NaN when not from a sync or already reviewed


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
        self._df = sort_points_index(df)
        self._changed = False
        self._save_path: Path | None = None
        self._experiment_name: str = ""
        # Latest sync id in the project (set by Project); points with this
        # sync id are "new".
        self.latest_sync: int | None = None
        self.latest_sync_label: str = ""  # e.g. "sync 3 (2026-10-05 14:22)", for tooltips

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
        # experiment/trial are names, not numbers: all-digit ones ("20260619")
        # are read as int64 and would not match the strings from trials.csv.
        df.index = df.index.set_levels(
            [df.index.levels[0].astype(str),  # experiment
             df.index.levels[1].astype(str),  # trial
             df.index.levels[2].astype(int),  # frame
             df.index.levels[3].astype(int),  # instance
             df.index.levels[4].astype(str)],  # bodypart
        )
        return normalize_columns(df)

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
                    source = row[(cam, "source")]
                    score = row[(cam, "score")]
                    sync = row[(cam, "sync")]
                except KeyError:
                    continue
                cam_data[cam] = PointData(
                    x=float(x) if pd.notna(x) else np.nan,
                    y=float(y) if pd.notna(y) else np.nan,
                    xr=float(xr) if pd.notna(xr) else np.nan,
                    yr=float(yr) if pd.notna(yr) else np.nan,
                    state=str(state) if pd.notna(state) else "not_set",
                    source=str(source) if pd.notna(source) else "",
                    score=float(score) if pd.notna(score) else np.nan,
                    sync=float(sync) if pd.notna(sync) else np.nan,
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
        """Place or move a point.  Creates the row if needed.

        Every edit made in label3d records source=label3d and clears the
        point's sync id, so a corrected point is no longer new.
        """
        idx = (self._experiment_name, trial, frame, instance, bodypart)
        if idx not in self._df.index:
            self._ensure_row(idx)
        self._df.at[idx, (camera, "x")] = x
        self._df.at[idx, (camera, "y")] = y
        self._df.at[idx, (camera, "state")] = state
        self._set_label3d(idx, camera)
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
        self._set_label3d(idx, camera)
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
        self._df.at[idx, (camera, "source")] = ""
        self._df.at[idx, (camera, "score")] = np.nan
        self._df.at[idx, (camera, "sync")] = np.nan
        self._mark_changed()
        self._invalidate_cache(trial, frame)

    def accept_points(
        self,
        trial: str,
        frame: int,
        instance: int | None = None,
        bodypart: str | None = None,
    ) -> int:
        """Mark points as reviewed by clearing their sync id.  Returns points accepted.

        With instance and bodypart: that bodypart in every camera.  Without:
        every point in the frame.  x/y, state and source are left unchanged.
        """
        try:
            frame_df = self._df.loc[(self._experiment_name, trial, frame)]
        except KeyError:
            return 0
        n = 0
        for inst, bp in frame_df.index:
            if instance is not None and (int(inst) != instance or str(bp) != bodypart):
                continue
            idx = (self._experiment_name, trial, frame, inst, bp)
            for cam in self.cameras:
                if pd.notna(self._df.at[idx, (cam, "sync")]):
                    self._df.at[idx, (cam, "sync")] = np.nan
                    n += 1
        if n:
            self._mark_changed()
            self._invalidate_cache(trial, frame)
        return n

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
        self._ensure_rows(
            [(self._experiment_name, trial, frame, new_idx, bp) for bp in bodyparts]
        )
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
            for cam in self.cameras:
                if self._df.at[new_key, (cam, "state")] in LABELED_STATES:
                    self._set_label3d(new_key, cam)
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

    def trial_rows(self, trial: str, frames: list[int] | None = None) -> pd.DataFrame:
        """Rows of one trial (optionally only some frames), full index kept."""
        try:
            rows = self._df.xs(
                (self._experiment_name, trial), level=INDEX_NAMES[:2], drop_level=False
            )
        except KeyError:
            return self._df.iloc[:0]
        if frames is not None:
            rows = rows[rows.index.get_level_values("frame").isin(frames)]
        return rows

    def instance_rows(self, trial: str, frame: int, instance: int) -> pd.DataFrame:
        """Rows of one instance in one frame, full index kept."""
        try:
            return self._df.xs(
                (self._experiment_name, trial, frame, instance),
                level=INDEX_NAMES[:4], drop_level=False,
            )
        except KeyError:
            return self._df.iloc[:0]

    def set_triangulation(
        self,
        index: pd.MultiIndex,
        p3d: np.ndarray,
        reproj: dict[str, np.ndarray],
    ) -> None:
        """Write 3D points (N, 3) and each camera's reprojections (N, 2) for index rows.

        NaN values are written as well, which clears earlier results.
        """
        rows = self._df.index.get_indexer(index)
        columns = self._df.columns
        for k, coord in enumerate(THREE_D_COORDS):
            self._df.iloc[rows, columns.get_loc(("3d", coord))] = p3d[:, k]
        for cam, xy in reproj.items():
            self._df.iloc[rows, columns.get_loc((cam, "xr"))] = xy[:, 0]
            self._df.iloc[rows, columns.get_loc((cam, "yr"))] = xy[:, 1]
        self._cache_key = None
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

    def get_new_frames(self, trial: str) -> dict[int, str]:
        """Frames of a trial with new points: frame → 'user' or 'predicted'.

        'user' when any new point in the frame is a SLEAP user label,
        'predicted' when all of them are predictions.
        """
        if self.latest_sync is None:
            return {}
        try:
            trial_df = self._df.loc[(self._experiment_name, trial)]
        except KeyError:
            return {}
        result: dict[int, str] = {}
        frames = trial_df.index.get_level_values("frame")
        for cam in self.cameras:
            if (cam, "sync") not in trial_df.columns:
                continue
            is_new = (trial_df[(cam, "sync")] == self.latest_sync).to_numpy()
            if not is_new.any():
                continue
            is_user = (trial_df[(cam, "source")] == SOURCE_SLEAP_USER).to_numpy()
            for f, user in zip(frames[is_new], is_user[is_new]):
                f = int(f)
                if user:
                    result[f] = "user"
                else:
                    result.setdefault(f, "predicted")
        return result

    def trials_with_new_points(self) -> list[str]:
        """Trials (in index order) of the store's experiment that have new points."""
        if self.latest_sync is None or self._df.empty:
            return []
        mask = np.zeros(len(self._df), dtype=bool)
        for cam in self.cameras:
            if (cam, "sync") in self._df.columns:
                mask |= (self._df[(cam, "sync")] == self.latest_sync).to_numpy()
        if not mask.any():
            return []
        trials = self._df.index.get_level_values("trial")[mask]
        return list(dict.fromkeys(str(t) for t in trials))

    def get_next_new_frame(self, trial: str, current_frame: int) -> int | None:
        """Smallest frame > current_frame with new points, or None."""
        for f in sorted(self.get_new_frames(trial)):
            if f > current_frame:
                return f
        return None

    def get_prev_new_frame(self, trial: str, current_frame: int) -> int | None:
        """Largest frame < current_frame with new points, or None."""
        for f in sorted(self.get_new_frames(trial), reverse=True):
            if f < current_frame:
                return f
        return None

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
        self._ensure_rows([idx])

    def _ensure_rows(self, idxs: list[tuple]) -> None:
        """Create rows with NaN/not_set for any of idxs that don't exist.

        New rows are appended, then the index is re-sorted if they landed out
        of order, so partial-key lookups stay fast (no lexsort warning).
        """
        idxs = [idx for idx in idxs if idx not in self._df.index]
        if not idxs:
            return
        row = pd.Series(index=self._df.columns, dtype=object)
        for cam in self.cameras:
            row[(cam, "x")] = np.nan
            row[(cam, "y")] = np.nan
            row[(cam, "xr")] = np.nan
            row[(cam, "yr")] = np.nan
            row[(cam, "state")] = "not_set"
            row[(cam, "source")] = ""
            row[(cam, "score")] = np.nan
            row[(cam, "sync")] = np.nan
        row[("3d", "x")] = np.nan
        row[("3d", "y")] = np.nan
        row[("3d", "z")] = np.nan
        new_idx = pd.MultiIndex.from_tuples(idxs, names=INDEX_NAMES)
        new_rows = pd.DataFrame([row] * len(idxs), index=new_idx, columns=self._df.columns)
        in_order = self._df.empty or (
            self._df.index[-1][:4] <= min(idx[:4] for idx in idxs)
        )
        self._df = pd.concat([self._df, new_rows])
        if not in_order:
            self._df = sort_points_index(self._df)

    def _set_label3d(self, idx: tuple, camera: str) -> None:
        """Record a label3d edit: source=label3d, no score, no sync id."""
        self._df.at[idx, (camera, "source")] = SOURCE_LABEL3D
        self._df.at[idx, (camera, "score")] = np.nan
        self._df.at[idx, (camera, "sync")] = np.nan

    def _mark_changed(self) -> None:
        self._changed = True
        self.data_changed.emit()

    def _auto_save(self) -> None:
        self.save_if_changed()

    def save_if_changed(self) -> None:
        """Save immediately if there are unsaved changes."""
        if self._changed and self._save_path is not None:
            # Backup before overwriting; replace() (unlike rename()) overwrites
            # an existing .bak on Windows too
            bak = self._save_path.with_suffix(".csv.bak")
            if self._save_path.exists():
                self._save_path.replace(bak)
            self.to_csv(self._save_path)

    # -- Merge -----------------------------------------------------------------

    def replace_df(self, df: pd.DataFrame) -> None:
        """Replace all data (e.g. with a merge result), clear the cache, mark changed."""
        self._df = sort_points_index(df)
        self._cache.clear()
        self._cache_key = None
        self._mark_changed()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give every camera the full CAM_COORDS set, in order, with consistent dtypes.

    Files written before the source/score/sync columns existed get them filled
    in: points with a labeled state get source=label3d, score and sync NaN.
    """
    cameras = [c for c in df.columns.get_level_values(0).unique() if c != "3d"]
    for cam in cameras:
        if (cam, "state") not in df.columns:
            df[(cam, "state")] = "not_set"
        if (cam, "source") not in df.columns:
            labeled = df[(cam, "state")].isin(LABELED_STATES)
            df[(cam, "source")] = np.where(labeled, SOURCE_LABEL3D, "")
        for coord in FLOAT_COORDS:
            if (cam, coord) not in df.columns:
                df[(cam, coord)] = np.nan
            else:
                df[(cam, coord)] = pd.to_numeric(df[(cam, coord)], errors="coerce").astype(float)
        df[(cam, "source")] = df[(cam, "source")].fillna("").astype(str)
    columns = [(cam, coord) for cam in cameras for coord in CAM_COORDS]
    columns += [("3d", coord) for coord in THREE_D_COORDS]
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df.reindex(
        columns=pd.MultiIndex.from_tuples(columns, names=["view", "coord"])
    )

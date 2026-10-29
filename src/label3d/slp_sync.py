"""SLP file mapping and merge rules for syncing the points files with SLEAP files.

The mapping assigns each (experiment, trial, camera) cell — one video — to exactly
one SLP file.  It is stored in project.toml under [[slp.files]].

compute_merge() combines labels read from SLP files with the current points.
It is pure (no Qt objects created) and safe to call from a background thread.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from label3d.points import (
    HIDDEN_COORDS,
    INDEX_NAMES,
    SOURCE_SLEAP_PRED,
    SOURCE_SLEAP_USER,
    TEXT_COORDS,
    USER_STATES,
)

Cell = tuple[str, str, str]  # (experiment, trial, camera)

WRITTEN_STATES = ("set", "predicted")
COORD_TOL = 0.01  # pixels


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

@dataclass
class SlpBlock:
    """One experiment × a list of trials × a list of cameras."""

    experiment: str
    trials: list[str]
    cameras: list[str]

    def cells(self) -> list[Cell]:
        return [(self.experiment, t, c) for t in self.trials for c in self.cameras]


@dataclass
class SlpFileSpec:
    """One SLP file and the blocks of cells it contains."""

    path: Path  # relative to project root
    blocks: list[SlpBlock] = field(default_factory=list)

    def cells(self) -> list[Cell]:
        return [cell for block in self.blocks for cell in block.cells()]


@dataclass
class SlpMapping:
    """Assignment of (experiment, trial, camera) cells to SLP files."""

    files: list[SlpFileSpec] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict) -> SlpMapping:
        """Read the mapping from a parsed project.toml dict."""
        files = []
        for f in config.get("slp", {}).get("files", []):
            blocks = [
                SlpBlock(
                    experiment=str(b["experiment"]),
                    trials=[str(t) for t in b.get("trials", [])],
                    cameras=[str(c) for c in b.get("cameras", [])],
                )
                for b in f.get("blocks", [])
            ]
            files.append(SlpFileSpec(path=Path(f["path"]), blocks=blocks))
        return cls(files=files)

    @classmethod
    def from_cell_map(
        cls, cell_map: dict[Cell, Path], cell_order: list[Cell] | None = None
    ) -> SlpMapping:
        """Build a mapping from a cell → path dict, compacting cells into blocks.

        cell_order sets the trial and camera order within blocks (defaults to
        the dict order).
        """
        order = cell_order if cell_order is not None else list(cell_map)
        by_path: dict[Path, list[Cell]] = {}
        for cell in order:
            if cell in cell_map:
                by_path.setdefault(Path(cell_map[cell]), []).append(cell)
        return cls(files=[
            SlpFileSpec(path=path, blocks=compact_blocks(cells))
            for path, cells in by_path.items()
        ])

    def to_config(self) -> dict:
        """Return the [slp] section for project.toml."""
        return {
            "files": [
                {
                    "path": spec.path.as_posix(),
                    "blocks": [
                        {
                            "experiment": b.experiment,
                            "trials": list(b.trials),
                            "cameras": list(b.cameras),
                        }
                        for b in spec.blocks
                    ],
                }
                for spec in self.files
            ]
        }

    def cell_map(self) -> dict[Cell, Path]:
        """Return cell → file path (relative to project root)."""
        return {cell: spec.path for spec in self.files for cell in spec.cells()}

    def duplicate_cells(self) -> list[Cell]:
        """Cells assigned to more than one file (or twice within one file)."""
        seen: set[Cell] = set()
        dupes: list[Cell] = []
        for spec in self.files:
            for cell in spec.cells():
                if cell in seen and cell not in dupes:
                    dupes.append(cell)
                seen.add(cell)
        return dupes

    def missing_cells(self, trials_df: pd.DataFrame, cameras: list[str]) -> list[Cell]:
        """Cells that have a video in trials.csv but no assigned file."""
        assigned = self.cell_map()
        return [cell for cell in video_cells(trials_df, cameras) if cell not in assigned]

    def is_complete(self, trials_df: pd.DataFrame, cameras: list[str]) -> bool:
        """True when every video has exactly one file."""
        return (
            bool(self.files)
            and not self.missing_cells(trials_df, cameras)
            and not self.duplicate_cells()
        )


SLP_DIR = Path("slp")
# SLEAP keeps the models trained from a .slp file in this folder beside it, so
# each .slp file in slp/ gets a folder of its own: slp/{stem}/{stem}.slp
MODELS_DIR = "models"
BACKUP_DIR = "backup"


def slp_file_path(stem: str) -> Path:
    """The path, relative to the project root, of a new .slp file in its own folder."""
    return SLP_DIR / stem / f"{stem}.slp"


def find_slp_files(
    directory: Path, skip: tuple[str, ...] = (MODELS_DIR,), exclude: Path | None = None
) -> list[Path]:
    """The .slp files under directory, searched recursively, sorted.

    Folders named in skip are not searched (SLEAP writes the training and
    validation labels of each model as .slp files in its models folder), nor is
    exclude.  The writer's temporary .tmp.slp files are left out.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(directory):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in skip and here / d != exclude]
        found += [
            here / f for f in filenames if f.endswith(".slp") and not f.endswith(".tmp.slp")
        ]
    return sorted(found)

# key -> (label, default file name pattern); placeholders {experiment}, {trial}, {camera}
PRESETS: dict[str, tuple[str, str]] = {
    "per_camera": ("One file per camera", "{camera}.slp"),
    "per_experiment_camera": ("One file per camera per experiment", "{experiment}_{camera}.slp"),
    "per_trial_camera": ("One file per trial × camera", "{experiment}_{trial}_{camera}.slp"),
    "single": ("One file for everything", "labels.slp"),
}


class SlpMappingEditor:
    """Cell-level editing of an SlpMapping (used by the Set up Sleap files dialog).

    Only cells with a video in trials.csv are kept.  Files left with no cells
    are dropped after every edit.
    """

    def __init__(self, mapping: SlpMapping, trials_df: pd.DataFrame, cameras: list[str]):
        self.cells_info = video_cells(trials_df, cameras)
        self._rank = {cell: i for i, cell in enumerate(self.cells_info)}
        self._files: dict[Path, list[Cell]] = {}
        for spec in mapping.files:
            cells = self._files.setdefault(spec.path, [])
            cells.extend(c for c in spec.cells() if c in self._rank and c not in cells)
        self._drop_empty()

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    def cells_of(self, path: Path) -> list[Cell]:
        return sorted(self._files.get(path, []), key=self._rank.__getitem__)

    def files_of(self, cell: Cell) -> list[Path]:
        return [path for path, cells in self._files.items() if cell in cells]

    def mapping(self) -> SlpMapping:
        return SlpMapping(files=[
            SlpFileSpec(path=path, blocks=compact_blocks(self.cells_of(path)))
            for path in self._files
        ])

    def apply_preset(self, pattern: str) -> None:
        """Replace all assignments using a file name pattern.

        Raises KeyError/ValueError/IndexError for an invalid pattern.
        """
        files: dict[Path, list[Cell]] = {}
        for cell in self.cells_info:
            experiment, trial, camera = cell
            name = pattern.format(experiment=experiment, trial=trial, camera=camera).strip()
            if not name.endswith(".slp"):
                name += ".slp"
            files.setdefault(SLP_DIR / Path(name).stem / name, []).append(cell)
        self._files = files

    def move_cells(self, cells: list[Cell], path: Path) -> None:
        """Assign cells to path (creating the file if needed), removing them elsewhere."""
        moving = [c for c in dict.fromkeys(cells) if c in self._rank]
        if not moving:
            return
        moving_set = set(moving)
        for p in self._files:
            self._files[p] = [c for c in self._files[p] if c not in moving_set]
        self._files.setdefault(path, []).extend(moving)
        self._drop_empty()

    def new_file(self, cells: list[Cell], stem: str = "new") -> Path | None:
        """Move cells into a new, uniquely named file in its own folder.  Returns its path."""
        if not any(c in self._rank for c in cells):
            return None
        path = self.unique_path(stem)
        self.move_cells(cells, path)
        return path

    def split_by_camera(self, path: Path) -> list[Path]:
        """Split a file into one file per camera.  Returns the new paths."""
        cells = self.cells_of(path)
        cameras = list(dict.fromkeys(c[2] for c in cells))
        if len(cameras) < 2:
            return []
        return [
            self.new_file([c for c in cells if c[2] == cam], f"{path.stem}_{cam}")
            for cam in cameras
        ]

    def split_by_trial(self, path: Path, trials: set[tuple[str, str]]) -> Path | None:
        """Move the given (experiment, trial) rows of a file into a new file.

        Returns None if that would move none or all of the file's cells.
        """
        cells = self.cells_of(path)
        moving = [c for c in cells if (c[0], c[1]) in trials]
        if not moving or len(moving) == len(cells):
            return None
        return self.new_file(moving, path.stem)

    def join(self, paths: list[Path]) -> Path | None:
        """Move all cells of paths[1:] into paths[0]."""
        paths = [p for p in paths if p in self._files]
        if len(paths) < 2:
            return None
        for p in paths[1:]:
            self.move_cells(self.cells_of(p), paths[0])
        return paths[0]

    def rename(self, old: Path, new: Path) -> Path:
        """Rename a file, keeping its position.  Raises ValueError for a bad name.

        File names are unique, whatever their folders: the dialog lists files by name.
        """
        if not new.name or new.stem == "":
            raise ValueError("File name cannot be empty")
        if new.suffix != ".slp":
            new = new.with_name(new.name + ".slp")
        if new != old and any(
            p != old and p.name.lower() == new.name.lower() for p in self._files
        ):
            raise ValueError(f"{new.name} already exists")
        self._files = {(new if p == old else p): cells for p, cells in self._files.items()}
        return new

    def unique_path(self, stem: str) -> Path:
        """slp/{stem}/{stem}.slp, numbered so that neither its folder nor its name is taken."""
        folders = {p.parent.as_posix().lower() for p in self._files}
        names = {p.name.lower() for p in self._files}
        path = slp_file_path(stem)
        n = 2
        while path.parent.as_posix().lower() in folders or path.name.lower() in names:
            path = slp_file_path(f"{stem}_{n}")
            n += 1
        return path

    def errors(self) -> list[str]:
        """Problems that must be fixed before the mapping can be applied."""
        counts: dict[Cell, int] = {}
        for cells in self._files.values():
            for cell in cells:
                counts[cell] = counts.get(cell, 0) + 1
        errors = []
        unassigned = sum(1 for cell in self.cells_info if cell not in counts)
        if unassigned:
            errors.append(f"{unassigned} video(s) unassigned")
        duplicated = sum(1 for n in counts.values() if n > 1)
        if duplicated:
            errors.append(f"{duplicated} video(s) assigned to more than one file")
        seen: set[str] = set()
        for path in self._files:
            if path.suffix != ".slp":
                errors.append(f"File name must end in .slp: {path.name}")
            key = path.as_posix().lower()
            if key in seen:
                errors.append(f"File names differ only by case: {path.name}")
            seen.add(key)
        return errors

    def _drop_empty(self) -> None:
        self._files = {p: cells for p, cells in self._files.items() if cells}


def video_cells(trials_df: pd.DataFrame, cameras: list[str]) -> dict[Cell, tuple[str, int]]:
    """Return cell → (video filename, start frame) for every trial video.

    Calibration takes (is_calibration rows) are skipped — they carry no labels.
    """
    cells: dict[Cell, tuple[str, int]] = {}
    has_experiment = "experiment" in trials_df.columns
    if "is_calibration" in trials_df.columns:
        is_calib = (
            trials_df["is_calibration"].astype(str).str.lower().isin(("true", "1", "yes"))
        )
        trials_df = trials_df[~is_calib]
    for _, row in trials_df.iterrows():
        experiment = str(row["experiment"]) if has_experiment else "default"
        trial = str(row["trial"])
        for cam in cameras:
            if cam not in row.index or pd.isna(row[cam]) or not str(row[cam]).strip():
                continue
            start_val = row.get(f"{cam}_start", 0)
            start = 0 if pd.isna(start_val) else int(start_val)
            cells[(experiment, trial, cam)] = (str(row[cam]), start)
    return cells


def compact_blocks(cells: list[Cell]) -> list[SlpBlock]:
    """Group cells into as few (trials × cameras) blocks as possible.

    Trials with the same camera set (within one experiment) share a block, so a
    rectangular assignment becomes one block.  Camera order follows first
    appearance in cells.
    """
    cam_rank: dict[str, int] = {}
    trial_cams: dict[tuple[str, str], list[str]] = {}
    for experiment, trial, cam in cells:
        cam_rank.setdefault(cam, len(cam_rank))
        cams = trial_cams.setdefault((experiment, trial), [])
        if cam not in cams:
            cams.append(cam)

    blocks: dict[tuple[str, tuple[str, ...]], SlpBlock] = {}
    for (experiment, trial), cams in trial_cams.items():
        cam_key = tuple(sorted(cams, key=cam_rank.__getitem__))
        key = (experiment, cam_key)
        if key not in blocks:
            blocks[key] = SlpBlock(experiment=experiment, trials=[], cameras=list(cam_key))
        blocks[key].trials.append(trial)
    return list(blocks.values())


# ---------------------------------------------------------------------------
# Per-camera frame helpers (shared with SleapAdapter.save)
# ---------------------------------------------------------------------------

def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with str experiment/trial/bodypart and int frame/instance levels."""
    idx = df.index
    new_index = pd.MultiIndex.from_arrays(
        [
            idx.get_level_values("experiment").astype(str),
            idx.get_level_values("trial").astype(str),
            idx.get_level_values("frame").astype(int),
            idx.get_level_values("instance").astype(int),
            idx.get_level_values("bodypart").astype(str),
        ],
        names=INDEX_NAMES,
    )
    return df.set_axis(new_index, axis=0)


def camera_cells(
    frame_df: pd.DataFrame | None, cam: str
) -> dict[tuple[int, str], tuple[float, float, str]]:
    """Return (instance, bodypart) → (x, y, state) for one camera in one frame."""
    if frame_df is None or frame_df.empty or (cam, "state") not in frame_df.columns:
        return {}
    instances = frame_df.index.get_level_values("instance")
    bodyparts = frame_df.index.get_level_values("bodypart")
    xs = pd.to_numeric(frame_df[(cam, "x")], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(frame_df[(cam, "y")], errors="coerce").to_numpy(dtype=float)
    states = frame_df[(cam, "state")].to_numpy()
    return {
        (int(i), str(b)): (float(x), float(y), s if isinstance(s, str) else "not_set")
        for i, b, x, y, s in zip(instances, bodyparts, xs, ys, states)
    }


def camera_scores(frame_df: pd.DataFrame | None, cam: str) -> dict[tuple[int, str], float]:
    """Return (instance, bodypart) → SLEAP score for one camera in one frame (NaN if none)."""
    if frame_df is None or frame_df.empty or (cam, "score") not in frame_df.columns:
        return {}
    instances = frame_df.index.get_level_values("instance")
    bodyparts = frame_df.index.get_level_values("bodypart")
    scores = pd.to_numeric(frame_df[(cam, "score")], errors="coerce").to_numpy(dtype=float)
    return {(int(i), str(b)): float(sc) for i, b, sc in zip(instances, bodyparts, scores)}


def camera_sources(frame_df: pd.DataFrame | None, cam: str) -> dict[tuple[int, str], str]:
    """Return (instance, bodypart) → source for one camera in one frame."""
    if frame_df is None or frame_df.empty or (cam, "source") not in frame_df.columns:
        return {}
    instances = frame_df.index.get_level_values("instance")
    bodyparts = frame_df.index.get_level_values("bodypart")
    sources = frame_df[(cam, "source")].to_numpy()
    return {
        (int(i), str(b)): (src if isinstance(src, str) else "")
        for i, b, src in zip(instances, bodyparts, sources)
    }


def camera_hidden(
    frame_df: pd.DataFrame | None, cam: str
) -> dict[tuple[int, str], tuple[float, float, float]]:
    """Return (instance, bodypart) → (px, py, pscore), the prediction hidden under a
    user instance, for one camera in one frame.  Points with none are left out."""
    if frame_df is None or frame_df.empty or (cam, "px") not in frame_df.columns:
        return {}
    instances = frame_df.index.get_level_values("instance")
    bodyparts = frame_df.index.get_level_values("bodypart")
    cols = [
        pd.to_numeric(frame_df[(cam, c)], errors="coerce").to_numpy(dtype=float)
        for c in HIDDEN_COORDS
    ]
    return {
        (int(i), str(b)): (float(px), float(py), float(ps))
        for i, b, px, py, ps in zip(instances, bodyparts, *cols)
        if math.isfinite(px) and math.isfinite(py)
    }


CameraData = tuple[
    dict[tuple[int, str], tuple[float, float, str]],  # camera_cells
    dict[tuple[int, str], float],                      # camera_scores
    dict[tuple[int, str], str],                        # camera_sources
    dict[tuple[int, str], tuple[float, float, float]], # camera_hidden
]
NO_CAMERA_DATA: CameraData = ({}, {}, {}, {})


def frame_keys(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    """(experiment, trial, frame) of each row of df, in row order."""
    idx = df.index
    return list(zip(
        idx.get_level_values("experiment").tolist(),
        idx.get_level_values("trial").tolist(),
        idx.get_level_values("frame").tolist(),
    ))


def frame_camera_data(
    df: pd.DataFrame, cameras: list[str]
) -> dict[tuple[str, str, int], dict[str, CameraData]]:
    """camera_cells, camera_scores, camera_sources and camera_hidden for every frame
    of df at once.

    Returns (experiment, trial, frame) → camera → (cells, scores, sources, hidden),
    the dicts those functions return for that frame's rows.  A camera with no
    state column is left out.  Each column is read once, which is far faster
    than slicing df frame by frame.  df must have a normalized index.
    """
    out: dict[tuple[str, str, int], dict[str, CameraData]] = {}
    if df.empty:
        return out
    keys = frame_keys(df)
    instances = df.index.get_level_values("instance").tolist()
    bodyparts = df.index.get_level_values("bodypart").tolist()
    n = len(df)
    for cam in cameras:
        if (cam, "state") not in df.columns:
            continue
        xs = pd.to_numeric(df[(cam, "x")], errors="coerce").to_numpy(dtype=float).tolist()
        ys = pd.to_numeric(df[(cam, "y")], errors="coerce").to_numpy(dtype=float).tolist()
        states = df[(cam, "state")].tolist()
        scores = (
            pd.to_numeric(df[(cam, "score")], errors="coerce").to_numpy(dtype=float).tolist()
            if (cam, "score") in df.columns else [math.nan] * n
        )
        sources = df[(cam, "source")].tolist() if (cam, "source") in df.columns else [""] * n
        hidden = [
            pd.to_numeric(df[(cam, c)], errors="coerce").to_numpy(dtype=float).tolist()
            if (cam, c) in df.columns else [math.nan] * n
            for c in HIDDEN_COORDS
        ]
        for key, i, b, x, y, s, sc, src, px, py, ps in zip(
            keys, instances, bodyparts, xs, ys, states, scores, sources, *hidden
        ):
            by_cam = out.get(key)
            if by_cam is None:
                by_cam = out[key] = {}
            data = by_cam.get(cam)
            if data is None:
                data = by_cam[cam] = ({}, {}, {}, {})
            data[0][(i, b)] = (x, y, s if isinstance(s, str) else "not_set")
            data[1][(i, b)] = sc
            data[2][(i, b)] = src if isinstance(src, str) else ""
            if math.isfinite(px) and math.isfinite(py):
                data[3][(i, b)] = (px, py, ps)
    return out


def written_instances(
    cells: dict[tuple[int, str], tuple[float, float, str]]
) -> list[tuple[int, bool]]:
    """Instances written to an SLP file for one camera, in order, as (instance, is_user).

    An instance is written when at least one bodypart is set or predicted with
    finite coordinates.  is_user is True when any bodypart is set or not_visible.
    SleapAdapter.save() writes exactly these instances, in this order, so the
    n-th instance in an SLP frame corresponds to the n-th entry here.
    """
    written: set[int] = set()
    user: set[int] = set()
    for (inst, _), (x, y, state) in cells.items():
        if state in WRITTEN_STATES and math.isfinite(x) and math.isfinite(y):
            written.add(inst)
        if state in USER_STATES:
            user.add(inst)
    return [(inst, inst in user) for inst in sorted(written)]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    """Outcome of merging SLP labels into the current points.

    Counts are kept per (experiment, trial) in trial_counts, with keys
    new_user (SLEAP user labels taken), new_pred (predictions where the store
    had none), updated_pred (predictions that replaced a moved prediction)
    and conflicts.
    """

    df: pd.DataFrame            # store df with non-conflicting SLP data applied
    df_overwrite: pd.DataFrame  # same, plus conflicting cells taken from SLP
    conflicts: list[tuple] = field(default_factory=list)  # (exp, trial, frame, inst, bp, cam)
    changed_frames: set[tuple[str, str, int]] = field(default_factory=set)
    # Frames where a cell was given this merge's sync id (excluding conflicts)
    new_frames: set[tuple[str, str, int]] = field(default_factory=set)
    trial_counts: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)
    n_score_updates: int = 0    # predictions that kept their x/y but got a new score
    sync_id: int | None = None
    # SLP points left out because their camera's skeleton lacks the node:
    # (camera, node) -> count
    dropped: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def conflict_frames(self) -> set[tuple[str, str, int]]:
        return {c[:3] for c in self.conflicts}

    def _total(self, key: str) -> int:
        return sum(c.get(key, 0) for c in self.trial_counts.values())

    @property
    def n_new(self) -> int:
        """SLEAP user labels taken from the SLP files."""
        return self._total("new_user")

    @property
    def n_new_pred(self) -> int:
        return self._total("new_pred")

    @property
    def n_updated_pred(self) -> int:
        return self._total("updated_pred")

    @property
    def n_predictions(self) -> int:
        """Predicted cells added or replaced."""
        return self.n_new_pred + self.n_updated_pred

    @property
    def has_changes(self) -> bool:
        """True when applying df (without overwrite) changes the store."""
        return bool(self.changed_frames) or self.n_score_updates > 0

    def count(self, frame_key: tuple[str, str, int], kind: str) -> None:
        counts = self.trial_counts.setdefault(frame_key[:2], {})
        counts[kind] = counts.get(kind, 0) + 1


# A re-run prediction that moved less than this keeps its x/y, source and sync
# id, so re-running inference does not mark every prediction as new.
PREDICTION_MOVE_PX = 0.5

SLEAP_SOURCES = {"set": SOURCE_SLEAP_USER, "predicted": SOURCE_SLEAP_PRED}


def compute_merge(
    store_df: pd.DataFrame,
    slp_df: pd.DataFrame,
    sync_id: int | None = None,
    progress: Callable[[int, int], object] | None = None,
) -> MergeResult:
    """Merge labels read from SLP files into the store DataFrame.

    Merge is per camera cell.  For each (frame, camera) present in the SLP data:

    - SLP instances are matched to store instances by position: the n-th SLP
      instance is the n-th instance that save() would write for that camera
      (see written_instances), then store instances with no data in that
      camera, then new instance indices.
    - If the store or the SLP file has user data in that frame and camera,
      SLP predictions are ignored and SLP user points are merged:

        store set / predicted, same coords (±0.01 px) → no change
        store missing / not_set / predicted  → take SLP
        store set, different coords          → conflict
        store not_visible                    → conflict
        SLP point missing (NaN)              → keep store, except that a
                                               store prediction is cleared

      The prediction hidden under each SLP user instance (its from_predicted)
      replaces the one hidden under the store instance, without conflicts.

    - Otherwise SLP predictions replace the store's predictions for that camera.
      A stored prediction within PREDICTION_MOVE_PX keeps its x/y, source and
      sync id; only a SLEAP prediction's score is updated.

    Every cell taken from SLP gets source sleap_user / sleap_pred, the SLP
    score, and sync = sync_id (NaN when sync_id is None).  Frames or cameras
    absent from the SLP data are left unchanged.

    progress, if given, is called as progress(done, total) with the number of
    SLP frames merged so far.
    """
    store_df = normalize_index(store_df)
    if slp_df.empty:
        return MergeResult(df=store_df, df_overwrite=store_df, sync_id=sync_id)
    slp_df = normalize_index(slp_df)
    sync_value = float(sync_id) if sync_id is not None else np.nan

    cameras = [
        c for c in store_df.columns.get_level_values(0).unique()
        if c != "3d" and (c, "state") in slp_df.columns
    ]
    slp_frames = frame_camera_data(slp_df, cameras)
    store_frames = frame_camera_data(store_df, cameras)
    store_instances: dict[tuple[str, str, int], set[int]] = {}
    if not store_df.empty:
        for key, inst in zip(
            frame_keys(store_df), store_df.index.get_level_values("instance").tolist()
        ):
            store_instances.setdefault(key, set()).add(inst)

    result = MergeResult(df=store_df, df_overwrite=store_df, sync_id=sync_id)
    updates: dict[tuple[tuple, str], dict] = {}
    conflict_updates: dict[tuple[tuple, str], dict] = {}

    def taken(x: float, y: float, state: str, score: float) -> dict:
        return {
            "x": x, "y": y, "state": state, "source": SLEAP_SOURCES[state],
            "score": score if state == "predicted" else np.nan, "sync": sync_value,
        }

    n_frames = len(slp_frames)
    step = max(1, n_frames // 200)
    for n_done, (key, slp_by_cam) in enumerate(slp_frames.items()):
        if progress is not None and n_done % step == 0:
            progress(n_done, n_frames)
        store_by_cam = store_frames.get(key, {})
        frame_instances = sorted(store_instances.get(key, ()))
        next_new = frame_instances[-1] + 1 if frame_instances else 0

        for cam in cameras:
            slp_cells, slp_scores, _, slp_hidden = slp_by_cam.get(cam, NO_CAMERA_DATA)
            slp_insts = written_instances(slp_cells)
            if not slp_insts:
                continue
            store_cells, store_scores, store_sources, store_hidden = store_by_cam.get(
                cam, NO_CAMERA_DATA
            )
            store_insts = [inst for inst, _ in written_instances(store_cells)]
            targets = store_insts + [i for i in frame_instances if i not in store_insts]

            def target(pos: int) -> int:
                if pos < len(targets):
                    return targets[pos]
                return next_new + pos - len(targets)

            slp_by_inst: dict[int, list[tuple[str, float, float, str]]] = {}
            for (inst, bp), (x, y, state) in slp_cells.items():
                slp_by_inst.setdefault(inst, []).append((bp, x, y, state))

            store_has_user = any(s in USER_STATES for _, _, s in store_cells.values())
            slp_has_user = any(is_user for _, is_user in slp_insts)

            if store_has_user or slp_has_user:
                for pos, (slp_inst, is_user) in enumerate(slp_insts):
                    if not is_user:
                        continue
                    inst = target(pos)
                    for bp, sx, sy, sstate in slp_by_inst[slp_inst]:
                        idx = (*key, inst, bp)
                        new_hidden = slp_hidden.get((slp_inst, bp))
                        if not _same_hidden(store_hidden.get((inst, bp)), new_hidden):
                            px, py, ps = new_hidden or (np.nan, np.nan, np.nan)
                            updates.setdefault((idx, cam), {}).update(
                                {"px": px, "py": py, "pscore": ps}
                            )
                            result.changed_frames.add(key)
                        cx, cy, cstate = store_cells.get((inst, bp), (np.nan, np.nan, "not_set"))
                        if sstate != "set" or not (math.isfinite(sx) and math.isfinite(sy)):
                            if cstate == "predicted":
                                # the SLEAP user instance has no point here
                                updates.setdefault((idx, cam), {}).update({
                                    "x": np.nan, "y": np.nan, "state": "not_set",
                                    "source": "", "score": np.nan, "sync": np.nan,
                                })
                                result.changed_frames.add(key)
                            continue
                        same = abs(cx - sx) <= COORD_TOL and abs(cy - sy) <= COORD_TOL
                        if cstate in ("set", "predicted") and same:
                            # predicted points in a user instance come back as set
                            continue
                        if cstate in ("not_set", "predicted"):
                            updates.setdefault((idx, cam), {}).update(
                                taken(sx, sy, "set", np.nan)
                            )
                            result.count(key, "new_user")
                            result.changed_frames.add(key)
                            result.new_frames.add(key)
                        else:
                            conflict_updates[(idx, cam)] = taken(sx, sy, "set", np.nan)
                            result.conflicts.append((*idx, cam))
                            result.count(key, "conflicts")
            else:
                new_preds: dict[tuple[int, str], tuple[float, float, float]] = {}
                for pos, (slp_inst, _) in enumerate(slp_insts):
                    inst = target(pos)
                    for bp, sx, sy, sstate in slp_by_inst[slp_inst]:
                        if sstate == "predicted" and math.isfinite(sx) and math.isfinite(sy):
                            score = slp_scores.get((slp_inst, bp), np.nan)
                            new_preds[(inst, bp)] = (sx, sy, score)

                for (inst, bp), (_, _, cstate) in store_cells.items():
                    if cstate == "predicted" and (inst, bp) not in new_preds:
                        updates[((*key, inst, bp), cam)] = {
                            "x": np.nan, "y": np.nan, "state": "not_set",
                            "source": "", "score": np.nan, "sync": np.nan,
                        }
                        result.changed_frames.add(key)

                for (inst, bp), (sx, sy, score) in new_preds.items():
                    idx = (*key, inst, bp)
                    cx, cy, cstate = store_cells.get((inst, bp), (np.nan, np.nan, "not_set"))
                    if cstate == "predicted":
                        if (
                            abs(cx - sx) <= PREDICTION_MOVE_PX
                            and abs(cy - sy) <= PREDICTION_MOVE_PX
                        ):
                            old_score = store_scores.get((inst, bp), np.nan)
                            # not np.isclose(score, old_score), without its per-call cost
                            if (
                                store_sources.get((inst, bp)) == SOURCE_SLEAP_PRED
                                and math.isfinite(score)
                                and not (
                                    math.isfinite(old_score)
                                    and abs(score - old_score) <= 1e-8 + 1e-5 * abs(old_score)
                                )
                            ):
                                updates[(idx, cam)] = {"score": score}
                                result.n_score_updates += 1
                            continue
                        result.count(key, "updated_pred")
                    else:
                        result.count(key, "new_pred")
                    updates[(idx, cam)] = taken(sx, sy, "predicted", score)
                    result.changed_frames.add(key)
                    result.new_frames.add(key)

    if progress is not None:
        progress(n_frames, n_frames)
    result.df = _apply_updates(store_df, updates)
    result.df_overwrite = (
        _apply_updates(result.df, conflict_updates) if conflict_updates else result.df
    )
    return result


def _same_hidden(
    a: tuple[float, float, float] | None, b: tuple[float, float, float] | None
) -> bool:
    """True when two hidden predictions (px, py, pscore) are the same, None for none."""
    if a is None or b is None:
        return a is b
    if abs(a[0] - b[0]) > COORD_TOL or abs(a[1] - b[1]) > COORD_TOL:
        return False
    if math.isfinite(a[2]) != math.isfinite(b[2]):
        return False
    return not math.isfinite(a[2]) or abs(a[2] - b[2]) <= 1e-6


def _apply_updates(
    base_df: pd.DataFrame, updates: dict[tuple[tuple, str], dict]
) -> pd.DataFrame:
    """Return a copy of base_df with (index, camera) → {coord: value} updates applied."""
    if not updates:
        return base_df

    cameras = [c for c in base_df.columns.get_level_values(0).unique() if c != "3d"]
    new_rows = list(dict.fromkeys(idx for idx, _ in updates if idx not in base_df.index))
    if new_rows:
        added = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(new_rows, names=INDEX_NAMES),
            columns=base_df.columns,
            dtype=object,
        )
        for cam in cameras:
            added[(cam, "state")] = "not_set"
            if (cam, "source") in added.columns:
                added[(cam, "source")] = ""
        df = pd.concat([base_df, added]) if not base_df.empty else added
        df = df.sort_index()
    else:
        df = base_df.copy()

    # Numeric columns must be float to accept NaN and sub-pixel values
    for col in df.columns:
        if col[1] not in TEXT_COORDS and df[col].dtype != float:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    by_cam_coord: dict[tuple[str, str], list[tuple[tuple, object]]] = {}
    for (idx, cam), values in updates.items():
        for coord, value in values.items():
            by_cam_coord.setdefault((cam, coord), []).append((idx, value))
    for col, items in by_cam_coord.items():
        if col not in df.columns:
            continue
        index = pd.MultiIndex.from_tuples([it[0] for it in items], names=INDEX_NAMES)
        df.loc[index, col] = [it[1] for it in items]
    return df

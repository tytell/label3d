"""SLP file mapping and merge rules for syncing points.csv with SLEAP files.

The mapping assigns each (experiment, trial, camera) cell — one video — to exactly
one SLP file.  It is stored in project.toml under [[slp.files]].

compute_merge() combines labels read from SLP files with the current points.
It is pure (no Qt objects created) and safe to call from a background thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from label3d.points import INDEX_NAMES

Cell = tuple[str, str, str]  # (experiment, trial, camera)

USER_STATES = ("set", "not_visible")
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
            files.setdefault(SLP_DIR / name, []).append(cell)
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

    def new_file(self, cells: list[Cell], stem: str = "new", parent: Path = SLP_DIR) -> Path | None:
        """Move cells into a new, uniquely named file.  Returns its path."""
        if not any(c in self._rank for c in cells):
            return None
        path = self.unique_path(stem, parent)
        self.move_cells(cells, path)
        return path

    def split_by_camera(self, path: Path) -> list[Path]:
        """Split a file into one file per camera.  Returns the new paths."""
        cells = self.cells_of(path)
        cameras = list(dict.fromkeys(c[2] for c in cells))
        if len(cameras) < 2:
            return []
        return [
            self.new_file([c for c in cells if c[2] == cam], f"{path.stem}_{cam}", path.parent)
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
        return self.new_file(moving, path.stem, path.parent)

    def join(self, paths: list[Path]) -> Path | None:
        """Move all cells of paths[1:] into paths[0]."""
        paths = [p for p in paths if p in self._files]
        if len(paths) < 2:
            return None
        for p in paths[1:]:
            self.move_cells(self.cells_of(p), paths[0])
        return paths[0]

    def rename(self, old: Path, new: Path) -> Path:
        """Rename a file, keeping its position.  Raises ValueError for a bad name."""
        if not new.name or new.stem == "":
            raise ValueError("File name cannot be empty")
        if new.suffix != ".slp":
            new = new.with_name(new.name + ".slp")
        if new != old and new in self._files:
            raise ValueError(f"{new.name} already exists")
        self._files = {(new if p == old else p): cells for p, cells in self._files.items()}
        return new

    def unique_path(self, stem: str, parent: Path = SLP_DIR) -> Path:
        path = parent / f"{stem}.slp"
        n = 2
        taken = {p.as_posix().lower() for p in self._files}
        while path.as_posix().lower() in taken:
            path = parent / f"{stem}_{n}.slp"
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
    """Return cell → (video filename, start frame) for every video in trials.csv."""
    cells: dict[Cell, tuple[str, int]] = {}
    has_experiment = "experiment" in trials_df.columns
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
        if state in WRITTEN_STATES and np.isfinite(x) and np.isfinite(y):
            written.add(inst)
        if state in USER_STATES:
            user.add(inst)
    return [(inst, inst in user) for inst in sorted(written)]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    """Outcome of merging SLP labels into the current points."""

    df: pd.DataFrame            # store df with non-conflicting SLP data applied
    df_overwrite: pd.DataFrame  # same, plus conflicting cells taken from SLP
    n_new: int = 0              # user cells taken from SLP
    n_predictions: int = 0      # predicted cells added or replaced
    conflicts: list[tuple] = field(default_factory=list)  # (exp, trial, frame, inst, bp, cam)
    changed_frames: set[tuple[str, str, int]] = field(default_factory=set)

    @property
    def conflict_frames(self) -> set[tuple[str, str, int]]:
        return {c[:3] for c in self.conflicts}


def compute_merge(store_df: pd.DataFrame, slp_df: pd.DataFrame) -> MergeResult:
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
        SLP point missing (NaN)              → keep store

    - Otherwise SLP predictions replace the store's predictions for that camera.

    Frames or cameras absent from the SLP data are left unchanged.
    """
    store_df = normalize_index(store_df)
    if slp_df.empty:
        return MergeResult(df=store_df, df_overwrite=store_df)
    slp_df = normalize_index(slp_df)

    cameras = [
        c for c in store_df.columns.get_level_values(0).unique()
        if c != "3d" and (c, "state") in slp_df.columns
    ]
    frame_levels = ["experiment", "trial", "frame"]
    store_groups = (
        {k: g for k, g in store_df.groupby(level=frame_levels, sort=False)}
        if not store_df.empty else {}
    )

    result = MergeResult(df=store_df, df_overwrite=store_df)
    updates: dict[tuple[tuple, str], tuple[float, float, str]] = {}
    conflict_updates: dict[tuple[tuple, str], tuple[float, float, str]] = {}

    for key, slp_group in slp_df.groupby(level=frame_levels, sort=False):
        store_group = store_groups.get(key)
        frame_instances = (
            sorted(set(store_group.index.get_level_values("instance").tolist()))
            if store_group is not None else []
        )
        next_new = frame_instances[-1] + 1 if frame_instances else 0

        for cam in cameras:
            slp_cells = camera_cells(slp_group, cam)
            slp_insts = written_instances(slp_cells)
            if not slp_insts:
                continue
            store_cells = camera_cells(store_group, cam)
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
                        if sstate != "set" or not (np.isfinite(sx) and np.isfinite(sy)):
                            continue
                        idx = (*key, inst, bp)
                        cx, cy, cstate = store_cells.get((inst, bp), (np.nan, np.nan, "not_set"))
                        same = abs(cx - sx) <= COORD_TOL and abs(cy - sy) <= COORD_TOL
                        if cstate in ("set", "predicted") and same:
                            # predicted points in a user instance come back as set
                            continue
                        if cstate in ("not_set", "predicted"):
                            updates[(idx, cam)] = (sx, sy, "set")
                            result.n_new += 1
                            result.changed_frames.add(key)
                        else:
                            conflict_updates[(idx, cam)] = (sx, sy, "set")
                            result.conflicts.append((*idx, cam))
            else:
                new_preds: dict[tuple[int, str], tuple[float, float]] = {}
                for pos, (slp_inst, _) in enumerate(slp_insts):
                    inst = target(pos)
                    for bp, sx, sy, sstate in slp_by_inst[slp_inst]:
                        if sstate == "predicted" and np.isfinite(sx) and np.isfinite(sy):
                            new_preds[(inst, bp)] = (sx, sy)

                for (inst, bp), (_, _, cstate) in store_cells.items():
                    if cstate == "predicted" and (inst, bp) not in new_preds:
                        updates[((*key, inst, bp), cam)] = (np.nan, np.nan, "not_set")
                        result.changed_frames.add(key)

                for (inst, bp), (sx, sy) in new_preds.items():
                    cx, cy, cstate = store_cells.get((inst, bp), (np.nan, np.nan, "not_set"))
                    if (
                        cstate == "predicted"
                        and abs(cx - sx) <= COORD_TOL
                        and abs(cy - sy) <= COORD_TOL
                    ):
                        continue
                    updates[((*key, inst, bp), cam)] = (sx, sy, "predicted")
                    result.n_predictions += 1
                    result.changed_frames.add(key)

    result.df = _apply_updates(store_df, updates)
    result.df_overwrite = (
        _apply_updates(result.df, conflict_updates) if conflict_updates else result.df
    )
    return result


def _apply_updates(
    base_df: pd.DataFrame, updates: dict[tuple[tuple, str], tuple[float, float, str]]
) -> pd.DataFrame:
    """Return a copy of base_df with (index, camera) → (x, y, state) updates applied."""
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
        df = pd.concat([base_df, added]) if not base_df.empty else added
        df = df.sort_index()
    else:
        df = base_df.copy()

    # Coordinate columns must be float to accept NaN and sub-pixel values
    for col in df.columns:
        if col[1] != "state" and df[col].dtype != float:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    by_cam: dict[str, list[tuple[tuple, float, float, str]]] = {}
    for (idx, cam), (x, y, state) in updates.items():
        by_cam.setdefault(cam, []).append((idx, x, y, state))
    for cam, items in by_cam.items():
        index = pd.MultiIndex.from_tuples([it[0] for it in items], names=INDEX_NAMES)
        df.loc[index, (cam, "x")] = [it[1] for it in items]
        df.loc[index, (cam, "y")] = [it[2] for it in items]
        df.loc[index, (cam, "state")] = [it[3] for it in items]
    return df

"""SLEAP format adapter: import/export .slp files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import sleap_io as sio

import logging
logger = logging.getLogger(__name__)

from label3d.adapters.base import PoseAdapter
from label3d.points import (
    CAM_COORDS,
    INDEX_NAMES,
    SOURCE_SLEAP_PRED,
    SOURCE_SLEAP_USER,
    THREE_D_COORDS,
)
from label3d.slp_sync import (
    Cell,
    SlpMapping,
    camera_cells,
    camera_scores,
    normalize_index,
    video_cells,
    written_instances,
)


class SleapAdapter(PoseAdapter):
    """Import and export SLEAP .slp pose files."""

    @classmethod
    def name(cls) -> str:
        return "sleap"

    @classmethod
    def file_filter(cls) -> str:
        return "SLEAP files (*.slp)"

    def load(
        self,
        files: list[Path] | None,
        trials: pd.DataFrame,
        cameras: list[str],
        mapping: SlpMapping | None = None,
        root: Path | None = None,
        unmatched: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load .slp files and return a points-compatible DataFrame.

        Each .slp file may cover one or more cameras (matched by video basename
        against trials.csv).  When mapping is given, files is ignored: the
        mapping's files (relative to root) are read, and videos in a file that
        are not assigned to that file are skipped.

        Within each frame and camera, instances with no finite points are
        skipped and the rest are numbered 0, 1, … in file order. A prediction
        that a user instance was corrected from (its from_predicted) is
        skipped too: as in SLEAP, the correction replaces it rather than
        becoming a second instance.

        Frames before a video's {cam}_start or after its {cam}_end are dropped.
        If unmatched is given, basenames of videos that match no trial video
        are appended to it.
        """
        cells_info = video_cells(trials, cameras)
        cell_ends = _cell_ends(trials, cameras)
        basename_to_cells = basename_lookup(trials, cameras)
        rows: list[dict] = []

        for slp_path, allowed in self._sources(files, mapping, root):
            logger.debug(f"Loading SLEAP file: {slp_path}")
            labels = sio.load_file(str(slp_path))
            logger.debug(f"{len(labels.labeled_frames)} labeled frames")
            warned: set[str] = set()

            for lf in labels.labeled_frames:
                if not isinstance(lf.video.filename, str):
                    continue
                video_basename = Path(lf.video.filename).name
                matches = [
                    m for m in basename_to_cells.get(video_basename, []) if m in cells_info
                ]
                if not matches and unmatched is not None and video_basename not in unmatched:
                    unmatched.append(video_basename)
                if allowed is not None:
                    kept = [m for m in matches if m in allowed]
                    if matches and not kept and video_basename not in warned:
                        logger.warning(
                            f"{slp_path.name}: video {video_basename} is not assigned "
                            "to this file, skipping"
                        )
                        warned.add(video_basename)
                    matches = kept

                replaced = {
                    id(inst.from_predicted)
                    for inst in lf.instances
                    if not isinstance(inst, sio.PredictedInstance)
                    and inst.from_predicted is not None
                }

                for cell in matches:
                    experiment_name, trial_name, cam_name = cell
                    global_frame = lf.frame_idx - cells_info[cell][1]
                    end = cell_ends.get(cell)
                    if global_frame < 0 or (end is not None and lf.frame_idx > end):
                        continue

                    inst_idx = 0
                    for inst in lf.instances:
                        if id(inst) in replaced:
                            continue
                        pts = inst.numpy()  # (n_nodes, 2), NaN for missing
                        finite = np.isfinite(pts).all(axis=1)
                        if not finite.any():
                            continue
                        is_predicted = isinstance(inst, sio.PredictedInstance)
                        scores = (
                            inst.numpy(scores=True)[:, 2] if is_predicted
                            else np.full(len(pts), np.nan)
                        )
                        source = SOURCE_SLEAP_PRED if is_predicted else SOURCE_SLEAP_USER

                        for node_idx, node_name in enumerate(inst.skeleton.node_names):
                            if finite[node_idx]:
                                state = "predicted" if is_predicted else "set"
                                x_val = float(pts[node_idx, 0])
                                y_val = float(pts[node_idx, 1])
                                score = float(scores[node_idx])
                            else:
                                state = "not_set"
                                x_val = np.nan
                                y_val = np.nan
                                score = np.nan

                            rows.append({
                                "experiment": experiment_name,
                                "trial": trial_name,
                                "frame": global_frame,
                                "instance": inst_idx,
                                "bodypart": node_name,
                                "camera": cam_name,
                                "x": x_val,
                                "y": y_val,
                                "state": state,
                                "source": source if state != "not_set" else "",
                                "score": score,
                            })
                        inst_idx += 1

        return _rows_to_dataframe(rows, cameras)

    def build_file_map(
        self,
        files: list[Path],
        trials_df: pd.DataFrame,
        cameras: list[str],
    ) -> dict[tuple[str, str, str], Path]:
        """Return a mapping of (experiment, trial, camera) -> source .slp file path.

        Reads the video references embedded in each SLP file and matches them
        against trials_df using the same basename lookup used during load.
        """
        basename_to_trial_cam = basename_lookup(trials_df, cameras)
        file_map: dict[tuple[str, str, str], Path] = {}
        for slp_path in files:
            labels = sio.load_file(str(slp_path))
            for video in labels.videos:
                basename = Path(video.filename).name
                for key in basename_to_trial_cam.get(basename, []):
                    file_map[key] = slp_path
        return file_map

    def save(
        self,
        df: pd.DataFrame,
        cameras: list[str],
        skeleton=None,
        trials_df: pd.DataFrame | None = None,
        skeletons=None,
        file_map: dict[tuple[str, str, str], Path] | None = None,
        output_dir: Path | None = None,
        mapping: SlpMapping | None = None,
        root: Path | None = None,
        **kwargs,
    ) -> int:
        """Write .slp files from the points DataFrame.

        Targets, in order of precedence:
        - mapping (with root): one file per SlpFileSpec, holding all its videos
        - file_map: cells grouped by their source path
        - output_dir (Phase 16 export view): one {camera}_{trial}.slp per cell with data

        Each file is written once with all of its videos.  If the file already
        exists, its skeleton, videos (matched by basename), suggestions, tracks
        and provenance are kept; labeled frames are replaced.  A new file gets
        its cells' skeleton from skeletons (a ProjectSkeleton), else skeleton
        for every file, else the df's bodyparts; raises ValueError if its cells
        have different skeletons.  Only the nodes of a file's skeleton are written.  Videos no longer
        assigned to the file are dropped.  Files are written even when they
        have no labeled frames.  Returns the number of files written.
        """
        if trials_df is None:
            raise ValueError("trials_df is required")

        cells_info = {
            cell: info for cell, info in video_cells(trials_df, cameras).items()
        }
        df = normalize_index(df)
        trial_groups = (
            {k: g for k, g in df.groupby(level=["experiment", "trial"], sort=False)}
            if not df.empty else {}
        )

        targets: dict[Path, list[Cell]] = {}
        if mapping is not None:
            if root is None:
                raise ValueError("root is required with mapping")
            for spec in mapping.files:
                targets[root / spec.path] = [c for c in spec.cells() if c in cells_info]
        elif file_map is not None:
            for cell in cells_info:
                if cell in file_map:
                    targets.setdefault(Path(file_map[cell]), []).append(cell)
                elif (cell[0], cell[1]) in trial_groups:
                    logger.warning(f"No SLP file found for {cell}, skipping")
        elif output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            for cell in cells_info:
                experiment_name, trial_name, cam = cell
                if (experiment_name, trial_name) in trial_groups:
                    targets[output_dir / f"{cam}_{trial_name}.slp"] = [cell]
        else:
            raise ValueError("One of mapping, file_map or output_dir must be provided")

        made: dict[int, object] = {}
        for path, cells in targets.items():
            new_skeleton = _new_file_skeleton(path, cells, skeletons, skeleton)
            key = id(new_skeleton)
            if key not in made:
                made[key] = _make_skeleton(new_skeleton, df)
            self._write_file(path, cells, trial_groups, made[key], cells_info, root)
        return len(targets)

    # -- Internal --------------------------------------------------------------

    def _sources(
        self,
        files: list[Path] | None,
        mapping: SlpMapping | None,
        root: Path | None,
    ) -> list[tuple[Path, set[Cell] | None]]:
        """Return (file path, allowed cells or None) pairs to read."""
        if mapping is None:
            return [(Path(f), None) for f in files or []]
        if root is None:
            raise ValueError("root is required with mapping")
        sources = []
        for spec in mapping.files:
            path = root / spec.path
            if not path.exists():
                logger.warning(f"Mapped SLP file does not exist, treating as empty: {path}")
                continue
            sources.append((path, set(spec.cells())))
        return sources

    def _write_file(
        self,
        path: Path,
        cells: list[Cell],
        trial_groups: dict[tuple[str, str], pd.DataFrame],
        sio_skeleton,
        cells_info: dict[Cell, tuple[str, int]],
        root: Path | None,
    ) -> None:
        existing = sio.load_file(str(path)) if path.exists() else None

        # An existing file keeps its skeleton, whatever the project's
        skeleton = sio_skeleton
        if existing is not None and existing.skeletons:
            skeleton = existing.skeletons[0]

        existing_videos = {}
        if existing is not None:
            for video in existing.videos:
                if isinstance(video.filename, str):
                    existing_videos[Path(video.filename).name] = video

        videos: list = []
        labeled_frames: list = []
        for cell in cells:
            filename, start = cells_info[cell]
            video = existing_videos.get(Path(filename).name)
            if video is None:
                video_filename = (
                    str((root / "experiments" / cell[0] / "videos" / filename).absolute())
                    if root is not None else filename
                )
                video = sio.Video(filename=video_filename, open_backend=False)
            if not any(v is video for v in videos):
                videos.append(video)
            labeled_frames.extend(
                _cell_frames(cell, video, start, trial_groups, skeleton)
            )

        suggestions = []
        if existing is not None:
            suggestions = [
                s for s in existing.suggestions if any(s.video is v for v in videos)
            ]

        labels = sio.Labels(
            labeled_frames=labeled_frames,
            videos=videos,
            skeletons=[skeleton],
            tracks=list(existing.tracks) if existing is not None else [],
            suggestions=suggestions,
            provenance=dict(existing.provenance) if existing is not None else {},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.stem + ".tmp.slp")
        sio.save_file(labels, str(tmp_path), verbose=False)
        os.replace(tmp_path, path)


def _new_file_skeleton(path: Path, cells: list[Cell], skeletons, skeleton):
    """The label3d Skeleton a new file at path is created with (None: df bodyparts)."""
    if skeletons is None or not cells:
        return skeleton if skeletons is None else skeletons.union
    found = []
    for cell in cells:
        sk = skeletons.skeleton_for(cell)
        if not any(sk is f or sk.same_topology(f) for f in found):
            found.append(sk)
    if len(found) > 1:
        names = ", ".join(sk.name or "unnamed" for sk in found)
        raise ValueError(f"{path.name}: its videos have different skeletons ({names})")
    return found[0]


def _make_skeleton(skeleton, df: pd.DataFrame):
    """Build a sleap_io Skeleton from a label3d Skeleton (or df bodyparts)."""
    sio_skeleton = sio.Skeleton(name=(skeleton.name if skeleton else None) or "label3d")
    if skeleton is None:
        for bp in df.index.get_level_values("bodypart").unique():
            sio_skeleton.add_node(str(bp))
    else:
        for node in skeleton.nodes:
            sio_skeleton.add_node(node)
        for src, dst in skeleton.edges:
            sio_skeleton.add_edge(src, dst)
        for a, b in skeleton.symmetries:
            sio_skeleton.add_symmetry(a, b)
    return sio_skeleton


def _cell_frames(
    cell: Cell,
    video,
    start: int,
    trial_groups: dict[tuple[str, str], pd.DataFrame],
    skeleton,
) -> list:
    """Build LabeledFrames for one (experiment, trial, camera) cell."""
    experiment_name, trial_name, cam = cell
    group = trial_groups.get((experiment_name, trial_name))
    if group is None or (cam, "state") not in group.columns:
        return []

    node_names = skeleton.node_names
    labeled_frames = []
    for frame_idx, frame_df in group.groupby(level="frame", sort=True):
        cells = camera_cells(frame_df, cam)
        scores = camera_scores(frame_df, cam)
        instances = []
        for inst, is_user in written_instances(cells):
            # not_visible points are written as NaN (SLEAP's "not visible")
            pts = np.full((len(node_names), 2), np.nan, dtype=np.float32)
            point_scores = np.zeros(len(node_names), dtype=np.float64)
            for bp_idx, bp in enumerate(node_names):
                x, y, state = cells.get((inst, bp), (np.nan, np.nan, "not_set"))
                if state in ("set", "predicted") and np.isfinite(x) and np.isfinite(y):
                    pts[bp_idx] = [x, y]
                    # Predictions without a stored score (label3d guesses) count as 1
                    score = scores.get((inst, bp), np.nan)
                    point_scores[bp_idx] = score if np.isfinite(score) else 1.0
            if is_user:
                instances.append(sio.Instance.from_numpy(pts, skeleton=skeleton))
            else:
                visible = np.isfinite(pts).all(axis=1)
                instances.append(sio.PredictedInstance.from_numpy(
                    pts,
                    skeleton=skeleton,
                    point_scores=point_scores,
                    score=float(point_scores[visible].mean()) if visible.any() else 0.0,
                ))
        if instances:
            labeled_frames.append(sio.LabeledFrame(
                video=video, frame_idx=int(frame_idx) + start, instances=instances
            ))
    return labeled_frames


def _cell_ends(trials: pd.DataFrame, cameras: list[str]) -> dict[Cell, int]:
    """(experiment, trial, camera) → last video frame of the trial, where {cam}_end is set."""
    ends: dict[Cell, int] = {}
    has_experiment = "experiment" in trials.columns
    for _, row in trials.iterrows():
        experiment_name = str(row["experiment"]) if has_experiment else "default"
        for cam in cameras:
            end = row.get(f"{cam}_end")
            if end is not None and pd.notna(end) and str(end).strip() != "":
                ends[(experiment_name, str(row["trial"]), cam)] = int(float(end))
    return ends


def basename_lookup(
    trials: pd.DataFrame, cameras: list[str]
) -> dict[str, list[tuple[str, str, str]]]:
    """Build video_basename -> [(experiment_name, trial_name, camera_name)] lookup."""
    lookup: dict[str, list[tuple[str, str, str]]] = {}
    has_experiment = "experiment" in trials.columns
    for _, row in trials.iterrows():
        experiment_name = str(row["experiment"]) if has_experiment else "default"
        trial_name = str(row["trial"])
        for cam in cameras:
            if cam in row and pd.notna(row[cam]):
                basename = Path(str(row[cam])).name
                lookup.setdefault(basename, []).append((experiment_name, trial_name, cam))
    return lookup


def _rows_to_dataframe(
    rows: list[dict], cameras: list[str]
) -> pd.DataFrame:
    """Convert flat row dicts to the canonical MultiIndex DataFrame."""
    if not rows:
        col_tuples = []
        for cam in cameras:
            for coord in CAM_COORDS:
                col_tuples.append((cam, coord))
        for coord in THREE_D_COORDS:
            col_tuples.append(("3d", coord))
        columns = pd.MultiIndex.from_tuples(col_tuples, names=["view", "coord"])
        index = pd.MultiIndex.from_tuples([], names=INDEX_NAMES)
        return pd.DataFrame(index=index, columns=columns)

    # Collect unique (experiment, trial, frame, instance, bodypart) keys
    keys: dict[tuple, dict[str, tuple]] = {}
    for r in rows:
        key = (r.get("experiment", "default"), r["trial"], r["frame"], r["instance"], r["bodypart"])
        if key not in keys:
            keys[key] = {}
        keys[key][r["camera"]] = (
            r["x"], r["y"], r["state"], r.get("source", ""), r.get("score", np.nan)
        )

    # Build column MultiIndex
    col_tuples = []
    for cam in cameras:
        for coord in CAM_COORDS:
            col_tuples.append((cam, coord))
    for coord in THREE_D_COORDS:
        col_tuples.append(("3d", coord))
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["view", "coord"])

    # Build rows
    data = []
    idx_tuples = []
    for key, cam_data in keys.items():
        row_vals = []
        for cam in cameras:
            if cam in cam_data:
                x, y, state, source, score = cam_data[cam]
                row_vals.extend([x, y, np.nan, np.nan, state, source, score, np.nan])
            else:
                row_vals.extend([np.nan, np.nan, np.nan, np.nan, "not_set", "", np.nan, np.nan])
        # 3d columns: NaN (caller fills after triangulation)
        row_vals.extend([np.nan, np.nan, np.nan])
        data.append(row_vals)
        idx_tuples.append(key)

    index = pd.MultiIndex.from_tuples(idx_tuples, names=INDEX_NAMES)
    return pd.DataFrame(data, index=index, columns=columns)

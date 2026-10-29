"""SLEAP format adapter: import/export .slp files."""

from __future__ import annotations

import os
from collections.abc import Callable
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
    camera_hidden,
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
        progress: Callable[[int, int, str], object] | None = None,
    ) -> pd.DataFrame:
        """Load .slp files and return a points-compatible DataFrame.

        Each .slp file may cover one or more cameras (matched by video basename
        against trials.csv).  When mapping is given, files is ignored: the
        mapping's files (relative to root) are read, and videos in a file that
        are not assigned to that file are skipped.

        Within each frame and camera, instances with no finite points are
        skipped and the rest are numbered 0, 1, … in file order. A prediction
        that a user instance was made from (its from_predicted) is not an
        instance of its own: as in SLEAP, the user instance replaces it, and
        the prediction is kept as that instance's hidden prediction (px, py,
        pscore), which comes back if the user instance is deleted.

        Frames before a video's {cam}_start or after its {cam}_end are dropped.
        If unmatched is given, basenames of videos that match no trial video
        are appended to it.

        progress, if given, is called as progress(done, total, text) with the
        labeled frames read so far in the current file (total 0 while a file
        is being opened).
        """
        cells_info = video_cells(trials, cameras)
        cell_ends = _cell_ends(trials, cameras)
        basename_to_cells = basename_lookup(trials, cameras)
        rows: list[dict] = []

        sources = self._sources(files, mapping, root)
        for n_file, (slp_path, allowed) in enumerate(sources, start=1):
            text = f"Reading {slp_path.name} (file {n_file} of {len(sources)})…"
            if progress is not None:
                progress(0, 0, text)
            logger.debug(f"Loading SLEAP file: {slp_path}")
            labels = sio.load_file(str(slp_path))
            n_frames = len(labels.labeled_frames)
            logger.debug(f"{n_frames} labeled frames")
            step = max(1, n_frames // 100)
            warned: set[str] = set()

            for n_done, lf in enumerate(labels.labeled_frames):
                if progress is not None and n_done % step == 0:
                    progress(n_done, n_frames, text)
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
                        hidden = (
                            {} if is_predicted or inst.from_predicted is None
                            else _prediction_points(inst.from_predicted)
                        )

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

                            px, py, pscore = hidden.get(node_name, (np.nan, np.nan, np.nan))
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
                                "px": px,
                                "py": py,
                                "pscore": pscore,
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
        - output_dir (Phase 17 export view): one {camera}_{trial}.slp per cell with data

        Each file is written once with all of its videos.  If the file already
        exists, its skeletons, videos (matched by basename), suggestions,
        tracks, sessions, ROIs, masks and provenance are kept; the labeled
        frames inside the trials' frame ranges are replaced.  Everything the
        project does not cover is kept untouched: videos that are not trial
        videos, with their frames, and the frames of trial videos outside
        every trial's range.  A new file gets
        its cells' skeleton from skeletons (a ProjectSkeleton), else skeleton
        for every file, else the df's bodyparts; raises ValueError if its cells
        have different skeletons.  Only the nodes of a file's skeleton are written.  Trial
        videos no longer assigned to the file are dropped.  Files are written even when they
        have no labeled frames.  Returns the number of files written.
        """
        if trials_df is None:
            raise ValueError("trials_df is required")

        cells_info = {
            cell: info for cell, info in video_cells(trials_df, cameras).items()
        }
        ranges = _video_ranges(cells_info, _cell_ends(trials_df, cameras))
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
            self._write_file(path, cells, trial_groups, made[key], cells_info, ranges, root)
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
        ranges: dict[str, list[tuple[int, int | None]]],
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

        written: list = []
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
            if not any(v is video for v in written):
                written.append(video)
            labeled_frames.extend(
                _cell_frames(cell, video, start, trial_groups, skeleton)
            )

        if existing is None:
            labels = sio.Labels(
                labeled_frames=labeled_frames, videos=written, skeletons=[skeleton]
            )
        else:
            labels = _merge_existing(existing, written, labeled_frames, skeleton, ranges, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.stem + ".tmp.slp")
        sio.save_file(labels, str(tmp_path), verbose=False)
        os.replace(tmp_path, path)


def _video_ranges(
    cells_info: dict[Cell, tuple[str, int]], cell_ends: dict[Cell, int]
) -> dict[str, list[tuple[int, int | None]]]:
    """Trial video basename -> the (first, last) video frames of its trials; last None if open."""
    ranges: dict[str, list[tuple[int, int | None]]] = {}
    for cell, (filename, start) in cells_info.items():
        ranges.setdefault(Path(filename).name, []).append((start, cell_ends.get(cell)))
    return ranges


def _video_name(video) -> str | None:
    return Path(video.filename).name if isinstance(video.filename, str) else None


def _in_ranges(frame_idx: int, ranges: list[tuple[int, int | None]]) -> bool:
    return any(
        frame_idx >= first and (last is None or frame_idx <= last) for first, last in ranges
    )


def _merge_existing(
    existing, written: list, labeled_frames: list, skeleton, ranges, path: Path
):
    """Labels with the new frames of the written videos and the rest of existing kept.

    Videos that are not trial videos (their basename is in no trial, or they
    are image sequences) are kept with all their frames, in the file's order;
    trial videos are kept only if written, with their frames outside every
    trial's range.  Suggestions, ROIs and masks of the kept videos are kept,
    unless they belong to a replaced instance, and sessions whose frames are
    all kept.
    """
    written_ids = {id(v) for v in written}
    videos = [
        v for v in existing.videos if id(v) in written_ids or _video_name(v) not in ranges
    ]
    videos += [v for v in written if not any(v is k for k in videos)]
    video_ids = {id(v) for v in videos}

    kept_frames = [
        lf for lf in existing.labeled_frames
        if id(lf.video) in video_ids
        and not _in_ranges(lf.frame_idx, ranges.get(_video_name(lf.video), []))
    ]
    kept_frame_ids = {id(lf) for lf in kept_frames}
    kept_instance_ids = {id(inst) for lf in kept_frames for inst in lf.instances}
    order = {id(v): i for i, v in enumerate(videos)}
    frames = sorted(
        labeled_frames + kept_frames, key=lambda lf: (order[id(lf.video)], lf.frame_idx)
    )

    def kept(item) -> bool:
        return (item.video is None or id(item.video) in video_ids) and (
            item.instance is None or id(item.instance) in kept_instance_ids
        )

    sessions = []
    for session in existing.sessions:
        if all(id(v) in video_ids for v in session.videos) and all(
            id(lf) in kept_frame_ids
            for group in session.frame_groups.values()
            for lf in group.labeled_frames
        ):
            sessions.append(session)
        else:
            logger.warning(f"{path.name}: dropping a recording session whose frames were rewritten")

    return sio.Labels(
        labeled_frames=frames,
        videos=videos,
        skeletons=list(existing.skeletons) or [skeleton],
        tracks=list(existing.tracks),
        suggestions=[s for s in existing.suggestions if id(s.video) in video_ids],
        sessions=sessions,
        provenance=dict(existing.provenance),
        rois=[r for r in existing.rois if kept(r)],
        masks=[m for m in existing.masks if kept(m)],
    )


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
        hidden = camera_hidden(frame_df, cam)
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
                # The prediction hidden under the user instance is written too,
                # linked by from_predicted, so SLEAP hides it as its own GUI does
                pred = None
                pred_pts = np.full((len(node_names), 2), np.nan, dtype=np.float32)
                pred_scores = np.zeros(len(node_names), dtype=np.float64)
                for bp_idx, bp in enumerate(node_names):
                    if (inst, bp) in hidden:
                        px, py, pscore = hidden[(inst, bp)]
                        pred_pts[bp_idx] = [px, py]
                        pred_scores[bp_idx] = pscore if np.isfinite(pscore) else 1.0
                if np.isfinite(pred_pts).all(axis=1).any():
                    pred = _predicted_instance(pred_pts, pred_scores, skeleton)
                    instances.append(pred)
                instances.append(
                    sio.Instance.from_numpy(pts, skeleton=skeleton, from_predicted=pred)
                )
            else:
                instances.append(_predicted_instance(pts, point_scores, skeleton))
        if instances:
            labeled_frames.append(sio.LabeledFrame(
                video=video, frame_idx=int(frame_idx) + start, instances=instances
            ))
    return labeled_frames


def _predicted_instance(pts: np.ndarray, point_scores: np.ndarray, skeleton):
    """A PredictedInstance whose score is the mean of its visible points' scores."""
    visible = np.isfinite(pts).all(axis=1)
    return sio.PredictedInstance.from_numpy(
        pts,
        skeleton=skeleton,
        point_scores=point_scores,
        score=float(point_scores[visible].mean()) if visible.any() else 0.0,
    )


def _prediction_points(pred) -> dict[str, tuple[float, float, float]]:
    """Node name → (x, y, score) of a PredictedInstance's visible points."""
    pts = pred.numpy(scores=True)
    return {
        name: (float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
        for i, name in enumerate(pred.skeleton.node_names)
        if np.isfinite(pts[i, :2]).all()
    }


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
        keys[key][r["camera"]] = r

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
            r = cam_data.get(cam, {"state": "not_set", "source": ""})
            row_vals.extend(r.get(coord, np.nan) for coord in CAM_COORDS)
        # 3d columns: NaN (caller fills after triangulation)
        row_vals.extend([np.nan, np.nan, np.nan])
        data.append(row_vals)
        idx_tuples.append(key)

    index = pd.MultiIndex.from_tuples(idx_tuples, names=INDEX_NAMES)
    return pd.DataFrame(data, index=index, columns=columns)

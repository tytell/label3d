"""SLEAP format adapter: import/export .slp files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from label3d.adapters.base import PoseAdapter
from label3d.points import CAM_COORDS, INDEX_NAMES, THREE_D_COORDS


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
        files: list[Path],
        trials: pd.DataFrame,
        cameras: list[str],
    ) -> pd.DataFrame:
        """Load .slp files and return a points-compatible DataFrame.

        Each .slp file may cover one or more cameras (matched by video basename
        against trials.csv).
        """
        import sleap_io as sio

        # Build a lookup: video_basename -> list of (trial_name, camera_name)
        basename_to_trial_cam = _build_basename_lookup(trials, cameras)

        rows: list[dict] = []

        for slp_path in files:
            labels = sio.load_file(str(slp_path))
            node_names = [n.name for n in labels.skeleton.nodes]

            for lf in labels.labeled_frames:
                # Match video to (trial, camera)
                video_basename = Path(lf.video.filename).name
                matches = basename_to_trial_cam.get(video_basename, [])
                if not matches:
                    continue

                for trial_name, cam_name in matches:
                    frame_idx = lf.frame_idx
                    # Adjust frame index by start offset
                    start_col = f"{cam_name}_start"
                    if start_col in trials.columns:
                        trial_row = trials[trials["trial"] == trial_name]
                        if not trial_row.empty:
                            start = int(trial_row.iloc[0].get(start_col, 0) or 0)
                            global_frame = frame_idx - start
                        else:
                            global_frame = frame_idx
                    else:
                        global_frame = frame_idx

                    if global_frame < 0:
                        continue

                    for inst_idx, inst in enumerate(lf.instances):
                        is_predicted = hasattr(inst, "score")
                        pts = inst.numpy()  # (n_nodes, 2), NaN for missing

                        for node_idx, node_name in enumerate(node_names):
                            x, y = pts[node_idx]
                            if np.isnan(x) or np.isnan(y):
                                state = "predicted" if is_predicted else "not_set"
                                x_val = np.nan
                                y_val = np.nan
                            else:
                                state = "predicted" if is_predicted else "set"
                                x_val = float(x)
                                y_val = float(y)

                            rows.append({
                                "trial": trial_name,
                                "frame": global_frame,
                                "instance": inst_idx,
                                "bodypart": node_name,
                                "camera": cam_name,
                                "x": x_val,
                                "y": y_val,
                                "state": state,
                            })

        return _rows_to_dataframe(rows, cameras)

    def save(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        cameras: list[str],
        skeleton=None,
        trials_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """Write per-(trial, camera) .slp files from the points DataFrame."""
        import sleap_io as sio

        output_dir.mkdir(parents=True, exist_ok=True)

        # Build sleap skeleton
        if skeleton is None:
            bodyparts = df.index.get_level_values("bodypart").unique().tolist()
            sio_skeleton = sio.Skeleton(name="label3d")
            for bp in bodyparts:
                sio_skeleton.add_node(bp)
        else:
            sio_skeleton = sio.Skeleton(name="label3d")
            for node in skeleton.nodes:
                sio_skeleton.add_node(node)
            for src, dst in skeleton.edges:
                sio_skeleton.add_edge(src, dst)

        # Group by trial
        for trial_name in df.index.get_level_values("trial").unique():
            trial_df = df.loc[trial_name]

            for cam in cameras:
                if cam not in df.columns.get_level_values(0):
                    continue

                # Resolve video path for the Video reference
                video_filename = ""
                if trials_df is not None:
                    trial_row = trials_df[trials_df["trial"] == trial_name]
                    if not trial_row.empty and cam in trial_row.columns:
                        video_filename = str(trial_row.iloc[0][cam])

                video = sio.Video(filename=video_filename)
                node_names = [n.name for n in sio_skeleton.nodes]

                labeled_frames = []
                for frame_idx in trial_df.index.get_level_values("frame").unique():
                    frame_df = trial_df.loc[frame_idx]
                    instances = []

                    for inst_idx in frame_df.index.get_level_values("instance").unique():
                        inst_df = frame_df.loc[inst_idx]
                        pts = np.full((len(node_names), 2), np.nan, dtype=np.float32)

                        any_set = False
                        all_predicted = True
                        for bp_idx, bp in enumerate(node_names):
                            if bp in inst_df.index:
                                row = inst_df.loc[bp]
                                x = row.get((cam, "x"), np.nan)
                                y = row.get((cam, "y"), np.nan)
                                state = row.get((cam, "state"), "not_set")
                                if pd.notna(x) and pd.notna(y):
                                    pts[bp_idx] = [float(x), float(y)]
                                if state in ("set", "occluded", "not_set"):
                                    all_predicted = False
                                if state == "set":
                                    any_set = True

                        if all_predicted:
                            inst = sio.PredictedInstance.from_numpy(
                                pts, skeleton=sio_skeleton, score=1.0
                            )
                        else:
                            inst = sio.Instance.from_numpy(
                                pts, skeleton=sio_skeleton
                            )
                        instances.append(inst)

                    if instances:
                        lf = sio.LabeledFrame(
                            video=video, frame_idx=int(frame_idx), instances=instances
                        )
                        labeled_frames.append(lf)

                if labeled_frames:
                    labels = sio.Labels(labeled_frames=labeled_frames)
                    slp_path = output_dir / f"{cam}_{trial_name}.slp"
                    sio.save_file(labels, str(slp_path))


def _build_basename_lookup(
    trials: pd.DataFrame, cameras: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """Build video_basename -> [(trial_name, camera_name)] lookup."""
    lookup: dict[str, list[tuple[str, str]]] = {}
    for _, row in trials.iterrows():
        trial_name = str(row["trial"])
        for cam in cameras:
            if cam in row and pd.notna(row[cam]):
                basename = Path(str(row[cam])).name
                lookup.setdefault(basename, []).append((trial_name, cam))
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

    # Collect unique (trial, frame, instance, bodypart) keys
    keys: dict[tuple, dict[str, tuple]] = {}
    for r in rows:
        key = (r["trial"], r["frame"], r["instance"], r["bodypart"])
        if key not in keys:
            keys[key] = {}
        keys[key][r["camera"]] = (r["x"], r["y"], r["state"])

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
                x, y, state = cam_data[cam]
                row_vals.extend([x, y, np.nan, np.nan, state])
            else:
                row_vals.extend([np.nan, np.nan, np.nan, np.nan, "not_set"])
        # 3d columns: NaN (caller fills after triangulation)
        row_vals.extend([np.nan, np.nan, np.nan])
        data.append(row_vals)
        idx_tuples.append(key)

    index = pd.MultiIndex.from_tuples(idx_tuples, names=INDEX_NAMES)
    return pd.DataFrame(data, index=index, columns=columns)

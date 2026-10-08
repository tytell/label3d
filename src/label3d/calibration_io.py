"""Calibration I/O: save/load detected Charuco corners, compute reprojection errors."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aniposelib.boards import merge_rows

logger = logging.getLogger(__name__)


# corners.csv columns. One file per camera holds every take's detections;
# `detected` is the time of the detection run that produced a take's rows.
CORNER_COLUMNS = ["take", "frame", "corner_id", "x", "y", "detected"]


def corners_path(cam_dir: Path) -> Path:
    """Path to a camera's corners file."""
    return cam_dir / "corners.csv"


def _rows_to_df(rows: list[dict], detected: str) -> pd.DataFrame:
    records = []
    for row in rows:
        framenum = row.get("framenum")
        ids, corners = row.get("ids"), row.get("corners")
        if not isinstance(framenum, tuple) or ids is None or corners is None:
            continue
        take, frame = framenum
        ids_flat = np.asarray(ids, dtype=np.int32).ravel()
        corners_flat = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        for corner_id, (x, y) in zip(ids_flat, corners_flat):
            records.append((str(take), int(frame), int(corner_id), float(x), float(y), detected))
    return pd.DataFrame(records, columns=CORNER_COLUMNS)


def _read_corners_df(cam_dir: Path) -> pd.DataFrame | None:
    """The corners file as a DataFrame."""
    csv_path = corners_path(cam_dir)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, dtype={"take": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CORNER_COLUMNS)
    except Exception as exc:
        logger.warning("Failed to load corners from %s: %s", csv_path, exc)
        return None
    return df


def save_corners(
    cam_dir: Path,
    rows: list[dict],
    detected: datetime | str,
    takes: list[str] | None = None,
) -> None:
    """Save detected Charuco corners for one camera.

    Replaces the rows of `takes` (default: the takes present in `rows`) and
    keeps every other take's rows. Each row needs 'framenum' = (take, frame),
    'ids' and 'corners', as aniposelib's detection produces them.
    """
    if isinstance(detected, datetime):
        detected = detected.isoformat(timespec="seconds")
    new = _rows_to_df(rows, detected)
    if takes is None:
        takes = sorted(set(new["take"]))
    old = _read_corners_df(cam_dir)
    if old is not None and not old.empty:
        old = old[~old["take"].astype(str).isin(takes)]
        new = pd.concat([old, new], ignore_index=True) if not new.empty else old
    new.to_csv(corners_path(cam_dir), index=False)


def corner_detections(cam_dir: Path) -> dict[str, str]:
    """take -> timestamp of the detection run its saved corners came from."""
    df = _read_corners_df(cam_dir)
    if df is None or df.empty:
        return {}
    return {str(t): str(d) for t, d in df.groupby("take")["detected"].first().items()}


def load_corners(cam_dir: Path, takes: list[str] | None = None) -> list[dict] | None:
    """Load saved Charuco corners for one camera, optionally only some takes.

    Returns a list of row dicts compatible with aniposelib's merge_rows —
    each dict has 'corners' (N,1,2), 'ids' (N,1) and 'framenum', a
    (take, frame) tuple. Returns None if the file is absent.
    """
    df = _read_corners_df(cam_dir)
    if df is None:
        return None
    if takes is not None:
        df = df[df["take"].astype(str).isin([str(t) for t in takes])]
    if df.empty:
        return []

    rows: list[dict] = []
    for (take, frame), group in df.groupby(["take", "frame"], sort=False):
        ids_arr = group["corner_id"].to_numpy(dtype=np.int32)
        xs = group["x"].to_numpy(dtype=np.float32)
        ys = group["y"].to_numpy(dtype=np.float32)
        n = len(ids_arr)
        rows.append({
            "corners": np.column_stack([xs, ys]).reshape(n, 1, 2),
            "ids": ids_arr.reshape(n, 1),
            "framenum": (str(take), int(frame)),
        })
    return rows


def compute_frame_errors(
    cgroup,
    board,
    rows_list: list[list],
    cam_names: list[str],
) -> tuple[float, dict]:
    """Compute per-corner reprojection errors from a calibrated CameraGroup.

    Returns (mean_error_px, frame_errors) where frame_errors maps
    merged_frame_idx -> cam_name -> ndarray of per-corner errors (NaN = undetected).
    """
    try:
        merged = merge_rows(rows_list, cam_names=cam_names)
    except Exception as exc:
        logger.warning("compute_frame_errors: merge_rows failed: %s", exc)
        return 0.0, {}

    try:
        objp_all = board.get_object_points().reshape(-1, 3).astype(np.float64)
    except Exception as exc:
        logger.warning("compute_frame_errors: get_object_points failed: %s", exc)
        return 0.0, {}

    frame_errors: dict[int, dict[str, np.ndarray]] = {}
    n_corners = len(objp_all)
    all_errors: list[float] = []

    for fi, merged_row in enumerate(merged):
        frame_errors[fi] = {}

        for cam_name, cam in zip(cam_names, cgroup.cameras):
            errors = np.full(n_corners, np.nan)

            if cam_name not in merged_row:
                frame_errors[fi][cam_name] = errors
                continue

            row = merged_row[cam_name]
            filled = row.get("filled")
            if filled is None:
                frame_errors[fi][cam_name] = errors
                continue

            pts2d = np.asarray(filled, dtype=np.float64).reshape(-1, 2)
            mask = ~np.any(np.isnan(pts2d), axis=1)
            if np.sum(mask) < 4:
                frame_errors[fi][cam_name] = errors
                continue

            K = np.array(cam.get_camera_matrix(), dtype=np.float64)
            dist = np.array(cam.get_distortions(), dtype=np.float64)

            objp_det = objp_all[mask]
            imgp_det = pts2d[mask].reshape(-1, 1, 2)

            try:
                ok, rvec, tvec = cv2.solvePnP(objp_det, imgp_det, K, dist)
            except Exception as exc:
                logger.warning("solvePnP raised for %s frame %d: %s", cam_name, fi, exc)
                ok = False

            if not ok:
                rvec_raw = row.get("rvec")
                tvec_raw = row.get("tvec")
                if rvec_raw is None or tvec_raw is None:
                    frame_errors[fi][cam_name] = errors
                    continue
                try:
                    rvec = np.asarray(rvec_raw, dtype=np.float64).reshape(3, 1)
                    tvec = np.asarray(tvec_raw, dtype=np.float64).reshape(3, 1)
                except Exception:
                    frame_errors[fi][cam_name] = errors
                    continue

            try:
                projected, _ = cv2.projectPoints(objp_all, rvec, tvec, K, dist)
                projected = projected.reshape(-1, 2)
            except Exception as exc:
                logger.warning("projectPoints failed for %s frame %d: %s", cam_name, fi, exc)
                frame_errors[fi][cam_name] = errors
                continue

            for j in range(n_corners):
                if not np.isnan(pts2d[j, 0]):
                    dx = projected[j, 0] - pts2d[j, 0]
                    dy = projected[j, 1] - pts2d[j, 1]
                    errors[j] = float(np.sqrt(dx * dx + dy * dy))

            frame_errors[fi][cam_name] = errors
            all_errors.extend(errors[~np.isnan(errors)].tolist())

    mean_error = float(np.mean(all_errors)) if all_errors else 0.0
    return mean_error, frame_errors

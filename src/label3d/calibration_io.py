"""Calibration I/O: save/load detected Charuco corners, compute reprojection errors."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aniposelib.boards import merge_rows

logger = logging.getLogger(__name__)


def corners_path(cam_dir: Path, take: str | None = None) -> Path:
    """Path to a camera's corners file.

    One take: corners_{take}.csv. The joint solution, or whichever single take
    the user selects as final: corners.csv.
    """
    return cam_dir / (f"corners_{take}.csv" if take else "corners.csv")


def save_corners(cam_dir: Path, rows: list[dict], take: str | None = None) -> None:
    """Save detected Charuco corners for one camera.

    Schema: source_key (str), corner_id (int), x (float), y (float).
    source_key is "{take}:{frame_in_video}" for video-based calibration, or an
    image basename for the deprecated image path.
    """
    records = []
    for row in rows:
        fname = row.get("fname")
        framenum = row.get("framenum")

        if framenum is not None and isinstance(framenum, tuple):
            take_name, fi = framenum
            source_key = f"{take_name}:{fi}"
        elif fname is not None:
            source_key = Path(fname).name
        elif framenum is not None:
            source_key = f"{take or ''}:{int(framenum)}"
        else:
            continue

        ids = row.get("ids")
        corners = row.get("corners")
        if ids is None or corners is None:
            continue

        ids_flat = np.asarray(ids, dtype=np.int32).ravel()
        corners_flat = np.asarray(corners, dtype=np.float32).reshape(-1, 2)

        for corner_id, (x, y) in zip(ids_flat, corners_flat):
            records.append({
                "source_key": source_key,
                "corner_id": int(corner_id),
                "x": float(x),
                "y": float(y),
            })

    df = pd.DataFrame(records, columns=["source_key", "corner_id", "x", "y"])
    df.to_csv(corners_path(cam_dir, take), index=False)


def load_corners(cam_dir: Path, take: str | None = None) -> list[dict] | None:
    """Load saved Charuco corners for one camera.

    Returns a list of row dicts compatible with aniposelib's merge_rows —
    each dict has 'corners' (N,1,2), 'ids' (N,1), and either 'fname' (Path)
    or 'framenum' (a (take, frame) tuple). Returns None if the file is absent.
    """
    csv_path = corners_path(cam_dir, take)
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to load corners from %s: %s", csv_path, exc)
        return None

    if df.empty:
        return []

    # Collect unique source_keys in original order
    seen: list[str] = []
    for key in df["source_key"].astype(str):
        if key not in seen:
            seen.append(key)

    rows: list[dict] = []
    for i, source_key in enumerate(seen):
        group = df[df["source_key"].astype(str) == source_key]
        ids_arr = group["corner_id"].to_numpy(dtype=np.int32)
        xs = group["x"].to_numpy(dtype=np.float32)
        ys = group["y"].to_numpy(dtype=np.float32)

        n = len(ids_arr)
        corners = np.column_stack([xs, ys]).reshape(n, 1, 2)
        ids = ids_arr.reshape(n, 1)

        row: dict = {"corners": corners, "ids": ids}

        # Reconstruct source reference: "{take}:{frame}" → video frame, else an
        # image basename from the deprecated image path. merge_rows always keys
        # on 'framenum', so image rows fall back to the sequence index.
        parts = source_key.rsplit(":", 1)
        if len(parts) == 2 and parts[1].isdigit():
            row["framenum"] = (parts[0], int(parts[1]))
        else:
            row["framenum"] = i
            row["fname"] = cam_dir / source_key

        rows.append(row)

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

"""Instance initialization strategies for pre-populating new instances.

Three strategies:

random     — scatter bodyparts around a center point using Gaussian noise;
             if skeleton.node_offsets_cm is defined, those cm offsets are
             projected through the camera at a nominal depth instead.

average    — compute each bodypart's mean offset from the first skeleton node
             over labeled instances in this experiment+camera; place the first
             node at center_px and the rest at their mean offsets.
             Falls back to random when fewer than 2 qualifying instances exist.

reprojected — triangulate the selected instance from its existing cameras and
              project the 3D positions into the target camera.  Only fills
              bodyparts that are currently not_set; never overwrites set/predicted.
              Returns None if triangulation is impossible.

availability — available_strategies() checks which strategies are usable given
               the current state; used to enable/disable menu items.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import pandas as pd

from label3d.geometry import triangulate_point

if TYPE_CHECKING:
    from label3d.geometry import CameraParams
    from label3d.points import PointsStore
    from label3d.skeleton import Skeleton


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_center(frame_size: tuple[int, int]) -> tuple[float, float]:
    w, h = frame_size
    return (w / 2.0, h / 2.0)


def _pixel_to_world_at_depth(
    cam: CameraParams,
    px: float,
    py: float,
    depth: float,
) -> np.ndarray:
    """Back-project a pixel to a world point at the given camera-space depth.

    Returns a (3,) world-space array.
    """
    pt = np.array([[[px, py]]], dtype=np.float64)
    # Normalised camera coords (removes K and distortion)
    pt_norm = cv2.undistortPoints(pt, cam.matrix, cam.dist)  # (1, 1, 2)
    ray_cam = np.array([pt_norm[0, 0, 0], pt_norm[0, 0, 1], 1.0]) * depth
    # Camera → world: R.T @ (p_cam − t)
    return cam.R.T @ (ray_cam - cam.tvec.ravel())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_random(
    skeleton: Skeleton,
    camera_name: str,
    cameras: dict[str, CameraParams] | None,
    center_px: tuple[float, float] | None,
    frame_size: tuple[int, int],
    rng: np.random.Generator | None = None,
    nodes: Collection[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return {bodypart: (x, y)} with random placement around center_px.

    If skeleton.node_offsets_cm covers all nodes and calibration is provided,
    each bodypart's cm offset is projected through the camera at a nominal depth
    (50 cm) with 10% jitter.  Otherwise uses an isotropic Gaussian with
    sigma = min(w, h) / 8. nodes, if given, limits the result to the nodes
    the camera has.
    """
    if rng is None:
        rng = np.random.default_rng()

    w, h = frame_size
    cx, cy = center_px if center_px is not None else _frame_center(frame_size)

    result: dict[str, tuple[float, float]] = {}

    cam = cameras.get(camera_name) if cameras else None
    node_offsets: dict[str, list[float]] = getattr(skeleton, "node_offsets_cm", {})
    wanted = [bp for bp in skeleton.nodes if nodes is None or bp in nodes]

    if (
        cam is not None
        and node_offsets
        and all(bp in node_offsets for bp in wanted)
    ):
        try:
            nominal_depth = 50.0  # cm
            world_center = _pixel_to_world_at_depth(cam, cx, cy, nominal_depth)

            projected: list[np.ndarray] = []
            for bp in wanted:
                offset = np.array(node_offsets[bp], dtype=float)
                world_pt = world_center + offset
                pix = cam.project(world_pt.reshape(1, 3))[0]
                projected.append(pix)

            pts = np.array(projected)  # (n, 2)
            span = max(
                float(pts[:, 0].max() - pts[:, 0].min()),
                float(pts[:, 1].max() - pts[:, 1].min()),
                1.0,
            )
            sigma = span * 0.1

            for bp, pix in zip(wanted, projected):
                jitter = rng.normal(0.0, sigma, size=2)
                x = float(np.clip(pix[0] + jitter[0], 0, w))
                y = float(np.clip(pix[1] + jitter[1], 0, h))
                result[bp] = (x, y)
            return result
        except Exception:
            pass  # Fall through to Gaussian

    # Isotropic Gaussian fallback
    sigma = min(w, h) / 8.0
    for bp in wanted:
        x = float(np.clip(rng.normal(cx, sigma), 0, w))
        y = float(np.clip(rng.normal(cy, sigma), 0, h))
        result[bp] = (x, y)

    return result


def init_average(
    points_store: PointsStore,
    experiment: str,
    camera_name: str,
    skeleton: Skeleton,
    center_px: tuple[float, float] | None,
    frame_size: tuple[int, int],
    nodes: Collection[str] | None = None,
) -> tuple[dict[str, tuple[float, float]], bool]:
    """Return (coords, used_fallback) for average-relative placement.

    Only nodes (if given: the nodes the camera has) are placed. The first of
    them in skeleton order is the anchor.  For every instance in the
    experiment whose anchor is placed (set/predicted) in camera_name, each
    other placed bodypart's offset from the anchor is collected; the offsets
    are averaged per bodypart.  The anchor goes at center_px, or, without one,
    at the anchor's mean position in this camera (frame centre if it has never
    been placed), and every other bodypart at anchor + mean offset.  A bodypart
    with no offsets gets a random position.

    used_fallback is True when fewer than 2 instances give offsets (random
    placement is used instead).
    """
    w, h = frame_size
    wanted = [bp for bp in skeleton.nodes if nodes is None or bp in nodes]
    anchor = wanted[0] if wanted else ""

    mean_offsets = pd.DataFrame(columns=["dx", "dy"], dtype=float)
    anchor_xy = np.empty((0, 2))
    n_sources = 0

    try:
        exp_df = points_store.df.loc[experiment]
    except KeyError:
        exp_df = None

    if anchor and exp_df is not None and (camera_name, "state") in exp_df.columns:
        placed = exp_df[(camera_name, "state")].isin(["set", "predicted"])
        xy = (
            exp_df.loc[placed, [(camera_name, "x"), (camera_name, "y")]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        xy.columns = ["x", "y"]
        # Index (trial, frame, instance), one row per placed bodypart
        xy = xy.reset_index(level="bodypart")
        anchors = xy.loc[xy["bodypart"] == anchor, ["x", "y"]]
        anchor_xy = anchors.to_numpy(dtype=float)

        others = xy.loc[xy["bodypart"] != anchor].join(
            anchors, rsuffix="_anchor", how="inner"
        )
        others["dx"] = others["x"] - others["x_anchor"]
        others["dy"] = others["y"] - others["y_anchor"]
        mean_offsets = others.groupby("bodypart")[["dx", "dy"]].mean()
        n_sources = others.index.unique().size

    if center_px is not None:
        cx, cy = center_px
    elif len(anchor_xy):
        cx, cy = (float(v) for v in anchor_xy.mean(axis=0))
    else:
        cx, cy = _frame_center(frame_size)

    random_coords = init_random(
        skeleton, camera_name, None, (cx, cy), frame_size, nodes=wanted
    )
    if n_sources < 2:
        return random_coords, True

    result: dict[str, tuple[float, float]] = {}
    for bp in wanted:
        if bp == anchor:
            x, y = cx, cy
        elif bp in mean_offsets.index:
            x = cx + float(mean_offsets.at[bp, "dx"])
            y = cy + float(mean_offsets.at[bp, "dy"])
        else:
            x, y = random_coords[bp]
        result[bp] = (float(np.clip(x, 0, w)), float(np.clip(y, 0, h)))

    return result, False


def init_reprojected(
    points_store: PointsStore,
    experiment: str,
    trial: str,
    frame: int,
    instance: int,
    target_camera: str,
    cameras: dict[str, CameraParams],
    available: dict[str, Collection[str]] | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Triangulate instance from other cameras and project into target_camera.

    Returns {bodypart: (x, y)} for bodyparts that could be reprojected, or None
    if no bodypart had ≥2 source-camera observations.  Coordinates are clipped
    to the target camera frame bounds. available (camera -> nodes it has) limits
    both the bodyparts placed in target_camera and the source points used.
    """
    target_cam = cameras.get(target_camera)
    if target_cam is None:
        return None

    w, h = target_cam.size

    try:
        inst_df = points_store.df.loc[(experiment, trial, frame, instance)]
        bodyparts = inst_df.index.tolist()
    except KeyError:
        return None

    result: dict[str, tuple[float, float]] = {}

    def has(cam_name: str, bp: str) -> bool:
        return available is None or bp in available.get(cam_name, ())

    for bp in bodyparts:
        if not has(target_camera, bp):
            continue
        picks_2d: dict[str, tuple[float, float]] = {}
        for cam_name, cam in cameras.items():
            if cam_name == target_camera or not has(cam_name, bp):
                continue
            pt = points_store.get_point(trial, frame, instance, bp, cam_name)
            if pt.state in ("set", "predicted") and not np.isnan(pt.x):
                picks_2d[cam_name] = (pt.x, pt.y)

        if len(picks_2d) < 2:
            continue

        xyz = triangulate_point(cameras, picks_2d)
        if xyz is None:
            continue

        # Reject points behind the camera
        cam_pt = target_cam.R @ xyz + target_cam.tvec.ravel()
        if cam_pt[2] <= 0:
            continue

        pix = target_cam.project(xyz.reshape(1, 3))[0]
        px = float(np.clip(pix[0], 0, w))
        py = float(np.clip(pix[1], 0, h))
        result[bp] = (px, py)

    return result if result else None


def available_strategies(
    points_store: PointsStore,
    experiment: str,
    trial: str,
    frame: int,
    instance: int | None,
    camera_name: str,
) -> set[Literal["random", "average", "reprojected"]]:
    """Return the set of strategies currently usable.

    "random"      — always available.
    "average"     — available if any instance in this experiment+camera has at
                    least one set/predicted point (init_average will fall back to
                    random with a status message if < 2 qualifying instances are
                    found).
    "reprojected" — available if instance is not None and that instance has
                    set/predicted points in ≥2 cameras other than camera_name
                    in this frame.
    """
    result: set[Literal["random", "average", "reprojected"]] = {"random"}

    # Average: quick check — any labeled point in experiment+camera
    try:
        exp_df = points_store.df.loc[experiment]
        if (camera_name, "state") in exp_df.columns:
            if exp_df[(camera_name, "state")].isin(["set", "predicted"]).any():
                result.add("average")
    except KeyError:
        pass

    # Reprojected: selected instance has ≥2 other cameras with valid data
    if instance is not None and instance >= 0:
        try:
            inst_df = points_store.df.loc[(experiment, trial, frame, instance)]
            other_cam_count = 0
            for cam in points_store.cameras:
                if cam == camera_name:
                    continue
                if (cam, "state") not in inst_df.columns:
                    continue
                if inst_df[(cam, "state")].isin(["set", "predicted"]).any():
                    other_cam_count += 1
            if other_cam_count >= 2:
                result.add("reprojected")
        except KeyError:
            pass

    return result

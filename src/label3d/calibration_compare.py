"""Compare calibrations computed from different calibration takes.

Intrinsics should be identical across takes of one experiment; extrinsics are
what move if a camera is bumped between the start and the end of an experiment.
So the comparison looks at each camera *pair*: how much the relative rotation
turned, and how much the baseline between the two cameras changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

# Defaults for the [calibration] section of project.toml
DEFAULT_ROTATION_DRIFT_DEG = 0.5
DEFAULT_BASELINE_DRIFT_PCT = 1.0


@dataclass
class PairDrift:
    """How one camera pair's relative pose differs between two calibrations."""

    cam_a: str
    cam_b: str
    rotation_deg: float
    baseline_ref_cm: float
    baseline_other_cm: float

    @property
    def baseline_delta_cm(self) -> float:
        return abs(self.baseline_other_cm - self.baseline_ref_cm)

    @property
    def baseline_pct(self) -> float:
        mean = (self.baseline_ref_cm + self.baseline_other_cm) / 2.0
        return 100.0 * self.baseline_delta_cm / mean if mean > 0 else 0.0

    def exceeds(self, rotation_deg: float, baseline_pct: float) -> bool:
        return self.rotation_deg > rotation_deg or self.baseline_pct > baseline_pct


def _extrinsics(cgroup) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """camera name -> (R, t) mapping world points to that camera's frame."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cam in cgroup.cameras:
        rvec = np.asarray(cam.get_rotation(), dtype=np.float64).ravel()
        tvec = np.asarray(cam.get_translation(), dtype=np.float64).ravel()
        out[cam.get_name()] = (_rodrigues(rvec), tvec)
    return out


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation vector to matrix, without pulling in cv2 for three lines."""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _relative_pose(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pose of camera b relative to camera a."""
    R_a, t_a = a
    R_b, t_b = b
    R_ab = R_b @ R_a.T
    t_ab = t_b - R_ab @ t_a
    return R_ab, t_ab


def _angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    R = R1.T @ R2
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compare_calibrations(reference, other) -> list[PairDrift]:
    """Per camera-pair drift of `other` relative to `reference`.

    Only cameras present in both are compared. Pairs come out in a stable
    order so the comparison table does not reshuffle between runs.
    """
    ref_ext = _extrinsics(reference)
    other_ext = _extrinsics(other)
    shared = [name for name in ref_ext if name in other_ext]

    drifts: list[PairDrift] = []
    for cam_a, cam_b in combinations(shared, 2):
        R_ref, t_ref = _relative_pose(ref_ext[cam_a], ref_ext[cam_b])
        R_oth, t_oth = _relative_pose(other_ext[cam_a], other_ext[cam_b])
        drifts.append(PairDrift(
            cam_a=cam_a,
            cam_b=cam_b,
            rotation_deg=_angle_between(R_ref, R_oth),
            baseline_ref_cm=float(np.linalg.norm(t_ref)),
            baseline_other_cm=float(np.linalg.norm(t_oth)),
        ))
    return drifts


def drift_thresholds(config: dict) -> tuple[float, float]:
    """(rotation_deg, baseline_pct) from the [calibration] section."""
    calib = config.get("calibration", {})
    return (
        float(calib.get("rotation_drift_deg", DEFAULT_ROTATION_DRIFT_DEG)),
        float(calib.get("baseline_drift_pct", DEFAULT_BASELINE_DRIFT_PCT)),
    )


def drift_warnings(
    take_name: str, drifts: list[PairDrift], rotation_deg: float, baseline_pct: float
) -> list[str]:
    """Human-readable warnings for the pairs that exceed a threshold."""
    messages = []
    for d in drifts:
        if not d.exceeds(rotation_deg, baseline_pct):
            continue
        messages.append(
            f"{take_name}: {d.cam_a}-{d.cam_b} differs from the joint solution by "
            f"{d.rotation_deg:.2f}deg rotation and {d.baseline_pct:.1f}% baseline "
            f"({d.baseline_ref_cm:.1f} -> {d.baseline_other_cm:.1f} cm)"
        )
    return messages

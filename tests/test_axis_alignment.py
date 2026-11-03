"""Axis alignment geometry (Phase 16.1)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from label3d.geometry import (
    AlignmentError,
    CameraParams,
    align_points,
    axis_vector,
    compute_axis_alignment,
    primary_axis_order,
    scale_check,
    triangulate_pick,
    unalign_points,
)

# The object's frame in calibration coordinates: p_cal = R_TRUE @ p_obj + O_TRUE
R_TRUE, _ = cv2.Rodrigues(np.array([0.3, -0.5, 0.8]))
O_TRUE = np.array([4.0, -3.0, 2.0])


def look_at(name: str, position, target=(0.0, 0.0, 0.0)) -> CameraParams:
    """A camera at position looking at target (OpenCV camera frame)."""
    position = np.asarray(position, dtype=float)
    z = np.asarray(target, dtype=float) - position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    rvec, _ = cv2.Rodrigues(R)
    return CameraParams(
        name=name,
        matrix=np.array([[800.0, 0, 640], [0, 800.0, 480], [0, 0, 1]]),
        dist=np.zeros(5),
        rvec=rvec.ravel(),
        tvec=-R @ position,
        size=(1280, 960),
    )


@pytest.fixture
def cameras():
    return {
        "cam0": look_at("cam0", [80.0, 0.0, 30.0]),
        "cam1": look_at("cam1", [0.0, 80.0, 40.0]),
        "cam2": look_at("cam2", [-60.0, -50.0, 50.0]),
    }


def to_cal(p_obj) -> np.ndarray:
    return R_TRUE @ np.asarray(p_obj, dtype=float) + O_TRUE


def click(cameras, p_obj, noise=0.0, rng=None) -> dict[str, tuple[float, float]]:
    """Each camera's pixel for an object point, optionally with noise."""
    picks = {}
    for name, cam in cameras.items():
        xy = cam.project(to_cal(p_obj)[np.newaxis])[0]
        if noise:
            xy = xy + rng.normal(0.0, noise, 2)
        picks[name] = (float(xy[0]), float(xy[1]))
    return picks


def triangulated(cameras, p_obj, noise=0.0, rng=None) -> np.ndarray:
    xyz, _errors = triangulate_pick(cameras, click(cameras, p_obj, noise, rng))
    return xyz


def angle_deg(a, b) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class TestTriangulatePick:
    def test_point_and_errors(self, cameras):
        p = np.array([1.0, 2.0, 3.0])
        xyz, errors = triangulate_pick(cameras, click(cameras, p))
        np.testing.assert_allclose(xyz, to_cal(p), atol=1e-6)
        assert set(errors) == set(cameras)
        assert max(errors.values()) < 1e-6

    def test_one_camera_fails(self, cameras):
        assert triangulate_pick(cameras, {"cam0": (640.0, 480.0)}) is None


class TestModes:
    def test_four_points(self, cameras):
        origin = triangulated(cameras, [0, 0, 0])
        axes = {
            k: axis_vector(origin, triangulated(cameras, 8.0 * np.eye(3)[i]))
            for i, k in enumerate("xyz")
        }
        result = compute_axis_alignment(origin, axes)
        np.testing.assert_allclose(result.R_align, R_TRUE, atol=1e-6)
        np.testing.assert_allclose(result.origin, O_TRUE, atol=1e-6)
        assert result.derived == ""

    def test_seven_points_ignore_origin_offset(self, cameras):
        # Axis points lie on lines parallel to the axes but displaced from
        # the origin; only near and far set the direction.
        origin = triangulated(cameras, [0, 0, 0])
        offsets = {"x": [0, 1.5, -0.5], "y": [2.0, 0, 1.0], "z": [-1.0, 0.7, 0]}
        axes = {}
        for i, k in enumerate("xyz"):
            unit = np.eye(3)[i]
            near = triangulated(cameras, np.add(offsets[k], 2.0 * unit))
            far = triangulated(cameras, np.add(offsets[k], 10.0 * unit))
            axes[k] = axis_vector(origin, near, far)
        result = compute_axis_alignment(origin, axes)
        np.testing.assert_allclose(result.R_align, R_TRUE, atol=1e-6)
        np.testing.assert_allclose(result.origin, O_TRUE, atol=1e-6)

    @pytest.mark.parametrize("missing", ["x", "y", "z"])
    def test_third_axis_from_cross_product(self, cameras, missing):
        origin = triangulated(cameras, [0, 0, 0])
        axes = {
            k: axis_vector(origin, triangulated(cameras, 6.0 * np.eye(3)[i]))
            for i, k in enumerate("xyz") if k != missing
        }
        result = compute_axis_alignment(origin, axes)
        assert result.derived == missing
        assert np.linalg.det(result.R_align) == pytest.approx(1.0)
        np.testing.assert_allclose(result.R_align, R_TRUE, atol=1e-6)


class TestChecks:
    def test_left_handed_raises(self):
        axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, -1.0])}
        with pytest.raises(AlignmentError, match="left-handed"):
            compute_axis_alignment(np.zeros(3), axes)

    def test_flip_makes_left_handed_clicks_right_handed(self):
        axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, -1.0])}
        result = compute_axis_alignment(np.zeros(3), axes, flip={"z"})
        np.testing.assert_allclose(result.R_align, np.eye(3), atol=1e-12)

    def test_parallel_axes_raise(self):
        axes = {"x": np.array([1.0, 0, 0]), "y": np.array([2.0, 0, 0])}
        with pytest.raises(AlignmentError, match="parallel"):
            compute_axis_alignment(np.zeros(3), axes)

    def test_one_axis_raises(self):
        with pytest.raises(AlignmentError, match="two axes"):
            compute_axis_alignment(np.zeros(3), {"x": np.array([1.0, 0, 0])})

    def test_zero_length_raises(self):
        axes = {"x": np.zeros(3), "y": np.array([0, 1.0, 0])}
        with pytest.raises(AlignmentError, match="zero length"):
            compute_axis_alignment(np.zeros(3), axes)


class TestPrimaryAxis:
    AXES = {"x": np.array([10.0, 0, 0]), "y": np.array([0, 5.0, 0]), "z": np.array([0, 0, 7.0])}

    def test_longest(self):
        assert primary_axis_order(self.AXES) == ["x", "z", "y"]

    def test_lowest_error(self):
        errors = {"x": 0.9, "y": 0.2, "z": 0.5}
        assert primary_axis_order(self.AXES, errors, "lowest_error") == ["y", "z", "x"]

    def test_lowest_error_needs_errors(self):
        with pytest.raises(AlignmentError):
            primary_axis_order(self.AXES, {"x": 0.1}, "lowest_error")

    def test_explicit(self):
        assert primary_axis_order(self.AXES, rule="y") == ["y", "x", "z"]

    def test_derived_axis_cannot_be_primary(self):
        axes = {"x": self.AXES["x"], "y": self.AXES["y"]}
        with pytest.raises(AlignmentError, match="cross product"):
            primary_axis_order(axes, rule="z")

    def test_unknown_rule(self):
        with pytest.raises(ValueError):
            primary_axis_order(self.AXES, rule="shortest")

    def test_noisy_clicks_keep_primary_exact(self):
        # Axes 3° and 5° off orthogonal: the primary is kept, the others move
        tilt = np.radians(3.0)
        axes = {
            "x": np.array([1.0, 0, 0]) * 12,
            "y": np.array([np.sin(tilt), np.cos(tilt), 0]) * 8,
            "z": np.array([0, np.sin(np.radians(5.0)), np.cos(np.radians(5.0))]) * 6,
        }
        result = compute_axis_alignment(np.zeros(3), axes, rule="x")
        np.testing.assert_allclose(result.R_align[:, 0], [1.0, 0, 0], atol=1e-12)
        assert result.primary == "x" and result.secondary == "y"
        assert result.moved_deg["x"] == pytest.approx(0.0, abs=1e-9)
        assert result.moved_deg["y"] == pytest.approx(3.0)
        assert result.moved_deg["z"] == pytest.approx(5.0, abs=0.01)
        assert result.axis_angles_deg["xy"] == pytest.approx(87.0)
        np.testing.assert_allclose(result.R_align.T @ result.R_align, np.eye(3), atol=1e-12)

    def test_noisy_triangulated_clicks(self, cameras):
        rng = np.random.default_rng(20261103)
        origin = triangulated(cameras, [0, 0, 0], 0.5, rng)
        axes = {
            k: axis_vector(origin, triangulated(cameras, 10.0 * np.eye(3)[i], 0.5, rng))
            for i, k in enumerate("xyz")
        }
        result = compute_axis_alignment(origin, axes)
        assert np.linalg.det(result.R_align) == pytest.approx(1.0)
        for i in range(3):
            assert angle_deg(result.R_align[:, i], R_TRUE[:, i]) < 1.0


class TestScaleCheck:
    def test_measured_against_known(self):
        axes = {"x": np.array([10.2, 0, 0]), "y": np.array([0, 5.0, 0])}
        check = scale_check(axes, {"x": 10.0, "z": 4.0})
        assert set(check) == {"x"}
        assert check["x"]["measured_cm"] == pytest.approx(10.2)
        assert check["x"]["error_pct"] == pytest.approx(2.0)

    def test_mis_scaled_rig(self, cameras):
        # A rig whose translations are 5% too large measures lengths 5% too long
        scaled = {
            name: CameraParams(cam.name, cam.matrix, cam.dist, cam.rvec, cam.tvec * 1.05, cam.size)
            for name, cam in cameras.items()
        }
        a, _ = triangulate_pick(scaled, click(cameras, [0, 0, 0]))
        b, _ = triangulate_pick(scaled, click(cameras, [10.0, 0, 0]))
        check = scale_check({"x": axis_vector(a, b)}, {"x": 10.0})
        assert check["x"]["error_pct"] == pytest.approx(5.0, abs=0.01)


class TestTransform:
    def test_round_trip_and_distances(self):
        rng = np.random.default_rng(1103)
        xyz = rng.normal(0.0, 20.0, (50, 3))
        aligned = align_points(xyz, R_TRUE, O_TRUE)
        np.testing.assert_allclose(unalign_points(aligned, R_TRUE, O_TRUE), xyz, atol=1e-9)
        np.testing.assert_allclose(
            np.linalg.norm(aligned[1:] - aligned[:-1], axis=1),
            np.linalg.norm(xyz[1:] - xyz[:-1], axis=1),
        )

    def test_object_points_come_out_in_object_frame(self):
        p_obj = np.array([[0.0, 0, 0], [3.0, 0, 0], [0, 2.0, 5.0]])
        p_cal = np.array([to_cal(p) for p in p_obj])
        np.testing.assert_allclose(align_points(p_cal, R_TRUE, O_TRUE), p_obj, atol=1e-12)

    def test_nan_passes_through(self):
        xyz = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
        out = align_points(xyz, R_TRUE, O_TRUE)
        assert np.isfinite(out[0]).all()
        assert np.isnan(out[1]).all()

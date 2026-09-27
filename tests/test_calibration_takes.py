"""Tests for Phase 9 calibration takes: pairing, ordering, and corners I/O."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from label3d.calibration_compare import noise_adjusted_thresholds, run_spread
from label3d.calibration_io import corners_path, load_corners, save_corners
from label3d.project import Project
from label3d.widgets.calibration_view import (
    CalibFrameProvider,
    _build_takes,
    _scan_take_media,
    nearest_row,
    rekey_frame_errors,
    take_range,
)
from label3d.widgets.frame_slider import FrameSlider
from label3d.workers import calibration_worker
from label3d.workers.calibration_worker import (
    JOINT_TAKE,
    CalibrationTake,
    DetectionWorker,
    TakeVideo,
    rows_in_range,
)

CAMERAS = ["cam1", "cam2"]


def _write_project(root: Path, calib_names: dict[str, dict[str, str]]) -> Path:
    """Build a minimal project; calib_names is {take: {camera: filename}}."""
    header = ["experiment", "trial", "is_calibration"]
    for cam in CAMERAS:
        header += [cam, f"{cam}_start", f"{cam}_end"]

    rows = ["exp1,trial1,False," + ",".join(f"{c}_trial1.mp4,0," for c in CAMERAS)]
    for take, cam_files in calib_names.items():
        cells = ",".join(f"{cam_files[c]},0," for c in CAMERAS)
        rows.append(f"exp1,{take},True,{cells}")

    (root / "trials.csv").write_text(",".join(header) + "\n" + "\n".join(rows) + "\n")

    videos = root / "experiments" / "exp1" / "videos"
    videos.mkdir(parents=True)
    for cam in CAMERAS:
        (videos / f"{cam}_trial1.mp4").write_bytes(b"\x00" * 32)

    for cam in CAMERAS:
        cam_dir = root / "experiments" / "exp1" / "calibration" / cam
        cam_dir.mkdir(parents=True)
        for cam_files in calib_names.values():
            (cam_dir / cam_files[cam]).write_bytes(b"\x00" * 32)
    return root


class TestTakePairing:
    def test_takes_pair_by_name_not_sort_order(self, tmp_path, qapp):
        """The bug this replaces: cameras whose files sort differently.

        cam1 sorts PostCal before PreCal; cam2 sorts the reverse. Position in a
        sorted folder listing would pair PostCal with PreCal.
        """
        from label3d.widgets.calibration_view import _build_takes

        _write_project(tmp_path, {
            "PreCal": {"cam1": "b_PreCal.mp4", "cam2": "a_PreCal.mp4"},
            "PostCal": {"cam1": "a_PostCal.mp4", "cam2": "z_PostCal.mp4"},
        })
        project = Project.load(tmp_path)
        takes = {t.name: t for t in _build_takes(project)}

        assert set(takes) == {"PreCal", "PostCal"}
        assert takes["PreCal"].videos["cam1"].path.name == "b_PreCal.mp4"
        assert takes["PreCal"].videos["cam2"].path.name == "a_PreCal.mp4"
        assert takes["PostCal"].videos["cam1"].path.name == "a_PostCal.mp4"
        assert takes["PostCal"].videos["cam2"].path.name == "z_PostCal.mp4"

    def test_take_videos_resolve_into_calibration_dir(self, tmp_path, qapp):
        from label3d.widgets.calibration_view import _build_takes

        _write_project(tmp_path, {
            "PreCal": {"cam1": "p1.mp4", "cam2": "p2.mp4"},
        })
        project = Project.load(tmp_path)
        take = _build_takes(project)[0]
        for cam, tv in take.videos.items():
            assert tv.path.parent.name == cam
            assert tv.path.parent.parent.name == "calibration"
            assert tv.path.is_file()

    def test_missing_camera_leaves_take_incomplete(self, tmp_path, qapp):
        """A camera absent from a take must not shift another camera's video in."""
        from label3d.widgets.calibration_view import _build_takes

        header = ["experiment", "trial", "is_calibration"]
        for cam in CAMERAS:
            header += [cam, f"{cam}_start", f"{cam}_end"]
        (tmp_path / "trials.csv").write_text(
            ",".join(header) + "\n"
            "exp1,trial1,False,cam1_trial1.mp4,0,,cam2_trial1.mp4,0,\n"
            "exp1,PreCal,True,pre1.mp4,0,,,,\n"
        )
        videos = tmp_path / "experiments" / "exp1" / "videos"
        videos.mkdir(parents=True)
        for cam in CAMERAS:
            (videos / f"{cam}_trial1.mp4").write_bytes(b"\x00" * 32)
        cam_dir = tmp_path / "experiments" / "exp1" / "calibration" / "cam1"
        cam_dir.mkdir(parents=True)
        (cam_dir / "pre1.mp4").write_bytes(b"\x00" * 32)

        project = Project.load(tmp_path)
        take = _build_takes(project)[0]
        assert set(take.videos) == {"cam1"}


class TestCornersIO:
    def _rows(self, take: str) -> list[dict]:
        return [
            {
                "framenum": (take, frame),
                "ids": np.array([[0], [1]], dtype=np.int32),
                "corners": np.array(
                    [[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32
                ),
            }
            for frame in (0, 20)
        ]

    def test_per_take_file_naming(self, tmp_path):
        assert corners_path(tmp_path, "PreCal").name == "corners_PreCal.csv"
        assert corners_path(tmp_path).name == "corners.csv"

    def test_round_trip_preserves_take_key(self, tmp_path):
        save_corners(tmp_path, self._rows("PreCal"), take="PreCal")
        assert (tmp_path / "corners_PreCal.csv").is_file()

        loaded = load_corners(tmp_path, take="PreCal")
        assert [r["framenum"] for r in loaded] == [("PreCal", 0), ("PreCal", 20)]
        np.testing.assert_allclose(loaded[0]["corners"].reshape(-1, 2),
                                   [[1.0, 2.0], [3.0, 4.0]])

    def test_takes_do_not_collide(self, tmp_path):
        save_corners(tmp_path, self._rows("PreCal"), take="PreCal")
        save_corners(tmp_path, self._rows("PostCal"), take="PostCal")
        joint = self._rows("PreCal") + self._rows("PostCal")
        save_corners(tmp_path, joint)

        assert [r["framenum"] for r in load_corners(tmp_path, take="PostCal")] == [
            ("PostCal", 0), ("PostCal", 20),
        ]
        keys = [r["framenum"] for r in load_corners(tmp_path)]
        assert keys == [("PreCal", 0), ("PreCal", 20), ("PostCal", 0), ("PostCal", 20)]

    def test_merge_rows_pairs_by_take(self, tmp_path):
        """Two cameras' rows merge only where take AND frame agree."""
        from aniposelib.boards import merge_rows

        cam1 = self._rows("PreCal") + self._rows("PostCal")
        cam2 = self._rows("PreCal")
        merged = merge_rows([cam1, cam2], cam_names=["cam1", "cam2"])

        both = [m for m in merged if len(m) == 2]
        only1 = [m for m in merged if set(m) == {"cam1"}]
        assert len(both) == 2   # the two PreCal frames
        assert len(only1) == 2  # PostCal, which cam2 never recorded

    def test_missing_file_returns_none(self, tmp_path):
        assert load_corners(tmp_path, take="Nope") is None


# --- Drift comparison between takes ---


class _FakeCamera:
    def __init__(self, name, rvec, tvec):
        self._name, self._r, self._t = name, np.asarray(rvec, float), np.asarray(tvec, float)

    def get_name(self):
        return self._name

    def get_rotation(self):
        return self._r

    def get_translation(self):
        return self._t


class _FakeGroup:
    def __init__(self, cameras):
        self.cameras = cameras


def _group(specs):
    return _FakeGroup([_FakeCamera(n, r, t) for n, r, t in specs])


BASE = [
    ("cam1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ("cam2", [0.0, 0.5, 0.0], [-100.0, 0.0, 50.0]),
]


class TestDriftComparison:
    def test_identical_calibrations_have_no_drift(self):
        from label3d.calibration_compare import compare_calibrations

        drifts = compare_calibrations(_group(BASE), _group(BASE))
        assert len(drifts) == 1
        assert drifts[0].rotation_deg == pytest.approx(0.0, abs=1e-9)
        assert drifts[0].baseline_pct == pytest.approx(0.0, abs=1e-9)

    def test_rotated_camera_reports_rotation(self):
        from label3d.calibration_compare import compare_calibrations

        moved = list(BASE)
        # Rotate cam2 by 2 degrees about y, on top of its existing 0.5 rad.
        moved[1] = ("cam2", [0.0, 0.5 + np.radians(2.0), 0.0], [-100.0, 0.0, 50.0])

        drift = compare_calibrations(_group(BASE), _group(moved))[0]
        assert drift.rotation_deg == pytest.approx(2.0, abs=1e-6)

    def test_moved_camera_reports_baseline_change(self):
        from label3d.calibration_compare import compare_calibrations

        moved = list(BASE)
        moved[1] = ("cam2", [0.0, 0.5, 0.0], [-110.0, 0.0, 50.0])

        drift = compare_calibrations(_group(BASE), _group(moved))[0]
        assert drift.baseline_other_cm != pytest.approx(drift.baseline_ref_cm)
        assert drift.baseline_delta_cm > 0

    def test_warnings_fire_only_past_threshold(self):
        from label3d.calibration_compare import compare_calibrations, drift_warnings

        nudged = list(BASE)
        nudged[1] = ("cam2", [0.0, 0.5 + np.radians(0.1), 0.0], [-100.0, 0.0, 50.0])
        small = compare_calibrations(_group(BASE), _group(nudged))
        assert drift_warnings("PreCal", small, 0.5, 1.0) == []

        bumped = list(BASE)
        bumped[1] = ("cam2", [0.0, 0.5 + np.radians(3.0), 0.0], [-100.0, 0.0, 50.0])
        big = compare_calibrations(_group(BASE), _group(bumped))
        messages = drift_warnings("PreCal", big, 0.5, 1.0)
        assert len(messages) == 1
        assert "PreCal" in messages[0] and "cam1-cam2" in messages[0]

    def test_thresholds_read_from_config(self):
        from label3d.calibration_compare import drift_thresholds

        assert drift_thresholds({}) == (0.5, 1.0)
        assert drift_thresholds(
            {"calibration": {"rotation_drift_deg": 2.0, "baseline_drift_pct": 5.0}}
        ) == (2.0, 5.0)

    def test_only_shared_cameras_compared(self):
        from label3d.calibration_compare import compare_calibrations

        three = BASE + [("cam3", [0.1, 0.0, 0.0], [0.0, -80.0, 40.0])]
        drifts = compare_calibrations(_group(three), _group(BASE))
        assert {(d.cam_a, d.cam_b) for d in drifts} == {("cam1", "cam2")}


# --- Browsing frames one take at a time ---


def _merged(take: str, frame: int, cams=("cam1", "cam2")) -> dict:
    return {c: {"framenum": (take, frame)} for c in cams}


class TestTakeDisplayFilter:
    def test_row_take_reads_any_camera(self):
        from label3d.widgets.calibration_view import row_take

        assert row_take(_merged("PreCal", 20)) == "PreCal"
        # A row only one camera saw still reports its take
        assert row_take({"cam2": {"framenum": ("PostCal", 40)}}) == "PostCal"
        assert row_take({}) is None

    def test_filter_selects_one_take(self):
        from label3d.widgets.calibration_view import take_row_indices

        rows = [
            _merged("PreCal", 0), _merged("PostCal", 0),
            _merged("PreCal", 20), _merged("PostCal", 20),
        ]
        assert take_row_indices(rows, "PreCal") == [0, 2]
        assert take_row_indices(rows, "PostCal") == [1, 3]

    def test_joint_shows_every_frame(self):
        from label3d.widgets.calibration_view import JOINT_TAKE, take_row_indices

        rows = [_merged("PreCal", 0), _merged("PostCal", 0)]
        assert take_row_indices(rows, JOINT_TAKE) == [0, 1]
        assert take_row_indices(rows, None) == [0, 1]
        assert take_row_indices(rows, "") == [0, 1]

    def test_indices_map_back_to_unfiltered_rows(self):
        """The map is what keeps reprojection errors aligned when filtering."""
        from label3d.widgets.calibration_view import take_row_indices

        rows = [_merged("A", 0), _merged("B", 0), _merged("B", 20), _merged("A", 20)]
        keep = take_row_indices(rows, "B")
        assert keep == [1, 2]
        assert [rows[i] for i in keep] == [rows[1], rows[2]]

    def test_unknown_take_shows_nothing(self):
        from label3d.widgets.calibration_view import take_row_indices

        assert take_row_indices([_merged("PreCal", 0)], "Missing") == []


def _write_counter_video(path: Path, n_frames: int) -> Path:
    """Grey video whose frame i has brightness 10*i, so reads reveal the index."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 24)
    )
    for i in range(n_frames):
        writer.write(np.full((24, 32, 3), 10 * i, dtype=np.uint8))
    writer.release()
    return path


def _frame_index(img: np.ndarray) -> int:
    return round(float(img.mean()) / 10)


@pytest.fixture
def offset_take(tmp_path) -> CalibrationTake:
    """cam2 started recording 3 frames before cam1, and stopped earlier.

    cam1 has synchronized frames 0-19, cam2 only 0-13.
    """
    return CalibrationTake(
        name="PreCal",
        videos={
            "cam1": TakeVideo(_write_counter_video(tmp_path / "c1.avi", 20), start=0),
            "cam2": TakeVideo(_write_counter_video(tmp_path / "c2.avi", 17), start=3),
        },
    )


class _AlwaysDetects:
    """Stand-in board that finds one corner on every frame."""

    def detect_image(self, frame):
        return np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.int32)

    def fill_points_rows(self, rows):
        return rows


class TestSynchronizedFrames:
    def test_detection_keys_rows_by_synchronized_frame(self, offset_take, qapp):
        worker = DetectionWorker(["cam1", "cam2"], {}, [offset_take], skip=5)
        board = _AlwaysDetects()
        frames = {
            cam: [r["framenum"] for r in worker._detect_video(board, tv, offset_take)]
            for cam, tv in offset_take.videos.items()
        }
        assert frames["cam1"] == [("PreCal", s) for s in (0, 5, 10, 15)]
        # cam2 reads video frames 3, 8, 13 — the same instants as cam1's 0, 5, 10
        assert frames["cam2"] == [("PreCal", s) for s in (0, 5, 10)]

    def test_browse_rows_use_synchronized_frames(self, offset_take):
        merged, _, lengths = _scan_take_media([offset_take], ["cam1", "cam2"], skip=5)
        keys = [row["cam1"]["framenum"] for row in merged]
        assert keys == [("PreCal", s) for s in (0, 5, 10, 15)]
        assert "cam2" not in merged[-1]   # cam2 ends at synchronized frame 13
        assert lengths == {"PreCal": 20}  # the longest camera sets the take length

    def test_provider_reads_frame_plus_camera_offset(self, offset_take):
        provider = CalibFrameProvider()
        provider.setup([_merged("PreCal", 5)], {}, [offset_take])
        try:
            assert _frame_index(provider.get_frame("cam1", 0)) == 5
            assert _frame_index(provider.get_frame("cam2", 0)) == 8
        finally:
            provider.close()

    def test_camera_without_detection_still_shows_its_frame(self, offset_take):
        """A merged row only cam1 detected in must still show cam2's image."""
        provider = CalibFrameProvider()
        provider.setup([_merged("PreCal", 5, cams=("cam1",))], {}, [offset_take])
        try:
            assert _frame_index(provider.get_frame("cam2", 0)) == 8
        finally:
            provider.close()


class TestNearestRow:
    ROWS = [_merged("A", 0), _merged("A", 20), _merged("B", 0), _merged("B", 40)]

    def test_snaps_to_closest_frame_in_take(self):
        assert nearest_row(self.ROWS, "A", 7) == 0
        assert nearest_row(self.ROWS, "A", 13) == 1
        assert nearest_row(self.ROWS, "B", 25) == 3

    def test_ignores_other_takes(self):
        """B's frame 0 is closer to 3 than A's 20, but the slider spans A only."""
        rows = [_merged("A", 20), _merged("B", 0)]
        assert nearest_row(rows, "A", 3) == 0

    def test_no_rows_in_take(self):
        assert nearest_row(self.ROWS, "C", 0) is None
        assert nearest_row([], "A", 0) is None


class TestTakeRange:
    def test_detection_scans_only_the_range(self, offset_take, qapp):
        offset_take.start, offset_take.end = 4, 12
        worker = DetectionWorker(["cam1", "cam2"], {}, [offset_take], skip=5)
        board = _AlwaysDetects()
        for tv in offset_take.videos.values():
            frames = [r["framenum"][1] for r in worker._detect_video(board, tv, offset_take)]
            # Strides from the range start, so both cameras sample the same instants
            assert frames == [4, 9]

    def test_browse_rows_follow_the_range(self, offset_take):
        offset_take.start, offset_take.end = 4, 12
        merged, _, lengths = _scan_take_media([offset_take], ["cam1", "cam2"], skip=5)
        assert [row["cam1"]["framenum"][1] for row in merged] == [4, 9]
        assert lengths == {"PreCal": 20}  # the slider still spans the whole take

    def test_range_read_from_config(self):
        config = {"calibration": {"ranges": {"exp1": {"PreCal": {"start": 30, "end": 900}}}}}
        assert take_range(config, "exp1", "PreCal") == (30, 900)
        assert take_range(config, "exp1", "PostCal") == (0, None)
        assert take_range({}, "exp1", "PreCal") == (0, None)

    def test_build_takes_applies_saved_range(self, tmp_path, qapp):
        _write_project(tmp_path, {"PreCal": {"cam1": "p1.mp4", "cam2": "p2.mp4"}})
        (tmp_path / "project.toml").write_text(
            '[calibration.ranges.exp1.PreCal]\nstart = 10\nend = 50\n'
        )
        take = _build_takes(Project.load(tmp_path))[0]
        assert (take.start, take.end) == (10, 50)

    def test_rows_in_range(self):
        take = CalibrationTake("A", start=10, end=30)
        rows = {"cam1": [
            {"framenum": ("A", 0)}, {"framenum": ("A", 10)},
            {"framenum": ("A", 30)}, {"framenum": ("A", 31)},
            {"framenum": ("B", 0)},   # no range known for B: kept
        ]}
        kept = rows_in_range(rows, [take])["cam1"]
        assert [r["framenum"] for r in kept] == [("A", 10), ("A", 30), ("B", 0)]


class TestRekeyFrameErrors:
    def test_errors_follow_rows_after_filtering(self):
        """Position 0 of the filtered rows is frame ("A", 20), index 1 in full."""
        cams = ["cam1", "cam2"]
        full = [_merged("A", 0), _merged("A", 20), _merged("A", 40)]
        filtered = {c: [{"framenum": ("A", 20)}, {"framenum": ("A", 40)}] for c in cams}
        rekeyed = rekey_frame_errors({0: "e20", 1: "e40"}, filtered, cams, full)
        assert rekeyed == {1: "e20", 2: "e40"}


class TestSliderBounds:
    def test_bounds_clamp_to_range(self, qapp):
        slider = FrameSlider()
        slider.set_range(100)
        slider.set_bounds(-5, 500)
        assert slider.bounds() == (0, 100)
        slider.set_bounds(60, 40)          # end before start collapses onto start
        assert slider.bounds() == (60, 60)
        slider.set_range(50)               # shrinking the range re-clamps
        assert slider.bounds() == (50, 50)

    def test_dragging_a_bound_emits_without_seeking(self, qapp):
        slider = FrameSlider()
        slider.resize(212, slider.height())   # 200 px of track: 2 px per frame
        slider.set_range(100)
        slider.set_bounds(20, 80)
        seeks, bounds = [], []
        slider.frame_changed.connect(seeks.append)
        slider.bounds_changed.connect(lambda s, e: bounds.append((s, e)))

        x_start = slider._frame_to_x(20)
        assert slider._bound_at(x_start) == "start"
        slider._dragging = "start"
        slider._drag_bound_to(slider._frame_to_x(50))
        slider._drag_bound_to(slider._frame_to_x(95))   # can't pass the end marker
        assert bounds == [(50, 80), (80, 80)]
        assert seeks == []

    def test_no_bounds_means_no_grab(self, qapp):
        slider = FrameSlider()
        slider.resize(212, slider.height())
        slider.set_range(100)
        assert slider._bound_at(slider._frame_to_x(0)) is None


def _rotated(deg: float):
    """BASE with cam2 turned `deg` degrees further about y."""
    return _group([BASE[0], ("cam2", [0.0, 0.5 + np.radians(deg), 0.0], BASE[1][2])])


class TestRunSpread:
    def test_spread_is_largest_pairwise_drift(self):
        rot, base = run_spread([_group(BASE), _rotated(0.3), _rotated(1.0)])
        assert rot == pytest.approx(1.0, abs=1e-6)
        assert base == pytest.approx(0.0, abs=1e-9)

    def test_single_run_has_no_spread(self):
        assert run_spread([_group(BASE)]) == (0.0, 0.0)

    def test_thresholds_rise_above_twice_the_spread(self):
        # configured 0.5 deg / 1 %, runs disagreed by up to 1.2 deg / 0.2 %
        rot, base = noise_adjusted_thresholds(0.5, 1.0, [(1.2, 0.2), (0.4, 0.1)])
        assert rot == pytest.approx(2.4)
        assert base == pytest.approx(1.0)   # spread too small to matter

    def test_no_spread_keeps_configured_thresholds(self):
        assert noise_adjusted_thresholds(0.5, 1.0, [(0.0, 0.0)]) == (0.5, 1.0)


def _run_worker(monkeypatch, take_names, errors, n_repeats):
    """Run MultiTakeCalibrationWorker with calibrate_rows returning `errors` in turn."""
    results_iter = iter(errors)
    monkeypatch.setattr(calibration_worker, "_build_board", lambda bp: None)
    monkeypatch.setattr(
        calibration_worker,
        "calibrate_rows",
        lambda cams, sizes, rows, board: (_rotated(0.0), next(results_iter), {}),
    )
    rows = {"cam1": [{"framenum": (t, 0)} for t in take_names]}
    worker = calibration_worker.MultiTakeCalibrationWorker(
        ["cam1", "cam2"], {}, rows, {}, take_names, n_repeats=n_repeats
    )
    out = []
    worker.finished.connect(out.append)
    worker.run()
    return out[0]


class TestRepeatedCalibration:
    def test_keeps_lowest_error_run(self, monkeypatch, qapp):
        results = _run_worker(monkeypatch, ["PreCal"], [0.9, 0.4, 0.7], n_repeats=3)
        assert [r.name for r in results] == ["PreCal"]
        assert results[0].error == pytest.approx(0.4)

    def test_single_take_has_no_joint_solution(self, monkeypatch, qapp):
        results = _run_worker(monkeypatch, ["PreCal"], [0.5], n_repeats=1)
        assert [r.name for r in results] == ["PreCal"]

    def test_several_takes_add_joint_last(self, monkeypatch, qapp):
        results = _run_worker(
            monkeypatch, ["PreCal", "PostCal"], [0.5, 0.6, 0.3, 0.2, 0.4, 0.5], n_repeats=2
        )
        assert [r.name for r in results] == ["PreCal", "PostCal", JOINT_TAKE]
        assert [round(r.error, 1) for r in results] == [0.5, 0.2, 0.4]

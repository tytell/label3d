"""Tests for calibration refinement with labelled points (Phase 12)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from aniposelib.cameras import Camera, CameraGroup

from label3d.calibration_compare import compare_calibrations
from label3d.calibration_io import save_corners
from label3d.calibration_refine import (
    PointSelection,
    RefineOptions,
    align_to,
    apply_similarity,
    charuco_points,
    charuco_rows,
    count_points,
    error_stats,
    frames_by_trial,
    gather_points,
    holdout_frames,
    holdout_mask,
    load_points_record,
    matching_charuco_entry,
    point_errors,
    refine,
    run_stage2,
    save_points_record,
    save_refinement,
    usable_mask,
)
from label3d.calibration_registry import (
    REFINED_DIR,
    camera_dir,
    find_calibration,
    list_calibrations,
    load_calibration,
    save_calibration,
)
from label3d.calibration_solve import CalibrationTake, take_detections, take_ranges
from label3d.points import INDEX_NAMES, PointsStore
from label3d.project import Project
from label3d.widgets.refine_dialog import RefineDialog
from label3d.workers.calibration_worker import RefineWorker
from synthetic import build_cameras, load_config
from tests.conftest import requires_test_project, TEST_PROJECT

CAMS = ["cam1", "cam2", "cam3"]


def _points_df(rows: list[dict]) -> pd.DataFrame:
    """A points DataFrame from {trial, frame, bodypart, cam: (x, y, state, source, score)}."""
    df = PointsStore.empty_df(CAMS)
    records = {}
    for r in rows:
        key = ("exp1", r["trial"], r["frame"], 0, r["bodypart"])
        rec = {col: np.nan for col in df.columns}
        for cam in CAMS:
            rec[(cam, "state")] = "not_set"
            rec[(cam, "source")] = ""
            if cam in r:
                x, y, state, source, score = r[cam]
                rec.update({
                    (cam, "x"): x, (cam, "y"): y, (cam, "state"): state,
                    (cam, "source"): source, (cam, "score"): score,
                })
        records[key] = rec
    out = pd.DataFrame.from_dict(records, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=INDEX_NAMES)
    out.columns = df.columns
    return out


def _selection_df() -> pd.DataFrame:
    user = lambda x: (x, x + 1, "set", "label3d", np.nan)          # noqa: E731
    sleap_user = lambda x: (x, x + 1, "set", "sleap_user", np.nan)  # noqa: E731
    pred = lambda x, s: (x, x + 1, "predicted", "sleap_pred", s)    # noqa: E731
    guess = lambda x: (x, x + 1, "predicted", "label3d", np.nan)    # noqa: E731
    return _points_df([
        # three label3d user points
        {"trial": "t1", "frame": 0, "bodypart": "a", "cam1": user(1), "cam2": user(2), "cam3": user(3)},
        # one user point plus a label3d reprojection guess: only 1 camera counts
        {"trial": "t1", "frame": 0, "bodypart": "b", "cam1": user(4), "cam2": guess(5)},
        # SLEAP predictions, one above and one below the threshold
        {"trial": "t1", "frame": 1, "bodypart": "a", "cam1": pred(6, 0.9), "cam2": pred(7, 0.5)},
        # SLEAP user labels in two cameras, another trial
        {"trial": "t2", "frame": 3, "bodypart": "a", "cam2": sleap_user(8), "cam3": sleap_user(9)},
    ])


class TestPointSelection:
    def test_sources_and_min_cameras(self):
        df = _selection_df()
        sel = PointSelection(sources={"label3d", "sleap_user"})
        p2ds, index = gather_points(df, CAMS, sel)
        assert list(index) == [("t1", 0, 0, "a"), ("t2", 3, 0, "a")]
        assert p2ds.shape == (3, 2, 2)
        np.testing.assert_allclose(p2ds[:, 0, 0], [1, 2, 3])
        assert np.isnan(p2ds[0, 1, 0]) and p2ds[1, 1, 0] == 8

    def test_reprojection_guesses_left_out(self):
        df = _selection_df()
        mask = usable_mask(df, CAMS, PointSelection(sources={"label3d"}, min_cameras=1))
        assert mask.loc[("t1", 0, 0, "b"), "cam1"]
        assert not mask.loc[("t1", 0, 0, "b"), "cam2"]

    def test_score_threshold(self):
        df = _selection_df()
        sel = PointSelection(sources={"sleap_pred"}, score_min=0.8, min_cameras=1)
        p2ds, index = gather_points(df, CAMS, sel)
        assert list(index) == [("t1", 1, 0, "a")]
        assert p2ds[0, 0, 0] == 6 and np.isnan(p2ds[1, 0, 0])
        sel.score_min = 0.4
        p2ds, _ = gather_points(df, CAMS, sel)
        assert p2ds[1, 0, 0] == 7

    def test_min_cameras_three(self):
        df = _selection_df()
        sel = PointSelection(sources={"label3d", "sleap_user"}, min_cameras=3)
        _, index = gather_points(df, CAMS, sel)
        assert list(index) == [("t1", 0, 0, "a")]

    def test_trials_filter(self):
        df = _selection_df()
        sel = PointSelection(sources={"label3d", "sleap_user"}, trials=["t2"])
        _, index = gather_points(df, CAMS, sel)
        assert list(index) == [("t2", 3, 0, "a")]

    def test_count_points(self):
        df = _selection_df()
        counts = count_points(df, CAMS, PointSelection(sources={"label3d", "sleap_user"}))
        assert counts == {("cam1", "cam2"): 1, ("cam1", "cam3"): 1, ("cam2", "cam3"): 2}


class TestHoldout:
    def _index(self):
        keys = [("t1", f, 0, bp) for f in range(20) for bp in ("a", "b")]
        keys += [("t2", f, 0, "a") for f in range(10)]
        return pd.MultiIndex.from_tuples(keys)

    def test_split_by_frame(self):
        index = self._index()
        frames = holdout_frames(index, 0.2, seed=7)
        assert len(frames) == 6   # 20% of 30 frames
        mask = holdout_mask(index, frames)
        # both bodyparts of a held-out frame are held out together
        for t, f, _, bp in index:
            if t == "t1":
                assert mask[list(index).index((t, f, 0, "a"))] == mask[list(index).index((t, f, 0, "b"))]

    def test_reproducible(self):
        index = self._index()
        assert holdout_frames(index, 0.2, seed=7) == holdout_frames(index, 0.2, seed=7)
        assert holdout_frames(index, 0.2, seed=7) != holdout_frames(index, 0.2, seed=8)

    def test_edge_cases(self):
        index = self._index()
        assert holdout_frames(index, 0.0, seed=1) == set()
        one = pd.MultiIndex.from_tuples([("t1", 0, 0, "a")])
        assert holdout_frames(one, 0.5, seed=1) == set()
        two = pd.MultiIndex.from_tuples([("t1", 0, 0, "a"), ("t1", 1, 0, "a")])
        assert len(holdout_frames(two, 0.9, seed=1)) == 1

    def test_frames_by_trial(self):
        assert frames_by_trial(self._index())["t2"] == list(range(10))


class TestPointsRecord:
    def test_round_trip(self, tmp_path):
        df = _selection_df()
        sel = PointSelection(sources={"label3d", "sleap_user", "sleap_pred"}, score_min=0.8, min_cameras=1)
        p2ds, index = gather_points(df, CAMS, sel)
        holdout = np.array([False, True, False, True][: len(index)])
        path = tmp_path / "x_points.csv"
        save_points_record(path, p2ds, index, CAMS, df, holdout)

        record = pd.read_csv(path)
        assert list(record.columns) == [
            "trial", "frame", "instance", "bodypart", "camera", "x", "y", "source", "score", "holdout",
        ]
        assert set(record["source"]) == {"label3d", "sleap_user", "sleap_pred"}

        p2ds_back, index_back, holdout_back = load_points_record(path, CAMS)
        assert list(index_back) == list(index)
        np.testing.assert_array_equal(holdout_back, holdout)
        np.testing.assert_allclose(p2ds_back, p2ds, equal_nan=True)


# -- Refinement pipeline -------------------------------------------------------

SYNTHETIC_CONFIG = Path(__file__).parent.parent / "synthetic_config.toml"


def _true_rig(names=("cam1", "cam2", "cam3")) -> CameraGroup:
    """The synthetic generator's camera rig."""
    cams = build_cameras(load_config(str(SYNTHETIC_CONFIG)))
    return CameraGroup([
        Camera(name=n, size=tuple(cams[n]["resolution"]), matrix=cams[n]["K"],
               dist=cams[n]["dist"], rvec=cams[n]["rvec"], tvec=cams[n]["t"])
        for n in names
    ])


def _animal_points(rig: CameraGroup, n_frames=150, n_parts=4, noise_px=0.3, seed=11):
    """A moving 'animal': n_parts points per frame near the origin, projected into rig."""
    rng = np.random.default_rng(seed)
    centre = np.cumsum(rng.normal(0, 0.4, size=(n_frames, 3)), axis=0)
    centre = np.clip(centre, -6, 6)
    offsets = rng.normal(0, 2.0, size=(n_parts, 3))
    p3ds = (centre[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    p2ds = np.array([cam.project(p3ds).reshape(-1, 2) for cam in rig.cameras])
    p2ds += rng.normal(0, noise_px, size=p2ds.shape)
    keys = [("t1", f, 0, f"bp{b}") for f in range(n_frames) for b in range(n_parts)]
    return p2ds, pd.MultiIndex.from_tuples(keys, names=["trial", "frame", "instance", "bodypart"])


def _perturbed(rig: CameraGroup, cam=1, rot_deg=1.0, shift_cm=1.0) -> CameraGroup:
    """rig with one camera turned and moved a little."""
    out = rig.copy()
    c = out.cameras[cam]
    R = cv2.Rodrigues(np.asarray(c.get_rotation(), dtype=float))[0]
    dR = cv2.Rodrigues(np.radians([rot_deg, -rot_deg / 2, 0.0]))[0]
    c.set_rotation(cv2.Rodrigues(dR @ R)[0].ravel())
    c.set_translation(np.asarray(c.get_translation(), dtype=float) + [shift_cm, 0, -shift_cm])
    return out


def _max_rotation_drift(a: CameraGroup, b: CameraGroup) -> float:
    return max(d.rotation_deg for d in compare_calibrations(a, b))


class TestRecovery:
    def test_moves_perturbed_camera_toward_truth(self):
        truth = _true_rig()
        start = _perturbed(truth)
        p2ds, index = _animal_points(truth)
        options = RefineOptions(n_repeats=2, seed=5)
        result = refine(start, p2ds, index, options)

        assert _max_rotation_drift(truth, result.cgroup) < 0.5 * _max_rotation_drift(truth, start)
        assert result.errors["holdout"] < result.errors["holdout_before"]
        assert result.errors["holdout"] < 1.0
        assert result.holdout.any() and not result.holdout.all()
        assert abs(result.scale - 1.0) < 0.05

    def test_charuco_errors_reported(self):
        truth = _true_rig()
        start = _perturbed(truth)
        p2ds, index = _animal_points(truth, n_frames=60)
        corners, _ = _animal_points(truth, n_frames=10, seed=3, noise_px=0.1)
        result = refine(start, p2ds, index, RefineOptions(n_repeats=1, seed=2), charuco_p2ds=corners)
        assert result.errors["charuco"] < result.errors["charuco_before"]

    def test_no_points_raises(self):
        truth = _true_rig()
        empty = np.full((3, 0, 2), np.nan)
        with pytest.raises(ValueError):
            refine(truth, empty, pd.MultiIndex.from_tuples([], names=["trial", "frame", "instance", "bodypart"]),
                   RefineOptions(n_repeats=1, seed=1))


class TestAlignment:
    def _scaled_moved(self, rig: CameraGroup) -> CameraGroup:
        out = rig.copy()
        R = cv2.Rodrigues(np.radians([10.0, -5.0, 20.0]))[0]
        apply_similarity(out, 0.7, R, np.array([3.0, -2.0, 5.0]))
        return out

    @pytest.mark.parametrize("names", [("cam1", "cam2", "cam3"), ("cam1", "cam2")])
    def test_recovers_reference_frame(self, names):
        rig = _true_rig(names)
        moved = self._scaled_moved(rig)
        scale = align_to(moved, rig)
        assert abs(scale - 1 / 0.7) < 1e-6
        for a, b in zip(rig.cameras, moved.cameras):
            np.testing.assert_allclose(b.get_translation(), a.get_translation(), atol=1e-6)
            np.testing.assert_allclose(
                cv2.Rodrigues(np.asarray(b.get_rotation(), dtype=float))[0],
                cv2.Rodrigues(np.asarray(a.get_rotation(), dtype=float))[0], atol=1e-6)

    def test_similarity_keeps_projections(self):
        rig = _true_rig()
        p2ds, _ = _animal_points(rig, n_frames=5, noise_px=0.0)
        moved = self._scaled_moved(rig)
        assert error_stats(point_errors(moved, p2ds))[0] < 1e-6

    def test_keeps_baselines(self):
        truth = _true_rig()
        start = _perturbed(truth, rot_deg=0.5, shift_cm=0.5)
        p2ds, index = _animal_points(truth, n_frames=80)
        result = refine(start, p2ds, index, RefineOptions(n_repeats=1, seed=9))
        for d in compare_calibrations(start, result.cgroup):
            assert abs(d.baseline_pct) < 3.0


class TestSaving:
    def _result(self):
        truth = _true_rig()
        start = _perturbed(truth)
        p2ds, index = _animal_points(truth, n_frames=60)
        options = RefineOptions(n_repeats=2, seed=13)
        return start, p2ds, index, options, refine(start, p2ds, index, options)

    def _points_df(self, p2ds, index):
        """A points DataFrame holding p2ds as label3d user points."""
        df = PointsStore.empty_df(CAMS)
        records = {}
        for n, (t, f, i, bp) in enumerate(index):
            rec = {col: np.nan for col in df.columns}
            for c, cam in enumerate(CAMS):
                rec.update({(cam, "x"): p2ds[c, n, 0], (cam, "y"): p2ds[c, n, 1],
                            (cam, "state"): "set", (cam, "source"): "label3d"})
            records[("exp1", t, f, i, bp)] = rec
        out = pd.DataFrame.from_dict(records, orient="index")
        out.index = pd.MultiIndex.from_tuples(out.index, names=INDEX_NAMES)
        out.columns = df.columns
        return out

    def test_metadata_and_reproducible(self, tmp_path):
        start, p2ds, index, options, result = self._result()
        parent = save_calibration(tmp_path, start, "charuco-all", "charuco",
                                  charuco_takes=["PreCal"])
        sel = PointSelection(sources={"label3d"})
        points_df = self._points_df(p2ds, index)
        entry = save_refinement(tmp_path, result, parent, "refine", sel, options, points_df)

        assert entry.path.parent.name == REFINED_DIR
        assert entry.parent == parent.id
        assert entry.charuco_takes == ["PreCal"]
        points = entry.metadata["points"]
        assert points["sources"] == ["label3d"]
        assert points["score_min"] == 0.0
        assert points["frames"]["t1"] == list(range(60))
        assert points["points_file"] == f"{entry.id}_points.csv"
        assert entry.metadata["options"]["seed"] == 13
        assert find_calibration(tmp_path, entry.id) is not None

        # Rerun Stage 2 from the parent and the record with the stored seed.
        record_p2ds, record_index, holdout = load_points_record(
            entry.path.parent / points["points_file"], CAMS)
        assert list(record_index) == list(index)
        np.testing.assert_array_equal(holdout, result.holdout)
        opts = entry.metadata["options"]
        rerun, _ = run_stage2(
            load_calibration(parent), record_p2ds[:, ~holdout],
            RefineOptions(only_extrinsics=opts["only_extrinsics"],
                          n_samp_full=opts["n_samp_full"], n_samp_iter=opts["n_samp_iter"]),
            opts["seed"] + opts["kept_repeat"],
        )
        saved = load_calibration(entry)
        for a, b in zip(saved.cameras, rerun.cameras):
            np.testing.assert_allclose(a.get_rotation(), b.get_rotation(), atol=1e-6)
            np.testing.assert_allclose(a.get_translation(), b.get_translation(), atol=1e-4)


class TestStage1:
    def _corners(self, exp_dir: Path, cam_names: list[str]) -> None:
        for c, cam in enumerate(cam_names):
            cam_dir = camera_dir(exp_dir, cam)
            cam_dir.mkdir(parents=True)
            rows = [
                {"framenum": (take, f), "ids": np.array([[0], [1]]),
                 "corners": np.array([[[10.0 * c + f, 5.0]], [[20.0, 5.0 + f]]])}
                for take in ("PreCal", "PostCal") for f in (0, 10, 20)
            ]
            save_corners(cam_dir, rows, "2026-10-01T10:00:00")

    def test_charuco_rows_and_points(self, tmp_path):
        self._corners(tmp_path, ["cam1", "cam2"])
        takes = [CalibrationTake("PreCal", start=0, end=10), CalibrationTake("PostCal")]
        rows = charuco_rows(tmp_path, ["cam1", "cam2"], takes, ["PreCal"])
        assert [r["framenum"] for r in rows["cam1"]] == [("PreCal", 0), ("PreCal", 10)]
        p2ds = charuco_points(rows, ["cam1", "cam2"])
        assert p2ds.shape == (2, 4, 2)   # 2 frames x 2 corners, both cameras

    def test_matching_entry(self, tmp_path):
        cams = ["cam1", "cam2"]
        self._corners(tmp_path, cams)
        takes = [CalibrationTake("PreCal", start=0, end=10), CalibrationTake("PostCal")]
        rig = _true_rig(tuple(cams))
        meta = {"charuco": {"detected": take_detections(tmp_path, cams, ["PreCal"]),
                            "take_ranges": take_ranges(takes, ["PreCal"])}}
        entry = save_calibration(tmp_path, rig, "charuco-PreCal", "charuco",
                                 charuco_takes=["PreCal"], **meta)
        entries = list_calibrations(tmp_path)
        assert matching_charuco_entry(entries, ["PreCal"], takes, tmp_path, cams).id == entry.id
        assert matching_charuco_entry(entries, ["PreCal", "PostCal"], takes, tmp_path, cams) is None
        takes[0].end = 20   # range changed since the solve
        assert matching_charuco_entry(entries, ["PreCal"], takes, tmp_path, cams) is None


class TestRefineWorker:
    def test_points_follow_calibration_camera_order(self, tmp_path, qapp):
        truth = _true_rig(("cam3", "cam1", "cam2"))   # not the project's order
        start = _perturbed(truth, cam=0)
        entry = save_calibration(tmp_path, start, "imported", "imported")
        p2ds, index = _animal_points(truth, n_frames=60)
        project_order = ["cam1", "cam2", "cam3"]
        p2ds_project = p2ds[[1, 2, 0]]

        worker = RefineWorker(
            tmp_path, project_order, p2ds_project, index,
            RefineOptions(n_repeats=1, seed=4), start_entry=entry,
        )
        out, errors = [], []
        worker.finished.connect(out.append)
        worker.error_occurred.connect(errors.append)
        worker.run()

        assert errors == []
        outcome = out[0]
        assert outcome.parent.id == entry.id and outcome.stage1 is None
        assert outcome.result.cgroup.get_names() == ["cam3", "cam1", "cam2"]
        assert outcome.result.errors["holdout"] < outcome.result.errors["holdout_before"]

    def test_reports_missing_corners(self, tmp_path, qapp):
        p2ds, index = _animal_points(_true_rig(), n_frames=10)
        worker = RefineWorker(
            tmp_path, ["cam1", "cam2", "cam3"], p2ds, index, RefineOptions(n_repeats=1, seed=1),
            takes=[CalibrationTake("PreCal")], take_names=["PreCal"], board_params={},
        )
        errors = []
        worker.error_occurred.connect(errors.append)
        worker.run()
        assert errors and "corners" in errors[0]


@requires_test_project
class TestRefineDialog:
    def test_run_and_save_active(self, tmp_path, qapp):
        root = tmp_path / "project"
        shutil.copytree(TEST_PROJECT, root)
        project = Project.load(root)
        parent_id = project.active_calibration_id()
        dialog = RefineDialog(project, [], {}, 20, 1)
        assert dialog._entry_radio.isChecked()   # no takes: start from an entry
        assert dialog._counts_table.rowCount() == 3

        dialog._repeats_spin.setValue(1)
        dialog._on_run()
        while dialog._worker.isRunning():
            qapp.processEvents()
            time.sleep(0.02)
        qapp.processEvents()
        assert dialog._save_active_btn.isEnabled()

        dialog._on_save(make_active=True)
        entry = dialog.saved_entry
        assert entry is not None and entry.parent == parent_id
        assert project.active_calibration_id() == entry.id
        assert (entry.path.parent / entry.metadata["points"]["points_file"]).exists()

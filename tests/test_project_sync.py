"""Tests for the sync round trip: read SLP -> merge -> write SLP (Phase 8.10)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import toml

from aniposelib.cameras import CameraGroup

from label3d.calibration_registry import save_calibration
from label3d.geometry import load_cameras
from label3d.points import CAM_COORDS, INDEX_NAMES, THREE_D_COORDS, PointsStore
from label3d.project import Project
from label3d.skeleton import Skeleton
from label3d.workers.triangulation_worker import AllExperimentsTriangulationWorker

sleap_io = pytest.importorskip("sleap_io", reason="sleap_io not installed")

EXP = "exp1"
CAMERAS = ["cam0", "cam1"]
BODYPARTS = ["head", "tail"]
TRIALS = ["t1", "t2"]


def _points_df() -> pd.DataFrame:
    """One user instance per trial, labeled in both cameras on frame 0."""
    col_tuples = [(c, coord) for c in CAMERAS for coord in CAM_COORDS]
    col_tuples += [("3d", coord) for coord in THREE_D_COORDS]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["view", "coord"])

    rows, index = [], []
    for t_i, trial in enumerate(TRIALS):
        for b_i, bp in enumerate(BODYPARTS):
            index.append((EXP, trial, 0, 0, bp))
            vals = []
            for c_i, _cam in enumerate(CAMERAS):
                x = 10.0 + 100 * t_i + 10 * b_i + c_i
                y = 20.0 + 100 * t_i + 10 * b_i + c_i
                vals += [x, y, np.nan, np.nan, "set", "label3d", np.nan, np.nan]
            vals += [np.nan, np.nan, np.nan]
            rows.append(vals)
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=INDEX_NAMES),
                        columns=columns)


@pytest.fixture
def project(tmp_path: Path, qapp) -> Project:
    """A minimal project: 2 trials x 2 cameras, one SLP file per camera."""
    root = tmp_path / "proj"
    exp_dir = root / "experiments" / EXP
    (exp_dir / "videos").mkdir(parents=True)
    (root / "slp").mkdir()

    rows = []
    for trial in TRIALS:
        row = {"experiment": EXP, "trial": trial}
        for cam in CAMERAS:
            row[cam] = f"{cam}_{trial}.mp4"
            row[f"{cam}_start"] = 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "trials.csv", index=False)

    (root / "skeleton.json").write_text(json.dumps({
        "nodes": BODYPARTS, "edges": [["head", "tail"]], "symmetries": [],
        "node_offsets_cm": {},
    }))

    config = {"project": {"name": "test"}, "slp": {"files": [
        {"path": f"slp/{cam}.slp",
         "blocks": [{"experiment": EXP, "trials": list(TRIALS), "cameras": [cam]}]}
        for cam in CAMERAS
    ]}}
    (root / "project.toml").write_text(toml.dumps(config))

    _points_df().to_csv(exp_dir / "points.csv")
    return Project.load(root)


class TestSyncRoundTrip:
    def test_mapping_is_complete(self, project):
        assert project.slp_mapping.is_complete(project.trials_df, project.cameras)

    def test_write_creates_every_mapped_file(self, project):
        assert project.write_slp_files() == 2
        assert sorted(p.name for p in project.slp_dir.glob("*.slp")) == ["cam0.slp", "cam1.slp"]

    def test_missing_file_is_read_as_empty(self, project):
        """A mapped file that does not exist yet reads as empty, not an error."""
        results = project.read_slp_for_sync()
        assert results[EXP].conflicts == []
        assert results[EXP].n_new == 0

    def test_round_trip_produces_no_changes(self, project):
        project.write_slp_files()
        results = project.read_slp_for_sync()
        assert results[EXP].conflicts == []
        assert results[EXP].changed_frames == set()
        assert results[EXP].n_new == 0
        assert results[EXP].n_predictions == 0

    def test_label_added_in_sleap_comes_back_as_new(self, project):
        """Clearing a camera's points locally means the SLP copy reads as new."""
        project.write_slp_files()

        df = project.points.df.copy()
        df[("cam1", "x")] = np.nan
        df[("cam1", "y")] = np.nan
        df[("cam1", "state")] = "not_set"
        project.points.replace_df(df)
        project.points.to_csv(project.experiments[EXP].root / "points.csv")

        results = project.read_slp_for_sync()
        assert results[EXP].n_new == len(TRIALS) * len(BODYPARTS)
        assert results[EXP].conflicts == []

        project.apply_merge(results, overwrite=False)
        assert project.points.df[("cam1", "state")].tolist() == ["set"] * 4

    def test_edited_point_is_a_conflict(self, project):
        project.write_slp_files()

        df = project.points.df.copy()
        df.loc[(EXP, "t1", 0, 0, "head"), ("cam0", "x")] = 999.0
        project.points.replace_df(df)
        project.points.to_csv(project.experiments[EXP].root / "points.csv")

        results = project.read_slp_for_sync()
        assert len(results[EXP].conflicts) == 1
        assert results[EXP].conflicts[0][:5] == (EXP, "t1", 0, 0, "head")

        # Keep label3d: the local value survives
        project.apply_merge(results, overwrite=False)
        assert project.points.df.loc[(EXP, "t1", 0, 0, "head"), ("cam0", "x")] == 999.0

    def test_overwrite_takes_the_sleap_value(self, project):
        project.write_slp_files()
        original = project.points.df.loc[(EXP, "t1", 0, 0, "head"), ("cam0", "x")]

        df = project.points.df.copy()
        df.loc[(EXP, "t1", 0, 0, "head"), ("cam0", "x")] = 999.0
        project.points.replace_df(df)
        project.points.to_csv(project.experiments[EXP].root / "points.csv")

        results = project.read_slp_for_sync()
        project.apply_merge(results, overwrite=True)
        assert project.points.df.loc[(EXP, "t1", 0, 0, "head"), ("cam0", "x")] == original


# ---------------------------------------------------------------------------
# Triangulate all experiments (Phase 8.10)
# ---------------------------------------------------------------------------

def _activate_calibration(project: Project, shift: float = 0.0) -> Path:
    """Two cameras 20 cm apart looking down +z, saved to the registry and made
    active. shift moves the second camera along x (a perturbed calibration)."""
    cal = {}
    for i, cam in enumerate(CAMERAS):
        cal[f"cam_{i}"] = {
            "name": cam,
            "size": [640, 480],
            "matrix": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "translation": [(-20.0 - shift) * i, 0.0, 100.0],
        }
    tmp = project.root / "cal.tmp.toml"
    tmp.write_text(toml.dumps(cal))
    cgroup = CameraGroup.load(str(tmp))
    tmp.unlink()
    entry = save_calibration(
        project.experiments[EXP].root, cgroup, f"test-{shift:g}", "charuco"
    )
    project.set_active_calibration(EXP, entry.id)
    return entry.path


class TestTriangulateAllExperiments:
    def test_worker_fills_3d_and_store_saves(self, project, tmp_path):
        from label3d.geometry import load_cameras
        from label3d.workers.triangulation_worker import (
            AllExperimentsTriangulationWorker,
            _labeled_trials,
        )

        calib_path = _activate_calibration(project)
        assert project.calibration_path(EXP) == calib_path

        store = project.points
        assert _labeled_trials(store) == list(TRIALS)
        assert store.df[("3d", "x")].isna().all()

        worker = AllExperimentsTriangulationWorker([(EXP, load_cameras(calib_path), store)])
        worker.run()  # run synchronously — no event loop needed

        assert store.df[("3d", "x")].notna().all()
        assert store.changed
        store.save_if_changed()
        saved = PointsStore.read_csv_df(project.experiments[EXP].root / "points.csv")
        assert saved[("3d", "x")].notna().all()


class TestSwitchCalibration:
    """Switching the active calibration changes the triangulated points (Phase 11.5)."""

    def _triangulate(self, project):
        cameras = load_cameras(project.calibration_path(EXP))
        worker = AllExperimentsTriangulationWorker(
            [(EXP, cameras, project.points)], first_trial="t2"
        )
        worker.run()
        assert worker.completed
        return project.points.df[[("cam0", "xr"), ("cam1", "xr"), ("3d", "x")]].copy()

    def test_perturbed_calibration_gives_different_reprojections(self, project):
        first = _activate_calibration(project)
        before = self._triangulate(project)
        project.set_triangulated_with(EXP, project.active_calibration_id(EXP))
        assert not project.triangulation_is_stale(EXP)

        _activate_calibration(project, shift=2.0)
        assert project.triangulation_is_stale(EXP)
        after = self._triangulate(project)
        assert not np.allclose(before.to_numpy(float), after.to_numpy(float))

        # Switching back reproduces the first result
        project.set_active_calibration(EXP, first.stem)
        again = self._triangulate(project)
        np.testing.assert_allclose(again.to_numpy(float), before.to_numpy(float))

    def test_interrupted_run_is_not_complete(self, project, monkeypatch):
        _activate_calibration(project)
        worker = AllExperimentsTriangulationWorker(
            [(EXP, load_cameras(project.calibration_path(EXP)), project.points)]
        )
        monkeypatch.setattr(worker, "isInterruptionRequested", lambda: True)
        worker.run()
        assert not worker.completed

    def test_pointer_writes_keep_calibration_settings(self, project):
        project.save_config_sections({"calibration": {"frame_skip": 7, "repeats": 2}})
        _activate_calibration(project)
        project.set_triangulated_with(EXP, "x")
        config = toml.load(project.root / "project.toml")
        assert config["calibration"]["frame_skip"] == 7
        assert config["calibration"]["repeats"] == 2
        assert config["calibration"]["triangulated_with"][EXP] == "x"
        assert config["calibration"]["active"][EXP] == project.active_calibration_id(EXP)


class TestSetSkeleton:
    def test_kept_and_added_nodes_delete_nothing(self, project):
        new = Skeleton(["tail", "head", "fin"], edges=[("head", "fin")])
        assert project.skeleton_removals(new) == {}
        before = project.points.df.copy()
        project.set_skeleton(new)
        pd.testing.assert_frame_equal(project.points.df, before)
        [entry] = toml.load(project.root / "project.toml")["skeleton"]["assigned"]
        assert entry["edges"] == [["head", "fin"]]
        assert entry["cameras"] == CAMERAS

    def test_removed_node_counts_and_deletes(self, project):
        new = Skeleton(["head", "fin"])
        assert project.skeleton_removals(new) == {"tail": len(TRIALS)}
        project.set_skeleton(new)
        assert set(project.points.df.index.get_level_values("bodypart")) == {"head"}
        on_disk = PointsStore.read_csv_df(project.experiments[EXP].root / "points.csv")
        assert set(on_disk.index.get_level_values("bodypart")) == {"head"}
        assert project.skeleton.nodes == ["head", "fin"]

    def test_unlabelled_rows_are_not_counted(self, project):
        df = project.points.df.copy()
        tail = df.index.get_level_values("bodypart") == "tail"
        for cam in CAMERAS:
            df.loc[tail, (cam, "state")] = "not_set"
            df.loc[tail, (cam, "x")] = np.nan
        project.points.replace_df(df)
        assert project.skeleton_removals(Skeleton(["head"])) == {}

    def test_existing_slp_files_keep_their_skeleton(self, project):
        project.write_slp_files()   # .slp files carry "tail"
        project.load_skeletons()
        project.set_skeleton(Skeleton(["head"]))
        # The assignment is for cells without a file; the files keep "tail"
        assert project.skeletons.by_camera["cam1"].nodes == ["head"]
        assert "tail" in project.skeletons.nodes_for((EXP, TRIALS[0], "cam1"))
        assert Skeleton.from_slp(project.root / "slp/cam1.slp").nodes == BODYPARTS

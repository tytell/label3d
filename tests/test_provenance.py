"""Tests for point provenance, sync history and prediction import (Phase 10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from label3d.adapters.sleap_adapter import SleapAdapter
from label3d.points import PointsStore, normalize_columns
from label3d.slp_sync import PREDICTION_MOVE_PX, compute_merge

from tests.test_project_sync import BODYPARTS, CAMERAS, EXP, TRIALS, project  # noqa: F401
from tests.test_slp_sync import _df

sio = pytest.importorskip("sleap_io", reason="sleap_io not installed")

NAN = float("nan")


def _cell(df, frame, inst, bp, cam, coord, trial="t1", exp=EXP):
    return df.loc[(exp, trial, frame, inst, bp), (cam, coord)]


def _predictions_file(path: Path, video_name: str, frame_points: dict) -> Path:
    """Write an .slp holding PredictedInstances, like sleap-nn track output.

    frame_points: {frame_idx: [(x, y, score) per bodypart]}
    """
    skeleton = sio.Skeleton(list(BODYPARTS))
    video = sio.Video(filename=f"/cluster/data/{video_name}", open_backend=False)
    frames = []
    for frame_idx, pts in frame_points.items():
        arr = np.array([[x, y] for x, y, _ in pts], dtype=np.float32)
        scores = np.array([sc for _, _, sc in pts])
        inst = sio.PredictedInstance.from_numpy(
            arr, skeleton=skeleton, point_scores=scores, score=float(scores.mean())
        )
        frames.append(sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst]))
    sio.save_file(
        sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton]), str(path)
    )
    return path


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class TestAdapterProvenance:
    def test_load_gives_sources_and_scores(self, tmp_path):
        trials = pd.DataFrame([{"experiment": EXP, "trial": "t1", "cam0": "v0.mp4",
                                "cam0_start": 0}])
        skeleton = sio.Skeleton(list(BODYPARTS))
        video = sio.Video(filename="v0.mp4", open_backend=False)
        user = sio.Instance.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32),
                                       skeleton=skeleton)
        pred = sio.PredictedInstance.from_numpy(
            np.array([[5, 6], [7, 8]], dtype=np.float32), skeleton=skeleton,
            point_scores=np.array([0.9, 0.4]), score=0.65,
        )
        labels = sio.Labels(
            labeled_frames=[
                sio.LabeledFrame(video=video, frame_idx=0, instances=[user]),
                sio.LabeledFrame(video=video, frame_idx=1, instances=[pred]),
            ],
            videos=[video], skeletons=[skeleton],
        )
        path = tmp_path / "a.slp"
        sio.save_file(labels, str(path))

        df = SleapAdapter().load([path], trials, ["cam0"])
        assert _cell(df, 0, 0, "head", "cam0", "source") == "sleap_user"
        assert np.isnan(_cell(df, 0, 0, "head", "cam0", "score"))
        assert _cell(df, 1, 0, "head", "cam0", "source") == "sleap_pred"
        assert _cell(df, 1, 0, "head", "cam0", "score") == pytest.approx(0.9)
        assert _cell(df, 1, 0, "tail", "cam0", "score") == pytest.approx(0.4)

    def test_save_load_keeps_prediction_scores(self, project):  # noqa: F811
        df = project.points.df.copy()
        idx = (EXP, "t1", 0, 0, "head")
        for cam in CAMERAS:
            for bp, score in (("head", 0.8), ("tail", 0.3)):
                df.loc[(EXP, "t1", 0, 0, bp), (cam, "state")] = "predicted"
                df.loc[(EXP, "t1", 0, 0, bp), (cam, "source")] = "sleap_pred"
                df.loc[(EXP, "t1", 0, 0, bp), (cam, "score")] = score
        project.points.replace_df(df)
        project.write_slp_files()

        loaded = SleapAdapter().load(
            None, project.trials_df, project.cameras,
            mapping=project.slp_mapping, root=project.root,
        )
        assert loaded.loc[idx, ("cam0", "score")] == pytest.approx(0.8)
        assert loaded.loc[(EXP, "t1", 0, 0, "tail"), ("cam1", "score")] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestMergeProvenance:
    def test_new_user_label_gets_source_and_sync(self):
        store = _df({})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "set")}})
        r = compute_merge(store, slp, sync_id=7)
        assert _cell(r.df, 0, 0, "head", "cam1", "source") == "sleap_user"
        assert _cell(r.df, 0, 0, "head", "cam1", "sync") == 7
        assert r.new_frames == {(EXP, "t1", 0)}
        assert r.trial_counts == {(EXP, "t1"): {"new_user": 1}}

    def test_new_prediction_gets_score_and_sync(self):
        store = _df({})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted", "sleap_pred", 0.75)}})
        r = compute_merge(store, slp, sync_id=3)
        assert _cell(r.df, 0, 0, "head", "cam1", "source") == "sleap_pred"
        assert _cell(r.df, 0, 0, "head", "cam1", "score") == pytest.approx(0.75)
        assert _cell(r.df, 0, 0, "head", "cam1", "sync") == 3
        assert r.n_new_pred == 1 and r.n_updated_pred == 0

    def test_no_sync_id_marks_nothing_new(self):
        r = compute_merge(_df({}), _df({(0, 0, "head"): {"cam1": (10, 20, "set")}}))
        assert np.isnan(_cell(r.df, 0, 0, "head", "cam1", "sync"))
        assert _cell(r.df, 0, 0, "head", "cam1", "source") == "sleap_user"

    def test_prediction_moved_less_than_tolerance_is_not_new(self):
        small = PREDICTION_MOVE_PX / 2
        store = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted", "sleap_pred", 0.5, 2)}})
        slp = _df({(0, 0, "head"): {"cam1": (10 + small, 20, "predicted", "sleap_pred", 0.9)}})
        r = compute_merge(store, slp, sync_id=3)
        assert _cell(r.df, 0, 0, "head", "cam1", "x") == 10
        assert _cell(r.df, 0, 0, "head", "cam1", "sync") == 2
        assert _cell(r.df, 0, 0, "head", "cam1", "score") == pytest.approx(0.9)
        assert r.n_predictions == 0 and not r.new_frames
        assert r.n_score_updates == 1 and r.has_changes

    def test_prediction_moved_more_than_tolerance_is_updated(self):
        store = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted", "sleap_pred", 0.5, 2)}})
        slp = _df({(0, 0, "head"): {"cam1": (15, 20, "predicted", "sleap_pred", 0.9)}})
        r = compute_merge(store, slp, sync_id=3)
        assert _cell(r.df, 0, 0, "head", "cam1", "x") == 15
        assert _cell(r.df, 0, 0, "head", "cam1", "sync") == 3
        assert r.n_updated_pred == 1 and r.n_new_pred == 0

    def test_label3d_guess_round_trip_keeps_source(self):
        """A label3d reprojection guess is written as a prediction; it must not come back as SLEAP's."""
        store = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted", "label3d")}})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "predicted", "sleap_pred", 1.0)}})
        r = compute_merge(store, slp, sync_id=3)
        assert _cell(r.df, 0, 0, "head", "cam1", "source") == "label3d"
        assert np.isnan(_cell(r.df, 0, 0, "head", "cam1", "sync"))
        assert not r.has_changes

    def test_use_sleap_on_conflict_gives_new_sleap_user_point(self):
        store = _df({(0, 0, "head"): {"cam1": (1, 2, "set")}})
        slp = _df({(0, 0, "head"): {"cam1": (10, 20, "set")}})
        r = compute_merge(store, slp, sync_id=4)
        assert _cell(r.df, 0, 0, "head", "cam1", "source") == "label3d"
        assert _cell(r.df_overwrite, 0, 0, "head", "cam1", "source") == "sleap_user"
        assert _cell(r.df_overwrite, 0, 0, "head", "cam1", "sync") == 4
        assert r.trial_counts[(EXP, "t1")]["conflicts"] == 1

    def test_label3d_point_round_trip_keeps_source(self, project):  # noqa: F811
        project.write_slp_files()
        results = project.read_slp_for_sync()
        assert not results[EXP].has_changes
        assert (project.points.df[("cam0", "source")] == "label3d").all()


# ---------------------------------------------------------------------------
# Sync history
# ---------------------------------------------------------------------------

def _clear_camera(project, cam):  # noqa: F811
    df = project.points.df.copy()
    df[(cam, "x")] = np.nan
    df[(cam, "y")] = np.nan
    df[(cam, "state")] = "not_set"
    df[(cam, "source")] = ""
    project.points.replace_df(df)
    project.points.to_csv(project.experiments[EXP].root / "points.csv")


class TestSyncHistory:
    def test_read_with_changes_records_a_sync(self, project):  # noqa: F811
        project.write_slp_files()
        _clear_camera(project, "cam1")
        assert project.latest_sync_id() is None

        results = project.read_slp_for_sync()
        assert results[EXP].sync_id == 1
        project.apply_merge(results, overwrite=False)

        assert project.latest_sync_id() == 1
        history = project.sync_history()
        assert sorted(history["trial"]) == sorted(TRIALS)
        assert history["n_new_user"].sum() == len(TRIALS) * len(BODYPARTS)
        assert set(history["kind"]) == {"sync"}

        store = project.points
        assert store.latest_sync == 1
        assert store.get_new_frames("t1") == {0: "user"}
        assert store.trials_with_new_points() == TRIALS

    def test_read_without_changes_adds_nothing(self, project):  # noqa: F811
        project.write_slp_files()
        _clear_camera(project, "cam1")
        project.apply_merge(project.read_slp_for_sync(), overwrite=False)
        project.write_slp_files()

        results = project.read_slp_for_sync()
        assert results[EXP].sync_id == 2
        project.apply_merge(results, overwrite=False)
        assert project.latest_sync_id() == 1
        assert project.points.get_new_frames("t1") == {0: "user"}

    def test_setup_import_marks_nothing_new(self, project):  # noqa: F811
        project.write_slp_files()
        _clear_camera(project, "cam1")
        results = project.read_slp_for_sync(sync=False)
        project.apply_merge(results, overwrite=True)
        assert project.latest_sync_id() is None
        assert (project.points.df[("cam1", "source")] == "sleap_user").all()
        assert project.points.get_new_frames("t1") == {}


# ---------------------------------------------------------------------------
# Import predictions
# ---------------------------------------------------------------------------

class TestImportPredictions:
    def test_import_matches_by_basename_and_drops_out_of_range(self, project, tmp_path):  # noqa: F811
        trials = project.trials_df.copy()
        trials["cam0_end"] = 5
        trials.to_csv(project.trials_csv_path, index=False)
        project.trials_df = trials

        path = _predictions_file(tmp_path / "out.slp", "cam0_t2.mp4", {
            3: [(50, 60, 0.9), (70, 80, 0.8)],
            9: [(1, 1, 0.9), (2, 2, 0.9)],   # past cam0_end
        })
        stranger = _predictions_file(tmp_path / "other.slp", "unknown.mp4",
                                     {0: [(1, 1, 0.5), (2, 2, 0.5)]})
        before = path.read_bytes()

        results, unmatched = project.read_predictions_for_import([path, stranger])
        assert unmatched == ["unknown.mp4"]
        r = results[EXP]
        assert r.new_frames == {(EXP, "t2", 3)}
        assert r.n_new_pred == len(BODYPARTS)

        project.apply_merge(results, overwrite=False, kind="import")
        df = project.points.df
        assert df.loc[(EXP, "t2", 3, 0, "head"), ("cam0", "source")] == "sleap_pred"
        assert df.loc[(EXP, "t2", 3, 0, "head"), ("cam0", "score")] == pytest.approx(0.9)
        assert df.loc[(EXP, "t2", 3, 0, "head"), ("cam0", "sync")] == 1
        assert (EXP, "t2", 9, 0, "head") not in df.index
        assert project.sync_history()["kind"].tolist() == ["import"]

        # The imported file is untouched and not added to the mapping
        assert path.read_bytes() == before
        assert all(spec.path.name != "out.slp" for spec in project.slp_mapping.files)

        # The next Sync writes the predictions into the mapped file
        project.write_slp_files()
        labels = sio.load_file(str(project.slp_dir / "cam0.slp"))
        frames = [lf for lf in labels.labeled_frames
                  if Path(lf.video.filename).name == "cam0_t2.mp4" and lf.frame_idx == 3]
        assert len(frames) == 1
        assert isinstance(frames[0].instances[0], sio.PredictedInstance)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@pytest.fixture
def store(qapp):
    s = PointsStore.create_empty(["cam1", "cam2"], [], experiment_name=EXP)
    df = _df({
        (0, 0, "head"): {"cam1": (10, 20, "predicted", "sleap_pred", 0.9, 5)},
        (0, 0, "tail"): {"cam1": (11, 21, "set", "sleap_user", NAN, 5)},
        (4, 0, "head"): {"cam1": (12, 22, "predicted", "sleap_pred", 0.9, 5)},
        (8, 0, "head"): {"cam1": (13, 23, "predicted", "sleap_pred", 0.9, 4)},
    })
    s.replace_df(normalize_columns(df))
    s.latest_sync = 5
    return s


class TestStoreProvenance:
    def test_edit_sets_label3d_and_clears_sync(self, store):
        store.set_point("t1", 0, 0, "head", "cam1", 30, 40, state="set")
        pt = store.get_point("t1", 0, 0, "head", "cam1")
        assert pt.source == "label3d" and np.isnan(pt.sync) and np.isnan(pt.score)

    def test_not_visible_and_delete_clear_sync(self, store):
        store.set_not_visible("t1", 0, 0, "head", "cam1")
        assert np.isnan(store.get_point("t1", 0, 0, "head", "cam1").sync)
        store.delete_point("t1", 4, 0, "head", "cam1")
        pt = store.get_point("t1", 4, 0, "head", "cam1")
        assert pt.source == "" and np.isnan(pt.sync)

    def test_new_frames(self, store):
        assert store.get_new_frames("t1") == {0: "user", 4: "predicted"}

    def test_accept_point(self, store):
        assert store.accept_points("t1", 0, 0, "head") == 1
        pt = store.get_point("t1", 0, 0, "head", "cam1")
        assert np.isnan(pt.sync) and pt.source == "sleap_pred" and pt.state == "predicted"
        assert store.get_new_frames("t1") == {0: "user", 4: "predicted"}

    def test_accept_frame(self, store):
        assert store.accept_points("t1", 0) == 2
        assert store.get_new_frames("t1") == {4: "predicted"}

    def test_next_prev_new_frame(self, store):
        assert store.get_next_new_frame("t1", 0) == 4
        assert store.get_next_new_frame("t1", 4) is None
        assert store.get_prev_new_frame("t1", 4) == 0
        assert store.get_prev_new_frame("t1", 0) is None

    def test_old_csv_gets_columns_filled_in(self, tmp_path, qapp):
        cols = pd.MultiIndex.from_tuples(
            [("cam0", c) for c in ("x", "y", "xr", "yr", "state")]
            + [("3d", c) for c in ("x", "y", "z")], names=["view", "coord"],
        )
        idx = pd.MultiIndex.from_tuples(
            [(EXP, "t1", 0, 0, "head"), (EXP, "t1", 0, 0, "tail")],
            names=["experiment", "trial", "frame", "instance", "bodypart"],
        )
        old = pd.DataFrame(
            [[1, 2, NAN, NAN, "set", NAN, NAN, NAN],
             [NAN, NAN, NAN, NAN, "not_set", NAN, NAN, NAN]],
            index=idx, columns=cols,
        )
        path = tmp_path / "points.csv"
        old.to_csv(path)
        df = PointsStore.read_csv_df(path)
        assert df.loc[(EXP, "t1", 0, 0, "head"), ("cam0", "source")] == "label3d"
        assert df.loc[(EXP, "t1", 0, 0, "tail"), ("cam0", "source")] == ""
        assert df[("cam0", "sync")].isna().all()
        assert [c for v, c in df.columns if v == "cam0"] == [
            "x", "y", "xr", "yr", "state", "source", "score", "sync"
        ]

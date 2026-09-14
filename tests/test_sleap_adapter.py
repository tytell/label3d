"""Tests for SleapAdapter write-back to mapped SLP files (Phase 8.7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio

from label3d.adapters.sleap_adapter import SleapAdapter
from label3d.points import CAM_COORDS, INDEX_NAMES, THREE_D_COORDS
from label3d.skeleton import Skeleton
from label3d.slp_sync import SlpBlock, SlpFileSpec, SlpMapping, compute_merge

NAN = float("nan")
EXP = "exp1"
CAMERAS = ["cam1", "cam2"]
SKELETON = Skeleton(nodes=["head", "tail"], edges=[("head", "tail")], symmetries=[])


def _trials_df(starts=None) -> pd.DataFrame:
    starts = starts or {}
    rows = []
    for t in ["t1", "t2"]:
        row = {"experiment": EXP, "trial": t}
        for c in CAMERAS:
            row[c] = f"{c}_{t}.mp4"
            row[f"{c}_start"] = starts.get((t, c), 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _df(cells: dict, experiment=EXP) -> pd.DataFrame:
    """cells: {(trial, frame, instance, bodypart): {camera: (x, y, state)}}"""
    col_tuples = [(c, coord) for c in CAMERAS for coord in CAM_COORDS]
    col_tuples += [("3d", coord) for coord in THREE_D_COORDS]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["view", "coord"])
    idx, data = [], []
    for (trial, frame, inst, bp), cams in cells.items():
        row = []
        for c in CAMERAS:
            x, y, s = cams.get(c, (NAN, NAN, "not_set"))
            row += [x, y, NAN, NAN, s]
        row += [NAN, NAN, NAN]
        idx.append((experiment, trial, frame, inst, bp))
        data.append(row)
    return pd.DataFrame(
        data, index=pd.MultiIndex.from_tuples(idx, names=INDEX_NAMES), columns=columns
    )


def _one_file_per_camera() -> SlpMapping:
    return SlpMapping(files=[
        SlpFileSpec(Path("slp/cam1.slp"), [SlpBlock(EXP, ["t1", "t2"], ["cam1"])]),
        SlpFileSpec(Path("slp/cam2.slp"), [SlpBlock(EXP, ["t1", "t2"], ["cam2"])]),
    ])


def _save(tmp_path, df, mapping, trials=None) -> int:
    return SleapAdapter().save(
        df, cameras=CAMERAS, skeleton=SKELETON, trials_df=trials if trials is not None else _trials_df(),
        mapping=mapping, root=tmp_path,
    )


def _frames_for(labels, basename):
    return [lf for lf in labels.labeled_frames if Path(lf.video.filename).name == basename]


class TestSaveInstanceTypes:
    def test_user_instance_exported_as_instance(self, tmp_path):
        df = _df({("t1", 0, 0, "head"): {"cam1": (10, 20, "set")},
                  ("t1", 0, 0, "tail"): {"cam1": (30, 40, "predicted")}})
        _save(tmp_path, df, _one_file_per_camera())
        labels = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        (lf,) = _frames_for(labels, "cam1_t1.mp4")
        (inst,) = lf.instances
        assert type(inst) is sio.Instance
        np.testing.assert_allclose(inst.numpy(), [[10, 20], [30, 40]])

    def test_predicted_instance_exported_as_predicted(self, tmp_path):
        df = _df({("t1", 0, 0, "head"): {"cam1": (10, 20, "predicted")}})
        _save(tmp_path, df, _one_file_per_camera())
        labels = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        (lf,) = labels.labeled_frames
        assert isinstance(lf.instances[0], sio.PredictedInstance)

    def test_all_not_set_instance_omitted(self, tmp_path):
        df = _df({("t1", 0, 0, "head"): {}, ("t1", 1, 0, "head"): {"cam1": (1, 2, "set")}})
        _save(tmp_path, df, _one_file_per_camera())
        labels = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        assert [lf.frame_idx for lf in labels.labeled_frames] == [1]

    def test_not_visible_written_as_nan(self, tmp_path):
        df = _df({("t1", 0, 0, "head"): {"cam1": (10, 20, "not_visible")},
                  ("t1", 0, 0, "tail"): {"cam1": (30, 40, "set")}})
        _save(tmp_path, df, _one_file_per_camera())
        labels = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        pts = labels.labeled_frames[0].instances[0].numpy()
        assert np.isnan(pts[0]).all()
        np.testing.assert_allclose(pts[1], [30, 40])


class TestSaveFiles:
    def test_shared_file_keeps_all_videos_and_frames(self, tmp_path):
        mapping = SlpMapping(files=[
            SlpFileSpec(Path("slp/all.slp"), [SlpBlock(EXP, ["t1", "t2"], CAMERAS)]),
        ])
        df = _df({
            (t, 3, 0, "head"): {c: (1, 2, "set") for c in CAMERAS} for t in ["t1", "t2"]
        })
        assert _save(tmp_path, df, mapping) == 1
        labels = sio.load_file(str(tmp_path / "slp/all.slp"))
        assert sorted(Path(v.filename).name for v in labels.videos) == [
            "cam1_t1.mp4", "cam1_t2.mp4", "cam2_t1.mp4", "cam2_t2.mp4",
        ]
        assert len(labels.labeled_frames) == 4

    def test_frame_offset(self, tmp_path):
        trials = _trials_df(starts={("t1", "cam1"): 10})
        df = _df({("t1", 5, 0, "head"): {"cam1": (1, 2, "set")}})
        mapping = _one_file_per_camera()
        _save(tmp_path, df, mapping, trials)

        labels = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        assert [lf.frame_idx for lf in labels.labeled_frames] == [15]

        loaded = SleapAdapter().load(None, trials, CAMERAS, mapping=mapping, root=tmp_path)
        assert loaded.index.get_level_values("frame").unique().tolist() == [5]

    def test_empty_file_written(self, tmp_path):
        assert _save(tmp_path, _df({}), _one_file_per_camera()) == 2
        labels = sio.load_file(str(tmp_path / "slp/cam2.slp"))
        assert len(labels.labeled_frames) == 0
        assert len(labels.videos) == 2
        assert labels.skeleton.node_names == ["head", "tail"]

    def test_video_path_points_into_experiment_videos(self, tmp_path):
        _save(tmp_path, _df({}), _one_file_per_camera())
        labels = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        assert Path(labels.videos[0].filename) == (
            tmp_path / "experiments" / EXP / "videos" / "cam1_t1.mp4"
        ).absolute()

    def test_existing_suggestions_preserved(self, tmp_path):
        mapping = _one_file_per_camera()
        _save(tmp_path, _df({}), mapping)
        path = tmp_path / "slp/cam1.slp"
        labels = sio.load_file(str(path))
        labels.suggestions.append(sio.SuggestionFrame(video=labels.videos[0], frame_idx=9))
        sio.save_file(labels, str(path))

        _save(tmp_path, _df({("t1", 0, 0, "head"): {"cam1": (1, 2, "set")}}), mapping)
        labels = sio.load_file(str(path))
        assert [(Path(s.video.filename).name, s.frame_idx) for s in labels.suggestions] == [
            ("cam1_t1.mp4", 9)
        ]
        assert len(labels.labeled_frames) == 1

    def test_video_dropped_when_reassigned(self, tmp_path):
        _save(tmp_path, _df({}), _one_file_per_camera())
        new_mapping = SlpMapping(files=[
            SlpFileSpec(Path("slp/cam1.slp"), [SlpBlock(EXP, ["t1"], ["cam1"])]),
        ])
        _save(tmp_path, _df({}), new_mapping)
        labels = sio.load_file(str(tmp_path / "slp/cam1.slp"))
        assert [Path(v.filename).name for v in labels.videos] == ["cam1_t1.mp4"]

    def test_build_file_map_routes_correctly(self, tmp_path):
        _save(tmp_path, _df({}), _one_file_per_camera())
        files = sorted((tmp_path / "slp").glob("*.slp"))
        file_map = SleapAdapter().build_file_map(files, _trials_df(), CAMERAS)
        assert file_map[(EXP, "t2", "cam1")] == tmp_path / "slp/cam1.slp"
        assert file_map[(EXP, "t1", "cam2")] == tmp_path / "slp/cam2.slp"

    def test_file_map_save_writes_shared_file_once(self, tmp_path):
        path = tmp_path / "shared.slp"
        file_map = {(EXP, t, c): path for t in ["t1", "t2"] for c in CAMERAS}
        df = _df({(t, 0, 0, "head"): {"cam1": (1, 2, "set")} for t in ["t1", "t2"]})
        n = SleapAdapter().save(
            df, cameras=CAMERAS, skeleton=SKELETON, trials_df=_trials_df(), file_map=file_map
        )
        assert n == 1
        assert len(sio.load_file(str(path)).labeled_frames) == 2


class TestLoad:
    def test_trial_lookup_uses_experiment(self, tmp_path):
        """Two experiments with the same trial name but different start frames."""
        trials = pd.DataFrame({
            "experiment": ["a", "b"],
            "trial": ["t1", "t1"],
            "cam1": ["a_cam1.mp4", "b_cam1.mp4"],
            "cam1_start": [0, 100],
        })
        sk = sio.Skeleton(["head"])
        frames = [
            sio.LabeledFrame(video=sio.Video(name, open_backend=False), frame_idx=100,
                             instances=[sio.Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=sk)])
            for name in ["a_cam1.mp4", "b_cam1.mp4"]
        ]
        path = tmp_path / "x.slp"
        sio.save_file(sio.Labels(labeled_frames=frames), str(path))

        loaded = SleapAdapter().load([path], trials, ["cam1"])
        assert set(loaded.index.droplevel(["instance", "bodypart"])) == {
            ("a", "t1", 100), ("b", "t1", 0),
        }

    def test_mapping_skips_unassigned_video(self, tmp_path):
        mapping = SlpMapping(files=[
            SlpFileSpec(Path("slp/both.slp"), [SlpBlock(EXP, ["t1"], CAMERAS)]),
        ])
        df = _df({("t1", 0, 0, "head"): {"cam1": (1, 2, "set"), "cam2": (3, 4, "set")}})
        _save(tmp_path, df, mapping)

        cam1_only = SlpMapping(files=[
            SlpFileSpec(Path("slp/both.slp"), [SlpBlock(EXP, ["t1"], ["cam1"])]),
        ])
        loaded = SleapAdapter().load(None, _trials_df(), CAMERAS, mapping=cam1_only, root=tmp_path)
        assert (loaded[("cam1", "state")] == "set").any()
        assert not (loaded[("cam2", "state")] == "set").any()

    def test_missing_mapped_file_treated_as_empty(self, tmp_path):
        loaded = SleapAdapter().load(
            None, _trials_df(), CAMERAS, mapping=_one_file_per_camera(), root=tmp_path
        )
        assert loaded.empty


class TestRoundTrip:
    def test_save_load_merge_is_identity(self, tmp_path):
        """Writing the store and reading it back must not change or conflict."""
        store = _df({
            # instance 0 only in cam2, instance 1 in both — cam1 has an instance gap
            ("t1", 0, 0, "head"): {"cam2": (1, 2, "set")},
            ("t1", 0, 0, "tail"): {"cam2": (3, 4, "not_visible")},
            ("t1", 0, 1, "head"): {"cam1": (30, 40, "set"), "cam2": (5, 6, "set")},
            ("t1", 0, 1, "tail"): {"cam1": (31, 41, "predicted"), "cam2": (7, 8, "set")},
            # predictions only
            ("t1", 4, 0, "head"): {"cam1": (9, 9, "predicted"), "cam2": (8, 8, "predicted")},
            ("t2", 2, 0, "head"): {"cam1": (11, 12, "predicted")},
            # user + prediction in the same frame
            ("t2", 7, 0, "head"): {"cam1": (13, 14, "set")},
            ("t2", 7, 1, "head"): {"cam1": (15, 16, "predicted")},
        })
        mapping = SlpMapping(files=[
            SlpFileSpec(Path("slp/early.slp"), [SlpBlock(EXP, ["t1"], CAMERAS)]),
            SlpFileSpec(Path("slp/late.slp"), [SlpBlock(EXP, ["t2"], CAMERAS)]),
        ])
        trials = _trials_df(starts={("t1", "cam2"): 3})
        _save(tmp_path, store, mapping, trials)

        loaded = SleapAdapter().load(None, trials, CAMERAS, mapping=mapping, root=tmp_path)
        r = compute_merge(store, loaded)
        assert r.conflicts == []
        assert r.n_new == 0 and r.n_predictions == 0
        assert r.changed_frames == set()

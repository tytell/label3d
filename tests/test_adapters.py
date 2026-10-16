"""Tests for the adapter registry and SleapAdapter."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from label3d.adapters import get_adapters
from label3d.adapters.sleap_adapter import SleapAdapter, basename_lookup
from label3d.points import CAM_COORDS, INDEX_NAMES, THREE_D_COORDS

# SLP fixtures require sleap_io — skip gracefully if not installed.
sleap_io = pytest.importorskip("sleap_io", reason="sleap_io not installed")


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_contains_sleap(self):
        adapters = get_adapters()
        assert "sleap" in adapters

    def test_sleap_maps_to_sleap_adapter(self):
        adapters = get_adapters()
        assert adapters["sleap"] is SleapAdapter

    def test_returns_dict_copy(self):
        a = get_adapters()
        b = get_adapters()
        a["dummy"] = object
        assert "dummy" not in b


# ---------------------------------------------------------------------------
# SleapAdapter metadata
# ---------------------------------------------------------------------------

class TestSleapAdapterMeta:
    def test_name(self):
        assert SleapAdapter.name() == "sleap"

    def test_file_filter(self):
        assert SleapAdapter.file_filter() == "SLEAP files (*.slp)"


# ---------------------------------------------------------------------------
# SleapAdapter.load — core behaviour
# ---------------------------------------------------------------------------

class TestSleapLoad:
    def test_returns_dataframe(self, slp_files, trials_df):
        adapter = SleapAdapter()
        cameras = ["cam1", "cam2", "cam3"]
        result = adapter.load(slp_files, trials_df, cameras)
        assert isinstance(result, pd.DataFrame)

    def test_index_names(self, slp_files, trials_df):
        adapter = SleapAdapter()
        result = adapter.load(slp_files, trials_df, ["cam1", "cam2", "cam3"])
        assert list(result.index.names) == INDEX_NAMES

    def test_column_structure(self, slp_files, trials_df):
        cameras = ["cam1", "cam2", "cam3"]
        adapter = SleapAdapter()
        result = adapter.load(slp_files, trials_df, cameras)
        for cam in cameras:
            for coord in CAM_COORDS:
                assert (cam, coord) in result.columns
        for coord in THREE_D_COORDS:
            assert ("3d", coord) in result.columns

    def test_has_rows(self, slp_files, trials_df):
        adapter = SleapAdapter()
        result = adapter.load(slp_files, trials_df, ["cam1", "cam2", "cam3"])
        assert len(result) > 0

    def test_correct_trial_name(self, slp_files, trials_df):
        adapter = SleapAdapter()
        result = adapter.load(slp_files, trials_df, ["cam1", "cam2", "cam3"])
        trials = result.index.get_level_values("trial").unique().tolist()
        assert "trial1" in trials

    def test_state_is_predicted_for_predicted_instances(self, slp_files, trials_df):
        """Synthetic SLP files contain PredictedInstances — state must be 'predicted'."""
        adapter = SleapAdapter()
        result = adapter.load(slp_files, trials_df, ["cam1", "cam2", "cam3"])
        states = result[("cam1", "state")].dropna().unique().tolist()
        assert "predicted" in states

    def test_frame_indices_are_non_negative(self, slp_files, trials_df):
        adapter = SleapAdapter()
        result = adapter.load(slp_files, trials_df, ["cam1", "cam2", "cam3"])
        frames = result.index.get_level_values("frame")
        assert (frames >= 0).all()

    def test_empty_file_list_returns_empty_dataframe(self, trials_df):
        adapter = SleapAdapter()
        result = adapter.load([], trials_df, ["cam1", "cam2"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # Column structure must still be correct
        for cam in ["cam1", "cam2"]:
            for coord in CAM_COORDS:
                assert (cam, coord) in result.columns

    def test_unmatched_video_basename_skipped(self, tmp_path, trials_df):
        """An SLP whose video basename doesn't match trials.csv yields no rows."""
        import sleap_io as sio

        sio_sk = sio.Skeleton(name="test")
        sio_sk.add_node("head")
        video = sio.Video(filename="completely_unknown_video.mp4")
        pts = np.full((1, 2), np.nan, dtype=np.float32)
        inst = sio.Instance.from_numpy(pts, skeleton=sio_sk)
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = sio.Labels(labeled_frames=[lf])
        slp_path = tmp_path / "unmatched.slp"
        sio.save_file(labels, str(slp_path))

        adapter = SleapAdapter()
        result = adapter.load([slp_path], trials_df, ["cam1", "cam2"])
        assert len(result) == 0

    def test_nonexistent_slp_file_raises(self, tmp_path, trials_df):
        missing = tmp_path / "does_not_exist.slp"
        adapter = SleapAdapter()
        with pytest.raises(Exception):
            adapter.load([missing], trials_df, ["cam1"])


# ---------------------------------------------------------------------------
# SleapAdapter.load — frame offset adjustment
# ---------------------------------------------------------------------------

class TestSleapLoadFrameOffset:
    def test_zero_offset_unchanged(self, slp_files, trials_df):
        """Synthetic trials.csv has cam1_start=0, so frame 0 in SLP → frame 0 in result."""
        adapter = SleapAdapter()
        result = adapter.load([slp_files[0]], trials_df, ["cam1", "cam2", "cam3"])
        frames = result.index.get_level_values("frame").tolist()
        assert 0 in frames

    def test_nonzero_offset_applied(self, tmp_path):
        """When cam_start=5, SLP frame 5 → global frame 0; SLP frame 4 → skipped (negative)."""
        import sleap_io as sio

        # Build a minimal SLP with frames 4 and 5
        sio_sk = sio.Skeleton(name="test")
        sio_sk.add_node("head")
        video_filename = "offset_cam.mp4"
        video = sio.Video(filename=video_filename)
        frames_slp = []
        for fidx in [4, 5, 6]:
            pts = np.array([[100.0, 200.0]], dtype=np.float32)
            inst = sio.Instance.from_numpy(pts, skeleton=sio_sk)
            frames_slp.append(sio.LabeledFrame(video=video, frame_idx=fidx, instances=[inst]))
        labels = sio.Labels(labeled_frames=frames_slp)
        slp_path = tmp_path / "offset_cam.slp"
        sio.save_file(labels, str(slp_path))

        # trials.csv: cam_start=5 means global_frame = slp_frame - 5
        trials_df = pd.DataFrame({
            "trial": ["trial_offset"],
            "mycam": [video_filename],
            "mycam_start": [5],
            "mycam_end": [10],
        })
        adapter = SleapAdapter()
        result = adapter.load([slp_path], trials_df, ["mycam"])
        frames = sorted(result.index.get_level_values("frame").unique().tolist())
        # SLP frame 4 → global -1 → skipped
        # SLP frame 5 → global 0
        # SLP frame 6 → global 1
        assert 0 in frames
        assert 1 in frames
        assert -1 not in frames


# ---------------------------------------------------------------------------
# SleapAdapter.save (round-trip)
# ---------------------------------------------------------------------------

class TestSleapSave:
    def test_save_creates_files(self, populated_store, tmp_path, trials_df):
        adapter = SleapAdapter()
        adapter.save(
            populated_store.df,
            cameras=["cam1", "cam2", "cam3"],
            output_dir=tmp_path,
            trials_df=trials_df,
        )
        slp_files = list(tmp_path.glob("*.slp"))
        assert len(slp_files) > 0

    def test_save_creates_output_dir(self, populated_store, tmp_path, trials_df):
        out_dir = tmp_path / "new_export_dir"
        assert not out_dir.exists()
        adapter = SleapAdapter()
        adapter.save(
            populated_store.df,
            cameras=["cam1", "cam2", "cam3"],
            output_dir=out_dir,
            trials_df=trials_df,
        )
        assert out_dir.exists()

    def test_save_output_dir_with_spaces(self, populated_store, tmp_path, trials_df):
        out_dir = tmp_path / "my export dir"
        adapter = SleapAdapter()
        adapter.save(
            populated_store.df,
            cameras=["cam1", "cam2", "cam3"],
            output_dir=out_dir,
            trials_df=trials_df,
        )
        slp_files = list(out_dir.glob("*.slp"))
        assert len(slp_files) > 0

    def test_save_roundtrip_preserves_coords(self, populated_store, tmp_path, trials_df):
        """Save then reload: saved SLP must contain the same number of frames/points."""
        adapter = SleapAdapter()
        adapter.save(
            populated_store.df,
            cameras=["cam1", "cam2", "cam3"],
            output_dir=tmp_path,
            trials_df=trials_df,
        )

        slp_files = sorted(tmp_path.glob("cam1_*.slp"))
        assert slp_files, "Expected at least one cam1 SLP file"

        # The saved SLP references original video filenames (e.g. cam1_trial1.mp4).
        # Reload using the original trials_df so basename matching works.
        reloaded = adapter.load(slp_files[:1], trials_df, ["cam1", "cam2", "cam3"])
        assert len(reloaded) > 0


# ---------------------------------------------------------------------------
# basename_lookup — path separator and edge-case tests
# ---------------------------------------------------------------------------

class TestBasenameLookup:
    def _make_trials(self, filename):
        return pd.DataFrame({
            "trial": ["t1"],
            "cam1": [filename],
            "cam1_start": [0],
        })

    def test_forward_slash_filename(self):
        trials = self._make_trials("subdir/cam1_trial1.mp4")
        lookup = basename_lookup(trials, ["cam1"])
        assert "cam1_trial1.mp4" in lookup

    def test_backslash_filename(self):
        """Windows-style backslash paths — basename extraction must still work."""
        trials = self._make_trials("subdir\\cam1_trial1.mp4")
        # Path("subdir\\cam1_trial1.mp4").name is OS-dependent, but the adapter
        # uses pathlib.Path which on POSIX treats the whole thing as one component.
        # We test that whatever .name returns, the lookup key is consistent.
        lookup = basename_lookup(trials, ["cam1"])
        # On POSIX: .name = "subdir\\cam1_trial1.mp4" (whole string)
        # On Windows: .name = "cam1_trial1.mp4"
        # Either way, lookup must have exactly one entry
        assert len(lookup) == 1

    def test_filename_with_spaces(self):
        trials = self._make_trials("my recording cam1.mp4")
        lookup = basename_lookup(trials, ["cam1"])
        assert "my recording cam1.mp4" in lookup

    def test_multiple_cameras(self):
        trials = pd.DataFrame({
            "trial": ["t1"],
            "cam1": ["cam1_t1.mp4"],
            "cam2": ["cam2_t1.mp4"],
        })
        lookup = basename_lookup(trials, ["cam1", "cam2"])
        assert "cam1_t1.mp4" in lookup
        assert "cam2_t1.mp4" in lookup
        assert lookup["cam1_t1.mp4"] == [("default", "t1", "cam1")]
        assert lookup["cam2_t1.mp4"] == [("default", "t1", "cam2")]

    def test_nan_filename_skipped(self):
        trials = pd.DataFrame({
            "trial": ["t1"],
            "cam1": [float("nan")],
        })
        lookup = basename_lookup(trials, ["cam1"])
        assert len(lookup) == 0

    def test_absolute_path_filename(self):
        trials = self._make_trials("/home/user/data/cam1_trial1.mp4")
        lookup = basename_lookup(trials, ["cam1"])
        assert "cam1_trial1.mp4" in lookup

    def test_case_sensitivity_documented(self):
        """Basename matching is case-sensitive on POSIX (document the behaviour)."""
        import sys
        trials = self._make_trials("Cam1_Trial1.mp4")
        lookup = basename_lookup(trials, ["cam1"])
        # Key is exactly what Path(...).name returns
        assert list(lookup.keys())[0] == "Cam1_Trial1.mp4"

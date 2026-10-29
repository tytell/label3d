"""Setting up a project from existing .slp files, with videos referenced in place.

See docs/design/project_setup_from_sleap.md. The videos in these tests always live
outside the project tree, which is the case the feature exists for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio
import toml

from label3d.points import read_points_df
from label3d.project import Project
from label3d.setup_project import (
    SetupConfig,
    _compute_video_roots,
    _identify_camera_from_path,
    classify_slp_files,
    config_from_toml,
    execute_setup,
    import_labels,
    preview_assignment,
)

CAMERAS = ["cam1", "cam2"]
NODES = ["head", "tail"]


def _make_video(path: Path, n_frames: int = 5) -> Path:
    """Write a tiny real video file that sleap_io and OpenCV can both open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.zeros((n_frames, 16, 16, 1), dtype=np.uint8)
    sio.save_video(frames, str(path))
    return path


def _make_slp(slp_path: Path, videos: list[Path], labeled: bool = True) -> Path:
    """Write a .slp referencing the given videos by their real paths."""
    # sio.Skeleton mutates the list it is given into Node objects, so copy.
    skeleton = sio.Skeleton(list(NODES))
    skeleton.add_edge(NODES[0], NODES[1])
    sio_videos = [sio.Video.from_filename(str(v)) for v in videos]
    frames = []
    if labeled:
        for video in sio_videos:
            inst = sio.Instance.from_numpy(
                np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skeleton
            )
            frames.append(
                sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
            )
    labels = sio.Labels(
        labeled_frames=frames, videos=sio_videos, skeletons=[skeleton]
    )
    slp_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_file(labels, str(slp_path))
    return slp_path


@pytest.fixture
def source(tmp_path):
    """Videos in an external folder, one .slp per camera beside them."""
    rig = tmp_path / "rig"
    videos = {}
    for cam in CAMERAS:
        for trial in ("Trial1", "Trial2"):
            videos[(cam, trial)] = _make_video(rig / f"{cam}_{trial}.mp4")
    slp_dir = tmp_path / "labels"
    slps = [
        _make_slp(slp_dir / f"{cam}.slp", [videos[(cam, t)] for t in ("Trial1", "Trial2")])
        for cam in CAMERAS
    ]
    return {"rig": rig, "videos": videos, "slps": slps, "tmp": tmp_path}


def _config(source, **overrides) -> SetupConfig:
    defaults = dict(
        source_dir=source["tmp"],
        output_dir=source["tmp"] / "proj",
        camera_names=CAMERAS,
        calib_keyword="calib",
        match_strategy="keyword",
        trial_keyword="Trial",
        slp_paths=source["slps"],
        video_mode="reference",
        project_name="demo",
    )
    defaults.update(overrides)
    return SetupConfig(**defaults)


class TestHelpers:
    """Reference resolution itself lives in tests/test_video_reference.py."""

    def test_camera_from_filename(self):
        assert _identify_camera_from_path(Path("/d/cam2_Trial1.mp4"), CAMERAS) == "cam2"

    def test_camera_from_parent_directory(self):
        """Rigs that write each camera into its own folder."""
        assert _identify_camera_from_path(Path("/d/cam2/Trial1.mp4"), CAMERAS) == "cam2"

    def test_video_root_is_common_parent(self):
        trials = {"t1": {"cam1": Path("/data/rig/a.mp4"), "cam2": Path("/data/rig/b.mp4")}}
        assert _compute_video_roots(trials, {}) == {"default": Path("/data/rig")}

    def test_no_video_root_for_unrelated_trees(self):
        trials = {"t1": {"cam1": Path("/data/a.mp4"), "cam2": Path("/other/b.mp4")}}
        assert _compute_video_roots(trials, {}) == {}


class TestClassify:
    def test_groups_trials_across_cameras(self, source):
        assignment = classify_slp_files(_config(source))
        assert sorted(assignment.trials) == ["Trial1", "Trial2"]
        assert sorted(assignment.trials["Trial1"]) == CAMERAS

    def test_extracts_skeleton(self, source):
        assignment = classify_slp_files(_config(source))
        assert assignment.skeleton["nodes"] == NODES

    def test_maps_slp_files_to_cells(self, source):
        assignment = classify_slp_files(_config(source))
        assert assignment.slp_files["Trial1"]["cam1"].name == "cam1.slp"

    def test_computes_video_root(self, source):
        assignment = classify_slp_files(_config(source))
        assert assignment.video_roots["default"] == source["rig"].resolve()

    def test_missing_video_is_recorded_not_raised(self, source, tmp_path):
        gone = tmp_path / "gone"
        gone.mkdir()
        video = _make_video(gone / "cam1_Trial3.mp4")
        slp = _make_slp(tmp_path / "labels" / "extra.slp", [video])
        video.unlink()
        assignment = classify_slp_files(
            _config(source, slp_paths=[*source["slps"], slp])
        )
        assert len(assignment.missing_videos) == 1
        assert "not found" in " ".join(assignment.warnings)

    def test_every_video_reference_is_recorded(self, source, tmp_path):
        gone = tmp_path / "gone"
        gone.mkdir()
        video = _make_video(gone / "cam1_Trial3.mp4")
        stray = _make_video(tmp_path / "rig" / "other_Trial1.mp4")
        slp = _make_slp(tmp_path / "labels" / "extra.slp", [video, stray])
        video.unlink()
        assignment = classify_slp_files(
            _config(source, slp_paths=[*source["slps"], slp])
        )
        refs = [r for r in assignment.video_refs if r.slp_path == slp]
        assert [r.recorded.name for r in refs] == ["cam1_Trial3.mp4", "other_Trial1.mp4"]
        assert refs[0].resolution.path is None
        assert refs[1].resolution.rule == "recorded"
        assert refs[1].camera is None
        found = [r for r in assignment.video_refs if r.slp_path == source["slps"][0]]
        assert {r.camera for r in found} == {"cam1"}

    def test_slp_outside_listed_experiments_is_an_error(self, source, tmp_path):
        gone = tmp_path / "gone"
        gone.mkdir()
        video = _make_video(gone / "cam1_Trial3.mp4")
        stray = _make_slp(tmp_path / "labels" / "stray.slp", [video])
        video.unlink()
        config = _config(
            source, slp_paths=[*source["slps"], stray], experiment_keywords=["cam"],
            allow_missing_videos=True,
        )
        assignment = classify_slp_files(config)
        assert len(assignment.errors) == 1
        assert "'stray.slp' has no video in a trial of experiments cam" in assignment.errors[0]
        assert "Errors (setup can't go ahead)" in preview_assignment(config, assignment)
        with pytest.raises(ValueError, match="stray.slp"):
            execute_setup(config, assignment)

    def test_preview_counts_unresolved_without_verbose(self, source, tmp_path):
        gone = tmp_path / "gone"
        gone.mkdir()
        videos = [_make_video(gone / f"cam1_Trial{i}.mp4") for i in (3, 4)]
        extra = _make_slp(tmp_path / "labels" / "extra.slp", videos)
        for v in videos:
            v.unlink()
        config = _config(source, slp_paths=[*source["slps"], extra])
        assignment = classify_slp_files(config)
        brief = preview_assignment(config, assignment, verbosity=2)
        assert "  2 from extra.slp" in brief
        assert "cam1_Trial3.mp4" not in brief
        full = preview_assignment(config, assignment)
        assert full.count("cam1_Trial3.mp4") == 2  # the listing and its warning

    def test_preview_summary_groups_missing_cameras(self, source, tmp_path):
        """Verbosity 0: trial counts, trials grouped by the cameras they lack."""
        extra = _make_slp(
            tmp_path / "labels" / "extra.slp",
            [_make_video(tmp_path / "rig" / f"cam1_Trial{i}.mp4") for i in (3, 4)],
        )
        config = _config(source, slp_paths=[*source["slps"], extra])
        assignment = classify_slp_files(config)
        summary = preview_assignment(config, assignment, verbosity=0)
        assert "[default]  4 trial(s), root:" in summary
        assert "    2 trial(s) missing cam2" in summary
        assert "Trial3:" not in summary
        assert "SLP files (" not in summary
        assert "is missing cameras" not in summary  # the warnings repeat the group
        assert "Calibration: none found" in summary
        named = preview_assignment(config, assignment, verbosity=1)
        assert "    2 trial(s) missing cam2\n      Trial3, Trial4\n" in named
        assert "Trial3:" not in named
        assert "Trial3, Trial4" not in summary
        assert "Trial3:" in preview_assignment(config, assignment, verbosity=2)

    def test_unlisted_experiment_videos_left_out(self, source, tmp_path):
        other = [_make_video(tmp_path / "rig" / f"{c}_other_Trial5.mp4") for c in CAMERAS]
        mixed = _make_slp(tmp_path / "labels" / "mixed.slp", [*other, source["videos"][("cam1", "Trial1")]])
        config = _config(
            source, slp_paths=[*source["slps"], mixed], experiment_keywords=["Trial1", "Trial2"]
        )
        assignment = classify_slp_files(config)
        assert sorted(f.name for f in assignment.excluded) == sorted(v.name for v in other)
        assert set(assignment.trial_experiment.values()) == {"Trial1", "Trial2"}
        assert assignment.errors == []  # mixed.slp still has a video in Trial1

    def test_slp_files_in_listed_experiments_pass(self, source):
        assignment = classify_slp_files(_config(source, experiment_keywords=["cam"]))
        assert assignment.errors == []

    def test_different_skeletons_are_merged(self, source, tmp_path):
        other = _make_video(tmp_path / "rig2" / "cam1_Trial9.mp4")
        skeleton = sio.Skeleton(["nose", "fin"])
        video = sio.Video.from_filename(str(other))
        labels = sio.Labels(labeled_frames=[], videos=[video], skeletons=[skeleton])
        odd = tmp_path / "labels" / "odd.slp"
        sio.save_file(labels, str(odd))
        assignment = classify_slp_files(_config(source, slp_paths=[*source["slps"], odd]))
        assert assignment.skeleton["nodes"] == [*NODES, "nose", "fin"]
        assert assignment.slp_skeletons[odd].nodes == ["nose", "fin"]


class TestConfigFromToml:
    def _write(self, source, setup: dict) -> Path:
        path = source["tmp"] / "setup.toml"
        base = {
            "camera_names": CAMERAS,
            "slp_files": [str(p) for p in source["slps"]],
            "match_strategy": "regex",
            "trial_regex": "(Trial\\d+)",
        }
        path.write_text(toml.dumps({"setup": {**base, **setup}}))
        return path

    def test_reads_slp_settings(self, source):
        path = self._write(source, {"output_dir": str(source["tmp"] / "proj")})
        config = config_from_toml(path)
        assert config.slp_paths == [p.resolve() for p in source["slps"]]
        assert config.video_mode == "reference"
        assert config.trial_regex == "(Trial\\d+)"
        assert config.output_dir == (source["tmp"] / "proj").resolve()

    def _flat_folder(self, source) -> Path:
        """The .slp files beside an unlabelled trial and calibration videos."""
        flat = source["tmp"] / "labels"
        for cam in CAMERAS:
            _make_video(flat / f"{cam}_Trial3.mp4")
            _make_video(flat / f"{cam}_calib.mp4")
        setup = {"output_dir": str(source["tmp"] / "proj"), "source_dir": str(flat)}
        path = self._write(source, setup)
        raw = toml.load(path)
        del raw["setup"]["slp_files"]
        path.write_text(toml.dumps(raw))
        return path

    def test_slp_files_found_in_source_dir(self, source):
        config = config_from_toml(self._flat_folder(source))
        assert config.slp_paths == [(source["tmp"] / "labels").resolve()]
        assert config.include_source_videos

    def test_unreferenced_source_videos_become_trials(self, source):
        assignment = classify_slp_files(config_from_toml(self._flat_folder(source)))
        assert sorted(assignment.trials) == ["Trial1", "Trial2", "Trial3"]
        assert sorted(assignment.trials["Trial3"]) == CAMERAS
        assert "Trial3" not in assignment.slp_files

    def test_calibration_found_in_source_dir(self, source):
        assignment = classify_slp_files(config_from_toml(self._flat_folder(source)))
        assert sorted(assignment.exp_calibration["default"]) == CAMERAS

    def test_output_dir_required_by_default(self, source):
        with pytest.raises(KeyError):
            config_from_toml(self._write(source, {}))

    def test_reads_camera_aliases(self, source):
        path = self._write(source, {
            "output_dir": str(source["tmp"] / "proj"),
            "camera_aliases": {"cam1": ["c1", "first"], "cam2": "c2"},
        })
        assert config_from_toml(path).camera_aliases == {"cam1": ["c1", "first"], "cam2": ["c2"]}

    def test_empty_skeleton_table_uses_slp_skeletons(self, source):
        path = self._write(source, {"output_dir": str(source["tmp"] / "proj")})
        path.write_text(path.read_text() + "\n[skeleton]\n# file = 'skeleton.json'\n")
        config = config_from_toml(path)
        assert config.skeleton is None
        assert config.skeleton_cameras is None
        assert classify_slp_files(config).skeleton["nodes"] == NODES

    def test_skeleton_table_without_nodes_is_an_error(self, source):
        path = self._write(source, {"output_dir": str(source["tmp"] / "proj")})
        path.write_text(path.read_text() + "\n[skeleton]\ncameras = ['cam1']\n")
        with pytest.raises(ValueError, match="needs 'file' or 'nodes'"):
            config_from_toml(path)

    def test_output_dir_optional_for_inspection(self, source):
        config = config_from_toml(self._write(source, {}), require_output=False)
        assert sorted(classify_slp_files(config).trials) == ["Trial1", "Trial2"]


class TestExecute:
    def test_videos_are_not_copied(self, source):
        config = _config(source)
        execute_setup(config, classify_slp_files(config))
        assert not (config.output_dir / "experiments/default/videos").exists()
        assert (source["rig"] / "cam1_Trial1.mp4").exists()

    def test_trials_csv_holds_relative_cells(self, source):
        config = _config(source)
        out = execute_setup(config, classify_slp_files(config))
        text = (out / "trials.csv").read_text()
        assert "cam1_Trial1.mp4" in text

    def test_trials_csv_records_end_frames_and_mtimes(self, source):
        """Setup measures the referenced videos, so the first open needn't."""
        config = _config(source)
        out = execute_setup(config, classify_slp_files(config))
        df = pd.read_csv(out / "trials.csv")
        assert (df["cam1_end"] == 4).all()      # _make_video writes 5 frames
        assert df["cam1_mtime"].str.endswith("Z").all()
        assert not (out / "trials.csv.bak").exists()

    def test_project_toml_records_video_root(self, source):
        config = _config(source)
        out = execute_setup(config, classify_slp_files(config))
        data = toml.load(out / "project.toml")
        assert data["video_roots"]["default"] == str(source["rig"].resolve())

    def test_skeleton_comes_from_slp_files(self, source):
        config = _config(source)
        out = execute_setup(config, classify_slp_files(config))
        assert not (out / "skeleton.json").exists()
        assert Project.load(out).skeleton.nodes == NODES

    def test_slp_files_are_copied_not_moved(self, source):
        config = _config(source)
        out = execute_setup(config, classify_slp_files(config))
        assert (out / "slp" / "cam1" / "cam1.slp").exists()
        assert source["slps"][0].exists()

    def test_slp_mapping_written(self, source):
        config = _config(source)
        out = execute_setup(config, classify_slp_files(config))
        data = toml.load(out / "project.toml")
        assert data["slp"]["files"]

    def test_project_loads_and_finds_videos(self, source):
        config = _config(source)
        out = execute_setup(config, classify_slp_files(config))
        project = Project.load(out)
        path = project.video_path("Trial1", "cam1")
        assert path == source["rig"].resolve() / "cam1_Trial1.mp4"
        assert path.exists()

    def test_missing_videos_block_execution(self, source, tmp_path):
        config = _config(source)
        assignment = classify_slp_files(config)
        assignment.missing_videos.append((Path("/gone/x.mp4"), "cam1.slp"))
        with pytest.raises(FileNotFoundError, match="could not be resolved"):
            execute_setup(config, assignment)

    def test_allow_missing_videos_proceeds(self, source):
        """Work on what was found: the located videos still make a project."""
        config = _config(source, allow_missing_videos=True)
        assignment = classify_slp_files(config)
        assignment.missing_videos.append((Path("/gone/x.mp4"), "cam1.slp"))
        out = execute_setup(config, assignment)
        project = Project.load(out)
        assert project.video_path("Trial1", "cam1").exists()

    def test_one_camera_missing_leaves_its_cell_blank(self, source, tmp_path):
        """A camera whose video cannot be found drops out; the trial survives."""
        gone = tmp_path / "gone"
        video = _make_video(gone / "cam2_Trial7.mp4")
        kept = _make_video(source["rig"] / "cam1_Trial7.mp4")
        slp = _make_slp(tmp_path / "labels" / "trial7.slp", [video, kept])
        video.unlink()
        config = _config(
            source, slp_paths=[*source["slps"], slp], allow_missing_videos=True
        )
        assignment = classify_slp_files(config)
        assert assignment.missing_videos
        assert set(assignment.trials["Trial7"]) == {"cam1"}
        out = execute_setup(config, assignment)
        project = Project.load(out)
        assert project.video_path("Trial7", "cam1").exists()


class TestModelsFolders:
    """Each .slp file gets a folder of its own in slp/, with its SLEAP models folder."""

    @pytest.fixture
    def nested(self, source):
        """labels2/{cam}/{cam}.slp; cam1's folder has a models folder."""
        root = source["tmp"] / "labels2"
        for cam in CAMERAS:
            _make_slp(root / cam / f"{cam}.slp", [source["videos"][(cam, t)] for t in ("Trial1", "Trial2")])
        model = root / "cam1" / "models" / "run1"
        model.mkdir(parents=True)
        (model / "best.ckpt").write_bytes(b"weights")
        # SLEAP writes the model's training labels as .slp files in its folder
        _make_slp(model / "labels_gt.train.slp", [source["videos"][("cam1", "Trial1")]])
        return root

    def test_found_recursively_skipping_models(self, source, nested):
        assignment = classify_slp_files(_config(source, slp_paths=[nested]))
        assert not assignment.errors
        assert sorted(p.name for p in assignment.slp_skeletons) == ["cam1.slp", "cam2.slp"]

    def test_models_folder_copied_beside_its_slp(self, source, nested):
        config = _config(source, slp_paths=[nested])
        out = execute_setup(config, classify_slp_files(config))
        assert (out / "slp" / "cam1" / "cam1.slp").exists()
        assert (out / "slp" / "cam1" / "models" / "run1" / "best.ckpt").read_bytes() == b"weights"
        assert not (out / "slp" / "cam2" / "models").exists()
        assert (nested / "cam1" / "models" / "run1" / "best.ckpt").exists()
        files = {f["path"] for f in toml.load(out / "project.toml")["slp"]["files"]}
        assert files == {"slp/cam1/cam1.slp", "slp/cam2/cam2.slp"}
        assert [p.name for p in Project.load(out).slp_files()] == ["cam1.slp", "cam2.slp"]

    def test_preview_lists_models_folders(self, source, nested):
        config = _config(source, slp_paths=[nested])
        text = preview_assignment(config, classify_slp_files(config))
        assert "slp/cam1/models" in text

    def test_shared_models_folder_is_an_error(self, source):
        (source["slps"][0].parent / "models").mkdir()
        assignment = classify_slp_files(_config(source))
        assert any("models folder and several .slp files" in e for e in assignment.errors)

    def test_clashing_stems_numbered(self, source):
        slps = [
            _make_slp(
                source["tmp"] / folder / "labels.slp",
                [source["videos"][(cam, t)] for t in ("Trial1", "Trial2")],
            )
            for folder, cam in (("a", "cam1"), ("b", "cam2"))
        ]
        config = _config(source, slp_paths=slps)
        out = execute_setup(config, classify_slp_files(config))
        assert (out / "slp" / "labels" / "labels.slp").exists()
        assert (out / "slp" / "labels_2" / "labels.slp").exists()


class TestCalibration:
    def _write_calib(self, path: Path) -> Path:
        data = {
            cam: {
                "name": cam, "size": [16, 16], "matrix": [[1, 0, 8], [0, 1, 8], [0, 0, 1]],
                "distortions": [0.0] * 5, "rotation": [0.0] * 3, "translation": [0.0] * 3,
            }
            for cam in CAMERAS
        }
        path.write_text(toml.dumps(data))
        return path

    def test_existing_calibration_toml_is_adopted(self, source):
        self._write_calib(source["slps"][0].parent / "calibration.toml")
        config = _config(source)
        assignment = classify_slp_files(config)
        assert assignment.calibration_toml["default"].name == "calibration.toml"
        out = execute_setup(config, assignment)
        assert not (out / "experiments/default/calibration.toml").exists()
        project = Project.load(out)
        entry = project.active_calibration("default")
        assert entry.method == "imported"
        assert entry.metadata["source_path"].endswith("calibration.toml")
        assert project.has_calibration

    def test_unrelated_toml_is_not_mistaken_for_calibration(self, source):
        (source["slps"][0].parent / "notes.toml").write_text('[meta]\nauthor = "x"\n')
        assignment = classify_slp_files(_config(source))
        assert not assignment.calibration_toml

    def test_calibration_videos_found_beside_slp(self, source):
        for cam in CAMERAS:
            _make_video(source["slps"][0].parent / f"{cam}_calib.mp4")
        assignment = classify_slp_files(_config(source))
        assert sorted(assignment.calibration_takes["default"]["calib"]) == CAMERAS

    def test_no_calibration_warns(self, source):
        assignment = classify_slp_files(_config(source))
        assert any("no calibration" in w for w in assignment.warnings)


class TestImportLabels:
    def test_labels_land_in_points_file(self, source):
        config = _config(source)
        out = execute_setup(config, classify_slp_files(config))
        counts = import_labels(out)
        assert counts.get("default", 0) > 0
        points = out / "experiments" / "default" / "points.parquet"
        assert points.exists()
        assert "head" in read_points_df(points).index.get_level_values("bodypart")

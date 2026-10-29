"""Tests for detecting old project layouts and upgrading them (Phase 20).

The projects in tests/fixtures/old_projects were made by label3d 0.0.9 and
0.1.0 themselves (make_old_projects.py): their .slp copies lost a video and
frames 7-9 in an old Sync, and two predictions were labelled over in label3d.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio
import toml
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from label3d import export_cli, upgrade, upgrade_cli
from label3d.layout import (
    CORNERS,
    FRAME_NUMBERS,
    POINTS,
    RESTORE_SLP,
    SKELETON,
    SLP_FOLDERS,
    SYNC_HISTORY,
    UPGRADE_BACKUP_DIR,
    ProjectNeedsUpgrade,
    detect_version,
    plan_upgrade,
)
from label3d.points import read_points_df
from label3d.project import Project
from label3d.upgrade import earliest_backup, find_originals, read_points_csv, run_upgrade
from label3d.version import LAYOUT_VERSION
from label3d.widgets.upgrade_dialog import UpgradeDialog, upgrade_project

FIXTURES = Path(__file__).parent / "fixtures" / "old_projects"
VERSIONS = ["0.0.9", "0.1.0"]
CAMERAS = ["cam1", "cam2"]
# From old_sync.py: (frame, bodypart, camera, dx, dy) labelled over predictions in exp1/trial1
RELABELLED = [(2, "snout", "cam1", 3.0, -2.0), (4, "dorsal", "cam2", -1.5, 2.5)]
EXTRA_VIDEO = "{camera}_not_in_project.mp4"
LAST_TRIAL_FRAME = 6


def _copy(version: str, tmp_path: Path) -> tuple[Path, Path]:
    """A copy of the fixture: (project root, folder of original .slp files)."""
    shutil.copytree(FIXTURES / f"v{version}", tmp_path / version)
    return tmp_path / version / "project", tmp_path / version / "originals"


def _old_points(version: str, experiment: str) -> pd.DataFrame:
    folder = FIXTURES / f"v{version}" / "project" / "experiments" / experiment
    if version == "0.0.9":
        return read_points_csv(folder / "points.csv")
    return read_points_df(folder / "points.parquet")


def _originals(plan, folder: Path) -> dict[Path, Path]:
    found = find_originals([p.name for p in plan.restore], folder)
    return {p: found[p.name][0] for p in plan.restore}


def _upgrade(root: Path, originals: Path):
    plan = plan_upgrade(root)
    return run_upgrade(root, _originals(plan, originals))


def _digests(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).parts[0] != UPGRADE_BACKUP_DIR
    }


def _frames(labels: sio.Labels, video_name: str) -> dict[int, sio.LabeledFrame]:
    return {lf.frame_idx: lf for lf in labels.labeled_frames
            if Path(lf.video.filename).name == video_name}


# -- Detection ---------------------------------------------------------------


class TestDetection:
    @pytest.mark.parametrize("version", VERSIONS)
    def test_version_from_files(self, version):
        root = FIXTURES / f"v{version}" / "project"
        assert detect_version(root, toml.load(root / "project.toml")) == version

    def test_recorded_version_wins(self):
        root = FIXTURES / "v0.0.9" / "project"
        assert detect_version(root, {"label3d": {"project_version": "0.3.0"}}) == "0.3.0"

    def test_0_0_9_steps(self):
        plan = plan_upgrade(FIXTURES / "v0.0.9" / "project")
        assert plan.steps == [POINTS, CORNERS, SYNC_HISTORY, SKELETON, SLP_FOLDERS,
                              FRAME_NUMBERS, RESTORE_SLP]
        assert plan.restore == [Path("slp/cam1.slp"), Path("slp/cam2.slp")]

    def test_0_1_0_steps(self):
        plan = plan_upgrade(FIXTURES / "v0.1.0" / "project")
        assert plan.steps == [SLP_FOLDERS, FRAME_NUMBERS, RESTORE_SLP]

    def test_pre_registry_project_is_refused(self, tmp_path):
        root, _ = _copy("0.1.0", tmp_path)
        (root / "experiments" / "exp1" / "calibration.toml").write_text("")
        with pytest.raises(ValueError, match="Set the project up again"):
            plan_upgrade(root)

    @pytest.mark.parametrize("version", VERSIONS)
    def test_load_asks_for_an_upgrade(self, tmp_path, version, qapp):
        root, _ = _copy(version, tmp_path)
        before = _digests(root)
        with pytest.raises(ProjectNeedsUpgrade) as exc_info:
            Project.load(root)
        assert exc_info.value.plan.from_version == version
        assert _digests(root) == before


# -- The upgrade -----------------------------------------------------------------


@pytest.fixture(params=VERSIONS)
def upgraded(request, tmp_path, qapp):
    """(version, root, originals folder, result) for an upgraded copy of each fixture."""
    version = request.param
    root, originals = _copy(version, tmp_path)
    result = _upgrade(root, originals)
    return version, root, originals, result


class TestUpgrade:
    def test_loads_and_records_the_version(self, upgraded):
        _, root, _, _ = upgraded
        project = Project.load(root)
        assert project.config["label3d"]["project_version"] == LAYOUT_VERSION
        assert (root / "project.toml").read_text().startswith("# label3d ")
        assert plan_upgrade(root) is None

    def test_points_are_unchanged(self, upgraded, qapp):
        version, root, _, _ = upgraded
        project = Project.load(root)
        for name in ["exp1", "exp2"]:
            old, new = _old_points(version, name), project.experiment_points_df(name)
            pd.testing.assert_index_equal(new.index, old.index)
            for cam in CAMERAS:
                for coord in ["x", "y", "state", "source"]:
                    pd.testing.assert_series_equal(
                        new[(cam, coord)], old[(cam, coord)], check_names=False
                    )

    def test_predictions_under_user_labels_are_recovered(self, upgraded):
        _, root, _, _ = upgraded
        df = Project.load(root).experiment_points_df("exp1")
        for frame, bodypart, cam, dx, dy in RELABELLED:
            row = df.loc[("exp1", "trial1", frame, 0, bodypart)]
            assert row[(cam, "source")] == "label3d"
            assert row[(cam, "px")] == pytest.approx(row[(cam, "x")] - dx)
            assert row[(cam, "py")] == pytest.approx(row[(cam, "y")] - dy)
            assert row[(cam, "pscore")] == pytest.approx(0.8)  # the point's score in the original
        other = df[("cam1", "px")].drop(("exp1", "trial1", 2, 0, "snout"))
        assert other.isna().all()

    @pytest.mark.parametrize("cam", CAMERAS)
    def test_slp_keeps_what_was_outside_the_project(self, upgraded, cam):
        _, root, originals, _ = upgraded
        restored = sio.load_slp(str(root / "slp" / cam / f"{cam}.slp"))
        original = sio.load_slp(str(originals / f"{cam}.slp"))
        assert len(restored.videos) == len(original.videos) == 3

        extra = EXTRA_VIDEO.format(camera=cam)
        [frame] = _frames(restored, extra).values()
        [want] = _frames(original, extra).values()
        np.testing.assert_allclose(frame.instances[0].numpy(), want.instances[0].numpy())

        trial_video = f"{cam}_exp1_trial1.mp4"
        restored_frames, original_frames = _frames(restored, trial_video), _frames(original, trial_video)
        for idx in range(LAST_TRIAL_FRAME + 1, 10):
            np.testing.assert_allclose(restored_frames[idx].instances[0].numpy(),
                                       original_frames[idx].instances[0].numpy())
        assert {Path(s.video.filename).name for s in restored.suggestions} == {
            Path(s.video.filename).name for s in original.suggestions
        }

    def test_user_label_is_written_over_its_prediction(self, upgraded):
        _, root, _, _ = upgraded
        labels = sio.load_slp(str(root / "slp" / "cam1" / "cam1.slp"))
        frame = _frames(labels, "cam1_exp1_trial1.mp4")[2]
        users = [i for i in frame.instances if type(i) is sio.Instance]
        assert len(users) == 1 and users[0].from_predicted is not None

    def test_slp_files_have_folders_and_mapping(self, upgraded):
        _, root, _, _ = upgraded
        config = toml.load(root / "project.toml")
        assert [f["path"] for f in config["slp"]["files"]] == ["slp/cam1/cam1.slp",
                                                               "slp/cam2/cam2.slp"]
        assert not list((root / "slp").glob("*.slp"))

    def test_trials_numbered_from_1(self, upgraded):
        version, root, _, _ = upgraded
        raw = pd.read_csv(root / "trials.csv")
        assert raw["cam1_start"].tolist() == [1, 1]
        assert raw["cam1_end"].tolist() == [LAST_TRIAL_FRAME + 1] * 2
        assert toml.load(root / "project.toml")["trials"]["frame_base"] == 1

    def test_backup_holds_the_old_files(self, upgraded):
        version, root, _, result = upgraded
        assert result.backup_dir.parent == root / UPGRADE_BACKUP_DIR
        assert result.backup_dir.name.startswith(f"{version}_")
        old = FIXTURES / f"v{version}" / "project"
        for rel in ["project.toml", "trials.csv", "slp/cam1.slp", "slp/cam2.slp"]:
            assert (result.backup_dir / rel).read_bytes() == (old / rel).read_bytes()

    def test_sync_backups_stay(self, upgraded):
        _, root, _, _ = upgraded
        assert list((root / "slp" / "backup").glob("*/cam1.slp"))


class TestUpgrade009:
    @pytest.fixture
    def root(self, tmp_path, qapp):
        root, originals = _copy("0.0.9", tmp_path)
        _upgrade(root, originals)
        return root

    def test_no_csv_working_files_or_skeleton_json(self, root):
        assert not list(root.glob("experiments/*/points.csv*"))
        assert not list(root.glob("experiments/*/calibrations/*/corners.csv"))
        assert not (root / "sync_history.csv").exists()
        assert not (root / "skeleton.json").exists()

    def test_corners_converted(self, root):
        old = FIXTURES / "v0.0.9/project/experiments/exp1/calibrations/cam1/corners.csv"
        new = pd.read_parquet(root / "experiments/exp1/calibrations/cam1/corners.parquet")
        want = pd.read_csv(old, dtype={"take": str, "detected": str})
        assert new["take"].tolist() == want["take"].tolist()
        assert new["detected"].tolist() == want["detected"].tolist()
        np.testing.assert_allclose(new["x"], want["x"], rtol=1e-6)

    def test_sync_history_converted(self, root):
        history = Project.load(root).sync_history()
        old = pd.read_csv(FIXTURES / "v0.0.9/project/sync_history.csv")
        assert history["id"].tolist()[: len(old)] == old["id"].tolist()
        assert history["experiment"].tolist()[: len(old)] == old["experiment"].astype(str).tolist()

    def test_skeleton_in_project_toml(self, root):
        [entry] = toml.load(root / "project.toml")["skeleton"]["assigned"]
        assert entry["nodes"] == ["snout", "left_eye", "dorsal"]
        assert entry["cameras"] == CAMERAS


def test_refinement_records_converted(tmp_path, qapp):
    root, originals = _copy("0.0.9", tmp_path)
    refined = root / "experiments/exp1/calibrations/refined"
    refined.mkdir()
    (refined / "refine_1.toml").write_text(toml.dumps(
        {"metadata": {"id": "refine_1", "points": {"points_file": "refine_1_points.csv"}}}
    ))
    pd.DataFrame({
        "trial": ["trial1"], "frame": [2], "instance": [0], "bodypart": ["snout"],
        "camera": ["cam1"], "x": [1.5], "y": [2.5], "source": [np.nan], "score": [np.nan],
        "holdout": [False],
    }).to_csv(refined / "refine_1_points.csv", index=False)

    _upgrade(root, originals)
    data = toml.load(refined / "refine_1.toml")
    assert data["metadata"]["points"]["points_file"] == "refine_1_points.parquet"
    record = pd.read_parquet(refined / "refine_1_points.parquet")
    assert record["source"].tolist() == [""] and record["x"].tolist() == [1.5]
    assert not (refined / "refine_1_points.csv").exists()


# -- Choosing the originals -----------------------------------------------------


def test_keeping_the_project_copy(tmp_path, qapp):
    root, _ = _copy("0.1.0", tmp_path)
    plan = plan_upgrade(root)
    result = run_upgrade(root, {p: None for p in plan.restore})
    labels = sio.load_slp(str(root / "slp" / "cam1" / "cam1.slp"))
    assert len(labels.videos) == 2
    assert any("kept the project's copy" in m for m in result.messages)


def test_missing_original_is_refused_before_anything_changes(tmp_path):
    root, _ = _copy("0.1.0", tmp_path)
    before = _digests(root)
    with pytest.raises(ValueError, match="No original chosen"):
        run_upgrade(root, {})
    assert _digests(root) == before
    assert not (root / UPGRADE_BACKUP_DIR).exists()


def test_earliest_backup_is_the_original(tmp_path):
    root, originals = _copy("0.1.0", tmp_path)
    found = earliest_backup(root, "cam1.slp")
    assert found.parent.parent == root / "slp" / "backup"
    assert found.read_bytes() == (originals / "cam1.slp").read_bytes()
    assert earliest_backup(root, "other.slp") is None


def test_find_originals_lists_every_match(tmp_path):
    for sub in ["a", "b", "a/models"]:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
        (tmp_path / sub / "cam1.slp").write_bytes(b"")
    found = find_originals(["cam1.slp", "cam2.slp"], tmp_path)
    assert found["cam1.slp"] == [tmp_path / "a" / "cam1.slp", tmp_path / "b" / "cam1.slp"]
    assert found["cam2.slp"] == []


def test_models_folder_copied_beside_its_own_slp(tmp_path, qapp):
    root, originals = _copy("0.1.0", tmp_path)
    (originals / "cam1").mkdir()
    shutil.move(originals / "cam1.slp", originals / "cam1" / "cam1.slp")
    (originals / "cam1" / "models" / "run1").mkdir(parents=True)
    (originals / "cam1" / "models" / "run1" / "best.ckpt").write_text("weights")
    (originals / "models").mkdir()  # beside cam2.slp, but also beside nothing else
    _upgrade(root, originals)
    assert (root / "slp/cam1/models/run1/best.ckpt").read_text() == "weights"
    assert (root / "slp/cam2/models").is_dir()


def test_shared_models_folder_is_not_copied(tmp_path, qapp):
    root, originals = _copy("0.1.0", tmp_path)
    (originals / "models").mkdir()
    result = _upgrade(root, originals)
    assert not (root / "slp/cam1/models").exists()
    assert any("beside several .slp files" in m for m in result.messages)


# -- Rolling back ------------------------------------------------------------------


@pytest.mark.parametrize("step", ["_renumber_frames", "restore_slp_files"])
def test_failed_upgrade_leaves_the_project_as_it_was(tmp_path, qapp, monkeypatch, step):
    root, originals = _copy("0.0.9", tmp_path)
    before = _digests(root)
    plan = plan_upgrade(root)

    def fail(*args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(upgrade, step, fail)
    with pytest.raises(RuntimeError, match="stopped"):
        run_upgrade(root, _originals(plan, originals))
    assert _digests(root) == before
    assert not (root / "slp" / "cam1").exists()
    assert list((root / UPGRADE_BACKUP_DIR).iterdir())


# -- label3d-upgrade --------------------------------------------------------------


class TestUpgradeCli:
    def test_dry_run_changes_nothing(self, tmp_path, capsys):
        root, originals = _copy("0.1.0", tmp_path)
        before = _digests(root)
        assert upgrade_cli.main([str(root), "-n", "--slp-source", str(originals)]) == 0
        assert _digests(root) == before
        assert "slp/cam1.slp: from" in capsys.readouterr().out

    def test_slp_option_needed(self, tmp_path, capsys):
        root, _ = _copy("0.1.0", tmp_path)
        assert upgrade_cli.main([str(root)]) == 2
        assert "--slp-source" in capsys.readouterr().err
        assert not (root / UPGRADE_BACKUP_DIR).exists()

    def test_several_originals_is_an_error(self, tmp_path, capsys):
        root, originals = _copy("0.1.0", tmp_path)
        (originals / "copy").mkdir()
        shutil.copy2(originals / "cam1.slp", originals / "copy" / "cam1.slp")
        assert upgrade_cli.main([str(root), "--slp-source", str(originals)]) == 2
        assert "several originals" in capsys.readouterr().err

    @pytest.mark.parametrize("option", ["--slp-source", "--slp-from-backup"])
    def test_upgrades(self, tmp_path, qapp, option):
        root, originals = _copy("0.1.0", tmp_path)
        args = [str(root), option] + ([str(originals)] if option == "--slp-source" else [])
        assert upgrade_cli.main(args) == 0
        assert len(sio.load_slp(str(root / "slp/cam1/cam1.slp")).videos) == 3
        assert upgrade_cli.main([str(root)]) == 0  # nothing left to do

    def test_export_cli_points_to_the_upgrade(self, tmp_path, capsys):
        root, _ = _copy("0.1.0", tmp_path)
        assert export_cli.main([str(root)]) == 1
        assert "label3d-upgrade" in capsys.readouterr().err


# -- In the app ----------------------------------------------------------------------


class TestUpgradeDialog:
    def test_defaults_to_the_earliest_backup(self, tmp_path, qapp):
        root, _ = _copy("0.1.0", tmp_path)
        dialog = UpgradeDialog(plan_upgrade(root))
        sources = dialog.sources()
        assert list(sources) == [Path("slp/cam1.slp"), Path("slp/cam2.slp")]
        assert all(src.parent.parent == root / "slp" / "backup" for src in sources.values())

    def test_search_folder_selects_the_originals(self, tmp_path, qapp, monkeypatch):
        root, originals = _copy("0.1.0", tmp_path)
        dialog = UpgradeDialog(plan_upgrade(root))
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(originals))
        dialog._search_folder()
        assert dialog.sources() == {Path("slp/cam1.slp"): originals / "cam1.slp",
                                    Path("slp/cam2.slp"): originals / "cam2.slp"}

    def test_no_restore_section_without_files_to_restore(self, tmp_path, qapp):
        root, _ = _copy("0.0.9", tmp_path)
        config = toml.load(root / "project.toml")
        config.pop("slp")
        (root / "project.toml").write_text(toml.dumps(config))
        assert UpgradeDialog(plan_upgrade(root)).sources() == {}

    def test_upgrade_project_runs_the_worker(self, tmp_path, qapp, monkeypatch):
        root, _ = _copy("0.1.0", tmp_path)
        monkeypatch.setattr(UpgradeDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
        shown = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
        assert upgrade_project(plan_upgrade(root))
        assert "restored from" in shown[0]
        assert Project.load(root).config["label3d"]["project_version"] == LAYOUT_VERSION

    def test_failed_upgrade_is_reported(self, tmp_path, qapp, monkeypatch):
        root, _ = _copy("0.1.0", tmp_path)
        before = _digests(root)
        monkeypatch.setattr(UpgradeDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
        monkeypatch.setattr(upgrade, "_renumber_frames",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("stopped")))
        errors = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
        assert not upgrade_project(plan_upgrade(root))
        assert "stopped" in errors[0]
        assert _digests(root) == before

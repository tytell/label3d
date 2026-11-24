"""Incremental SLEAP discovery, scoped imports and rollback on existing projects."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio

from label3d import add_videos
from label3d.add_videos import (
    AddVideosError, CalibrationVideo, DiscoveryOptions, apply_additions,
    calibration_candidates, discover_videos,
)
from label3d.points import read_points_df
from label3d.project import Project, write_trials_csv
from label3d.setup_project import write_project_toml
from label3d.slp_sync import SlpMapping

CAMERAS = ["left", "right"]
OLD = "20260729"
NEW = "20260805"


def video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_video(np.zeros((5, 16, 16, 1), dtype=np.uint8), str(path))
    return path


def labels(path, paths, *, labeled=True, windows=False):
    skeleton = sio.Skeleton(["head", "tail"])
    skeleton.add_edge("head", "tail")
    videos, frames = [], []
    for i, path_video in enumerate(paths):
        name = str(path_video)
        if windows:
            name = "Z:\\NewData\\" + "\\".join(path_video.parts[path_video.parts.index("3_edited_fish_videos"):])
        v = sio.Video(filename=name, open_backend=False)
        videos.append(v)
        if labeled:
            instance = sio.PredictedInstance.from_numpy(
                np.array([[10.0 + i * 100, 20.0], [30.0, 40.0]]),
                skeleton=skeleton, point_scores=np.array([0.8, 0.9]), score=0.85,
            )
            frames.append(sio.LabeledFrame(video=v, frame_idx=0, instances=[instance]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_file(sio.Labels(videos=videos, skeletons=[skeleton], labeled_frames=frames), str(path))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def existing(tmp_path, qapp):
    collection = tmp_path / "3_edited_fish_videos"
    old_dir = collection / f"{OLD}_zebrafish_1_synced_trimmed"
    new_dir = collection / f"{NEW}_zebrafish_1_synced_trimmed"
    old = [video(old_dir / f"trial01_{cam}.mp4") for cam in CAMERAS]
    new = [video(new_dir / f"trial01_{cam}.mp4") for cam in CAMERAS]
    root = tmp_path / "project"
    root.mkdir()
    (root / 'experiments' / OLD).mkdir(parents=True)
    source = root / "slp" / "labels" / "labels.slp"
    labels(source, old)
    row = {"experiment": OLD, "trial": "trial01", "type": "trial"}
    for cam, path in zip(CAMERAS, old):
        row.update({cam: path.name, f"{cam}_start": 0, f"{cam}_end": 4})
    write_trials_csv(root / "trials.csv", pd.DataFrame([row]), backup=False)
    mapping = SlpMapping.from_cell_map({(OLD, "trial01", cam): source.relative_to(root) for cam in CAMERAS})
    write_project_toml(root, "existing", video_roots={OLD: old_dir}, slp_mapping=mapping.to_config())
    project = Project.load(root)
    project.apply_merge(project.read_slp_for_sync(), overwrite=False)
    labels(source, [*old, *new])
    return project, source, old, new


def test_discover_all_videos_even_without_labels(existing):
    project, source, old, new = existing
    labels(source, [*old, *new], labeled=False)
    before = {p: digest(p) for p in (project.trials_csv_path, project.root / 'project.toml', source)}
    found = discover_videos(project)
    assert found.registered == 2
    assert len(found.candidates) == 2
    assert {v.path for v in found.candidates} == set(new)
    assert {v.experiment for v in found.candidates} == {NEW}
    assert {v.trial for v in found.candidates} == {'trial01'}
    assert {v.camera for v in found.candidates} == set(CAMERAS)
    assert all(digest(p) == sha for p, sha in before.items())


def test_resolve_windows_paths_from_existing_collection(existing):
    project, source, old, new = existing
    labels(source, [*old, *new], windows=True)
    found = discover_videos(project)
    assert {v.path for v in found.candidates} == set(new)
    assert found.registered == 2
    result = apply_additions(project, found, found.candidates, {NEW: []})
    result.refresh(project)
    assert len(project.experiments) == 2
    assert not discover_videos(project).candidates


def test_add_preserves_old_points_slp_and_settings_and_is_idempotent(existing):
    project, source, old, new = existing
    project.save_config_sections({'charuco': {'squares_x': 8, 'squares_y': 10},
                                  'alignment': {OLD: {'sentinel': 'keep'}}})
    old_points = project.experiments[OLD].root / 'points.parquet'
    previous, slp_hash = old_points.read_bytes(), digest(source)
    found = discover_videos(project)
    result = apply_additions(project, found, found.candidates, {NEW: []})
    assert old_points.read_bytes() == previous
    assert digest(source) == slp_hash
    result.refresh(project)
    assert project.config['alignment'][OLD]['sentinel'] == 'keep'
    assert project.config['charuco']['squares_x'] == 8
    assert project.slp_mapping.is_complete(project.trials_df, CAMERAS)
    incoming = read_points_df(project.experiments[NEW].root / 'points.parquet')
    assert incoming.loc[(NEW, 'trial01', 0, 0, 'head'), ('left', 'x')] == 210.0
    assert incoming.loc[(NEW, 'trial01', 0, 0, 'head'), ('right', 'x')] == 310.0
    assert set(project.experiments) == {OLD, NEW}
    assert not discover_videos(project).candidates
    assert (result.backup_dir / 'trials.csv').exists()
    assert (result.backup_dir / 'manifest.json').exists()


def test_sync_identical_basenames_keeps_each_experiments_video(existing):
    project, source, old, new = existing
    found = discover_videos(project)
    result = apply_additions(project, found, found.candidates, {NEW: []})
    result.refresh(project)
    project.write_slp_files()
    loaded = sio.load_file(str(source))
    assert len(loaded.videos) == 4
    xs = {str(lf.video.filename): lf.instances[0].numpy()[0, 0] for lf in loaded.labeled_frames}
    assert xs[str(old[0])] == 10.0
    assert xs[str(new[0])] == 210.0
    assert not any(r.has_changes or r.conflicts for r in project.read_slp_for_sync().values())


@pytest.mark.parametrize('windows', [False, True])
def test_sync_before_registration_does_not_import_new_siblings_into_old_trials(existing, windows):
    project, source, old, new = existing
    labels(source, [*old, *new], windows=windows)
    results = project.read_slp_for_sync()
    assert not any(r.has_changes or r.conflicts for r in results.values())
    project.write_slp_files()
    assert len(sio.load_file(str(source)).videos) == 4


def test_ambiguous_basename_never_imports_into_two_experiments(tmp_path):
    from label3d.video_matching import AmbiguousVideoError, VideoMatcher
    trials = pd.DataFrame([
        {'experiment': 'a', 'trial': 'trial01', 'left': '/data/a/left.mp4', 'left_start': 0},
        {'experiment': 'b', 'trial': 'trial01', 'left': '/data/b/left.mp4', 'left_start': 0},
    ])
    matcher = VideoMatcher(trials, ['left'])
    with pytest.raises(AmbiguousVideoError):
        matcher.matches('left.mp4')
    assert matcher.matches('/data/b/left.mp4') == [('b', 'trial01', 'left')]


def test_calibration_files_need_no_experiment_keyword(existing, tmp_path):
    project, source, old, new = existing
    calib = [video(tmp_path / 'calibrations' / cam / f'cal01_{cam}.mp4') for cam in CAMERAS]
    candidates = calibration_candidates(project, calib, r'cal\d+')
    assert {v.take for v in candidates} == {'cal01'}
    found = discover_videos(project)
    result = apply_additions(project, found, found.candidates, {NEW: candidates})
    result.refresh(project)
    project.set_experiment(NEW)
    assert [v.name for v in project.calibration_takes] == ['cal01']
    for cam, path in zip(CAMERAS, calib):
        copied = project.video_path('cal01', cam)
        assert copied == project.experiments[NEW].root / 'calibrations' / cam / path.name
        assert copied.read_bytes() == path.read_bytes()


def test_calibration_choice_required_for_new_experiment(existing):
    project, *_ = existing
    found = discover_videos(project)
    with pytest.raises(AddVideosError, match='calibration'):
        apply_additions(project, found, found.candidates, {})


def test_incomplete_trials_require_explicit_confirmation(existing):
    project, *_ = existing
    found = discover_videos(project)
    with pytest.raises(AddVideosError, match='missing right'):
        apply_additions(project, found, found.candidates[:1], {NEW: []})
    result = apply_additions(project, found, found.candidates[:1], {NEW: []}, allow_incomplete=True)
    assert result.n_videos == 1


def test_duplicate_assignments_rejected_before_writing(existing):
    project, *_ = existing
    found = discover_videos(project)
    found.candidates[1].camera = found.candidates[0].camera
    before = project.trials_csv_path.read_bytes()
    with pytest.raises(AddVideosError, match='Several videos assigned'):
        apply_additions(project, found, found.candidates, {NEW: []})
    assert project.trials_csv_path.read_bytes() == before


def test_source_changed_after_scan_rejected(existing):
    project, source, old, new = existing
    found = discover_videos(project)
    labels(source, [*old, *new], labeled=False)
    with pytest.raises(AddVideosError, match='changed since discovery'):
        apply_additions(project, found, found.candidates, {NEW: []})


def test_failure_importing_does_not_change_project(existing, monkeypatch):
    project, source, *_ = existing
    found = discover_videos(project)
    before = {p: p.read_bytes() for p in (source, project.root / 'project.toml', project.trials_csv_path)}
    def fail(*args, **kwargs):
        raise RuntimeError('bad SLEAP data')
    monkeypatch.setattr(add_videos.SleapAdapter, 'load', fail)
    with pytest.raises(RuntimeError, match='bad SLEAP data'):
        apply_additions(project, found, found.candidates, {NEW: []})
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not (project.root / 'experiments' / NEW).exists()


def test_failed_commit_restores_replaced_files_and_removes_new_ones(existing, monkeypatch):
    project, source, *_ = existing
    found = discover_videos(project)
    before = {p: p.read_bytes() for p in (source, project.root / 'project.toml', project.trials_csv_path, project.sync_history_path)}
    replace = add_videos.os.replace
    calls = 0
    def fail_once(src, dest):
        nonlocal calls
        if '.add-videos-' in str(src):
            calls += 1
            if calls == 3:
                raise OSError('simulated write failure')
        return replace(src, dest)
    monkeypatch.setattr(add_videos.os, 'replace', fail_once)
    with pytest.raises(OSError, match='simulated write failure'):
        apply_additions(project, found, found.candidates, {NEW: []})
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not (project.root / 'experiments' / NEW / 'points.parquet').exists()
    assert set(project.experiments) == {OLD}


def test_new_trials_in_existing_experiment_keep_old_labels(existing):
    project, source, old, _ = existing
    additional = [video(old[0].parent / f'trial02_{cam}.mp4') for cam in CAMERAS]
    labels(source, [*old, *additional])
    before = project.points.df.copy()
    found = discover_videos(project)
    assert {v.experiment for v in found.candidates} == {OLD}
    result = apply_additions(project, found, found.candidates, {})
    result.refresh(project)
    assert project.trial_names() == ['trial01', 'trial02']
    pd.testing.assert_frame_equal(project.points.df.loc[before.index], before)
    assert result.n_experiments == 0


def test_unlabeled_videos_register_without_predictions(existing):
    project, source, old, new = existing
    labels(source, [*old, *new], labeled=False)
    found = discover_videos(project)
    result = apply_additions(project, found, found.candidates, {NEW: []})
    result.refresh(project)
    assert NEW in project.experiments
    assert project.experiments[NEW].trials[0].name == 'trial01'
    assert (project.experiments[NEW].root / 'points.parquet').exists()


def test_new_root_and_search_locations_use_named_drives(existing):
    from label3d.drives import current_platform
    project, source, old, new = existing
    collection = old[0].parent.parent
    project.save_config_sections({'drives': {current_platform(): {'lab': str(collection.parent)}}})
    options = DiscoveryOptions(search_dirs=[collection])
    found = discover_videos(project, options)
    result = apply_additions(project, found, found.candidates, {NEW: []})
    assert result.config['video_roots'][NEW].startswith('{lab}/')
    assert result.config['discovery']['search_dirs'] == ['{lab}/3_edited_fish_videos']
    result.refresh(project)
    assert not discover_videos(project).candidates


def test_explicitly_located_renamed_video_is_remembered_for_read_and_write(existing):
    project, source, old, new = existing
    # The stored references no longer exist and the recordings have new names.
    for path in new:
        path.rename(path.with_name(path.name.replace('trial01', 'renamed_trial01')))
    found = discover_videos(project)
    assert len(found.candidates) == 2
    assert all(v.path is None for v in found.candidates)
    for v, original in zip(found.candidates, new):
        v.path = original.with_name(original.name.replace('trial01', 'renamed_trial01'))
        v.camera = 'left' if 'left' in original.name else 'right'
        v.trial, v.issue = 'trial01', ''
    result = apply_additions(project, found, found.candidates, {NEW: []})
    result.refresh(project)
    assert len(project.config['slp_video_paths']['slp/labels/labels.slp']) == 2
    assert not discover_videos(project).candidates
    assert not any(r.has_changes for r in project.read_slp_for_sync().values())
    project.write_slp_files()
    loaded = sio.load_file(str(source))
    assert len(loaded.videos) == 4
    values = [lf.instances[0].numpy()[0, 0] for lf in loaded.labeled_frames]
    assert set(values) == {10.0, 110.0, 210.0, 310.0}


def test_add_missing_camera_to_existing_trial_preserves_other_camera(existing):
    project, source, old, new = existing
    project.trials_df.at[0, 'right'] = ''
    project.trials_df.at[0, 'right_start'] = pd.NA
    project.trials_df.at[0, 'right_end'] = pd.NA
    write_trials_csv(project.trials_csv_path, project.trials_df)
    mapping = project.slp_mapping.cell_map()
    del mapping[(OLD, 'trial01', 'right')]
    project.save_slp_mapping(SlpMapping.from_cell_map(mapping))
    # Keep only left camera data in the old store, like a partially set-up trial.
    df = project.points.df.copy()
    for column in df.columns:
        if column[0] == 'right':
            df[column] = 'not_set' if column[1] == 'state' else '' if column[1] == 'source' else np.nan
    project.store_experiment_df(OLD, df)
    labels(source, old)
    found = discover_videos(project)
    assert len(found.candidates) == 1
    result = apply_additions(project, found, found.candidates, {})
    result.refresh(project)
    assert project.points.get_point('trial01', 0, 0, 'head', 'left').x == 10
    assert project.points.get_point('trial01', 0, 0, 'head', 'right').x == 110
    assert project.slp_mapping.is_complete(project.trials_df, CAMERAS)


def test_recursive_calibration_folder_selection(existing, tmp_path, qapp, monkeypatch):
    from label3d.widgets.add_videos_dialog import CalibrationVideosDialog
    from PySide6.QtWidgets import QFileDialog
    project, *_ = existing
    folder = tmp_path / 'calibration recordings'
    for cam in CAMERAS:
        video(folder / cam / f'cal01_{cam}.mp4')
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *a, **k: str(folder))
    dialog = CalibrationVideosDialog(project, NEW, r'cal\d+')
    dialog._folder()
    assert dialog.table.rowCount() == 2
    dialog._use()
    assert len(dialog.selection) == 2
    assert {v.camera for v in dialog.selection} == set(CAMERAS)
    dialog.deleteLater()


def test_keyword_pairing_uses_camera_aliases(existing):
    project, source, old, new = existing
    project.save_config_sections({'camera_aliases': {'left': ['L'], 'right': ['R']}})
    additional = [video(new[0].parent / f'{cam}_trial02.mp4') for cam in ('L', 'R')]
    labels(source, [*old, *additional])
    found = discover_videos(project)
    assert {v.trial for v in found.candidates} == {'trial02'}
    assert {v.camera for v in found.candidates} == set(CAMERAS)


def test_dialog_lists_discovered_videos_and_allows_bulk_selection(existing, qapp):
    from label3d.widgets.add_videos_dialog import AddVideosDialog
    project, *_ = existing
    dialog = AddVideosDialog(project)
    # Populate synchronously without starting its scheduled background scan.
    worker = type('Result', (), {'error': '', 'result': discover_videos(project)})()
    dialog._scan_done(worker)
    assert dialog.table.rowCount() == 2
    assert len(dialog._selected()) == 2
    assert dialog.add_button.isEnabled()
    dialog._select_all(False)
    assert not dialog.add_button.isEnabled()
    dialog._select_all(True)
    assert dialog.add_button.isEnabled()
    dialog.table.item(1, 2).setText('another_trial')
    assert len(dialog._selected()) == 2
    dialog.deleteLater()


def test_main_window_refreshes_experiment_selector_without_another_global_read(existing, qapp, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from label3d.state import ProjectState
    from label3d.widgets import main_window
    project, source, *_ = existing
    found = discover_videos(project)
    result = apply_additions(project, found, found.candidates, {NEW: []})
    class Accepted:
        addition_result = result
        def __init__(self, *args):
            pass
        def exec(self):
            return QDialog.DialogCode.Accepted
    monkeypatch.setattr(main_window, 'AddVideosDialog', Accepted)
    receipts = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: receipts.append(args[-1]))
    window = main_window.MainWindow(project, ProjectState())
    reads = []
    monkeypatch.setattr(window, '_read_slp_on_open', lambda: reads.append(True))
    window._do_add_videos()
    assert [window._exp_combo.itemText(i) for i in range(window._exp_combo.count())] == [OLD, NEW]
    assert not window._exp_combo.isHidden()
    assert window._state.points_store is project.points
    assert window._state.skeletons is project.skeletons
    assert project.active_experiment == OLD
    assert not reads
    assert 'Imported 4 SLEAP point(s)' in receipts[0]
    window._frame_provider.close()
    window.deleteLater()

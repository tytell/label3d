# Phase 10 — Point Provenance and Reviewing New Points

**Goal**: After a sync with SLEAP, show the user which points are new and where every point came from. There are three sources: labelled by a user in label3d, labelled by a user in SLEAP, and predicted by SLEAP. Mark frames with new points on the frame slider, mark the points themselves in the camera views, and add ways to jump to new frames and accept them.

**Status**: Complete. See *As implemented* at the end.

**Compatibility**: The `points.csv` schema changes with no fallback. When an old file is read, the missing columns are filled in. Existing points get `source = label3d`, and there's no attempt to work out where they came from.

**Background**: Phase 8 deliberately kept no sync history: a sync is a plain comparison of SLP contents against `points.csv`. A point labelled in SLEAP comes in as `set`, which is the same as a point labelled in label3d. SLEAP point scores are dropped on load, and every prediction is written back with `score=1.0`. State alone can't separate the sources either, because label3d makes `predicted` points of its own (the reprojection guesses in `instance_init.py`).

---

## 10.1 Data model (`points.py`)

Each camera's coords go from `{x, y, xr, yr, state}` to `{x, y, xr, yr, state, source, score, sync}`.

| Column | Values | Meaning |
|---|---|---|
| `source` | `label3d`, `sleap_user`, `sleap_pred` | Where the current x/y came from. Empty for `not_set`. |
| `score` | float or NaN | SLEAP per-point score for `sleap_pred`; NaN otherwise |
| `sync` | int or empty | Id of the sync read (10.3) that last brought this cell in from SLEAP. Cleared when the user corrects or accepts the point. |

- `source` doesn't depend on `state`. A label3d reprojection guess is `state=predicted`, `source=label3d`.
- Every label3d edit (place, drag, mark not visible, reset) sets `source = label3d` and clears `sync`. That includes `set_point`, `set_state`, the `instance_init` strategies and deletes.
- `CAM_COORDS`, `empty_df()`, `read_csv_df()` and `PointData` gain the new fields. `read_csv_df()` fills in missing columns as described under Compatibility.
- **New-point test:** a point is *new* when its `sync` equals the latest sync id in `sync_history.csv` (10.3).

## 10.2 SLEAP adapter (`adapters/sleap_adapter.py`)

- **`load()`:** emit `source` (`sleap_user` for `sio.Instance`, `sleap_pred` for `sio.PredictedInstance`). Also emit `score`, taken from `inst.numpy(scores=True)` for predictions.
- **`save()`:** write the stored per-point scores into `PredictedInstance.from_numpy(..., point_scores=...)`. The instance score is the mean of the point scores, rather than a fixed 1.0.
- **Round trip:** a label3d `set` point is written as a user `Instance` and reads back as `sleap_user` at the same x/y. The merge treats that as "no change" (8.8) and keeps the stored `source = label3d`, so a round trip doesn't change any source.

## 10.3 Merge and sync history (`slp_sync.py`, `project.py`)

**`compute_merge()`**
- Every cell taken from SLP gets its `source` and `score` copied from the SLP data, and its `sync` set to the current sync id.
- A `predicted` → `predicted` replacement counts as updated only if the point moved by more than `PREDICTION_MOVE_PX` (default 0.5 px). Otherwise the stored x/y, `source` and `sync` stay as they are, so re-running inference doesn't mark every prediction as new. The score is still updated.
- Conflicts resolved with **Use SLEAP** count as new `sleap_user` points.
- `MergeResult` gains counts split by source: `n_new_user`, `n_new_pred`, `n_updated_pred`. It also gains `new_frames: dict[(experiment, trial), set[int]]`.

**Sync id and history**
- Every read that merges at least one cell (the startup read in 8.11, or Sync in 8.10) gets a new sync id and appends a row to `project/sync_history.csv`.
- Columns: `id, timestamp, experiment, trial, n_new_user, n_new_pred, n_updated_pred, n_conflicts`, with one row per (experiment, trial) that changed.
- A read that changes nothing adds no row, so the previous sync's new points stay new.
- **New method:** `Project.latest_sync_id() -> int | None`.
- `apply_merge()` is unchanged. The ids are already in the merged `df` for the active experiment and in the `points.csv` written for the others.

## 10.4 Import predictions (`main_window.py`, `project.py`, `adapters/sleap_adapter.py`)

**Why:** inference in the SLEAP GUI merges predictions into the open labels file, so they reach label3d through the mapped files and the normal sync. Command-line inference doesn't. `sleap-nn track -i labels.slp -m models/ -o out.slp` writes a **separate** predictions file, and running on raw videos (`-i video.mp4`) gives one file per video. label3d currently never reads these files, so step 4 of the workflow by command line and step 5 (inference on all frames) both need this step.

**Menu:** **File > Import predictions...** opens a file dialog for one or more `.slp` files, anywhere on disk. It's enabled whenever a project is open; it doesn't need a complete SLP mapping.

**Matching**
- Each file's videos are matched to (experiment, trial, camera) cells by basename with `_build_basename_lookup()`, the same way as `build_file_map()`. A video path stored on another machine, such as a cluster, still matches.
- Frames are converted with the trial's `cam_start`, as in `load()`. Frames before `cam_start` or after `cam_end` are dropped.
- Videos that match no cell are listed in the summary and skipped.

**Merge**
- The imported data goes through `compute_merge()` with a new sync id, exactly like a sync read.
  - Predictions come in as `sleap_pred` with their scores.
  - Any user instances in the file come in as `sleap_user`. A predictions file made from a labels file may contain them, and the normal merge rules decide what happens to them.
  - The frame rule from 8.8 still applies: predictions are ignored on frames where the store has a user instance.
- The conflict warning (8.10) and the summary dialog (10.5) are the same as for a sync.
- **New method:** `Project.read_predictions_for_import(paths) -> dict[str, MergeResult]`. It reuses `SleapAdapter.load(files=paths, ...)` with no mapping, then `compute_merge()` per experiment.
- It runs in `SlpReadWorker` (a `files=` argument is added), then `apply_merge()`, then re-triangulates the changed frames of the active experiment.

**What is written**
- The imported files are only read: they aren't added to the mapping and they aren't modified.
- The imported points are written into the mapped SLP files at the next **Sync with Sleap**, so SLEAP sees them for the next round of training or proofreading.
- The status bar notes this after an import: "Imported N predictions; Sync to write them to the SLP files".

## 10.5 Sync summary dialog (`widgets/sync_summary_dialog.py`)

Shown after any read that merged something, and after an import (10.4). It replaces the status-bar-only report.

**Sync on close.** The sync that runs when the app closes (8.12) still gets a sync id and a history row, but shows no dialog. Its summary is available on the next launch through **File > Last sync summary...**.

- **Table:** experiment × trial, with columns for new SLEAP user labels, new predictions, updated predictions, conflicts and frames with new points. Rows are sorted with the active experiment first.
- **Buttons:** **Go to first new frame** (switches trial if needed) and **Close**.
- **Menu:** the latest summary can be reopened from **File > Last sync summary...** (rebuilt from `sync_history.csv`).

## 10.6 Frame slider and navigation (`widgets/frame_slider.py`, `state.py`, `main_window.py`)

**Slider**
- Add a "new" row above the per-camera rows. It shows a tick for every frame of the current trial that has a new point in any camera. The tick is orange if any new point is a SLEAP user label, and blue (the SLEAP prediction colour) if all of them are predictions.
- The row is hidden when the trial has no new points, so the slider looks the same as now outside review.
- **API:** `set_new_marks(dict[int, str])`, with values `"user"` or `"predicted"`.

**Navigation**
- **Go > Next new frame** (`Shift+]`) and **Go > Previous new frame** (`Shift+[`), next to the existing `[` / `]` labelled-frame jumps.
- These live in `AppState` as `jump_to_next_new()` and `jump_to_prev_new()`, backed by `PointsStore.get_next_new_frame()` and `get_prev_new_frame()`.
- At the end of a trial they go on to the next trial that has new points.

## 10.7 Accepting points (`label_view.py`, `main_window.py`, `points.py`)

Accepting clears `sync` without changing x/y, `state` or `source`. It's how the user signals they've reviewed a point and it's fine as it is. Correcting a point (10.1) clears `sync` too.

| Action | Shortcut | Effect |
|---|---|---|
| Accept point | `A` | Clear `sync` on the selected bodypart in every camera |
| Accept frame | `Shift+A` | Clear `sync` on every point in the current frame |

- Both actions are in the **Instances** menu and the point right-click menu.
- **Store methods:** `PointsStore.accept_points(trial, frame, instance=None, bodypart=None)`.
- Accepting a prediction does **not** turn it into `set`. Turning a prediction into ground truth stays a separate decision, made by editing the point.

## 10.8 Marker styles (`widgets/point_item.py`, `skeleton_overlay.py`, `main_window.py`)

- **View > Color points by** is a pair of radio actions: **Reprojection error** (the default, as now) or **Source**. The choice is saved in the session settings.
- **Shape** still means state: a filled circle is `set`, a ring is `predicted`, grey is `not_visible`, and a small dot is `not_set`.
- **Colour in source mode:** label3d user, SLEAP user and SLEAP prediction each get one colour. The colours must stay distinguishable against video and from the error colours (green, yellow, red).
- **New points:** a square outline around the marker, in both colour modes.
- **Tooltip:** add `Source`, `Score` (predictions) and `New since sync <timestamp>`.
- `PointItem.configure()` gains `source`, `score` and `is_new`. `ColorMode` is passed down from `LabelView`.

## 10.9 Workflow guide, first draft (`docs/user_workflow.md`)

A first draft of the user-facing guide, written once 10.4–10.8 exist so it names real menus. It's finished in Phase 13.6.

1. **Set up a project:** from SLEAP files ([quick_setup_from_slp.md](../quick_setup_from_slp.md)) or from videos.
2. **Calibrate:** charuco takes, checking for drift between takes.
3. **Label in label3d,** then **File > Sync with Sleap**.
4. **Train and infer in SLEAP.** There are two routes:
   - **GUI:** open a mapped file in `slp/`, train, run inference and save. The predictions are in the mapped file, so label3d reads them on the next open or Sync.
   - **Command line:** `sleap-nn train` and `sleap-nn track`. The predictions are written to separate files, which label3d reads with **File > Import predictions...** (10.4).
5. **Review:** reopen label3d, read the sync summary, step through new frames with `Shift+]`, then correct or accept. Add more labels on frames the model gets wrong, sync, and train again.
6. **Final pass:** once the model is good enough, run inference on all frames of every video (command line, then Import predictions). Then refine the calibration (Phase 12) and batch-export (Phase 13).

Link it from the README and the DESIGN.md index.

## 10.10 Tests

- **Adapter:** `load()` gives `sleap_user`, `sleap_pred` and per-point scores, and `save()` → `load()` keeps the scores.
- **Merge:**
  - Source and sync on new cells.
  - A prediction moved less than the tolerance is not new.
  - **Use SLEAP** on a conflict gives a new `sleap_user` point.
  - A label3d point round trip keeps `source = label3d`.
- **History:** a read with changes adds rows and a new id. A read with no changes adds nothing, and `latest_sync_id()` doesn't change.
- **Import:**
  - A predictions file whose videos are referenced by a full path from another machine matches by basename.
  - Frames outside a trial's `cam_start` / `cam_end` are dropped.
  - Unmatched videos are reported.
  - Imported predictions get `sleap_pred`, scores and a new sync id.
  - The imported file is unchanged and is not added to the mapping.
  - After the next Sync the predictions are in the mapped file.
- **Store:**
  - An edit sets `source = label3d` and clears `sync`.
  - `accept_points()` for one point and for a whole frame.
  - The next/previous new frame lookups.
  - Old CSVs get the missing columns filled in.

## 10.11 Reference docs

Update the docs that describe current behaviour once the steps above are built:
- **`docs/data_model.md`:** the `points.csv` camera columns (`source`, `score`, `sync`) and `sync_history.csv`.
- **`docs/workflows.md`:**
  - The keyboard shortcut tables: `Shift+[` / `Shift+]`, `A` / `Shift+A`.
  - **File > Import predictions...** and **File > Last sync summary...**.
  - **View > Color points by**.
  - The review workflow.
- **CLAUDE.md:** the `points.csv` line under "Data files".

## Implementation order

1. 10.1 + 10.2 + tests (data in and out)
2. 10.3 + tests (merge and history)
3. 10.4 import predictions + tests
4. 10.7 store methods + 10.6 navigation
5. 10.8 markers, 10.6 slider row, 10.5 summary dialog
6. 10.11 reference docs, 10.9 guide draft

## As implemented

Differences from the plan above, and details it left open:

- **Sync id handling.** `read_slp_for_sync()` and `read_predictions_for_import()` give merged cells the *next* id, but `apply_merge(results, overwrite, kind)` records it in `sync_history.csv`. A cancelled or empty read therefore leaves no trace and doesn't use up an id.
  - `kind` is `open`, `sync`, `close`, `import` or `setup`.
  - The history has two columns the plan didn't list: `kind` and `n_frames`.
- **Project setup marks nothing new.** The initial label import (`setup_project.import_labels`) calls `read_slp_for_sync(sync=False)`. It records sources and scores, so a project built from existing SLEAP files doesn't start with every point new.
- **`sync` values.** `sync` is stored as a float column, so ids appear as `3.0` in `points.csv`.
- **Tooltip label.** The store's `latest_sync` and `latest_sync_label` (e.g. `import 4 (2026-10-05 14:22)`) are set by `Project` on load, on an experiment switch and after a merge.
- **label3d guesses keep their source.** A label3d reprojection guess is written to SLEAP as a prediction and read back at the same place. Any stored prediction within `PREDICTION_MOVE_PX` keeps its x/y, source and sync. Only a `sleap_pred`'s score is updated (`MergeResult.n_score_updates`).
- **`MergeResult` counts.** Counts are now per trial (`trial_counts`). `n_new`, `n_new_pred`, `n_updated_pred` and `n_predictions` are properties computed from them. `has_changes` covers updates that only change a score.
- **Trial bounds.** `SleapAdapter.load()` drops frames after `{cam}_end`, for every read, not just imports. It collects unmatched video basenames through an `unmatched` list argument instead of reading each file twice.
- **Navigation across trials.** Jumping past the end of a trial is handled in `MainWindow._jump_to_new()`, because only the main window can load another trial. `ProjectState.jump_to_next_new()` / `jump_to_prev_new()` handle the current trial and return False at its end.
  - The shortcuts are registered as both `}` / `{` and `Shift+]` / `Shift+[`, so they work whichever way Qt reports the key.
- **Accept.** Accept point and Accept frame are in the **Instances** menu only. The point right-click mentioned in 10.7 doesn't exist: right-click on a point toggles not visible.
- **When the summary dialog appears.** It's shown after the read on open, Sync, Import and Set up Sleap files. It isn't shown after the sync on close.
  - **Go to first new frame** is disabled when all the new points are in other experiments.
  - After an import that brought nothing in but had unmatched videos, a message box lists them instead.
- **Tests.** `tests/test_provenance.py` covers the adapter, merge, history, import and store parts (21 tests). The UI was smoke-tested offscreen; it has no automated tests.


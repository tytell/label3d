# Phase 12 — Refining the Calibration with Labelled Points

**Goal**: Refine a calibration using the bodypart points labelled or predicted in several cameras. A refinement starts from a charuco calibration computed by aniposelib's `calibrate_rows` on the user's chosen takes, or from an imported calibration. aniposelib's `bundle_adjust_iter` then adjusts it with the digitised points. Every result is saved to the registry (Phase 11) with a full record of its inputs, enough to rebuild it. The charuco solution it started from is kept too, so the user can switch between the original and any refinement of it, or between refinements that used different points.

**Status**: ✓ Complete. See *As implemented* at the end.

**Scope**: one experiment at a time. Calibrations are per experiment, so a refinement uses only that experiment's charuco takes and that experiment's points. The dialog works on the active experiment.

---

## 12.1 Two stages

```
Stage 1  charuco corners from takes T  ──calibrate_rows──▶  cgroup_charuco   (saved, method="charuco"; or an existing imported entry)
Stage 2  cgroup_charuco + points P     ──bundle_adjust_iter──▶  cgroup_refined  (saved, method="refine", parent=cgroup_charuco)
```

**Storage** (Phase 11.1): the Stage 1 result is an ordinary `charuco` entry in `calibrations/`. The Stage 2 result goes to `calibrations/refined/`, with `metadata.parent` set to the Stage 1 id and a `_points.csv` companion file (12.2). Several refinements can share one parent. Deleting the parent deletes them all (11.3).

**Stage 1**
- Corners for the chosen takes come from `calibrations/{cam}/corners.csv` (`camera_dir(exp_dir, cam)`) through `load_corners(cam_dir, takes=...)` (Phase 11.4). Frame skip is already applied at detection. The rows are trimmed to each take's start/end with `rows_in_range()`, as the Calibration tab does.
- Camera sizes come from the take videos, as in `_scan_take_media()`.
- They're calibrated with the existing `calibrate_rows()` path in `workers/calibration_worker.py`, including repeated runs and keeping the best.
- **Moved out of Qt classes so the refinement can reuse them:**
  - the repeat-and-keep-best loop, from `MultiTakeCalibrationWorker._calibrate_repeated` to a module function;
  - `take_range()` and `_build_takes()`, from `calibration_view.py`;
  - the `[metadata]` builder, from `CalibrationView._charuco_metadata`.
- The dialog offers to start from an existing `charuco` entry instead of recalibrating if the entry matches the current corners exactly. A match has:
  - the same takes;
  - its `metadata.charuco.detected` table (take → timestamp) equal to `corner_detections()` for those takes;
  - its `take_ranges` equal to the current ones.
- **Starting from an imported calibration.** Stage 1 can also be an existing `imported` entry, such as a calibration found during setup from SLP files or the synthetic `ground-truth`. It's used as is, with no recalibration. If it has no `charuco_takes`, the charuco-corner errors (12.3, item 2) are left out of the report and the metadata.
- A new Stage 1 result is saved as a `charuco` entry with the same `[metadata]` as one made in the Calibration tab: `charuco_takes`, `[metadata.charuco]` (board, frame skip, take ranges, detected, `n_rows`), and `errors.charuco` / `spread_rotation_deg` / `spread_baseline_pct`.

**Stage 2**
- `cgroup.bundle_adjust_iter(p2ds, extra=None, only_extrinsics=..., n_samp_full=..., n_samp_iter=...)` runs on a copy of the Stage 1 group. Defaults are `n_samp_full=1000`, `n_samp_iter=200`. It stops early once the median error is below `error_threshold` (0.3 px by default), which animal points will rarely reach.
- `p2ds` is a `C × N × 2` array of **raw pixel** coordinates, in the order of `cgroup.get_names()`. aniposelib undistorts when it triangulates and projects with distortion, so no undistortion is needed. Missing cameras are NaN.
- There's no `extra`. The board constraint applies only to charuco points, and Stage 1 has already used it.

## 12.2 Point selection (`calibration_refine.py`, no Qt)

```python
@dataclass
class PointSelection:
    sources: set[str]                 # subset of {"label3d", "sleap_user", "sleap_pred"}
    score_min: float = 0.0            # applies to sleap_pred only
    trials: list[str] | None = None   # None = all trials of the experiment
    min_cameras: int = 2

def gather_points(points_df, cam_names, sel) -> tuple[np.ndarray, pd.MultiIndex]
    # points_df: the experiment's rows of points.csv (5-level index, experiment dropped or fixed)
    # -> (p2ds C×N×2, index of the (trial, frame, instance, bodypart) behind each column)
```

Source names use the constants in `points.py` (`SOURCE_LABEL3D`, `SOURCE_SLEAP_USER`, `SOURCE_SLEAP_PRED`). SLEAP user labels arrive as `state=set` and predictions as `state=predicted`, with a `score` (`slp_sync.SLEAP_SOURCES`).

**Which observations count:**
- A camera observation counts if its `state` is `set` or `predicted`, its `source` is in `sel.sources`, its x/y are finite, and for `sleap_pred` its `score` is at least `score_min`.
- label3d reprojection guesses (`state=predicted`, `source=label3d`) are left out. They're derived from the calibration, so fitting to them would be circular.

**Which points count:** a column (one bodypart in one frame) is kept if it has at least `min_cameras` cameras that count.

**Counts for the dialog:** `count_points(...)` returns, per camera pair, the number of points seen by both. The dialog shows these live.

**Record of the points used** (`{id}_points.csv`, next to the refined entry)
- One row per camera observation that went into Stage 2 or the hold-out: `trial, frame, instance, bodypart, camera, x, y, source, score, holdout`.
- `holdout` is True for observations in held-out frames (12.3).
- The x/y are copied, not referenced, so the file still describes the fit after `points.csv` is edited or re-synced.
- With this file and the parent entry, Stage 2 can be rerun exactly. `metadata.points` (sources, score threshold, frames, count) is a summary for the tooltip and registry list. `metadata.points.points_file` names the file.
- `save_points_record(path, p2ds, index, cam_names, points_df, holdout)` and `load_points_record(path, cam_names) -> (p2ds, index, holdout)` live in `calibration_refine.py`. `holdout` is a bool per point; source and score are looked up in `points_df`.

## 12.3 Holding out frames, fixing scale, validating

**Hold-out**
- About 20% of frames (configurable) are kept out of Stage 2, and the split is by frame, not by point.
- Reported error is measured on the held-out frames. The error on the frames used for fitting always goes down, so it says little.
- The split uses a fixed seed stored in the metadata, so a run can be repeated.

**Fixing scale and world frame**
- Bundle adjustment on animal points alone has no metric scale, and it can shift the whole rig. The 3D points and extrinsics are free up to a similarity (7 degrees of freedom).
- After Stage 2, apply a similarity transform that maps the refined rig back onto `cgroup_charuco`. This keeps the charuco scale (cm) and world axes.
- With 3 or more cameras, this is Umeyama on camera centres. With 2 cameras, the centres leave the rotation about the baseline free. The rotation then comes from the camera orientations (the mean of `R_charuco_i · R_refined_iᵀ`), the scale from the baseline length, and the translation from the centroids. Tests should cover both cases. `test_project_sync.py` already builds a two-camera rig.
- The size of the transform's scale correction is reported, but it is not a warning sign by itself. The animal points carry no scale, so the rig can drift in size during bundle adjustment at no cost, and the alignment undoes that drift completely.
- What the alignment can't explain is the warning sign: a change in the shape of the rig. `alignment_residuals()` gives the RMS distance between aligned and charuco camera centres (cm, and % of the mean camera-pair baseline) and the RMS angle between their orientations. With 2 cameras the centres always fit, so only the orientation residual shows a change. The dialog warns above 1% of the baseline or 0.5°.

**Reported errors:**
1. **Held-out points:** median and 90th-percentile reprojection error, before (Stage 1) and after (Stage 2).
2. **Charuco corners from the chosen takes:** before and after. Stage 2 should not make these much worse.
3. **Per camera-pair drift from Stage 1:** from `calibration_compare.compare_calibrations()`, as in the take comparison.

**Repeats:** `bundle_adjust_iter` subsamples at random, so Stage 2 runs `n_repeats` times (default 3). The run with the lowest held-out error is kept. The run-to-run spread is reported as rotation and baseline spread (`calibration_compare.run_spread()`), as for charuco. aniposelib draws from NumPy's global random state (`np.random.random`, `np.random.choice`). Each repeat therefore calls `np.random.seed(seed + repeat)`, and the kept repeat number goes in `metadata.options`, so the kept run can be reproduced. aniposelib samples at most `n_samp_iter` (200) points per camera pair; when no pair of the fitted points has more, every repeat would use all the points and give the same result, so Stage 2 runs once (`repeats_differ()`), reports no spread, and records `n_repeats = 1`. The dialog disables Repeats in that case, judged on all selected points, with a tooltip saying why.

**Errors in `[metadata.errors]`** for a `refine` entry, all in px unless noted:
- `holdout`, `holdout_p90`: the refinement on the held-out points. `entry_error()` already shows `holdout` first.
- `holdout_before`, `holdout_p90_before`: Stage 1 on the same points.
- `charuco`, `charuco_before`: the charuco corners of the parent's takes.
- `points`: the frames used for fitting.
- `scale_correction`: the alignment scale, a ratio.
- `align_centre_cm`, `align_centre_pct`, `align_rotation_deg`: the alignment residuals.
- `spread_rotation_deg`, `spread_baseline_pct`: as for charuco; left out when Stage 2 ran once.

## 12.4 Refine dialog (`widgets/refine_dialog.py`) and worker (`workers/calibration_worker.py`)

Opened from the Calibration tab with **Refine with points...**.

| Control | Default |
|---|---|
| Charuco takes (checklist), or *Start from existing entry* combo (charuco or imported entries) | all takes |
| Sources: label3d user / SLEAP user / SLEAP predictions | first two checked |
| Prediction score ≥ | 0.8 |
| Trials (checklist) | all |
| Extrinsics only | on |
| Hold-out fraction | 0.2 |
| Repeats | 3 |
| Label | `refine` |

**Running a refinement**
- A live table shows usable points per camera pair. **Run** is never disabled: a refinement can run with any number of points. The pair counts are there so the user can judge the result.
- `RefineWorker(QThread)` runs Stage 1 if needed, then Stage 2, the alignment and the validation. It emits progress text and a `RefineResult`.

**Results view**
- The before and after error table, the per-pair drift, the scale correction, the alignment residuals, and the run-to-run spread when there were repeats.
- **Save** writes the Stage 1 entry if it's new, then the `refine` entry and its `_points.csv`. Both entries use `[metadata]` as in Phase 11.1, with `points.sources`, `score_min`, `frames` and `points_file` filled in, and the hold-out seed in `metadata.options`.
- **Save and make active** also moves the pointer (`Project.set_active_calibration`). As with **Make active** in the Calibration tab (Phase 11, *As implemented*), it doesn't re-triangulate. The stale-3D warning and its **re-triangulate** link show instead.
- **Discard** saves nothing.
- After a save, the Calibration tab emits `calibrations_changed`, so the registry list and the top-bar combo are refilled.

## 12.5 Tests (`tests/test_calibration_refine.py`)

Uses the synthetic generator: a known rig, charuco takes and a moving animal.

- **Point selection:** `gather_points` respects sources, the score threshold and `min_cameras`, and leaves out label3d reprojection guesses.
- **Recovery:** after a small rotation and translation of one camera in the Stage 1 group, Stage 2 with the true animal points moves it back toward the truth. Held-out error goes down.
- **Alignment:** the similarity alignment keeps the baseline lengths of the Stage 1 group within tolerance.
- **Hold-out:** the split is by frame and is reproducible from its seed.
- **Metadata:** a saved `refine` entry is in `refined/` and has its `parent`, sources, score threshold and frame list.
- **Points record:** `_points.csv` round trip. Rerunning Stage 2 from the parent and the record, with the stored seed, reproduces the saved calibration.

## 12.6 Reference docs

- **`docs/workflows.md`:** the Refine dialog, its options, and how to read the before and after errors.
- **`docs/data_model.md`:** the `_points.csv` columns and the `refine` keys in `[metadata.errors]` and `[metadata.options]`.
- **`docs/architecture.md`:** `calibration_refine.py` and `widgets/refine_dialog.py`.
- **`DESIGN.md`** and **CLAUDE.md:** the phase status.

## Implementation order

1. 12.2 `gather_points` / `count_points`, points record + tests
2. 12.1 + 12.3 `refine()` pipeline (no Qt) + recovery test
3. 12.4 worker and dialog
4. 12.6 reference docs

## As implemented

Differences from the plan above, and details it left open:

- **`calibration_solve.py`.** The Stage 1 helpers moved into this new module, which has no Qt: `CalibrationTake`, `TakeCalibration`, `JOINT_TAKE`, `build_board`, `calibrate_rows`, `calibrate_repeated`, `rows_for_take`, `rows_in_range`, `take_range`, `build_takes`, `take_camera_sizes`, `take_ranges`, `take_detections` and `charuco_metadata`. The Calibration tab and `MultiTakeCalibrationWorker` call them too.
- **Charuco error.** The charuco corners of the starting entry's takes are triangulated across cameras and reprojected, the same measure as for the animal points. The per-camera board-pose error from `compute_frame_errors` would not change when only extrinsics move.
- **Run is never disabled.** A refinement runs with any number of points; the pair table is for judging the result.
- **Camera order.** Points are gathered in the project's camera order and reordered to the starting calibration's (`get_names()`). An imported entry may list the cameras differently. A calibration with different cameras is an error.
- **Metadata.** `[metadata.points]` also records `min_cameras`. `[metadata.options]` also records `n_repeats`, `n_samp_full` and `n_samp_iter`. `[metadata.errors]` also has `points_before`. Non-finite errors (no held-out frames, no charuco takes) are left out. `calibration_registry.update_metadata()` adds `points_file` once the id is known.
- **Seed.** Drawn at random per run unless `RefineOptions.seed` is set. It seeds the hold-out split, and repeat `r` calls `np.random.seed(seed + r)`. `run_stage2()` reruns one repeat.
- **Dialog.** A matching charuco entry is offered as a **Reuse …** checkbox under the take list, checked by default. **Existing calibration** lists `charuco` and `imported` entries, with the active one selected. The dialog can't be closed while a run is in progress, since bundle adjustment can't be interrupted. If saving fails after the Stage 1 entry was written, that entry stays in the registry.
- **Alignment residuals and repeats.** Added after UI testing: the scale correction alone couldn't tell harmless scale drift from a change in the rig's shape, so the alignment residuals (12.3) are reported and warned on instead. Repeats run only when they can differ (12.3).
- **Calibration tab.** It loads the saved corners and the active calibration when it opens, and redraws the current frame when the active calibration changes, so a refinement made active can be checked on the charuco corners straight away.
- **Tests.** `tests/test_calibration_refine.py` (28 test functions) covers point selection, the hold-out split, the points record, recovery of a perturbed camera on the synthetic rig, alignment with three and two cameras and its residuals, charuco errors, running once when repeats can't differ, saving and exact reproduction from the record, Stage 1 corner loading and entry matching, `RefineWorker`, the dialog on a copy of the synthetic test project, and the Calibration tab restoring saved corners and the active calibration.

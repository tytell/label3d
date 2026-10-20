# Phase 9 — Calibration Takes

**Goal**: Treat calibration recordings as first-class, matched groups. An experiment normally has several calibration takes (typically one before and one after the trials). Match them across cameras the way trials are matched, record the matches in `trials.csv`, calibrate each take individually *and* all takes jointly, and flag the user when the takes disagree — a sign that a camera moved during the experiment.

**Status**: Complete.

**Background**: Calibration media currently live in `experiments/{exp}/calibration/{cam}/` as an unordered folder listing. `DetectionWorker` pairs files across cameras by their position in `sorted(cam_dir.iterdir())`, so two cameras whose calibration files sort differently silently merge corners from different moments in time, producing a wrong calibration with no error. Matching takes explicitly and recording them in `trials.csv` removes that failure mode.

**Compatibility**: This phase changes the `trials.csv` schema without a fallback path. Existing projects are updated by hand; no migration script and no support for the old format.

---

## 9.1 Deprecate calibration images

Calibration from image folders is retired. The code stays in place but must not run.

- [x] `setup_project.classify_files` — raise before `exp_calibration` is built if any calibration image is found
- [x] `workers/calibration_worker.DetectionWorker.run` — raise at the top of the `if image_files:` branch, leaving the `detect_images` call below it intact but unreachable
- [x] `widgets/calibration_view._scan_raw_media` — same guard; the `_media_is_images` helper becomes the error check rather than a UI toggle
- [x] Error message directs the user to the video workflow
- [x] Remove the image path from `docs/design/workflows.md` and `docs/design/data_model.md`, noting it is deprecated rather than deleting the description
- [x] *(added during implementation)* The synthetic generator emitted calibration images and so had nothing valid left to produce. `generate_charuco_videos()` renders the same board poses into one video per (camera, take), holding each pose for `frames_per_pose` frames so the default stride samples every pose. `generate_charuco_images()` is left in place, unused.

---

## 9.2 Calibration identification and take matching (`setup_project.py`)

Two stages: decide whether a video is calibration at all, then decide which take it belongs to.

### `SetupConfig` additions

```python
calib_keywords: list[str] = field(default_factory=list)   # e.g. ["PreCal", "PostCal"]
calib_regex: str = ""                                     # alternative to keywords
calib_match_strategy: str = ""                            # defaults to match_strategy
```

The existing `calib_keyword: str` is folded into `calib_keywords` so current config files keep working.

- [x] **Stage 1 — is it calibration?** A video matches if any entry of `calib_keywords` appears in the filename (case-insensitive) or `calib_regex` matches the stem. Replaces the single-substring test in `classify_files`
- [x] **Stage 2 — which take?** The matched keyword *is* the take name: `calib_keywords = ["PreCal", "PostCal"]` yields takes `PreCal` and `PostCal` with no numeric matching. With `calib_regex`, a named group `take` (falling back to group 1) supplies the name
- [x] **Fallback** when stage 2 cannot distinguish takes — a single generic keyword such as `calib` — run the calibration files alone through the existing `_match_by_number` / `_match_by_datestamp` / `_match_by_mtime` under `calib_match_strategy`
- [x] Experiment assignment uses `_identify_experiment`, camera assignment uses `_identify_camera` — identical to trials
- [x] Grouping key is `(experiment, take)`

### `FileAssignment` additions

```python
calibration_takes: dict[str, dict[str, dict[str, Path]]]  # experiment -> take -> {camera: path}
```

`exp_calibration` remains for placing files on disk.

- [x] Warn when a take is missing cameras, or when two files in one take claim the same camera

### Retire the split prompt

Multiple calibration prefixes within an experiment are now the normal case, not evidence of two projects.

- [x] Stop calling `split_project_groups`; leave the function and `_extract_calib_prefix` in place, unused
- [x] Drop `needs_split` / `calibration_groups` / `trial_prefix` from the classification flow and from `preview_assignment`
- [x] **The one error case**: a video that matches both an experiment keyword *and* a calibration keyword but whose take cannot be determined. Everything else is a trial
- [x] `preview_assignment` lists takes per experiment with their per-camera file counts

---

## 9.3 `trials.csv` schema

Add `is_calibration` as the third column. Calibration rows put the take name in `trial`.

```
experiment,trial,is_calibration,LateralOrtho,LateralOrtho_start,LateralOrtho_end,…
PS37,PS37_Trial7,False,PS37_Trial7_LateralOrtho_H265.mp4,0,,…
PS37,PreCal,True,PS37_PreCal_LateralOrtho.mp4,0,,…
PS37,PostCal,True,PS37_PostCal_LateralOrtho.mp4,0,,…
```

Calibration rows reuse the `{cam}` / `{cam}_start` / `{cam}_end` triples, so a take can be trimmed to the segment where the board is actually visible with no extra columns.

- [x] **`_extract_cameras` first** (`project.py`) — it currently treats every column that is not `trial`, `experiment`, or `*_start` / `*_end` as a camera name. Without adding `is_calibration` to the skip list, the new column becomes a phantom camera and breaks every downstream consumer. This must land before anything writes the column
- [x] `execute_setup` writes calibration rows alongside trial rows; files still move to `experiments/{exp}/calibration/{cam}/` (flat, unchanged layout — `trials.csv` is the sole source of truth for which file belongs to which take)
- [x] `save_config_toml` and the interactive CLI prompt for `calib_keywords` / `calib_regex`
- [x] Confirm `video_cells()` (`slp_sync.py`, `adapters/sleap_adapter.py`) skips calibration rows so SLP matching never tries to assign labels to a calibration video

---

## 9.4 Project model (`project.py`)

- [x] `_parse_trials` filters on `is_calibration`; a second pass builds the takes. `TrialInfo` is reused unchanged — a take is structurally identical to a trial
- [x] `ExperimentInfo.calibration_takes: list[TrialInfo]`
- [x] `Project.calibration_takes` for the active experiment
- [x] `Project.video_path()` branches on `is_calibration`: calibration files resolve under `calibration/{cam}/`, trials under `videos/`

---

## 9.5 Detection keyed by take (`workers/calibration_worker.py`, `widgets/calibration_view.py`)

- [x] `DetectionWorker` accepts an ordered `list[tuple[str, dict[str, Path]]]` of takes instead of scanning `sorted(cam_dir.iterdir())`
- [x] Row keys become `(take_name, framenum)` rather than `(video_idx, framenum)`, so cross-camera correspondence comes from the CSV by name and a camera missing from a take is detectable instead of silently shifting every later index
- [x] `CalibFrameProvider.get_frame` resolves `(take_name, frame)` through a per-take file map
- [x] `_scan_raw_media` builds its browsable frame list from the same take ordering
- [x] Honour `{cam}_start` / `{cam}_end` when striding a take
- [x] `calibration_io.save_corners` — `source_key` becomes `{take}:{frame}`

### Corners file naming

- [x] Per-take detections: `calibration/{cam}/corners_{take}.csv`
- [x] The joint result, or whichever single take the user selects as final: `calibration/{cam}/corners.csv`
- [x] `load_corners` gains an optional take argument; the no-argument call still reads `corners.csv`

---

## 9.6 Per-take and joint calibration

Run `CalibrationWorker` once per take and once over all takes merged.

- [x] Per-take results → `experiments/{exp}/calibration_takes/{take}.toml`, joint result alongside them
- [x] `experiments/{exp}/calibration.toml` stays the file the rest of the app reads — the user's chosen solution is written there, so no other consumer changes

### Drift detection

Intrinsics should be identical across takes; extrinsics are what move when a camera is bumped. For each camera pair, compare between takes:

- [x] relative rotation angle (degrees)
- [x] baseline length (cm)
- [x] mean reprojection error per take (already produced by `compute_frame_errors`)

- [x] Flag when rotation exceeds **0.5°** or baseline changes by more than **1%**; both thresholds configurable under `[calibration]` in `project.toml`

### UI

- [x] Take selector in the calibration panel: one entry per take plus "All takes (joint)". Filled as soon as the takes load, so it also filters the browsed frames before any calibration exists; selecting a take shows only that take's frames and, once calibrated, previews its solution
- [x] Comparison table — per-take reprojection error, rotation drift, baseline drift, each against the joint solution
- [x] Warning banner when a threshold trips, naming the camera pair and the takes involved
- [x] Selecting a take and saving writes that solution to `calibration.toml` and its corners to `corners.csv`

---

## Implementation order

1. §9.1 — image deprecation guards (small, unblocks the rest)
2. §9.3 `_extract_cameras` skip-list fix — must precede any write of the new column
3. §9.2 — take matching in `setup_project`
4. §9.3 — schema, `execute_setup`, CLI
5. §9.4 — project model
6. §9.5 — detection keyed by take (this is the step that fixes the silent mismatch bug)
7. §9.6 — per-take / joint calibration and the comparison UI

# Phase 13 — Batch Triangulation and Export

**Goal**: The final step of the workflow. Once the SLEAP model is trained and has run on every frame of every camera, triangulate everything with each experiment's active calibration. Then write one long-format CSV with the 2D and 3D coordinates of all points from all experiments, and bring the mapped SLP files up to date at the same time.

**Status**: Not started. This replaces the Export tab. Phase 15.1 stays as written for exporting to other pose formats, but it will be reached from the File menu too, not from a tab.

---

## 13.1 Fast batch triangulation (`geometry.py`, `workers/triangulation_worker.py`)

- **The problem:** with predictions on every frame, triangulating frame by frame is too slow. The cost is mostly per-point overhead: `get_point` lookups in pandas, building a sub-`CameraGroup` for each bodypart, and writing results back through `.at`.
- **The fix:** add `triangulate_trial(cameras: dict[str, CameraParams], points_store, trial) -> TrialTriangulation`. It matches the other geometry functions, which take `cameras` and build the group with `build_camera_group` (cameras sorted by name). For each instance, it:
  1. builds one `C × N × 2` array of raw pixels across all frames and bodyparts, with NaN where a camera's point doesn't count (see *Which points*);
  2. calls `cgroup.triangulate(..., undistort=True)` once. aniposelib's default mode still runs one SVD per point in Python. If that's too slow, replace it with a batched DLT (`np.linalg.svd` on a stacked `N × 2C × 4` array) in `geometry.py`;
  3. reprojects every point into every camera in one batch;
  4. writes `3d` x/y/z and every camera's `xr`/`yr` back into the DataFrame in one vectorised assignment, not through `set_3d` / `set_reprojected`. The trial's frame cache is invalidated once at the end.
- **Which points:** a camera's point counts when its state is `set` or `predicted` and x is not NaN. `not_visible` and `not_set` don't count. **The SLEAP score is not used:** every matching point is triangulated, whatever its score. Scores stay with the 2D points and go into the export, so they can be filtered there.
- **Fewer than 2 cameras:** the `3d` x/y/z and every camera's `xr`/`yr` are set to NaN, even if they held values from an earlier run.
- **Same rule for `triangulate_frame`:** it currently skips such a bodypart and leaves its old values. Change it to clear them too, so the per-frame and batch results agree, and so stale 3D never reaches the export.
- **Callers:**
  - `TriangulationWorker` (`Ctrl+Shift+T`, and `_merge_tri_worker` after a Sync) calls `triangulate_trial` once. It loses its per-frame `frame_done` signal, so `label_view` redraws the current frame when the worker finishes. Progress is reported per instance, or per trial for long runs.
  - `AllExperimentsTriangulationWorker` calls it once per (experiment, trial). Cancelling stops between trials. It already updates each experiment's `triangulated_with` after a complete run (`_on_triangulate_all_done`); that stays as it is.
  - The core function has no Qt in it, so the CLI (13.3) can call it directly.
- **Test:** on the synthetic project, start from a store that has old 3D values in some cells, including bodyparts with fewer than 2 cameras. Run `triangulate_frame` over every frame on one copy and `triangulate_trial` on another. The two must agree to within 1e-6, NaN where the other is NaN, and the cells with fewer than 2 cameras must be NaN.

## 13.2 Export format

One CSV in **long** format, with one row per (experiment, trial, frame, instance, bodypart, view) and a single header row. Columns, in this fixed order:

| Column | Notes |
|---|---|
| `experiment, trial, frame, instance, bodypart` | as in `points.csv`; `frame` is the trial frame |
| `view` | camera name, or `3d` |
| `video_frame` | cameras: the frame in that camera's video, `cam_start + frame` (from `trials.csv`); empty for `3d` |
| `x, y` | pixels for cameras; cm for `3d` |
| `z` | `3d` rows only; empty for cameras |
| `state` | `set`, `predicted` or `not_visible` (cameras); empty for `3d` |
| `source` | `label3d`, `sleap_user` or `sleap_pred` (cameras) |
| `score` | SLEAP score, as stored, for every point that has one. No point is dropped for its score; filter afterwards if needed |
| `xr, yr, error` | reprojection and its distance from (x, y) in px (cameras; empty when there is no 3D point) |
| `n_cameras` | `3d` rows: number of cameras used (state `set` or `predicted`) |
| `mean_error` | `3d` rows: mean of those cameras' `error` |
| `calibration` | `3d` rows: registry id of the calibration used (the experiment's `triangulated_with`) |

- `error`, `n_cameras`, `mean_error` and `calibration` are not stored in `points.csv`. They are computed at export time from the stored values and `project.toml`.
- `sync` (the id of the sync that brought a point in) is not exported. It only matters inside label3d.
- `not_set` cells are left out.
- A `3d` row is written only where all of x/y/z are present. Since triangulation clears the 3D point of anything seen by fewer than 2 cameras (13.1), every `3d` row is current.
- **An experiment with no active calibration** is exported with its camera rows only: no `3d` rows, and empty `xr, yr, error`.

## 13.3 Export dialog (`widgets/export_dialog.py`) and CLI

**Remove the Export tab** (the "Export (coming soon)" stub in `main_window.py`), so the tabs are Calibrate and Label. **File > Export points...** is the way in.

**When it's enabled:** only when no experiment's 3D points are stale, i.e. `triangulation_is_stale(name)` is false for every experiment. Otherwise it is greyed out, with a tooltip naming the stale experiments and pointing to **Instances > Triangulate all experiments**. Experiments without an active calibration don't block it (`triangulation_is_stale` is false for them); see below. The enabled state is refreshed wherever the calibration status is (`_update_calibration_status`) and after triangulation finishes.

**The dialog:**
- **Experiments:** a checklist, all checked by default.
- **Include predictions:** on by default. When off, points with `source = sleap_pred` are left out. There is no score threshold; scores are in the file for filtering later.
- **Only points with 3D:** off by default.
- **Output path:** default `project/exports/points_{YYYYmmdd-HHMMSS}.csv`.
- **On Export:**
  1. **Experiments without an active calibration:** if any are checked, a warning lists them and says only their 2D points will be exported. **Export anyway** continues; **Cancel** goes back to the dialog.
  2. **Check the SLP files for new points:** read the mapped SLP files the way Sync does (`read_slp_for_sync` + `compute_merge`). If any would bring in new or changed points (`MergeResult.has_changes`), stop with a warning that says how many points, in which experiments, and to run **Sync** and then triangulate first. Nothing is written.
  3. **Otherwise** write the CSV, and write the current points into the mapped SLP files (`project.write_slp_files()`), as the write half of a Sync does. No sync id is recorded in `sync_history.csv`, because nothing was read in.
- Shows progress, then a summary: rows written, per-experiment counts by state and source, which experiments were 2D only, and how many SLP files were written.

**CLI:** `uv run label3d-export <project> [-o out.csv] [--experiments a,b] [--no-predictions] [--only-3d] [--no-slp]`

- Runs the same checks as the dialog:
  - any checked experiment with stale 3D: exits with an error naming them;
  - an experiment without an active calibration: prints a warning, then exports its 2D points;
  - new points in the SLP files: exits with an error, nothing written.
- `--no-slp` skips the SLP check and the SLP write, for exporting without a mapping set up.
- It doesn't triangulate.
- Exit codes: 0 when the export is written, 1 on any of the errors above.

Both use the same function with no Qt in it, `export.export_points(project, options) -> ExportResult` (output path, row counts, 2D-only experiments, SLP files written), in a new `src/label3d/export.py`. The pre-flight checks are separate functions in the same module (`stale_experiments`, `uncalibrated_experiments`, `pending_slp_changes`), so the dialog can ask its questions between them. This lets step 5 of the workflow run after a large inference job without opening the GUI. The CLI is a second entry in `pyproject.toml` `[project.scripts]`.

## 13.4 Tests

- **13.1:** `triangulate_trial` matches `triangulate_frame` as described there. A bodypart that drops to fewer than 2 cameras has NaN `3d` and `xr`/`yr` after either function.
- **`export_points`:**
  - writes the documented columns in order;
  - its rows round-trip to the store values (x, y, state, source, score, xr, yr, 3D);
  - `video_frame` equals `cam_start + frame` for a camera with a nonzero `cam_start`;
  - the filters (experiments, `include_predictions`, `only_3d`) work;
  - points with low scores are still exported;
  - an experiment with no active calibration gives camera rows only.
- **Pre-flight checks:** `stale_experiments`, `uncalibrated_experiments` and `pending_slp_changes` each detect their case on a modified synthetic project. When there are pending SLP changes, `export_points` is never called by the dialog or the CLI.
- **SLP write:** after an export, reading the mapped SLP files with `compute_merge` finds no changes.
- **CLI:** runs on the synthetic project and exits with code 0. It exits with 1 when an experiment is stale.

## 13.5 Workflow guide, final version (`docs/user_workflow.md`)

Finish the Phase 10.9 draft:
- Add calibration switching (Phase 11) and refinement (Phase 12).
- Add the export step: triangulate all experiments, **Sync**, then **File > Export points...**. Include the column reference from 13.2 and the CLI.
- Remove the "Draft" note and the "coming in a later version" line.

## 13.6 Reference docs

- **`docs/workflows.md`:** replace the *Export Workflow (Step 3)* section, which describes the Export tab, with **File > Export points...**: when it's enabled, the pre-flight checks, the SLP write, the CLI, and the export column reference. Remove the Export tab from the window layout diagrams.
- **`docs/data_model.md` and `CLAUDE.md`:** add `exports/` to the project layout.
- **README:** the `label3d-export` entry point.

## Implementation order

1. 13.1: clear old values in `triangulate_frame`, then add `triangulate_trial`, its test, and switch both workers over
2. 13.2: `export.py` with `export_points` and the pre-flight checks, plus the 13.4 tests
3. 13.3: remove the Export tab; add the dialog, the menu item's enabled state, and the CLI entry point
4. 13.5 guide, 13.6 reference docs

---

## Deferred: scaling of `points.csv`

Not part of this phase; this is a to-do for when predictions on every frame make the app slow.

- **The risk:** `points.csv` is a MultiIndex CSV, read and written in full by every sync and every auto-save. With predictions on every frame it gets large. The largest project so far is `real_OMR/experiments/20260721`: 67,697 rows, 16 MB.
- **Measure first**, on a project with predictions on every frame. Time these steps separately:
  - reading and writing `points.csv`
  - the in-memory merge (`compute_merge`) and the frame cache
  - writing the mapped SLP files (`write_slp_files`), which every Sync and every export rewrites in full and which now hold every prediction
  - Import predictions (10.4) on full-length predictions files
- **If it's too slow**, the options are:
  - a binary store such as parquet (estimated at about one session of work);
  - splitting the store per trial;
  - making Sync and export write only the SLP files whose cells changed.

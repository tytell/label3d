# Phase 13 — Batch Triangulation and Export

**Goal**: The final step of the workflow. Once the SLEAP model is trained and has run on every frame of every camera, triangulate everything with each experiment's active calibration. Then write one long-format CSV with the 2D and 3D coordinates of all points from all experiments, and bring the mapped SLP files up to date at the same time.

**Status**: Code, tests and docs done; to be checked in the app before marking complete. This replaces the Export tab. Phase 15.1 stays as written for exporting to other pose formats, but it will be reached from the File menu too, not from a tab.

---

## 13.1 Fast batch triangulation (`geometry.py`, `workers/triangulation_worker.py`)

- **The problem:** with predictions on every frame, triangulating frame by frame is too slow. The cost is mostly per-point overhead: `get_point` lookups in pandas, building a sub-`CameraGroup` for each bodypart, and writing results back through `.at`.
- **The fix (done):** `triangulate_trial(cameras, points_store, trial, frames=None) -> int` in `geometry.py`. It takes `cameras` like the other geometry functions and returns the number of rows with a 3D point. Every row of the trial (each frame, instance and bodypart) is independent, so all rows go in one batch, via `triangulate_rows(cameras, df) -> (p3d, reproj)`:
  1. each camera's used points are undistorted to normalised coordinates (`cv2.undistortPoints`) and stacked into an `N × 2C × 4` DLT system, with zero rows for cameras that don't count;
  2. all rows are solved with one batched `np.linalg.svd`. This is the same DLT as aniposelib's `CameraGroup.triangulate`, without its per-point Python loop;
  3. every point is reprojected into every camera in one call per camera;
  4. `PointsStore.set_triangulation(index, p3d, reproj)` writes `3d` x/y/z and every camera's `xr`/`yr` in one positional assignment per column, and invalidates the frame cache.

  On `real_OMR/20260721` (67,694 rows) the whole experiment takes 0.4 s. `mean_reprojection_errors(df, reproj)` gives each row's mean error over the cameras used; the export uses it too.
- **Which points:** a camera's point counts when its state is `set` or `predicted` and x is not NaN. `not_visible` and `not_set` don't count. **The SLEAP score is not used:** every matching point is triangulated, whatever its score. Scores stay with the 2D points and go into the export, so they can be filtered there.
- **Fewer than 2 cameras:** the `3d` x/y/z and every camera's `xr`/`yr` are set to NaN, even if they held values from an earlier run.
- **Same rule for `triangulate_frame` (done):** it now runs the same batch code on one instance's rows, so it clears old values too and agrees with `triangulate_trial`. Stale 3D never reaches the export.
- **Callers (done):**
  - `TriangulationWorker` (`Ctrl+Shift+T`, and `_merge_tri_worker` after a Sync) calls `triangulate_trial` once. Its per-frame `frame_done` signal is gone; `label_view` redraws when the worker finishes.
  - `AllExperimentsTriangulationWorker` calls it once per (experiment, trial), and reports progress and its `finished` count in trials. Cancelling stops between trials. `_on_triangulate_all_done` still updates each experiment's `triangulated_with` after a complete run.
  - The core functions have no Qt in them, so the CLI (13.3) can call them directly.
- **Test (done, `tests/test_triangulation.py`):** on the synthetic project, start from a store with old 3D values everywhere and every third row seen by one camera only. `triangulate_trial` and `triangulate_frame` must both match a per-point reference through aniposelib (one sub-`CameraGroup` per point, as `triangulate_frame` worked before) to within 1e-6, and the one-camera rows must be NaN.

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
- **Include predictions off** drops camera rows with `source = sleap_pred`. `3d` rows stay; `n_cameras` and `mean_error` still describe the triangulation, which used every `set` and `predicted` point.
- **Done:** `src/label3d/export.py` (`export_points`, `ExportOptions`, `ExportResult`, the pre-flight checks) and `tests/test_export.py`.

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
  2. **Check the SLP files for new points:** read the mapped SLP files the way Sync does (`read_slp_for_sync` + `compute_merge`). If any would bring in new or changed points, new scores or conflicts (`MergeResult.has_changes` or `conflicts`), stop with a warning that says how many points, in which experiments, and to run **Sync** and then triangulate first. Nothing is written.
  3. **Otherwise** write the CSV, and write the current points into the mapped SLP files (`project.write_slp_files()`), as the write half of a Sync does. The SLP files are written only when the mapping is complete (every video has a file), the condition that enables **Sync**; otherwise the summary says they were not written. No sync id is recorded in `sync_history.csv`, because nothing was read in.
- Shows progress, then a summary: rows written, per-experiment counts by state and source, which experiments were 2D only, and how many SLP files were written.

**CLI:** `uv run label3d-export <project> [-o out.csv] [--experiments a,b] [--no-predictions] [--only-3d] [--no-slp]`

- Runs the same checks as the dialog:
  - any checked experiment with stale 3D: exits with an error naming them;
  - an experiment without an active calibration: prints a warning, then exports its 2D points;
  - new points in the SLP files: exits with an error, nothing written.
- `--no-slp` skips the SLP check and the SLP write, for exporting without a mapping set up.
- It doesn't triangulate.
- Exit codes: 0 when the export is written, 1 on any of the errors above.

**Done:** `widgets/export_dialog.py`, `workers/export_worker.py` (the SLP check and the export run in the background), `export_cli.py` (`label3d-export`), `tests/test_export_cli.py`. The CLI loads the project with `Project.load`, so like the app it needs the videos to be reachable.

Both use the same function with no Qt in it, `export.export_points(project, options) -> ExportResult` (output path, row counts, 2D-only experiments, SLP files written), in a new `src/label3d/export.py`. The pre-flight checks are separate functions in the same module (`stale_experiments`, `uncalibrated_experiments`, `pending_slp_changes`), so the dialog can ask its questions between them. This lets step 5 of the workflow run after a large inference job without opening the GUI. The CLI is a second entry in `pyproject.toml` `[project.scripts]`.

## 13.4 Tests

- **13.1:** done; see there.
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
- Add the export step: **Sync**, triangulate all experiments, then **File > Export points...**. Include the column reference from 13.2 and the CLI.
- Remove the "Draft" note and the "coming in a later version" line.

## 13.6 Reference docs

- **Done.** `docs/workflows.md` links to the column table in the user guide rather than repeating it.
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

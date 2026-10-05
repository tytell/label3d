# Phase 13 — Batch Triangulation and Export

**Goal**: The final step of the workflow. Once the SLEAP model is trained and has run on every frame of every camera, triangulate everything with each experiment's active calibration. Then write one long-format CSV with the 2D and 3D coordinates of all points from all experiments.

**Status**: Not started. This covers what the old export view (now Phase 15.1) planned for the final output. Phase 15.1 stays as written for exporting to other pose formats.

---

## 13.1 Fast batch triangulation (`geometry.py`, `workers/triangulation_worker.py`)

- **The problem:** with predictions on every frame, triangulating frame by frame is too slow.
- **The fix:** add `triangulate_trial(df_trial, cgroup) -> (p3d, xr, yr, err)`. For each instance, it builds one `C × N × 2` array of raw pixels across all frames and bodyparts, calls `cgroup.triangulate()` once, and reprojects in one batch. Fewer than 2 cameras means NaN.
- **Which points:** the same rules as now (`set` / `predicted`; `not_visible` excluded), with an optional prediction score threshold.
- **Callers:** `TriangulationWorker` and `AllExperimentsTriangulationWorker` use it. After a full run, each experiment's `triangulated_with` is updated (Phase 11.5).
- **Test:** the output must match the current per-frame result to within 1e-6 on the synthetic project.

## 13.2 Export format

One CSV in **long** format, with one row per (experiment, trial, frame, instance, bodypart, view) and a single header row.

| Column | Notes |
|---|---|
| `experiment, trial, frame, instance, bodypart` | as in `points.csv` |
| `view` | camera name, or `3d` |
| `x, y` | pixels for cameras; cm for `3d` |
| `z` | `3d` rows only; empty for cameras |
| `state` | `set`, `predicted` or `not_visible` (cameras); empty for `3d` |
| `source` | `label3d`, `sleap_user` or `sleap_pred` (cameras) |
| `score` | SLEAP score (predictions) |
| `xr, yr, error` | reprojection and its error in px (cameras) |
| `n_cameras` | `3d` rows: cameras used |
| `mean_error` | `3d` rows: mean reprojection error across those cameras |
| `calibration` | `3d` rows: registry id of the calibration used |

- `not_set` cells are left out.
- A `3d` row is written only where triangulation succeeded.
- The column order is fixed so scripts can rely on it.

## 13.3 Export dialog (`widgets/export_dialog.py`) and CLI

**File > Export points...** opens a dialog:
- **Experiments:** a checklist, all checked by default.
- **Include predictions:** on by default, with a score threshold.
- **Re-triangulate first:** on by default. It runs 13.1 for every experiment whose `triangulated_with` is not its active calibration.
- **Only points with 3D:** off by default.
- **Output path:** default `project/exports/points_{YYYYmmdd-HHMMSS}.csv`.
- Shows progress, then a summary: rows written, and per-experiment counts by state and source.

**CLI:** `uv run label3d-export <project> [-o out.csv] [--experiments a,b] [--no-predictions] [--score-min 0.5] [--no-retriangulate] [--only-3d]`

This uses the same non-Qt function, `export.export_points(project, options) -> Path`, so step 5 of the workflow can run after a large inference job without opening the GUI.

## 13.4 Scaling risk: size of `points.csv`

- **The problem:** with predictions on every frame, `points.csv` (a MultiIndex CSV, read and written in full by every sync and every auto-save) gets large.
- **Measure first:** before 13.1, time read, write and sync on a real project at full-trial inference.
- **If it's too slow:** move the per-experiment store to parquet. Keep CSV for export only.

## 13.5 Tests

- `triangulate_trial` matches the per-frame triangulation, as described in 13.1.
- `export_points` writes the documented columns in order. Its rows round-trip to the store values. Filters (experiments, predictions, score, `only_3d`) work.
- The CLI runs on the synthetic project and exits with code 0.

## 13.6 Workflow guide, final version (`docs/user_workflow.md`)

Finish the Phase 10.8 draft:
- Add calibration switching (Phase 11) and refinement (Phase 12).
- Add the export step, with the column reference from 13.2 and the CLI.

## Implementation order

1. 13.4 measurement
2. 13.1 vectorised triangulation + test
3. 13.2 `export_points` + 13.5 tests
4. 13.3 dialog and CLI entry point (`pyproject.toml` `[project.scripts]`)
5. 13.6 guide

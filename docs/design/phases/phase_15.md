# Phase 15 — Parquet Working Files

**Goal**: Store the project's working tables as Parquet instead of CSV, so they are smaller and faster to read and write, and keep their types. The final export can be written as Parquet or CSV, in the wide layout or a long one.

**Status**: Code, tests and docs done; to be checked in the app (open a project set up again, label, Sync, export in both formats and layouts) before marking complete. This follows from the deferred *scaling of `points.csv`* note in [phase_13.md](phase_13.md#deferred-scaling-of-pointscsv). The BGvPS experiments' `points.csv` files are 94 MB (`BG75`) and 101 MB (`PS21`), and every auto-save, Sync and export reads or writes them in full.

---

## 15.1 Which files change

| Was | Becomes | Notes |
|---|---|---|
| `experiments/{name}/points.csv` | `points.parquet` | wide layout kept (15.2) |
| `calibrations/{cam}/corners.csv` | `corners.parquet` | same columns |
| `sync_history.csv` | `sync_history.parquet` | same columns |
| `trials.csv` | unchanged | edited by hand |
| `refined/{id}_points.csv` | unchanged | small provenance record of a refinement, kept with its registry `.toml` |

- **CSV is no longer read.** A project with a `points.csv`, `corners.csv` or `sync_history.csv` but no Parquet file is not converted: `Project.load` raises before it opens any video, naming the files and saying the project has to be set up again (`_legacy_csv_files`). This is how the pre-registry layout is handled. There is no conversion code. Before setting BGvPS up again, run **Sync** in the current version so that every manual label is in the mapped `.slp` files.
- **Dependency:** add `pyarrow` to `pyproject.toml`. It is the pandas Parquet engine, and R's `arrow` package reads the same files.
- **File names** live in one place, as constants next to their readers: `POINTS_FILE` in `points.py`, `corners_path` in `calibration_io.py` (exists), and `sync_history_path` in `project.py` (exists). The many literal `"points.csv"` paths in `project.py`, `main_window.py`, `setup_project.py` and `synthetic/writers.py` use `POINTS_FILE`.

## 15.2 `points.parquet` layout (`points.py`)

In memory nothing changes: the 5-level row index and the `(view, coord)` column MultiIndex stay. Only the file is different.

- **On disk the table is flat**, so R and other tools read it without pandas metadata:
  - the five index levels are ordinary columns: `experiment, trial, frame, instance, bodypart`;
  - each `(view, coord)` column is named `{view}/{coord}` (`LateralOrtho/x`, `3d/z`). `/` can't be in a camera name, since camera names are folder names.
- **Types:** these are written with a fixed schema (`points_schema(cameras)`), so an empty store and an all-NaN column also round-trip:

  | Column | Type |
  |---|---|
  | `experiment, trial, bodypart` | string |
  | `frame, instance` | int64 |
  | `{cam}/x, y, xr, yr, score, sync`, `3d/x, y, z` | float64 |
  | `{cam}/state, source` | string |

  Parquet dictionary-encodes repeated strings on its own, so the Arrow types are plain strings; dictionary types would come back as pandas Categoricals. `sync` stays float64 as in memory, where NaN means no sync id. `normalize_columns` now also makes the `3d` columns float64, so a store and its file agree.

  Because the types are stored, the `astype(str)` fixes in `read_csv_df` for all-digit experiment and trial names are no longer needed.
- **Row groups:** rows are sorted as now (`sort_points_index`) and written in row groups of about 100k rows, so a reader filtering on `trial` skips the groups it doesn't need. Compression is zstd.
- **API (done):**
  - `PointsStore.from_parquet(path)`, `read_points_df(path, trials=None)` and `to_parquet(path)` replace `from_csv`, `read_csv_df` and `to_csv`.
  - `trials=` passes a filter to pyarrow. It is there for `experiment_points_df` and the export, but they can keep reading the whole file until timing (15.6) shows a need.
- **Safe saves (done):** `_auto_save` and `save_if_changed` used to rename the file to `.csv.bak` and then write. Now they write `points.parquet.tmp`, keep the previous file as `points.parquet.bak`, then `os.replace` the temporary file into place. A crash while writing leaves the old file intact. `Project.clear_unavailable` and `apply_merge`, which write inactive experiments' files, use the same helper (`write_points_file(df, path)`).

## 15.3 `corners.parquet` and `sync_history.parquet` (done)

The temporary-file write is shared by all three files: `parquet_io.write_table(table, path, backup=False)`. Only the points file keeps a `.bak`.

- **Corners (`calibration_io.py`):** `_read_corners_df` and `save_corners` switch to Parquet with the same `CORNER_COLUMNS`. `take` and `detected` are strings, `frame` and `corner_id` int32, and `x, y` float32. `save_corners` already rewrites the whole file, so its logic doesn't change.
- **Sync history (`project.py`):** `_record_sync` appends with `to_csv(mode="a")`, but Parquet can't be appended to. It reads the history, concatenates the new rows and writes the whole file, with the same temporary-file replace. The file has one row per (sync, experiment, trial), so it stays small.

## 15.4 Export: format and layout (`export.py`, export dialog, CLI) (done)

`ExportOptions` gains two fields:

- `format`: `"parquet"` (default) or `"csv"`;
- `layout`: `"wide"` (default, the current columns from `export_columns`) or `"long"`.

**Long layout:** one row per (experiment, trial, frame, instance, bodypart, view), with `view` a camera name or `3d`. Columns, in this order: `experiment, trial, frame, instance, bodypart, view, video_frame, x, y, z, state, source, score, xr, yr, error, n_cameras, mean_error, calibration`.
- Camera rows leave `z`, `n_cameras`, `mean_error` and `calibration` empty.
- `3d` rows leave `video_frame`, `state`, `source`, `score`, `xr`, `yr` and `error` empty.
- Units are given in the docs, not in the column names: px for camera rows and cm for `3d` rows, since one `x` column holds both.
- `not_set` camera rows are left out. A `3d` row is written only where x/y/z are all present.

The long table is built from the same per-experiment wide rows (`experiment_rows`) by `wide_to_long(rows, cameras)`, so the two layouts can't disagree. The filters (`include_predictions`, `only_3d`) work as now, before the reshape.

**File extension:** the format sets the extension. `default_output_path(project, format)` gives `exports/points_{timestamp}.parquet` or `.csv`. A user-supplied path gets the right extension (`with_format_suffix`): another format's extension is replaced, and any other suffix is kept and the extension added (`v1.2` → `v1.2.csv`). The summary shows the final path.

**Dialog (`widgets/export_dialog.py`):** a **Format** choice (Parquet / CSV) and a **Layout** choice (Wide / Long). Changing the format changes the extension in the path field, and the Browse filter follows the format.

**CLI (`export_cli.py`):** add `--format {parquet,csv}` and `--layout {wide,long}`, and correct the suffix of `-o` the same way.

## 15.5 Tests

- **Points:**
  - `to_parquet` → `from_parquet` round-trips a populated store exactly: index, columns, dtypes, NaN and every state and source;
  - an empty store round-trips;
  - all-digit experiment and trial names come back as strings;
  - a saved file has the flat column names and reads in pyarrow without pandas.
- **Safe save:** after a save, the `.bak` file holds the previous contents. A write that fails part way (patched to raise) leaves `points.parquet` unchanged.
- **CSV refused:** a project with only a `points.csv` raises on load with a message naming it.
- **Corners, sync history:** existing tests switch to Parquet. Recording two syncs keeps both sets of rows.
- **Export:**
  - both formats × both layouts write the documented columns in order;
  - long rows round-trip to the wide rows;
  - a path with the wrong suffix is corrected;
  - the CLI's `--format` and `--layout` work.
- **Fixtures:**
  - regenerate the synthetic test project with Parquet, since `synthetic/writers.py` writes `points.parquet`;
  - the hand-built projects in `test_project_sync.py` and `test_slp_setup.py` write Parquet;
  - `test_points_paths.py` (spaces, unicode, nested paths) switches to Parquet.

## 15.6 Timing (done)

- **After 15.2:** the same `PS21` points as Parquet are 18.9 MB. Writing takes 0.38 s; reading takes 0.05–0.13 s, and reading one trial (15,025 rows) 0.02 s.
- **Before:** time reading and writing `points.csv` on BGvPS `PS21`. Measured 2026-10-20: 455,342 rows, 27 columns (3 cameras), 106 MB. Reading with `read_csv_df` takes 0.73 s; writing with `to_csv` takes 4.26 s, and every auto-save, Sync and export of that experiment pays it.
- **After:** time the same with `points.parquet`, and compare the file sizes.
- Record the numbers here. If Sync is still slow, the remaining options from the Phase 13 note apply: split the store per trial, or write only the SLP files whose cells changed.

## 15.7 Docs (done)

- **`CLAUDE.md`, `docs/design/data_model.md`:** file names, the on-disk column naming (`{view}/{coord}`) and the types. Mark CSV as no longer supported.
- **`docs/user_workflow.md`:**
  - the export's format and layout choices;
  - the long layout's columns, next to the existing wide table;
  - how to open the files: `pd.read_parquet` and `arrow::read_parquet()` / `arrow::open_dataset()`, and viewers such as Tad or the DuckDB UI.
- **`docs/setup_from_slp.md`, `docs/design/workflows.md`, `docs/design/architecture.md`:** file names.
- **`phase_13.md`:** the deferred note points here.
- The finished phase docs (8–12) are left as written.

## Implementation order

1. 15.6 "before" timing; add `pyarrow`
2. 15.2: Parquet I/O in `points.py`, the safe-save helper, `POINTS_FILE` everywhere, and refuse CSV on load. Then the points tests and regenerated fixtures
3. 15.3: corners and sync history, with their tests
4. 15.4: export format and layout in `export.py`, then the dialog and the CLI, with their tests
5. 15.6 "after" timing; 15.7 docs

# label3d — Claude Code Context

## What this project is

A PySide6 desktop app for 3D point labeling across multiple synchronized calibrated video streams. It interfaces with sleap.ai but lives in a separate repository. Design index in [DESIGN.md](DESIGN.md); design docs in [docs/design/](docs/design), user docs in [docs/](docs).

## Iterative Workflow

1. **Calibrate** — detect Charuco board corners across cameras, run bundle adjustment via aniposelib
2. **Label** — display synchronized video frames, click to place skeleton keypoints, view epipolar lines and reprojection errors in real time (the 3D view panel is hidden; `SHOW_3D_VIEW` in `label_view.py`)
3. **Export** — **Sync** writes the mapped `.slp` files for SLEAP training/inference; at the end, **File > Export points...** (or `uv run label3d-export`) writes every point, 2D and 3D, to one Parquet or CSV file, in a wide or long layout
4. **Re-import** — train a SLEAP model on the exported `.slp` files, run inference, drop the resulting `.slp` predictions back into `project/slp/` (or a subfolder), then import again. Repeat from step 2 to review, correct, and add more ground-truth labels

## Key conventions

- **Units**: centimeters throughout (camera positions, body offsets, triangulated 3D coords)
- **World frame**: x forward, z up, y right; **Camera frame**: OpenCV convention (X right, Y down, Z forward)
- **Coordinates stored as raw pixels**; undistortion applied on the fly when needed for geometry
- **Video sync**: offset is implicit — `cam_start` values in `trials.csv` encode per-camera frame offset, no separate column

## Data files (hierarchical layout — all projects)

```
project/
├── project.toml, trials.csv, slp/, exports/
└── experiments/{name}/
    ├── calibrations/          # {camera}/ (take videos, corners.parquet), registry *.toml, refined/
    ├── videos/
    └── points.parquet
```

- `trials.csv` — **global**, at project root; columns `experiment, trial, is_calibration, cam0, cam0_start, cam0_end, cam0_mtime, cam1, …`; blank `_end` / `_mtime` cells are measured and written back on load (`check_trial_videos`), so later loads don't open the videos; trial names are local to each experiment (disambiguated by the `experiment` column). `is_calibration = True` rows are **calibration takes** (one synchronized calibration recording, e.g. `PreCal` / `PostCal`); their videos live in `calibrations/{cam}/`, not `videos/`. Any column that is not `experiment`, `trial`, `is_calibration` or a `*_start`/`*_end`/`*_mtime` suffix is read as a camera name
- `experiments/{name}/calibrations/` — the calibration registry (`calibration_registry.py`): every solution as an aniposelib `CameraGroup.dump()` file with provenance in `[metadata]`; refinements in `refined/`. The one in use is `[calibration.active]` in `project.toml`, and the one the 3D points came from is `[calibration.triangulated_with]`. Projects with the pre-registry layout (`calibration.toml`, `calibration_takes/`, `calibration/{cam}/`) are not supported; they are set up again
- **Working files are Parquet** (Phase 15): points, corners, sync history and refinement records. CSV versions are not read; a project that has one is refused on load and set up again. `trials.csv` stays CSV
- `calibrations/{cam}/corners.parquet` — every take's detected corners for one camera; columns `take, frame, corner_id, x, y, detected`
- **Calibration is video-only** — image-based calibration is deprecated and raises
- `experiments/{name}/videos/` — video files for that experiment's trials (no trials.csv here)
- `experiments/{name}/points.parquet` — in memory a pandas DataFrame with row index `(experiment, trial, frame, instance, bodypart)` (5-level); column MultiIndex `(view, coord)` where cameras have `{x, y, xr, yr, state, source, score, sync}` and `"3d"` pseudo-camera has `{x, y, z}`; `state` ∈ `{set, predicted, not_set, not_visible}`; `source` ∈ `{label3d, sleap_user, sleap_pred}`; `sync` is the id of the sync that brought a point in (a point is new when it equals the latest id). On disk it is flat: the index levels are columns and each column is named `{view}/{coord}` (`read_points_df`, `write_points_file` in `points.py`)
- `sync_history.parquet` — at project root; one row per (sync, experiment, trial) for each SLEAP read that brought in points
- **Skeletons** — no `skeleton.json`: each camera's skeleton comes from its `.slp` files or from `[[skeleton.assigned]]` in `project.toml` (File > Load skeleton...), with `[skeleton.node_offsets_cm]`; nodes are matched by name and the project skeleton is their union. A node a camera's skeleton lacks can't be labelled there and doesn't count for triangulation. Rules in [docs/design/data_model.md](docs/design/data_model.md#skeletons-phase-14)
- `slp/` — imported `.slp` files (global, at project root), each in its own folder, `slp/{stem}/{name}.slp`, with SLEAP's `models/` folder beside it. Found recursively, leaving out `models/` and `backup/` (`find_slp_files` in `slp_sync.py`). Sync rewrites only the project's part of a file: videos not in `trials.csv`, and frames outside a trial's `_start`–`_end`, are kept untouched
- `exports/` — point files (Parquet or CSV, wide or long) from **File > Export points...** / `label3d-export` (`export.py`; columns in [docs/user_workflow.md](docs/user_workflow.md#export-columns))

## External dependencies cloned locally (not in repo)

- `sleap/` — reference for `.slp` HDF5 format and keyboard shortcuts
- `aniposelib/` — calibration; local checkout is on `sys.path` in scripts

## Synthetic test data

See [docs/design/synthetic_data.md](docs/design/synthetic_data.md). Quick start:

```
uv run python generate_synthetic.py synthetic_config.toml
```

## Development

```
uv run label3d          # launches the app
uv run label3d-export <project>   # Parquet/CSV export, no GUI
uv run python <script>  # run any script in the project venv
```

## Implementation phases

Phases 1–12 complete. Phase 13, batch triangulation and export ([phase_13.md](docs/design/phases/phase_13.md)), is implemented and awaiting a check in the app. Phase 14, per-camera skeletons ([phase_14.md](docs/design/phases/phase_14.md)), is implemented and awaiting a check in the app. Phase 15, Parquet working files ([phase_15.md](docs/design/phases/phase_15.md)), is implemented and awaiting a check in the app. Deferred: axis alignment ([phase_16.md](docs/design/phases/phase_16.md); geometry helpers and CameraPanel pick mode exist, the UI widget does not), export to other pose formats and polish ([phase_17_18.md](docs/design/phases/phase_17_18.md)), multi-instance support ([phase_19.md](docs/design/phases/phase_19.md)). [docs/design/phases/phase_12.md](docs/design/phases/phase_12.md) covers the finished calibration refinement with labelled points, [phase_11.md](docs/design/phases/phase_11.md) the finished calibration registry, [phase_10.md](docs/design/phases/phase_10.md) the finished point provenance and review work, [phase_9.md](docs/design/phases/phase_9.md) the finished calibration-take work, [phase_8.md](docs/design/phases/phase_8.md) the finished SLP sync work.

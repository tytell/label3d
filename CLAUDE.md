# label3d — Claude Code Context

## What this project is

A PySide6 desktop app for 3D point labeling across multiple synchronized calibrated video streams. It interfaces with sleap.ai but lives in a separate repository. Design index in [DESIGN.md](DESIGN.md); detailed docs in [docs/](docs/).

## Iterative Workflow

1. **Calibrate** — detect Charuco board corners across cameras, run bundle adjustment via aniposelib
2. **Label** — display synchronized video frames, click to place skeleton keypoints, view epipolar lines and reprojection errors in real time (the 3D view panel is hidden; `SHOW_3D_VIEW` in `label_view.py`)
3. **Export** — write per-camera `.slp` files for SLEAP training/inference
4. **Re-import** — train a SLEAP model on the exported `.slp` files, run inference, drop the resulting `.slp` predictions back into `project/slp/`, then import again. Repeat from step 2 to review, correct, and add more ground-truth labels

## Key conventions

- **Units**: centimeters throughout (camera positions, body offsets, triangulated 3D coords)
- **World frame**: x forward, z up, y right; **Camera frame**: OpenCV convention (X right, Y down, Z forward)
- **Coordinates stored as raw pixels**; undistortion applied on the fly when needed for geometry
- **Video sync**: offset is implicit — `cam_start` values in `trials.csv` encode per-camera frame offset, no separate column

## Data files (hierarchical layout — all projects)

```
project/
├── project.toml, skeleton.json, trials.csv, slp/
└── experiments/{name}/
    ├── calibrations/          # {camera}/ (take videos, corners.csv), registry *.toml, refined/
    ├── videos/
    └── points.csv
```

- `trials.csv` — **global**, at project root; columns `experiment, trial, is_calibration, cam0, cam0_start, cam0_end, cam0_mtime, cam1, …`; blank `_end` / `_mtime` cells are measured and written back on load (`check_trial_videos`), so later loads don't open the videos; trial names are local to each experiment (disambiguated by the `experiment` column). `is_calibration = True` rows are **calibration takes** (one synchronized calibration recording, e.g. `PreCal` / `PostCal`); their videos live in `calibrations/{cam}/`, not `videos/`. Any column that is not `experiment`, `trial`, `is_calibration` or a `*_start`/`*_end`/`*_mtime` suffix is read as a camera name
- `experiments/{name}/calibrations/` — the calibration registry (`calibration_registry.py`): every solution as an aniposelib `CameraGroup.dump()` file with provenance in `[metadata]`; refinements in `refined/`. The one in use is `[calibration.active]` in `project.toml`, and the one the 3D points came from is `[calibration.triangulated_with]`. Never read or write `calibration.toml` / `calibration_takes/` (legacy, imported once on load)
- `calibrations/{cam}/corners.csv` — every take's detected corners for one camera; columns `take, frame, corner_id, x, y, detected`
- **Calibration is video-only** — image-based calibration is deprecated and raises
- `experiments/{name}/videos/` — video files for that experiment's trials (no trials.csv here)
- `experiments/{name}/points.csv` — pandas MultiIndex CSV; row index `(experiment, trial, frame, instance, bodypart)` (5-level); column MultiIndex `(view, coord)` where cameras have `{x, y, xr, yr, state, source, score, sync}` and `"3d"` pseudo-camera has `{x, y, z}`; `state` ∈ `{set, predicted, not_set, not_visible}`; `source` ∈ `{label3d, sleap_user, sleap_pred}`; `sync` is the id of the sync that brought a point in (a point is new when it equals the latest id)
- `sync_history.csv` — at project root; one row per (sync, experiment, trial) for each SLEAP read that brought in points
- `skeleton.json` — `{nodes, edges, symmetries, node_offsets_cm}` (shared at project root)
- `slp/` — imported `.slp` files (global, at project root)

## External dependencies cloned locally (not in repo)

- `sleap/` — reference for `.slp` HDF5 format and keyboard shortcuts
- `aniposelib/` — calibration; local checkout is on `sys.path` in scripts

## Synthetic test data

See [docs/synthetic_data.md](docs/synthetic_data.md). Quick start:

```
uv run python generate_synthetic.py synthetic_config.toml
```

## Development

```
uv run label3d          # launches the app
uv run python <script>  # run any script in the project venv
```

## Implementation phases

Phases 1–10 complete. Phases 11–13 are next, in order: calibration registry and switching ([docs/phases/phase_11.md](docs/phases/phase_11.md)), calibration refinement with labelled points ([phase_12.md](docs/phases/phase_12.md)), batch triangulation and export ([phase_13.md](docs/phases/phase_13.md)). Deferred: axis alignment ([phase_14.md](docs/phases/phase_14.md); geometry helpers and CameraPanel pick mode exist, the UI widget does not), export view and polish ([phase_15_16.md](docs/phases/phase_15_16.md)), multi-instance support ([phase_17.md](docs/phases/phase_17.md)). [docs/phases/phase_10.md](docs/phases/phase_10.md) covers the finished point provenance and review work, [phase_9.md](docs/phases/phase_9.md) the finished calibration-take work, [phase_8.md](docs/phases/phase_8.md) the finished SLP sync work.

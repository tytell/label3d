# label3d — Claude Code Context

## What this project is

A PySide6 desktop app for 3D point labeling across multiple synchronized calibrated video streams. It interfaces with sleap.ai but lives in a separate repository. Design index in [DESIGN.md](DESIGN.md); detailed docs in [docs/](docs/).

## Iterative Workflow

1. **Calibrate** — detect Charuco board corners across cameras, run bundle adjustment via aniposelib
2. **Label** — display synchronized video frames, click to place skeleton keypoints, view epipolar lines and 3D reconstruction in real time
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
    ├── calibration/{camera}/, videos/
    ├── calibration.toml
    └── points.csv
```

- `trials.csv` — **global**, at project root; columns `experiment, trial, cam0, cam0_start, cam0_end, cam1, …`; trial names are local to each experiment (disambiguated by the `experiment` column)
- `experiments/{name}/calibration.toml` — aniposelib `CameraGroup.dump()` format
- `experiments/{name}/videos/` — video files for that experiment's trials (no trials.csv here)
- `experiments/{name}/points.csv` — pandas MultiIndex CSV; row index `(experiment, trial, frame, instance, bodypart)` (5-level); column MultiIndex `(view, coord)` where cameras have `{x, y, xr, yr, state}` and `"3d"` pseudo-camera has `{x, y, z}`; `state` ∈ `{set, predicted, not_set, not_visible}`
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

Phases 1–7 complete. Phase 8 (axis alignment) next — geometry helpers and CameraPanel pick mode done, UI widget not yet created. See [docs/phases/phase_8.md](docs/phases/phase_8.md) for the current phase spec, [docs/phases/phase_9_11.md](docs/phases/phase_9_11.md) for upcoming work.

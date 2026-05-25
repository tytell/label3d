# label3d — Claude Code Context

## What this project is

A PySide6 desktop app for 3D point labeling across multiple synchronized calibrated video streams. It interfaces with sleap.ai but lives in a separate repository. Full design specification in [DESIGN.md](DESIGN.md).

## Three-step workflow

1. **Calibrate** — detect Charuco board corners across cameras, run bundle adjustment via aniposelib
2. **Label** — display synchronized video frames, click to place skeleton keypoints, view epipolar lines and 3D reconstruction in real time
3. **Export** — write per-camera `.slp` files for SLEAP training/inference

## Key conventions

- **Units**: centimeters throughout (camera positions, body offsets, triangulated 3D coords)
- **World frame**: x forward, z up, y right; **Camera frame**: OpenCV convention (X right, Y down, Z forward)
- **Coordinates stored as raw pixels**; undistortion applied on the fly when needed for geometry
- **Video sync**: offset is implicit — `cam_start` values in `trials.csv` encode per-camera frame offset, no separate column

## Data files

- `calibration.toml` — aniposelib `CameraGroup.dump()` format: sections `cam_0`, `cam_1`, …; keys `name`, `size`, `matrix`, `distortions`, `rotation` (Rodrigues), `translation`
- `videos/trials.csv` — columns `trial, cam0, cam0_start, cam0_end, cam1, …`; video filenames matched by name, not path
- `points.csv` — pandas MultiIndex CSV; row index `(trial, frame, instance, bodypart)`; column MultiIndex `(view, coord)` where cameras have `{x, y, xr, yr, state}` and `"3d"` pseudo-camera has `{x, y, z}`; `state` ∈ `{set, predicted, not_set, not_visible}`
- `skeleton.json` — `{nodes, edges, symmetries, node_offsets_cm}`

## External dependencies cloned locally (not in repo)

- `sleap/` — reference for `.slp` HDF5 format and keyboard shortcuts
- `aniposelib/` — calibration; local checkout is on `sys.path` in scripts

## Synthetic test data

```
uv run python generate_synthetic.py synthetic_config.toml
```

Writes `test_project/` with two synthetic camera views, known-correct ground truth (`points.csv`), calibration, and `.slp` files. Sanity checks (triangulation, epipolar, reprojection) should all pass at machine precision.

## Development

```
uv run label3d          # launches the app
uv run python <script>  # run any script in the project venv
```

## Implementation phases

See `DESIGN.md §Implementation Plan`. Current status: Phases 1–5 complete. Next: Phase 6 (export view) or Phase 7 polish items.

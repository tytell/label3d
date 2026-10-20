# label3d — Synthetic Data Generator

`generate_synthetic.py` creates a complete test project with known-correct ground truth: cameras, videos with rendered skeleton keypoints, a ground-truth calibration registry entry, `points.parquet`, `project.toml`, and `.slp` files (one per camera, whose skeletons can differ; see below).

```bash
uv run python generate_synthetic.py synthetic_config.toml
uv run python generate_synthetic.py synthetic_config.toml --no-charuco  # skip calibration images
```

Sanity checks (triangulation, epipolar, reprojection) should all pass at machine precision.

## Module structure

```
synthetic/
├── __init__.py              # re-exports main entry point
├── cameras.py               # look_at(), compute_intrinsics(), build_cameras()
├── motion.py                # rotation matrices, compute_body_poses(), compute_world_points()
├── projection.py            # project_points(), project_all_frames()
├── rendering.py             # render_frame(), write_videos()
├── writers.py               # write_calibration_toml(), write_trials_csv(), write_points_csv(),
│                            #   camera_skeleton(), write_slp_files(), write_project_toml()
├── checks.py                # _dlt_triangulate(), _epipolar_error(), run_sanity_checks()
└── charuco.py               # generate_charuco_images()
```

## Output structure

```
test_project/
├── trials.csv                    # global trials table (experiment column)
├── project.toml                  # [skeleton.node_offsets_cm]; [[skeleton.assigned]] with --no-slp
├── slp/
└── experiments/
    └── default/
        ├── calibrations/{cam}/*.mp4                     # charuco take videos
        ├── calibrations/ground-truth_{timestamp}.toml   # ground-truth camera params; active
        ├── points.parquet        # ground-truth points (5-level row index)
        └── videos/               # rendered trial videos
```

## Per-camera skeletons

A camera's `missing_nodes` leaves those nodes out of its skeleton, to test Phase 14:

```toml
[cameras.cam3]
missing_nodes = ["dorsal"]
```

That camera's `.slp` file gets a skeleton without them, named `synthetic_without_dorsal`, and no points for them; its cells in `points.parquet` are `not_set`. Edges touching a missing node are dropped from its skeleton.

`--no-slp` writes no `.slp` files. Each distinct camera skeleton goes to `project.toml` as a `[[skeleton.assigned]]` entry instead, as after **File > Load skeleton...**, for testing the workflow where the first Sync creates the files.

## Charuco image generation config (`synthetic_config.toml`)

```toml
[charuco]
squares_x = 6
squares_y = 6
square_length = 24.33          # mm
marker_length = 17.0           # mm
marker_bits = 5
dict_size = 50
num_images = 12
board_image_resolution = 800   # px
background_color = [40, 40, 40]
seed = 348294

[charuco.placement]
center = [0.0, 0.0, 0.0]
extent = [15.0, 15.0, 10.0]   # half-extents in cm
tilt_range_deg = 30.0
```

The `[charuco]` section is optional — if absent, calibration images are not generated.

## How charuco image generation works

1. **Generate flat board image** — `cv2.aruco.CharucoBoard.generateImage()` produces a high-resolution image of the board
2. **Compute board poses** — For each of `num_images` poses: pick a random position within the placement volume and random orientation (tilted up to `tilt_range_deg`). Uses seeded RNG for reproducibility
3. **Render into each camera** — For each (pose, camera) pair: reverse-map each output pixel through the camera model to the board plane, look up the board texture. **Horizontal flip**: the U coordinate is mirrored (`1 - u`) so ArUco markers read correctly through the camera
4. **Write output** — Save as `calibrations/{cam_name}/board_{NN:03d}.png`

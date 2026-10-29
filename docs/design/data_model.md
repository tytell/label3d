# label3d — Data Model

## Core Concepts

### Camera Calibration (aniposelib / OpenCV style)
Calibration is handled by **aniposelib**'s `CameraGroup` and `CharucoBoard` classes.

Each `Camera` stores:
- `matrix`: 3×3 intrinsic matrix K (fx, fy, cx, cy)
- `dist`: distortion coefficients — 5-element `[k1, k2, p1, p2, k3]` for standard, 4-element for fisheye
- `rvec`: rotation vector (Rodrigues, world-to-camera)
- `tvec`: translation vector (world-to-camera)
- `size`: `(width, height)` in pixels
- `name`: camera identifier string

A `CameraGroup` holds a list of `Camera` objects and is saved and loaded with `CameraGroup.dump()` / `CameraGroup.load()` (aniposelib format — see Calibration file schema below). Every solution an experiment produces is kept in its calibration registry (see Calibration outputs).

### Epipolar Geometry
- Clicking a point in camera A defines a **ray** in 3D space → projects to an **epipolar line** in camera B
- If the same physical point is clicked in cameras A and B, triangulated 3D position should be consistent
- **Reprojection error**: re-project triangulated point back into each camera and measure pixel distance from clicked point

### Point Labeling
- A **label** is a named 3D point (e.g. body landmark)
- Each label has one clicked 2D location per camera per frame (can be missing/occluded)
- Minimum 2 cameras needed to triangulate; more = lower error
- Points are defined by a skeleton, which represents certain points that are connected to one another and any symmetries (left/right)

---

## Project Layout

All projects use the **hierarchical layout**: a shared project root with `experiments/` subfolders, one per calibration group.

```
Project
├── project.toml             # top-level config: name, charuco params, [skeleton] (shared)
├── trials.csv               # global: all trials across all experiments (has experiment column)
├── slp/                     # optional: imported .slp files (global, not per-experiment)
├── sync_history.parquet     # one row per trial for each SLEAP read that brought in points
├── exports/                 # File > Export points... / label3d-export output: points_{timestamp}.parquet or .csv
└── experiments/
    ├── PS18/                # one subfolder per experiment
    │   ├── calibrations/    # everything about calibrating this experiment
    │   │   ├── cam0/        # take videos + corners.parquet for cam0
    │   │   ├── cam1/
    │   │   ├── *.toml       # registry: every solution, one .toml each
    │   │   └── refined/     # refinements (Phase 12), each with a _points.parquet
    │   ├── videos/          # video files for this experiment's trials
    │   └── points.parquet   # 2D + 3D labels for this experiment's trials
    ├── PS19/
    │   └── …
    └── BG03/
        └── …
```

**Key design properties:**
- `trials.csv` is **global** at the project root; it has an `experiment` column so all experiments' trials live in one file. Trial names may repeat across experiments and are disambiguated by the `experiment` value.
- Each experiment has its own calibration registry and `points.parquet` — per-experiment because each has a distinct calibration and independent label data.
- `.slp` files live in the top-level `slp/` folder, each in a folder of its own (`slp/{stem}/{name}.slp`) so that SLEAP's `models/` folder for it can sit beside it; `Project.slp_files()` finds them recursively, leaving out `models/` (SLEAP writes each model's training labels there as `.slp` files) and `backup/`. Older projects with the files directly in `slp/` still load. Sync rewrites only what the project covers: the labeled frames of trial videos inside each trial's `_start`–`_end`. Videos that are in no trial, with their frames and suggestions, and frames of trial videos outside every trial's range, are kept untouched, as are all skeletons, tracks, sessions, ROIs and masks that don't depend on rewritten frames (`_merge_existing` in `sleap_adapter.py`). The adapter receives the full global `trials.csv` and resolves which `(experiment, trial, camera)` each file belongs to by matching video basenames across all rows.
- The points row index is `(experiment, trial, frame, instance, bodypart)` — the `experiment` level is constant within a single file but allows cross-experiment aggregation via plain `pd.concat`.
- The calibration in use for each experiment is the registry entry named in `[calibration.active]` of `project.toml`; the geometry stack (triangulation, epipolar lines) always sees exactly one `CameraGroup` per experiment.

### Project configuration (`project.toml`)

Optional project-level configuration file. Loaded by `Project.load()` into `project.config: dict`.

```toml
[project]
name = "my_experiment"

[charuco]
squares_x = 6               # board squares along X
squares_y = 8               # board squares along Y
square_length = 30.0         # mm — side length of each chessboard square
marker_length = 22.0         # mm — side length of ArUco markers within squares
marker_bits = 4              # ArUco marker bit depth (4, 5, 6, or 7)
dict_size = 50               # ArUco dictionary size (50, 100, 250, 1000)

[calibration]
frame_skip = 20              # detect corners on every Nth video frame
repeats = 3                  # calibrate each take this many times, keep the best
rotation_drift_deg = 0.5     # flag takes whose relative rotation differs by more
baseline_drift_pct = 1.0     # flag takes whose camera baseline differs by more

[calibration.active]         # experiment -> registry id of the calibration in use
PS18 = "charuco-all_20260901-101500"
"20260619" = "refine_20261005-142210"    # all-digit names quoted

[calibration.triangulated_with]   # experiment -> id the 3D points were computed with
PS18 = "charuco-all_20260901-101500"
```

`[calibration.active]` and `[calibration.triangulated_with]` are written by `Project.set_active_calibration()` and `Project.set_triangulated_with()`. When the two differ for an experiment, the Label tab warns that its 3D points are stale.

The `[charuco]` section maps directly to `aniposelib.boards.CharucoBoard.__init__` parameters. Both sections are written back by the calibration panel when the user changes a value.

`[calibration]` drift thresholds are compared per camera *pair* between each take's solution and the joint solution: intrinsics should be identical across takes, so what moves when a camera is bumped is the relative rotation and the baseline between pairs. Exceeding either threshold flags the take in the comparison table.

### Calibration outputs

| Path | Contents |
|---|---|
| `experiments/{exp}/calibrations/{label}_{YYYYmmdd-HHMMSS}.toml` | one registry entry: a `charuco`, `imported` or `align` solution |
| `experiments/{exp}/calibrations/refined/{id}.toml` | a `refine` entry (Phase 12) |
| `experiments/{exp}/calibrations/refined/{id}_points.parquet` | the exact points that went into that refinement |
| `experiments/{exp}/calibrations/{cam}/corners.parquet` | every take's detected corners for one camera |

**Registry entries** are plain `CameraGroup.dump()` files. Provenance is in the `[metadata]` table, which aniposelib keeps on load (`cgroup.metadata`). The id is the file stem; it never changes, and renaming changes only `label`. Each Calibrate run saves one `charuco` entry per take (`charuco-{take}`) and one for the joint solution (`charuco-all`). Deleting an entry also deletes every entry derived from it through `parent`. The module is `calibration_registry.py`.

```toml
[metadata]
id = "charuco-all_20261005-142210"
label = "charuco-all"
created = "2026-10-05T14:22:10"
method = "charuco"                    # charuco | refine | align | imported
parent = ""                           # id this one was derived from (refine, align)
charuco_takes = ["PreCal", "PostCal"]
source_path = "…"                     # imported only: the file it came from

[metadata.charuco]                    # charuco only: enough to redo the solve
frame_skip = 20
n_rows = 412
board = { squares_x = 6, squares_y = 8, square_length = 30.0, marker_length = 22.0, marker_bits = 4, dict_size = 50 }
take_ranges = { PreCal = { start = 0, end = 1800 }, PostCal = { start = 0 } }
detected = { PreCal = "2026-10-05T14:20:03", PostCal = "2026-10-05T14:20:03" }

[metadata.errors]                     # px, and run-to-run spread over repeats
charuco = 0.41
spread_rotation_deg = 0.02
spread_baseline_pct = 0.1

[metadata.alignment]                  # align only (Phase 16); see the schema below
```

A `refine` entry (Phase 12) has `parent` set to the entry it started from and `charuco_takes` copied from it, plus:

```toml
[metadata.points]
sources = ["label3d", "sleap_user"]
score_min = 0.8                       # for sleap_pred only
min_cameras = 2
n_points = 1840
points_file = "refine_20261005-142210_points.parquet"

[metadata.points.frames]              # trial -> frames that had points
t1 = [10, 55, 90]

[metadata.options]
only_extrinsics = true
holdout = 0.2                         # fraction of frames held out
n_repeats = 3                         # 1 when no camera pair had more than n_samp_iter points
seed = 138368576                      # hold-out split; repeat r used seed + r
kept_repeat = 1
n_samp_full = 1000
n_samp_iter = 200

[metadata.errors]                     # px unless noted; *_before = the parent on the same points
holdout = 1.9
holdout_p90 = 4.2
holdout_before = 2.6
holdout_p90_before = 5.8
points = 1.7
points_before = 2.5
charuco = 0.45                        # charuco corners of the parent's takes, triangulated
charuco_before = 0.41
scale_correction = 1.002              # ratio
align_centre_cm = 0.12                # what the alignment leaves: RMS camera-centre distance
align_centre_pct = 0.25               # the same, % of the mean camera-pair baseline
align_rotation_deg = 0.04             # RMS camera orientation difference
spread_rotation_deg = 0.01            # spread_* left out when Stage 2 ran once
spread_baseline_pct = 0.05
```

**`{id}_points.parquet` columns:** `trial, frame, instance, bodypart, camera, x, y, source, score, holdout` (`RECORD_SCHEMA`). One row is one camera observation that went into the refinement, in raw pixels, copied so the file still describes the fit after the points change. `holdout` is True for observations in held-out frames. With the parent entry and the stored seed, this is enough to rerun the refinement exactly.

**`corners.parquet` columns:** `take` (string), `frame`, `corner_id` (int32), `x, y` (float32), `detected` (string). One row is one detected corner. Cross-camera correspondence is by take name and absolute video frame. `detected` is the time of the detection run that produced that take's rows. Re-detecting a take replaces only its rows, so a charuco entry whose `metadata.charuco.detected` no longer matches was solved from older corners. `load_corners(cam_dir, takes=…)` filters by take.

**Old projects.** Projects made before Phase 11 kept solutions in `calibration.toml` and `calibration_takes/`, and calibration media in `calibration/{cam}/`. None of these are read any more; set such projects up again.

### Videos and trials

`trials.csv` lives at the project root and has one row per trial **and one row per calibration take** across all experiments. Columns:

| experiment | trial | is_calibration | cam0 | cam0_start | cam0_end | cam1 | cam1_start | cam1_end | … |
|---|---|---|---|---|---|---|---|---|---|
| PS18 | PreCal | True | PS18_PreCal_cam0.mp4 | 0 | | PS18_PreCal_cam1.mp4 | 0 | | … |
| PS18 | trial1 | False | cam0_t1.mp4 | 0 | 500 | cam1_t1.mp4 | 3 | 503 | … |
| PS18 | trial2 | False | cam0_t2.mp4 | 0 | 480 | cam1_t2.mp4 | 0 | 480 | … |
| PS18 | PostCal | True | PS18_PostCal_cam0.mp4 | 0 | | PS18_PostCal_cam1.mp4 | 0 | | … |
| PS19 | trial1 | False | cam0_t1.mp4 | 0 | 600 | cam1_t1.mp4 | 0 | 600 | … |

- `_start` / `_end`: frames of the video file (inclusive), **numbered from 1** as the GUI shows them (`frames.FRAME_BASE`). Blank = start / end of file. `read_trials_csv` / `write_trials_csv` convert to and from the 0-based frames used everywhere in memory, in the working files, in exports and in `.slp` files. `project.toml` records `[trials] frame_base = 1`; a project without it has a 0-based `trials.csv` (written before the change), which `Project.load()` converts once, keeping the original as `trials.csv.bak`. Frame ranges in `[calibration.ranges]` stay 0-based.
- `_mtime`: the video's modification time when its `_end` was recorded (UTC, `2026-07-23T22:52:38Z`).
- **Filled in on load.** `Project.load()` checks every video (`check_trial_videos`). A blank `_end` is set to the video's last frame, and a blank `_mtime` to its modification time, and `trials.csv` is rewritten (the original is kept once as `trials.csv.bak`). Counting frames means opening each video, which is slow on a network share, so this happens once; later loads only check that each file exists and compare its time. A video whose time differs from `_mtime` gets a warning when the project opens, and nothing is changed. Project setup fills both columns when it writes `trials.csv`. The app shows a progress dialog when opening takes more than half a second.
- The per-camera offset is **implicit**: global frame `N` maps to video frame `_start + N` for each camera (both 0-based, in memory).
- **Trial length** = `min(_end_i - _start_i + 1)` across all cameras. Blank `_end` resolved via `cv2.CAP_PROP_FRAME_COUNT`.
- Camera ID columns must match the `name` field values in the calibration files.
- Any column that is not `experiment`, `trial`, `is_calibration`, or a `*_start` / `*_end` / `*_mtime` suffix is taken to be a camera name — new metadata columns must be added to `project.NON_CAMERA_COLUMNS`, new per-camera suffixes to `project.CAMERA_COLUMN_SUFFIXES`.

**Calibration takes** (`is_calibration = True`): a take is one synchronized calibration recording, typically one before and one after the trials. The `trial` column holds the take name (from the calibration keyword, e.g. `PreCal`). Takes reuse the camera / `_start` / `_end` triples, so a take can be trimmed to the segment where the board is visible. Take videos resolve relative to `experiments/{experiment}/calibrations/{camera}/`; trial videos relative to `experiments/{experiment}/videos/`. Takes are excluded from `Project.trials` and from SLP cell matching.

### Primary storage: `points.parquet`

One Parquet file per experiment, storing both 2D and 3D data (Phase 15). label3d no longer reads the CSV working files (`points.csv`, `corners.csv`, `sync_history.csv`): `Project.load` refuses a project that has one without its Parquet file, and the project is set up again.

In memory, as below. Rows indexed by `(experiment, trial, frame, instance, bodypart)`. Columns grouped by **view**: each real camera has `(x, y, xr, yr, state, source, score, sync, px, py, pscore)` coords; `"3d"` pseudo-camera has `(x, y, z)`.

```
                                          cam0                          cam1                3d
experiment trial  frame  instance  bodypart   x    y    xr   yr   state  source      score  sync   x   …   x    y    z
PS18       trial1 0      0         head       92   202  91   203  set    label3d                     88  …   1.2  0.4  3.1
PS18       trial1 7      0         head       95   205             predicted  sleap_pred  0.93   4     …
```

Column structure is a 2-level `MultiIndex (view, coord)`:
- Real cameras: `coord` ∈ `{x, y, xr, yr, state, source, score, sync, px, py, pscore}` — clicked coords, reprojected coords, point state, provenance, and the hidden prediction
- `3d` pseudo-camera: `coord` ∈ `{x, y, z}` — triangulated world coords

**Provenance columns** (per camera):

| Column | Values | Meaning |
|---|---|---|
| `source` | `label3d`, `sleap_user`, `sleap_pred`, empty | Where the x/y came from. Any edit in label3d sets `label3d`. A label3d reprojection guess is `state=predicted`, `source=label3d`. Empty for `not_set`. |
| `score` | float or NaN | SLEAP per-point score, for `sleap_pred` |
| `sync` | sync id or NaN | The sync read that last brought the point in from SLEAP. Cleared when the user corrects or accepts the point. |

A point is **new** when its `sync` equals the latest id in the sync history.

**Hidden prediction** (per camera): `px`, `py`, `pscore` hold the SLEAP prediction under a user instance, as SLEAP's `from_predicted` does. Double-clicking a predicted instance in a camera makes a user instance there: every predicted point is copied as `set` (source `label3d`) and the prediction moves to `px, py, pscore`. Deleting the user instance in that camera (or deleting its last user point) brings the prediction back. Sync writes the hidden prediction as a `PredictedInstance` with the user `Instance` linked to it, and reading a `.slp` file fills these columns from each user instance's `from_predicted`. They are NaN where no prediction is hidden, and are not exported.

An instance is **predicted** in a camera when it has predicted points there and no user points (`set`, `not_visible`). Predicted instances can be selected (for their epipolar lines) but not edited until they are double-clicked.

**Point state values:**

| State | Meaning |
|---|---|
| `set` | User has manually placed this point (ground truth) |
| `predicted` | Point imported from external tracker; not yet reviewed |
| `not_set` | Point not yet placed — x, y are NaN; instance is user-type |
| `not_visible` | User marked as not visible — coordinates kept, excluded from triangulation |

**On disk** the table is flat, so tools other than pandas read it as is: the five index levels are ordinary columns, and each `(view, coord)` column is named `{view}/{coord}` (`cam0/x`, `3d/z`). The schema is fixed (`points_schema`):

| Column | Type |
|---|---|
| `experiment`, `trial`, `bodypart` | string |
| `frame`, `instance` | int64 |
| `{cam}/x, y, xr, yr, score, sync, px, py, pscore`, `3d/x, y, z` | float64 |
| `{cam}/state`, `{cam}/source` | string |

Rows are sorted by (experiment, trial, frame, instance) and stored in row groups of 100,000, zstd-compressed, so reading one trial skips the rest. A save writes `points.parquet.tmp`, keeps the previous file as `points.parquet.bak` and then moves the new file into place (`write_points_file`, `parquet_io.write_table`).

Written/read with:
```python
from label3d.points import read_points_df, write_points_file

write_points_file(df, path)                      # MultiIndex DataFrame -> flat file
df = read_points_df(path)                        # flat file -> MultiIndex DataFrame
df = read_points_df(path, trials=["trial1"])     # only some trials

# Access one camera
cam0 = df["cam0"]                          # columns: x, y, xr, yr, state, source, score, sync, px, py, pscore

# Access 3D
xyz = df["3d"][["x", "y", "z"]]                         # world coords, shape (n_rows, 3)
```

Missing coordinates are `NaN`. Reprojected coords (`xr`, `yr`) are `NaN` until triangulation has been run.

### Skeletons (Phase 14)

Cameras can have different skeletons: a node a camera never sees is often left out of that camera's SLEAP model. Nodes are matched across skeletons by exact name. There is no `skeleton.json`; the skeletons come from the `.slp` files and from `[skeleton]` in `project.toml` (`Project.load_skeletons()`, which reads only the files' headers).

Each **cell** — an (experiment, trial, camera), i.e. one video — has one skeleton, the first of:

1. the skeleton of its mapped `.slp` file, if that file exists;
2. the skeleton **assigned** to its camera in `project.toml` (by **File > Load skeleton...** or the setup file's `[skeleton]`);
3. the skeleton shared by every existing `.slp` file of its camera, if they all agree;
4. the union.

A node is **available** in a cell when it is in the cell's skeleton. Existing `.slp` files always keep their own skeleton when written; a new file is created with its cells' skeleton, so an assigned skeleton only affects files not created yet. All the cells mapped to one file must have the same skeleton (checked by **Set up Sleap files**, Sync and export).

The **project skeleton** (`Project.skeleton`, shown in the points list) is the union of the files' and the assigned skeletons: nodes in the order of the files, then the assigned skeletons; edges and symmetries are the union. Skeletons may disagree on edges, which only affect drawing: each camera panel draws its own skeleton's edges, and the union's are used only for a camera that falls back to the union. The skeletons **conflict**, and Sync is refused until it is fixed, when two skeletons that both have nodes `a` and `b` disagree on whether `a`–`b` is a symmetry, or a node ends up with two symmetry partners.

`project.toml`:

```toml
[skeleton.node_offsets_cm]      # used by random instance placement
snout = [10.0, 0.0, 0.0]

[[skeleton.assigned]]           # one per loaded skeleton
name = "full"
cameras = ["cam0", "cam2"]      # a camera is in at most one entry
nodes = ["snout", "left_eye", "tail_tip"]
edges = [["snout", "left_eye"]]
symmetries = []
```

An older project's `skeleton.json` is copied once into `project.toml` (assigned to every camera) and then ignored.

A node available in fewer than 2 cameras of a trial is never triangulated (it can still be labelled in 2D): triangulation counts a camera's point only where its skeleton has the node. Points left in a cell whose skeleton lacks the node (after a skeleton changed) are kept in the points file but hidden, not triangulated, not written to `.slp` and left out of the export; **File > Remove points outside the skeletons...** deletes them (`Project.remove_points_outside`). SLP points of nodes a cell's skeleton lacks are dropped on import and counted in the sync summary.

### Sync history: `sync_history.parquet`

At the project root. Parquet can't be appended to, so each recorded sync rewrites the file; it stays small. One row per (sync, experiment, trial) for every SLEAP read that brought in new points (startup read, Sync, Import predictions, Set up, sync on close). A read that changes nothing adds no row.

| Column | Meaning |
|---|---|
| `id` | Sync id; the largest is the latest sync |
| `timestamp` | ISO time of the read |
| `kind` | `open`, `sync`, `close`, `import` or `setup` |
| `experiment`, `trial` | Trial the counts are for |
| `n_new_user` | SLEAP user labels taken (including conflicts resolved with Use SLEAP) |
| `n_new_pred` | Predictions where label3d had none |
| `n_updated_pred` | Predictions that moved more than 0.5 px from the stored prediction |
| `n_conflicts` | Points that differed from label3d |
| `n_frames` | Frames with new points |

The initial label import at project setup records sources but no sync, so nothing starts out new.

### Import / Export — Plugin Architecture

Format adapters live in `label3d/adapters/` — one class per external format, all implementing `PoseAdapter` ABC from `adapters/base.py`. Auto-discovered at startup.

```python
class PoseAdapter(ABC):
    @abstractmethod
    def load(self, files: list[Path], trials: pd.DataFrame, cameras: list[str]) -> pd.DataFrame:
        """Return points-compatible DataFrame with MultiIndex rows and columns."""

    @abstractmethod
    def save(self, df: pd.DataFrame, output_dir: Path, cameras: list[str], **kwargs) -> None:
        """Write clicked 2D points to the external format."""

    @classmethod
    def name(cls) -> str: ...

    @classmethod
    def file_filter(cls) -> str: ...
```

**SLEAP adapter — camera matching logic:**
1. For each `.slp` file, read `Video.filename` values embedded in the file
2. Match using **basename only** against video basenames across all rows of global `trials.csv`
3. Extract `(frame_idx, node_name, x, y)` per `(experiment, trial, camera)` with state `predicted`

### Calibration file schema (TOML)

Written by aniposelib's `CameraGroup.dump()`. Section names are `cam_0`, `cam_1`, … (zero-indexed).

```toml
[cam_0]
name = "cam0"
size = [1920, 1080]
matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
distortions = [k1, k2, p1, p2, k3]
rotation = [rx, ry, rz]      # Rodrigues vector, world-to-camera
translation = [tx, ty, tz]

[cam_1]
# …

[metadata]
# Registry provenance (see Calibration outputs); ignored by the geometry code

[metadata.alignment]
# align entries only; written by the axis alignment step
# origin: 3D point used as origin in the new coordinate system, in old world coords (cm)
origin = [ox, oy, oz]
# R_align: 3×3 rotation matrix whose columns are [x_hat, y_hat, z_hat] in OLD world space
# Transform new→old: p_old = R_align @ p_new + origin
# Transform old→new: p_new = R_align.T @ (p_old − origin)
R_align = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
```

`CameraGroup.load()` and `label3d`'s `load_cameras()` read every top-level table except `metadata` as a camera, so all other data must go under `[metadata]`.

---

## Project Setup Utility

### Overview

Before using label3d, users organize raw video files into the expected project structure.
To start from existing SLEAP labels instead, with the videos left where they are, see
[setup_from_slp.md](../setup_from_slp.md).
 The **setup utility** (`label3d/setup_project.py`) takes a flat folder of videos + optional `.slp` files and reorganizes them.

```bash
uv run python setup_project_cli.py                  # interactive mode
uv run python setup_project_cli.py config.toml      # config file mode
```

### Input parameters

| Parameter | Description |
|---|---|
| `source_dir` | Flat folder containing all videos and .slp files |
| `output_dir` | Path for the new project folder |
| `camera_names` | Camera name substrings found in filenames (e.g. `left, right, bottom`) |
| `calib_keyword` | Keyword identifying calibration files (e.g. `calib`, `charuco`) |
| `experiment_keywords` | *(optional)* e.g. `PS, BG`; groups files by experiment prefix |
| `match_strategy` | `keyword`, `number`, `datestamp`, `regex`, or `mtime` |

### Two-level file classification

When `experiment_keywords` is provided:

**Level 1 — Experiment identification**: Each keyword matched as case-insensitive prefix immediately followed by a number. Pattern: `(?:PS|BG)\d+`.

**Level 2 — Trial identification**: After the experiment identifier is stripped, `match_strategy` is applied to the remaining stem.

### Data model

```python
@dataclass
class ExperimentAssignment:
    name: str
    trials: dict[str, dict[str, Path]]        # trial_name → {camera: path}
    calibration: dict[str, list[Path]]        # camera → [calib files]
    unmatched: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```

`FileAssignment` has:
- `experiments: dict[str, ExperimentAssignment]` — always populated (at minimum one `"default"` entry)
- `slp_files: list[Path]` — global, not per-experiment

### Output

Files are **moved** (not copied). Creates:
- `experiments/{name}/calibrations/{camera}/` for calibration media
- `experiments/{name}/videos/` for video files
- `experiments/{name}/points.parquet` when the `.slp` labels are imported (`import_labels`); otherwise the app starts an empty one
- `trials.csv` (global) at project root with `experiment` column
- `project.toml` and `slp/` at project root; a setup file `[skeleton]` is written to `project.toml` as an assignment (to its `cameras`, or all of them); otherwise the skeletons come from the `.slp` files

Core module: `label3d/setup_project.py` — `SetupConfig`, `ExperimentAssignment`, `FileAssignment`, `classify_files()`, `preview_assignment()`, `execute_setup()`.

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

A `CameraGroup` holds a list of `Camera` objects and is saved/loaded as `calibration.toml` (aniposelib format — see Calibration file schema below).

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
├── project.toml             # top-level config: name, charuco params (shared)
├── skeleton.json            # shared skeleton across all experiments
├── trials.csv               # global: all trials across all experiments (has experiment column)
├── slp/                     # optional: imported .slp files (global, not per-experiment)
└── experiments/
    ├── PS18/                # one subfolder per experiment
    │   ├── calibration/     # calibration media for this experiment
    │   │   ├── cam0/
    │   │   └── cam1/
    │   ├── videos/          # video files for this experiment's trials
    │   ├── calibration.toml # written after calibrating this experiment
    │   └── points.csv       # 2D + 3D labels for this experiment's trials
    ├── PS19/
    │   └── …
    └── BG03/
        └── …
```

**Key design properties:**
- `trials.csv` is **global** at the project root; it has an `experiment` column so all experiments' trials live in one file. Trial names may repeat across experiments and are disambiguated by the `experiment` value.
- Each experiment has its own `calibration.toml` and `points.csv` — per-experiment because each has a distinct calibration and independent label data.
- `.slp` files live in the top-level `slp/` folder; the adapter receives the full global `trials.csv` and resolves which `(experiment, trial, camera)` each file belongs to by matching video basenames across all rows.
- `points.csv` row index is `(experiment, trial, frame, instance, bodypart)` — the `experiment` level is constant within a single file but allows cross-experiment aggregation via plain `pd.concat`.
- The calibration for each experiment is `experiments/{name}/calibration.toml`; the geometry stack (triangulation, epipolar lines) always sees exactly one `CameraGroup` per experiment.

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
```

The `[charuco]` section maps directly to `aniposelib.boards.CharucoBoard.__init__` parameters.

### Videos and trials

`trials.csv` lives at the project root and has one row per trial across all experiments. Columns:

| experiment | trial | cam0 | cam0_start | cam0_end | cam1 | cam1_start | cam1_end | … |
|---|---|---|---|---|---|---|---|---|
| PS18 | trial1 | cam0_t1.mp4 | 0 | 500 | cam1_t1.mp4 | 3 | 503 | … |
| PS18 | trial2 | cam0_t2.mp4 | 0 | 480 | cam1_t2.mp4 | 0 | 480 | … |
| PS19 | trial1 | cam0_t1.mp4 | 0 | 600 | cam1_t1.mp4 | 0 | 600 | … |

- `_start` / `_end`: frame indices into the video file (inclusive). Blank = start (0) / end of file.
- The per-camera offset is **implicit**: global frame `N` maps to video frame `_start + N` for each camera.
- **Trial length** = `min(_end_i - _start_i + 1)` across all cameras. Blank `_end` resolved via `cv2.CAP_PROP_FRAME_COUNT`.
- Camera ID columns must match the `name` field values in `calibration.toml`.
- Video filenames are resolved relative to `experiments/{experiment}/videos/`.

### Primary storage: `points.csv`

One file per experiment, storing both 2D and 3D data. Rows indexed by `(experiment, trial, frame, instance, bodypart)`. Columns grouped by **view**: each real camera has `(x, y, xr, yr, state)` coords; `"3d"` pseudo-camera has `(x, y, z)`.

```
                                          cam0                          cam1                3d
experiment trial  frame  instance  bodypart   x     y    xr   yr   state    x     y   xr  yr  state   x    y    z
PS18       trial1 0      0         head       92   202   91  203   set      88   199  87  200  set    1.2  0.4  3.1
```

Column structure is a 2-level `MultiIndex (view, coord)`:
- Real cameras: `coord` ∈ `{x, y, xr, yr, state}` — clicked coords, reprojected coords, and point state
- `3d` pseudo-camera: `coord` ∈ `{x, y, z}` — triangulated world coords

**Point state values:**

| State | Meaning |
|---|---|
| `set` | User has manually placed this point (ground truth) |
| `predicted` | Point imported from external tracker; not yet reviewed |
| `not_set` | Point not yet placed — x, y are NaN; instance is user-type |
| `occluded` | User marked as not visible — x, y are NaN, excluded from triangulation |

Written/read with:
```python
# Write
df.to_csv("points.csv")

# Read — 2-level column header, 5-level row index
df = pd.read_csv("points.csv", header=[0, 1], index_col=[0, 1, 2, 3, 4])

# Access one camera
cam0 = df["cam0"]                                        # columns: x, y, xr, yr, state

# Access 3D
xyz = df["3d"][["x", "y", "z"]]                         # world coords, shape (n_rows, 3)
```

Missing coordinates are `NaN`. Reprojected coords (`xr`, `yr`) are `NaN` until triangulation has been run.

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
# aniposelib internal; ignored by label3d

[alignment]
# Optional; written after axis alignment step
# origin: 3D point used as origin in the new coordinate system, in old world coords (cm)
origin = [ox, oy, oz]
# R_align: 3×3 rotation matrix whose columns are [x_hat, y_hat, z_hat] in OLD world space
# Transform new→old: p_old = R_align @ p_new + origin
# Transform old→new: p_new = R_align.T @ (p_old − origin)
R_align = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
```

`label3d`'s `load_cameras()` skips sections named `"metadata"` or `"alignment"`.

---

## Project Setup Utility

### Overview

Before using label3d, users organize raw video files into the expected project structure. The **setup utility** (`label3d/setup_project.py`) takes a flat folder of videos + optional `.slp` files and reorganizes them.

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
- `experiments/{name}/calibration/{camera}/` for calibration media
- `experiments/{name}/videos/` for video files
- `experiments/{name}/points.csv` placeholder (correct 5-level row index)
- `trials.csv` (global) at project root with `experiment` column
- `project.toml`, `skeleton.json`, `slp/` at project root

Core module: `label3d/setup_project.py` — `SetupConfig`, `ExperimentAssignment`, `FileAssignment`, `classify_files()`, `preview_assignment()`, `execute_setup()`.

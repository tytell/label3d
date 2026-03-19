# label3d — Design Document

## Purpose

A cross-platform desktop app for:
1. **Calibrating** multi-camera rigs using Charuco boards (via aniposelib)
2. **Labeling** 3D points by clicking corresponding points across synchronized calibrated video streams
3. **Importing / exporting** 2D pose data from external trackers (SLEAP, etc.) and producing triangulated 3D coordinates

---

## Core Concepts

### Support for multiple cameras
Requires at least two cameras, but can support any number.

### Camera Calibration (aniposelib / OpenCV style)
Calibration is handled by **aniposelib**'s `CameraGroup` and `CharucoBoard` classes.

Each `Camera` stores:
- `matrix`: 3×3 intrinsic matrix K (fx, fy, cx, cy)
- `dist`: distortion coefficients — 5-element `[k1, k2, p1, p2, k3]` for standard, 4-element for fisheye
- `rvec`: rotation vector (Rodrigues, world-to-camera)
- `tvec`: translation vector (world-to-camera)
- `size`: `(width, height)` in pixels
- `name`: camera identifier string

A `CameraGroup` holds a list of `Camera` objects and is saved/loaded as `calibration.toml` (aniposelib format — see Data Model section for full schema).

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

## Data Model

```
Project
├── calibration/             # calibration images or videos, one subfolder per camera
│   ├── cam0/
│   └── cam1/
├── videos/                  # trial videos, one subfolder or flat with trials.csv
│   └── trials.csv           # maps trial names → per-camera video filenames + frame ranges
├── calibration.toml         # written by the Calibrate step (aniposelib CameraGroup.dump)
├── skeleton.json            # node names, edges, symmetries (from SLEAP or manual definition)
└── points.csv               # 2D + 3D labels (MultiIndex CSV, see below)
```

### Videos and trials

`trials.csv` has one row per trial. Columns:

| trial | cam0 | cam0_start | cam0_end | cam1 | cam1_start | cam1_end | … |
|---|---|---|---|---|---|---|---|
| trial1 | cam0_t1.mp4 | 0 | 500 | cam1_t1.mp4 | 3 | 503 | … |

- `_start` / `_end`: frame indices into the video file (inclusive). Blank = start (0) / end of file.
- The per-camera offset is **implicit**: global frame `N` maps to video frame `_start + N` for each camera. If cam0 starts at frame 0 and cam1 starts at frame 3, cam1 is effectively offset by 3 frames — no separate offset column needed.
- **Trial length** (number of valid global frames) = `min(_end_i - _start_i + 1)` across all cameras for that trial. If `_end` is blank for a camera, its frame count is determined from the video file at load time via `cv2.VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT)`.
- Camera ID columns must match the `name` field values in `calibration.toml`.

### Primary storage: `points.csv`

A single file storing both 2D and 3D data. Rows are indexed by `(trial, frame, instance, bodypart)` — one row per bodypart per instance per frame. Columns are grouped by **view**, where each real camera has `(x, y, xr, yr, state)` coords and the reconstructed 3D is treated as a pseudo-camera with `(x, y, z)` coords.

```
                                        cam0                          cam1                3d
trial  frame  instance  bodypart        x     y    xr   yr   state    x     y   xr  yr  state   x    y    z
trial1 0      0         head            92   202   91  203   set      88   199  87  200  set    1.2  0.4  3.1
                        tail           152   158  151  159   set     148   154 149  155  set    2.1  0.8  4.2
                        spine          NaN   NaN  NaN  NaN   occluded NaN  NaN NaN  NaN  set    NaN  NaN  NaN
              1         head           NaN   NaN  NaN  NaN   predicted 91  201  90  202  predicted NaN NaN NaN
       1      0         head            93   201   92  202   set       …
trial2 0      0         head            …
```

Column structure is a 2-level `MultiIndex (view, coord)`:
- Real cameras: `coord` ∈ `{x, y, xr, yr, state}` — clicked coords, reprojected coords, and point state
- `3d` pseudo-camera: `coord` ∈ `{x, y, z}` — triangulated world coords

**Point state values** (per camera per bodypart):

| State | Meaning |
|---|---|
| `set` | User has manually placed this point (ground truth) |
| `predicted` | Point imported from external tracker; not yet reviewed |
| `not_set` | Point not yet placed — x, y are NaN; instance is user-type |
| `occluded` | User marked as not visible — x, y are NaN, excluded from triangulation |

When a user edits a `predicted` point (moves or places it), the state becomes `set`. An instance is considered a "user instance" if any camera has state `set` or `not_set` or `occluded`. It is a "predicted instance" if all cameras have state `predicted`.

Written/read with:
```python
# Write
df.to_csv("points.csv")

# Read — 2-level column header, 4-level row index
df = pd.read_csv("points.csv", header=[0, 1], index_col=[0, 1, 2, 3])
# → df.columns: MultiIndex([(cam0, x), (cam0, y), ..., (cam0, state), (3d, x), (3d, y), (3d, z)])
# → df.index:   MultiIndex([(trial1, 0, 0, head), (trial1, 0, 0, tail), ...])

# Access one camera
cam0 = df["cam0"]                         # columns: x, y, xr, yr, state
cam0_xy = cam0[["x", "y"]]               # clicked coords, shape (n_rows, 2)
cam0_set = cam0[cam0["state"] == "set"]   # only user-placed points

# Access 3D
xyz = df["3d"][["x", "y", "z"]]           # world coords, shape (n_rows, 3)

# Access one trial/frame, all instances
frame = df.loc[("trial1", 0)]             # index is now (instance, bodypart)

# Access a specific instance
inst = df.loc[("trial1", 0, 0)]          # index is now just bodypart
```

Missing coordinates are `NaN`. Reprojected coords (`xr`, `yr`) are `NaN` until triangulation has been run. The `3d` pseudo-camera has no `state` column.

### Import / Export — Plugin Architecture

Import and export are handled by **format adapters** — one class per external format, all implementing a common interface. This makes it easy to add support for DeepLabCut, DANNCE, Anipose, etc. alongside SLEAP.

```python
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

class PoseAdapter(ABC):
    """Convert between an external pose format and the canonical points DataFrame."""

    @abstractmethod
    def load(
        self,
        files: list[Path],
        trials: pd.DataFrame,       # contents of trials.csv
        cameras: list[str],         # ordered camera IDs from calibration.toml
    ) -> pd.DataFrame:
        """Read one or more external files and return a points-compatible DataFrame.

        files   — arbitrary number of format-specific files (e.g. one .slp per
                  network, which may cover one or several cameras). The adapter
                  matches each file to camera(s) by comparing the video filenames
                  embedded in the file against the video columns in trials.csv.
        trials  — the trials table, used to resolve file→camera→trial mapping.
        cameras — canonical camera list; output columns must cover these IDs.

        Returns a DataFrame with MultiIndex(trial, frame, instance, bodypart) rows
        and MultiIndex(view, coord) columns. Missing points are NaN.
        Reprojected coords (xr, yr) and 3d columns are omitted — caller fills them.
        """

    @abstractmethod
    def save(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        cameras: list[str],
        **kwargs,
    ) -> None:
        """Write clicked 2D points (x, y columns only) to the external format."""

    @classmethod
    def name(cls) -> str:
        """Short identifier shown in the UI (e.g. 'sleap', 'deeplabcut')."""
        return cls.__name__

    @classmethod
    def file_filter(cls) -> str:
        """Qt file dialog filter string, e.g. 'SLEAP files (*.slp)'."""
        return "All files (*)"
```

Adapters live in `label3d/adapters/` and are auto-discovered at startup:

```
label3d/
└── adapters/
    ├── __init__.py       # registry: scan subclasses of PoseAdapter
    ├── sleap.py          # SleapAdapter(PoseAdapter)
    └── deeplabcut.py     # DeepLabCutAdapter(PoseAdapter)  ← future
```

The UI **Import / Export** dialogs populate their format dropdown from the registry — adding a new format requires only dropping a new file in `adapters/`.

**SLEAP adapter — camera matching logic:**
1. For each `.slp` file, read the `Video.filename` values embedded in the file (these are full absolute paths)
2. Match using **basename only** (`Path(video.filename).name`) against the basename of the video paths in `trials.csv` — full paths differ between machines, only filenames are reliable
3. A single `.slp` can cover multiple cameras if they were tracked together; separate out by video reference
4. Extract `(frame_idx, node_name, x, y)` per `(trial, camera)` and populate the output DataFrame with state `predicted`

**Key invariant:** all adapters produce and consume the same `points` schema — `(trial, frame, instance, bodypart)` row index, `(view, coord)` column index, `NaN` for missing.

### Calibration file schema (TOML)

Written by aniposelib's `CameraGroup.dump()`. Section names are `cam_0`, `cam_1`, … (zero-indexed, with underscore). The `[metadata]` section is added by aniposelib and can be ignored.

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
```

Loaded via `CameraGroup.load(fname)`, which reads all sections except `metadata` in sorted order and reconstructs `Camera` objects. Camera IDs used throughout the app (in `trials.csv`, `points.csv` column headers) are taken from the `name` field, not the section name.

---

## Application Workflow

The app has three sequential top-level modes, accessible from a persistent toolbar or tab bar:

```
[ 1. Calibrate ]  [ 2. Label ]  [ 3. Export ]
```

---

## Calibration Workflow

### Overview

Uses `aniposelib.boards.CharucoBoard` for detection and `CameraGroup.calibrate_rows()` for bundle-adjusted calibration. The same multi-camera video grid used for labeling is reused here.

### Charuco board parameters

Configured in a settings panel before calibration. Maps directly to `CharucoBoard.__init__`:

| UI label | Parameter | Default |
|---|---|---|
| Squares X | `squaresX` | 6 |
| Squares Y | `squaresY` | 8 |
| Square length (mm) | `square_length` | 30 |
| Marker length (mm) | `marker_length` | 22 |
| ArUco bits | `marker_bits` | 4 |
| Dictionary size | `dict_size` | 50 |

```python
from aniposelib.boards import CharucoBoard
board = CharucoBoard(squaresX=6, squaresY=8,
                     square_length=30, marker_length=22,
                     marker_bits=4, dict_size=50)
```

### Calibration input

Calibration images or video frames are loaded **separately** from the trial videos. The project folder contains
either individual videos for each camera (stored in the calibration folder) or image folders for each camera.

```
Project
├── calibration/
│   ├── cam0/        # images or a video for cam0
│   ├── cam1/
│   └── …
├── calibration.toml # written here after calibration
```

For video input, `CharucoBoard.detect_video(vidname, skip=20)` samples every N frames. For image folders, `CharucoBoard.detect_images(images)` processes each file.

### Calibration UI

```
┌──────────────────────────────────────────────────────────────┐
│  Board: 6×8  sq=30mm  mk=22mm  4×50         [Run Calibration]│
├──────────────┬──────────────┬──────────────┬─────────────────┤
│   Camera 0   │   Camera 1   │   Camera 2   │   Camera 3      │
│              │              │              │                 │
│  [image]     │  [image]     │  [image]     │  [image]        │
│  ● detected  │  ● detected  │  ○ missed    │  ● detected     │
│  corners     │  corners     │              │  corners        │
│              │              │              │                 │
│  err: 0.8px  │  err: 1.1px  │  —           │  err: 0.7px     │
└──────────────┴──────────────┴──────────────┴─────────────────┘
│  [◀ Prev]  Frame 12 / 47  [Next ▶]   Per-camera mean error   │
│  Overall reprojection error: 0.9 px                [Save]    │
└──────────────────────────────────────────────────────────────┘
```

**Overlays on each camera panel (same `QGraphicsScene` as labeling):**
- **Green dots**: detected Charuco corners (`corners` array from `board.detect_image()`)
- **Corner IDs**: small text label next to each corner (from `ids`)
- **Red dots / projected crosses**: reprojected corners after calibration — visual diff shows error per corner
- **Per-camera error badge**: mean reprojection error in pixels shown on panel

### Calibration pipeline (background thread)

```python
from aniposelib.cameras import Camera, CameraGroup

# 1. Detect corners in all frames for all cameras
#    calibrate_rows expects a list aligned with camera_group.cameras
all_rows = []
for cam_name in cam_names:
    source = sources[cam_name]        # path to video or image folder
    rows = board.detect_video(source, skip=skip)
    # each row: {'framenum': int, 'corners': arr, 'ids': arr, ...}
    all_rows.append(rows)

# 2. Calibrate — bundle-adjusted, with intrinsics + extrinsics init
cameras = [Camera(name=n) for n in cam_names]
camera_group = CameraGroup(cameras)
camera_group.calibrate_rows(
    all_rows,          # list[list[dict]], one inner list per camera
    board,
    init_intrinsics=True,
    init_extrinsics=True,
)

# 3. Save (writes cam_0, cam_1, … sections + metadata)
camera_group.dump("calibration.toml")

# 4. Compute per-frame reprojection errors for display
#    Use camera_group.reprojection_error(p3ds, p2ds, mean=True)
```

### After calibration

- `calibration.toml` is written; the Label and Triangulate modes become available
- The calibration viewer remains accessible to inspect per-camera error distributions
- Re-running calibration overwrites `calibration.toml` (with confirmation prompt)

---

## Labeling Workflow (Step 2)

### Setup

Before labeling, the user must either:
- **Import a skeleton** from an existing `.slp` file (`sleap_io.load_file()` → `labels.skeleton`) — node names and edges are read and stored as project metadata
- **Define a skeleton** manually: add node names, draw edges, mark symmetry pairs

### Label view layout

```
+------------------------------------------------------------------+
| [ Calibrate ] [> Label ] [ Export ]   Trial: [trial1 v]          |
| Instance: [* User  o Predicted]   Bodypart: head (3 of 13)        |
| Reprojection error: 1.3 px mean            [H] Hide  [3D] Toggle |
+--------------+--------------+--------------+--------------+------+
|   Camera 0   |   Camera 1   |   Camera 2   |   Camera 3   |  3D  |
|              |              |              |              |      |
|  (video)     |  (video)     |  (video)     |  (video)     | axes |
|  * user      |  o predicted |  - epi line  |              |  *   |
|  - epi line  |  - epi line  |              |              | [cam]|
|              |              |              |              |      |
+--------------+--------------+--------------+--------------+------+
| <<  <  [Frame 042 / 500]  >  >>   [Jump: ___]   --seekbar-----  |
+------------------------------------------------------------------+
```

The **3D panel** (rightmost, toggleable) shows:
- Triangulated point positions, colored by bodypart
- Skeleton edges in 3D
- Camera frustums at their calibrated positions (toggled independently)
- World axes (X/Y/Z)

Implemented as a `pyqtgraph.opengl.GLViewWidget`. Camera frustums derived from `camera.matrix`, `camera.rvec`, `camera.tvec`, and `camera.size`.

### Menus

> Label3D
  - Quit

> File
  - New project...   # set up a new project directory
  - Open project...  # open an existing project directory
  - Export points...  # export all points to the file
  
> View
  - Cameras
    + All cameras
    ---
    + Cam1
    + Cam2
    etc
  - Layout
    + Grid  # arrange the camera views in a square pattern, as closely as possible
    + Horizontal  # arrange the camera views in a horizontal row
    + Vertical  # arrange the camera views in a vertical column

### Skeleton overlay

The **full skeleton is always drawn** on every camera panel for every frame, regardless of which bodypart is active:
- **Edges** (bones): lines between connected nodes using the skeleton topology
- **Nodes** (points): circles at each labeled position
- **Node name labels**: text drawn beside each node, always visible (toggled off with `Ctrl+Tab`)
- Unlabeled nodes are shown as a small dim placeholder at the expected position if triangulated 3D exists (reprojected); otherwise omitted

### Point coloring

Points are **colored by reprojection error**, not by bodypart identity:

| Color | Meaning |
|---|---|
| Green | Low error (< 1 px) |
| Yellow | Medium error (1–3 px) |
| Red | High error (> 3 px) |
| Grey | No reprojection error available (not yet triangulated, or only one camera labeled) |
| White outline | Currently selected point |

Thresholds are configurable in project settings. Edges are drawn in a neutral color (bodypart color or grey) regardless of error.

### Point hover tooltip

Hovering over any labeled point shows a tooltip:
```
head  —  cam0
Clicked:     (342.1, 201.4)
Reprojected: (341.3, 202.0)
Error:        1.1 px
```
Implemented via `QGraphicsItem.setToolTip()` or a custom `hoverEnterEvent` on the node `QGraphicsEllipseItem`.

### Point selection and epipolar lines

**Selection** is per-bodypart, across all cameras simultaneously:
- Clicking a point in any camera panel **selects that bodypart** globally
- The selected point is highlighted (white outline) in all camera panels
- **Epipolar lines** for the selected point are drawn in all other cameras where it has been labeled — one epipolar line per labeled camera, color-coded by source camera
- Clicking empty space deselects

Epipolar lines are only shown for the **selected** bodypart, not all bodyparts simultaneously, to avoid visual clutter.

### Instance types and point state

Each point in each camera view carries a `state` field (stored in the `(cam, state)` column of `points.csv`):

| State | Render style | Meaning |
|---|---|---|
| `set` | Solid filled circle | User-placed ground truth |
| `predicted` | Open circle (ring only) | Imported from external tracker; not yet reviewed |
| `not_set` | Dim placeholder (if 3D reproject available) or absent | Instance exists but point not placed in this view |
| `occluded` | Hollow circle with X, grey | User marked as not visible; excluded from triangulation |

An instance is "user type" if any view has state `set`, `not_set`, or `occluded`. It is "predicted type" if all views have state `predicted`. Editing (moving or placing) a `predicted` point in any camera automatically promotes that camera's state to `set`.

### Keyboard shortcuts (matches SLEAP)

**Frame navigation**

| Key | Action |
|---|---|
| `→` / `←` | Next / prev frame |
| `Ctrl+→` / `Ctrl+←` | ±4 frames |
| `Ctrl+Alt+→` / `Ctrl+Alt+←` | ±100 frames |
| `Home` / `End` | First / last frame |
| `Alt+→` / `Alt+←` | Next / prev labeled frame |
| `Space` / `Shift+Space` | Next / prev suggested frame |
| `Ctrl+J` | Go to frame (dialog) |
| `J` / `K` | Jump to start / end of labeled region |

**Bodypart and instance selection**

| Key | Action |
|---|---|
| `↑` / `↓` | Cycle active bodypart up / down through skeleton node order |
| `` ` `` | Cycle to next instance |
| `1`–`9` | Select instance by index |
| `Escape` | Deselect bodypart and instance |

**Instance management**

| Key | Action |
|---|---|
| `Ctrl+I` | Add new user instance |
| `Ctrl+Backspace` | Delete selected instance |
| `Ctrl+C` / `Ctrl+V` | Copy / paste instance |
| `Ctrl+T` | Transpose instance tracks |

**Point manipulation**

| Action | How |
|---|---|
| Place / move point | Left-click on or near active bodypart in any camera panel |
| Delete point | `Backspace` while bodypart selected |
| Mark as occluded | Right-click on placed point (stores NaN, excluded from triangulation) |
| Move whole instance | `Alt`+drag any point |
| Duplicate instance | `Ctrl`+drag |
| Predicted → User | Any left-click drag on a predicted point (automatic) |

**Triangulation**

| Key | Action |
|---|---|
| `T` | Triangulate current frame (when auto-triangulate is off) |
| `Ctrl+Shift+T` | Triangulate all frames in current trial (background thread) |

**Display toggles**

| Key | Action |
|---|---|
| `H` | Toggle show all instances |
| `Ctrl+Tab` | Toggle node name labels |
| `Ctrl+Shift+Tab` | Toggle skeleton edges |
| `Ctrl+=` | Fit view to instances |

### Bodypart cycling and selection

- `↑` / `↓` arrows cycle the **selected bodypart** through skeleton node order (cycles across all instances if multiple exist)
- Clicking any placed point in any camera panel **selects** that bodypart (and instance, if more than one); no auto-advance on point placement
- Selected bodypart shows white outline across all cameras and is highlighted in the 3D view
- Selection is used to show epipolar lines and reprojection error details; no other behavior changes
- Clicking a node name in the sidebar list jumps directly to it
- The selected bodypart is shown in the toolbar; pressing `Escape` deselects

### Occlusion

Right-click a placed point → toggled as **occluded**: coordinate retained in display but stored as `NaN` in `x`/`y` (excluded from triangulation). Renders as a hollow circle with an X, colored grey.

---

## Export Workflow (Step 3)

### Export view layout

```
+----------------------------------------------------------+
| [ Calibrate ] [ Label ] [> Export ]                       |
|                                                           |
|  Format:  [SLEAP (.slp) v]                                |
|  Trial:   [All trials   v]                                |
|  Include: [x] User  [x] Predicted                         |
|                                                           |
|  Output:  /path/to/project/export/          [Browse]      |
|                                                           |
|  Summary:                                                 |
|    Trials: 3    Cameras: 4    Frames labeled: 1,240        |
|    Bodyparts: 13                                          |
|    User pts: 8,320    Predicted pts: 14,560               |
|                                                           |
|                                    [  Export  ]           |
+----------------------------------------------------------+
```

### SLEAP export (SleapAdapter.save)

For each `(trial, camera)` pair → one `.slp` file:
- `Skeleton` built from bodypart names and edges stored in project metadata
- `Video` reference set to matching file from `trials.csv`
- One `LabeledFrame` per labeled frame
- `Instance` (user) or `PredictedInstance` (predicted) per instance, filled from `(x, y)` columns; `NaN` → invisible
- Saved via `sleap_io.save_file(labels, path)`

### Iterative workflow

```
Label (manual user frames)
        |
        v
Export -> .slp -> Train SLEAP model
        |
        v
Import SLEAP predictions -> predicted instances in points.csv
        |
        v
Review / correct in Label view -> promoted to user instances
        |
        v
Export again with more user labels -> retrain
        |  (repeat)
```

---

## Key Features

### Video synchronization
- `trials.csv` stores per-camera `_start` / `_end` frame indices; global frame `N` maps to video frame `_start + N`
- Differing `_start` values across cameras naturally encode the sync offset — no separate column needed
- No playback — frame-step only

### Skeleton and point rendering
- Full skeleton (all nodes + edges + name labels) always drawn on every camera panel
- Points colored by reprojection error: green < 1 px, yellow 1–3 px, red > 3 px, grey = no data
- User instances: filled circle; Predicted instances: open circle (ring)
- Hovering a point shows tooltip: bodypart name, camera, clicked coords, reprojected coords, error in px
- Selected bodypart highlighted with white outline across all cameras simultaneously

### Epipolar line visualization
- Fundamental matrix computed from `CameraGroup` for each camera pair
- Drawn only for the **selected** bodypart — one line per camera that has a labeled point for it
- Lines clipped to image bounds, color-coded by source camera

### Triangulation
- Clicked points are undistorted on the fly (`cv2.undistortPoints`) before triangulation — raw pixels stored, undistorted used for geometry
- `CameraGroup.triangulate(points, undistort=True)` (DLT, ≥2 views with state `set` or `predicted`) → fills `3d` columns
- `CameraGroup.reprojection_error()` → fills `xr`/`yr` columns for all cameras (including `not_set` views, which get reprojected coords even without a clicked point)
- **Auto-triangulate** (toggleable, default on): triangulates the **current frame only** after any point edit; per-frame mean error shown in status bar
- When auto-triangulate is off: `T` triangulates current frame manually
- `Ctrl+Shift+T` (or triggered at export time) triangulates **all frames** in the current trial — runs in a background thread with progress indicator

### 3D view
- `pyqtgraph.opengl.GLViewWidget` panel in resizable splitter column
- Points colored by **reprojection error** (same color scale as 2D panels)
- Skeleton edges drawn as lines; camera frustums toggled independently
- Selected bodypart highlighted in 3D as well

---

## Architecture

### Pattern

The app follows a **Model–Signal–View** pattern: a central `ProjectState` object holds all mutable state and emits Qt signals on changes. Views subscribe to those signals and update themselves. No view reads from another view directly.

### Package layout

Files marked ✓ exist; unmarked are planned for later phases.

```
label3d/
├── __init__.py                  # ✓
├── __main__.py                  # ✓ CLI entry: `label3d [project_dir]`
├── app.py                       # ✓ QApplication, argument parsing, launch
├── state.py                     # ✓ ProjectState: all mutable state + Qt signals
├── project.py                   # ✓ Project folder I/O: trials.csv, calibration, paths
├── points.py                    # Phase 2: PointsStore: DataFrame wrapper, CRUD, save/load
├── skeleton.py                  # Phase 2: Skeleton: nodes, edges, symmetry pairs
├── geometry.py                  # Phase 3/5: Fundamental matrix, epipolar lines, triangulation
├── video.py                     # ✓ VideoReader: cv2.VideoCapture wrapper, frame cache
├── widgets/
│   ├── __init__.py              # ✓
│   ├── main_window.py           # ✓ QMainWindow, tab bar, menus, top toolbar
│   ├── camera_panel.py          # ✓ QGraphicsView/Scene for one camera view
│   ├── camera_grid.py           # ✓ Nested QSplitter grid of CameraPanels
│   ├── frame_controls.py        # ✓ Bottom bar: navigation, seekbar, frame counter
│   ├── point_item.py            # Phase 3: QGraphicsEllipseItem subclass: one labeled point
│   ├── skeleton_overlay.py      # Phase 3: Manages all point_items + edges for one camera
│   ├── epipolar_overlay.py      # Phase 3: Epipolar line items for one camera
│   ├── view_3d.py               # Phase 5: pyqtgraph GLViewWidget wrapper
│   ├── bodypart_list.py         # Phase 3: Sidebar: bodypart names, click to select
│   ├── calibration_view.py      # Phase 4: Calibration mode: settings panel + camera grid
│   ├── label_view.py            # Phase 3: Labeling mode: grid + controls + sidebar
│   └── export_view.py           # Phase 6: Export mode: form + summary
├── adapters/
│   ├── __init__.py              # Phase 2: Registry: discover PoseAdapter subclasses
│   ├── base.py                  # Phase 2: PoseAdapter ABC
│   └── sleap_adapter.py         # Phase 2: SleapAdapter
└── workers/
    ├── __init__.py
    ├── calibration_worker.py    # Phase 4: QThread: detect + calibrate + dump
    └── triangulation_worker.py  # Phase 5: QThread: batch triangulate all frames
```

---

## Implementation Plan

### Phase 1 — Package scaffold and video display ✓

**Goal**: `uv run label3d /path/to/project` opens a window showing synchronized video frames in a grid, navigable with arrow keys.

#### 1.1 Package setup
- [x] `pyproject.toml` with dependencies: `PySide6`, `opencv-python-headless`, `numpy`, `pandas`, `sleap-io`, `toml`
- [x] `__main__.py`: parse CLI args (project dir), call `app.run()`
- [x] `app.py`: create `QApplication`, instantiate `MainWindow`, enter event loop
- [x] Verify: `uv run label3d test_project` launches and displays video

#### 1.2 Project model (`project.py`)
- [x] `Project` dataclass: `root: Path`, `cameras: list[str]`, `trials: list[TrialInfo]`, `camera_info: dict[str, CameraInfo]`, `has_calibration: bool`
- [x] `Project.load(path)`:
  - Read `trials.csv` → extract camera list from column names (strip `_start`/`_end` suffixes)
  - Compute trial lengths: `min(_end - _start + 1)` across cameras, resolving blank `_end` via `cv2.CAP_PROP_FRAME_COUNT`
  - Load `calibration.toml` if present → extract `CameraInfo` (name, size) per camera
  - Loading `points.csv` and full `CameraGroup` deferred to Phase 2
- [x] `Project.video_path(trial, camera)` → resolved absolute path
- [x] `Project.video_start_frame(trial, camera)` → start frame offset

#### 1.3 Video reader (`video.py`)
- [x] `VideoReader` class wrapping `cv2.VideoCapture`:
  - `__init__(path: Path)` — opens the file, reads total frame count
  - `read_frame(n: int) -> np.ndarray | None` — seeks to frame n, returns BGR array
  - LRU cache of last 8 frames per reader (handles ±1 stepping efficiently without re-seeking)
  - `close()` — releases the capture
- [x] `FrameProvider` holds a `dict[str, VideoReader]` (one per camera for current trial):
  - `get_frame(camera, global_frame) -> np.ndarray | None` — maps global frame → video frame via `_start` offset
  - `open_trial(camera_paths, offsets)` — closes old readers, opens new ones

#### 1.4 Camera panel (`widgets/camera_panel.py`)
- [x] `CameraPanel(QGraphicsView)`:
  - Creates its own `QGraphicsScene`
  - `QGraphicsPixmapItem` as background (the video frame)
  - `set_frame(frame: np.ndarray)` — takes BGR numpy array, converts to QPixmap internally
  - Zoom via `wheelEvent` (`self.scale()`); pan via `ScrollHandDrag`
  - `fitInView()` on initial load / double-click to reset
  - Camera name label in top-left corner (`ItemIgnoresTransformations`)
  - `save_view_state()` / `restore_view_state()` — preserve zoom/pan across view toggling

#### 1.5 Camera grid (`widgets/camera_grid.py`)
- [x] `CameraGrid(QWidget)`:
  - Creates N `CameraPanel` widgets arranged in **nested `QSplitter`s** (draggable dividers)
  - Three layout modes: `LayoutMode.GRID` / `HORIZONTAL` / `VERTICAL`
  - Single-camera solo view toggle: `show_single_camera(cam | None)` with state save/restore
  - `update_frame(global_frame: int)` — for each camera, get frame from `FrameProvider`, pass to panel
  - Splitter proportions preserved when switching layout modes

#### 1.6 Main window and frame controls (`widgets/main_window.py`, `widgets/frame_controls.py`)
- [x] `MainWindow(QMainWindow)`:
  - `QTabBar` at top: Calibrate / Label / Export (Label selected by default)
  - Central `QVBoxLayout`: top bar (tabs + view/trial combos) → camera grid → frame controls
  - Menu bar: Label3D (Quit), File (New/Open/Export — disabled stubs), View (Cameras, Layout)
  - View > Cameras: `QActionGroup` radio buttons synced with combo box
  - View > Layout: Grid / Horizontal / Vertical radio buttons
  - Status bar at bottom
- [x] `FrameControls(QWidget)`:
  - Buttons: start, -4, -1, +1, +4, end
  - `QSlider` seekbar
  - Frame counter label (current / max)
  - Keyboard shortcuts: `←`/`→`, `Ctrl+←`/`→`, `Ctrl+Alt+←`/`→`, `Home`/`End`

#### 1.7 Application state (`state.py`)
- [x] `ProjectState(QObject)`:
  - Properties with signals: `current_trial`, `current_frame`, `max_frame`, `selected_instance`, `selected_bodypart`, `auto_triangulate`
  - `frame_changed = Signal(int)` — emitted on any frame navigation
  - `trial_changed = Signal(str)` — emitted on trial switch
  - `selection_changed = Signal(int, str)` — (instance, bodypart) or (-1, "") for deselect
  - `points_changed = Signal(str, int, int, str)` — (trial, frame, instance, bodypart) when a point is edited
  - Frame navigation helpers: `step_frame(delta)`, `go_to_start()`, `go_to_end()`
- [x] All widgets connect to these signals; state object is passed to every widget that needs it

**Milestone**: Can open a project, see video frames in a grid, step through frames with keyboard. No overlays, no editing.

---

### Phase 2 — Data layer: skeleton, points, adapters

**Goal**: Solid data model that can be created, queried, mutated, and persisted. SLEAP import works.

**Integration with Phase 1**: `Project.load()` gains `skeleton` and `points` attributes loaded from `skeleton.json` and `points.csv` when present. `ProjectState` gains a `points_store: PointsStore` reference. The existing `MainWindow` and `CameraPanel` are not modified — Phase 2 is pure data layer.

#### 2.1 Skeleton (`skeleton.py`)
- [ ] `Skeleton` dataclass:
  - `nodes: list[str]` — ordered bodypart names (defines cycling order for `↑`/`↓` in Phase 3)
  - `edges: list[tuple[str, str]]` — (source, target) pairs
  - `symmetries: list[tuple[str, str]]` — left/right pairs (stored, not yet used)
- [ ] `Skeleton.from_sleap(slp_path: Path) -> Skeleton` — load from `.slp` file via `sleap_io`
- [ ] `Skeleton.from_json(path: Path) -> Skeleton` / `to_json(path: Path)` — load/save `skeleton.json`
- [ ] Stored as `skeleton.json` in the project folder

#### 2.2 Points store (`points.py`)
- [ ] `PointsStore` class wrapping `pd.DataFrame`:
  - `create_empty(cameras: list[str], bodyparts: list[str]) -> PointsStore`
  - `from_csv(path: Path) -> PointsStore` / `to_csv(path: Path)`
- [ ] **Point-level API**:
  - `get_point(trial, frame, instance, bodypart, camera) -> (x, y, xr, yr, state)`
  - `set_point(trial, frame, instance, bodypart, camera, x, y, state)` — upserts; creates row if needed
  - `set_occluded(trial, frame, instance, bodypart, camera)` — state→`occluded`, x/y→NaN
  - `delete_point(trial, frame, instance, bodypart, camera)` — state→`not_set`, x/y→NaN
- [ ] **Frame-level API**:
  - `get_frame_data(trial, frame) -> pd.DataFrame` — slice for one frame, all instances
  - `get_instances(trial, frame) -> list[int]` — instance indices present in this frame
- [ ] **Instance-level API**:
  - `add_instance(trial, frame) -> int` — returns new instance index (`max existing + 1`)
  - `delete_instance(trial, frame, instance)` — drops all rows for that instance
  - `copy_instance(trial, src_frame, src_instance, dst_frame) -> int`
- [ ] **3D API**:
  - `set_3d(trial, frame, instance, bodypart, x, y, z)`
  - `set_reprojected(trial, frame, instance, bodypart, camera, xr, yr)`
- [ ] **Dirty tracking**: flag for unsaved changes; auto-save timer (every 30s if dirty)
- [ ] Emit `QObject` signal on mutations (for UI refresh)

**Performance note**: The DataFrame is the single source of truth. For frame-level access during labeling, cache the current frame slice as a dict-of-dicts for O(1) lookups. Rebuild cache on frame change. Use `df.at[]` for point-level updates rather than building new DataFrames.

#### 2.3 Adapter framework (`adapters/`)
- [ ] `base.py`: `PoseAdapter` ABC (with updated `(trial, frame, instance, bodypart)` index)
- [ ] `__init__.py`: `get_adapters() -> dict[str, PoseAdapter]` — discover subclasses
- [ ] `sleap_adapter.py` — `SleapAdapter`:
  - **load**:
    1. For each `.slp` file: `sleap_io.load_file(path)` → `Labels`
    2. For each `Video` in labels: match `Path(video.filename).name` against `trials.csv` video basenames → find `(trial, camera)`
    3. For each `LabeledFrame`: iterate instances; `instance.numpy()` → `(num_nodes, 2)` with NaN for missing
    4. Map node names by `labels.skeleton.node_names` index
    5. Set state = `predicted` for `PredictedInstance`, `set` for `Instance`
    6. Build DataFrame with `(trial, frame, instance, bodypart)` index
  - **save**:
    1. Group points by `(trial, camera)`
    2. For each group: build `sleap_io.Labels` with `Skeleton`, `Video`, `LabeledFrame`s
    3. Instance type from per-camera states: any `set`/`occluded` → `Instance`; all `predicted` → `PredictedInstance`
    4. `NaN` coords → `visible=False` in SLEAP point
    5. `sleap_io.save_file(labels, path)`

**Milestone**: Can load a SLEAP project, see imported points in memory, save to `points.csv`, reload. No visualization yet.

---

### Phase 3 — Labeling: overlays, interaction, selection

**Goal**: Full labeling UI — see skeleton, click to place/move points, select bodyparts, see epipolar lines.

**Integration with Phase 1**: `CameraPanel.set_frame()` continues to set the background; skeleton overlays are added to the same `QGraphicsScene`. Mouse events in `CameraPanel` are extended — the existing `ScrollHandDrag` mode is used when no bodypart is selected; point interaction takes priority otherwise. The `CameraGrid.update_frame()` call chain is extended to also update overlays. `ProjectState.selection_changed` and `points_changed` signals (already defined) now drive overlay updates.

#### 3.1 Point item (`widgets/point_item.py`)
- [ ] `PointItem(QGraphicsEllipseItem)`:
  - Properties: `bodypart: str`, `instance: int`, `state: str`, `error: float | None`
  - Render by state: filled circle (`set`), ring (`predicted`), dim dot (`not_set` with reproject), hollow-X (`occluded`)
  - Color from reprojection error (green/yellow/red/grey)
  - White outline when selected
  - `hoverEnterEvent` → tooltip (bodypart, camera, coords, error)
  - Draggable: `ItemIsMovable` for `set`/`predicted` states; on drag end → update `PointsStore`
  - Size: ~8px radius at 1:1 zoom; constant screen-size via `ItemIgnoresTransformations`

#### 3.2 Skeleton overlay (`widgets/skeleton_overlay.py`)
- [ ] `SkeletonOverlay` manages `PointItem`s + `QGraphicsLineItem` edges for one camera panel:
  - `update(frame_data, skeleton, camera)`:
    - For each (instance, bodypart): create/update/hide `PointItem`
    - For each skeleton edge: draw line between the two node positions (if both exist)
    - Node name labels: `QGraphicsSimpleTextItem` beside each node
  - Item pool: reuse existing `QGraphicsItem`s rather than creating/destroying each frame
  - Instance visibility: `H` toggle shows/hides non-selected instances

#### 3.3 Mouse interaction in CameraPanel
- [ ] `mousePressEvent`:
  - Left-click near existing point (within ~15px threshold) → select that (instance, bodypart)
  - Left-click on empty space with bodypart selected → place point at click → `set_point(state=set)`
  - Left-click on empty space with nothing selected → deselect
  - Right-click on existing point → toggle `occluded` ↔ `set`
- [ ] `mouseMoveEvent` (while dragging): update point position in real-time (visual only)
- [ ] `mouseReleaseEvent`: commit new position to `PointsStore`; if was `predicted` → promote to `set`
- [ ] Hit testing: `scene.items(click_pos)` filtered to `PointItem` instances

#### 3.4 Selection model (in `state.py`)
- [ ] `selection_changed` signal carries `(instance: int, bodypart: str)`
- [ ] `↑`/`↓` cycle `bodypart` through `skeleton.nodes` (wrapping); if multiple instances, cycle through all (instance, bodypart) pairs in order
- [ ] `` ` `` cycles to next instance (same bodypart)
- [ ] `1`–`9` select instance by index
- [ ] `Escape` clears selection
- [ ] On selection change: all CameraPanels update overlay (white outline, epipolar lines)

#### 3.5 Epipolar lines (`geometry.py`, `widgets/epipolar_overlay.py`)
- [ ] `compute_fundamental_matrix(cam_a: Camera, cam_b: Camera) -> np.ndarray`:
  - From K, R, t for both cameras, compute 3×3 fundamental matrix F
  - `E = [t₂ - R₂·R₁ᵀ·t₁]× · R₂·R₁ᵀ`; `F = K₂⁻ᵀ · E · K₁⁻¹`
  - Cache per camera pair (recompute only on calibration change)
- [ ] `compute_epipolar_line(F, point_2d) -> (pt1, pt2)`:
  - `l = F @ [x, y, 1]` → line coefficients; clip to image bounds → two endpoints
- [ ] `EpipolarOverlay`:
  - For selected bodypart: in each camera with a `set`/`predicted` point, compute epipolar line in every *other* camera
  - `QGraphicsLineItem`, colored by source camera (consistent palette)
  - Uses undistorted point coordinates for accurate geometry
  - Updates on selection change or point edit

#### 3.6 Keyboard shortcuts (in `label_view.py`)
- [ ] Register `QShortcut`s for all keys from the keyboard shortcut tables
- [ ] Frame navigation → update `ProjectState.current_frame`
- [ ] Selection → update `ProjectState.selected_bodypart` / `selected_instance`
- [ ] Instance management: `Ctrl+I` → `add_instance()`, `Ctrl+Backspace` → `delete_instance()`
- [ ] Point manipulation: `Backspace` → `delete_point()`, etc.

#### 3.7 Bodypart list sidebar (`widgets/bodypart_list.py`)
- [ ] `QListWidget` showing skeleton node names
- [ ] Current selection highlighted; click to select bodypart
- [ ] Show per-bodypart status (how many cameras have data)

**Milestone**: Full labeling workflow — see skeleton on all cameras, click to place/move points, select bodyparts with keyboard/mouse, see error coloring and epipolar lines.

---

### Phase 4 — Calibration

**Goal**: Load calibration media, detect corners, run bundle adjustment, visualize results.

#### 4.1 Calibration view (`widgets/calibration_view.py`)
- [ ] Top panel: board settings form (spinboxes for Charuco params)
- [ ] Reuse `CameraGrid` to display calibration images/video frames
- [ ] "Detect Corners" button → launches `CalibrationWorker`
- [ ] "Run Calibration" button (enabled after detection)
- [ ] "Save" button → `CameraGroup.dump()`
- [ ] Per-camera error badge overlay; overall error in status bar

#### 4.2 Calibration worker (`workers/calibration_worker.py`)
- [ ] `CalibrationWorker(QThread)`:
  - Step 1: detect corners per camera (emit per-camera progress)
  - Step 2: `CameraGroup.calibrate_rows()` (emit progress)
  - Step 3: compute per-frame reprojection errors
  - Emit results: detected corners, calibrated `CameraGroup`, error arrays
- [ ] On completion: overlay detected corners (green dots), reprojected corners (red crosses), error vectors

#### 4.3 Calibration input discovery
- [ ] Scan `calibration/cam*/` subfolders
- [ ] If subfolder contains video files (`.mp4`/`.avi`) → `board.detect_video(path, skip=N)`
- [ ] If subfolder contains images (`.png`/`.jpg`) → `board.detect_images(paths)`
- [ ] Camera names derived from subfolder names; validated against `trials.csv`

---

### Phase 5 — Triangulation and 3D view

**Goal**: Real-time triangulation of current frame, batch triangulation, 3D visualization.

#### 5.1 Triangulation engine (`geometry.py`)
- [ ] `triangulate_frame(camera_group, points_store, trial, frame, instance)`:
  - For each bodypart: collect 2D points from cameras where state ∈ {`set`, `predicted`}
  - Undistort on the fly via `camera.undistort_points()`
  - If ≥2 cameras: `camera_group.triangulate()` → 3D point
  - Reproject into all cameras → fill `xr`, `yr`; compute per-camera error
  - Store results via `PointsStore.set_3d()` and `set_reprojected()`
  - Return per-bodypart error dict for status bar display
- [ ] Auto-triangulate toggle in toolbar (checkbox, default on)
- [ ] `triangulate_frame_auto()`: called after any point edit if toggle is on
- [ ] `T` shortcut: triangulate current frame manually

#### 5.2 Batch triangulation (`workers/triangulation_worker.py`)
- [ ] `TriangulationWorker(QThread)`:
  - Iterates all labeled frames in current trial (or all trials)
  - Calls `triangulate_frame()` for each
  - Emits progress (frame N of M)
  - Triggered by `Ctrl+Shift+T` or "Triangulate All" button
  - Auto-triggered before export if checkbox is set

#### 5.3 3D view (`widgets/view_3d.py`)
- [ ] `View3D(QWidget)` wrapping `pyqtgraph.opengl.GLViewWidget`:
  - `GLScatterPlotItem` for bodypart points (colored by reprojection error)
  - `GLLinePlotItem` for skeleton edges
  - Camera frustums: wireframe pyramids from K, R, t, image_size; toggled with checkbox
  - World axes: three colored lines at origin
  - `update_frame(frame_data_3d)`: rebuild scatter/line data from current frame's `3d` columns
  - Selected bodypart rendered larger / brighter
- [ ] Placed in a `QSplitter` alongside the camera grid; show/hide toggle
- [ ] Falls back gracefully if OpenGL is unavailable (disable button, show message)

---

### Phase 6 — Export

**Goal**: Export labeled data back to external formats.

#### 6.1 Export view (`widgets/export_view.py`)
- [ ] Format dropdown (populated from adapter registry)
- [ ] Trial selector (all / specific)
- [ ] Instance type checkboxes (user / predicted)
- [ ] Output directory browser
- [ ] Summary: trial count, camera count, labeled frames, point counts by state
- [ ] "Triangulate All Before Export" checkbox (default on)
- [ ] "Export" button → batch triangulation if needed → `adapter.save()`
- [ ] Progress bar during export

---

### Phase 7 — Polish and robustness

- [ ] **Undo/redo**: Command pattern — each point edit creates an `UndoCommand(trial, frame, instance, bodypart, camera, old_value, new_value)` on `QUndoStack`; `Ctrl+Z`/`Ctrl+Shift+Z`
- [ ] **Auto-save**: `PointsStore` writes `points.csv` every 30s if dirty + on app close; backup to `.points.csv.bak` before overwriting
- [ ] **Seekbar coloring**: color each frame tick by worst reprojection error in that frame (green/yellow/red/grey)
- [ ] **Suggested frames**: uniformly spaced unlabeled frames shown as ticks on seekbar; `Space`/`Shift+Space` navigate to them
- [ ] **Progress indicators**: "42% of frames labeled" per trial
- [ ] **Frame pre-caching**: on step to frame N, pre-decode N+1 through N+3 in a background thread for all cameras

---

## Build order (dependency graph)

```
1.1 pyproject.toml + entry point
 │
 ▼
1.2 project.py (trials.csv loader)
 │
 ├──────────────────────┐
 ▼                      ▼
1.3 video.py            1.7 state.py (ProjectState)
 │                      │
 └──────────┬───────────┘
            ▼
1.4–1.6 camera_panel → camera_grid → main_window + frame_controls
            │
            │  ★ MILESTONE: video grid with frame stepping works
            ▼
2.1 skeleton.py  +  2.2 points.py (PointsStore)
            │
            ▼
2.3 adapters/ (SLEAP import)
            │
            │  ★ MILESTONE: can import SLEAP data into points.csv
            ▼
3.1–3.2 point_item + skeleton_overlay (rendering)
            │
            ▼
3.3 mouse interaction (click to place/move)
            │
            ▼
3.4 selection model  +  3.5 epipolar lines
            │
            │  ★ MILESTONE: full labeling workflow
            ▼
5.1 triangulation engine → error coloring → tooltips
            │
            ├──────────────────────┐
            ▼                      ▼
4.1–4.3 calibration view     5.2–5.3 batch triangulation + 3D view
            │                      │
            └──────────┬───────────┘
                       ▼
              6.1 export view
                       │
                       │  ★ MILESTONE: complete app
                       ▼
                  7.* polish
```

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| QGraphicsScene rebuild slow with many items | Laggy frame stepping | Pool QGraphicsItems; update positions instead of recreating. Profile early with 20+ bodyparts. |
| cv2.VideoCapture random seek slow for compressed video | 100ms+ delay jumping frames | LRU frame cache (8 frames); pre-decode ahead; sequential reads for ±1 stepping |
| pandas MultiIndex CRUD slow for point-level edits | Laggy click-to-place | Cache current frame as dict-of-dicts; batch-write to DataFrame on frame change |
| `state` column (string) alongside numeric columns | Dtype confusion, slow filtering | Categorical dtype for state column; numeric columns stay float64 (separate MultiIndex columns) |
| pyqtgraph OpenGL unavailable on some systems | 3D view crashes | Make 3D panel optional; catch import error, disable button, show message |
| aniposelib API instability (research code) | Calibration breaks with updates | Pin version in pyproject.toml; wrap all calls in `calibration.py` shim layer |
| Large projects (many trials × long videos) | Memory pressure | Lazy loading: only open VideoCapture for current trial; only load frame slice from DataFrame |
| Instance index gaps after deletion | Confusing numbering | Allow gaps — instance index is an opaque ID, not a sequential counter; UI shows "Instance 1, 2, …" labels |

---

## Technology Stack

| Component | Library |
|---|---|
| GUI | PySide6 |
| Video decoding | OpenCV (`cv2.VideoCapture`) |
| Calibration | aniposelib (`CameraGroup`, `CharucoBoard`) |
| Geometry | OpenCV + NumPy |
| Data I/O | pandas, sleap_io, tomllib / tomli-w, numpy |
| 3D view | pyqtgraph GL |
| Package / deps | uv (`pyproject.toml` + `uv.lock`) |
| Testing | pytest (dev dependency) |

---

## Testing

### Framework

Tests use **pytest** (`tests/` directory, run with `uv run pytest`). `pytest` is a dev dependency declared under `[dependency-groups] dev` in `pyproject.toml`.

A session-scoped `QApplication` fixture in `tests/conftest.py` is required before any `QObject`, `Signal`, or `QTimer` can be instantiated (i.e. before any `PointsStore`). All test modules that use `PointsStore` must accept the `qapp` fixture to ensure this is set up.

### Test modules

| Module | What it covers |
|---|---|
| `tests/test_skeleton.py` | `Skeleton` load/save, node navigation, special-char node names |
| `tests/test_points.py` | `PointsStore` construction, full CRUD API, cache, dirty tracking, signals |
| `tests/test_points_paths.py` | OS-level path robustness: spaces, unicode, separators, missing paths |
| `tests/test_adapters.py` | Adapter registry, `SleapAdapter` load/save, basename matching edge cases |
| `tests/test_project_loading.py` | `Project.load()` integration: skeleton/points wiring, missing files, error messages |

### Key testing conventions

- **`tmp_path` fixture** for all file I/O — tests never write outside their temp directory
- **Path robustness** is tested explicitly: spaces in directory names, spaces in file names, unicode characters in paths, Windows-style `\` vs POSIX `/` separators in video filenames stored in `trials.csv`
- **Special characters in names** (spaces, colons, parens, commas in bodypart/node/trial names) are round-tripped through JSON and CSV to verify they survive serialization
- **Error paths**: `FileNotFoundError` is expected for missing files/directories; error messages must contain enough context for the user to diagnose the problem (path or filename in message). `ValueError` is expected when `to_csv()` is called with no path set
- **sleap_io dependency**: `SleapAdapter` tests use `pytest.importorskip("sleap_io")` — they are skipped if `sleap_io` is not installed rather than failing
- **Cache invalidation**: frame cache state is inspected directly via `_cache_key` to verify that mutations correctly invalidate it
- **Signal emission**: `changed` signal callbacks are counted via a list-append lambda to confirm mutations emit exactly one signal

### Running tests

```
uv run pytest               # all tests
uv run pytest -v            # verbose
uv run pytest tests/test_skeleton.py   # single module
uv run pytest -k "spaces"  # tests whose name contains "spaces"
```

### Synthetic data generator

The synthetic data generator (`generate_synthetic.py`) creates a complete test project with known ground truth: cameras, videos with rendered skeleton keypoints, `calibration.toml`, `points.csv`, `skeleton.json`, and `.slp` files. Configuration lives in `synthetic_config.toml`.

#### Module structure

As the generator grows, its functions are organized into a `synthetic/` package:

```
synthetic/
├── __init__.py              # re-exports main entry point
├── cameras.py               # look_at(), compute_intrinsics(), build_cameras()
├── motion.py                # rotation matrices, compute_body_poses(), compute_world_points()
├── projection.py            # project_points(), project_all_frames()
├── rendering.py             # render_frame(), write_videos()
├── writers.py               # write_calibration_toml(), write_trials_csv(), write_points_csv(),
│                            #   write_skeleton_json(), write_slp_files()
├── checks.py                # _dlt_triangulate(), _epipolar_error(), run_sanity_checks()
└── charuco.py               # generate_charuco_images() — synthetic calibration board images
```

`generate_synthetic.py` becomes a thin wrapper that imports from `synthetic` and calls `main()`.

#### Synthetic calibration images

The generator can produce Charuco board images for each camera, enabling end-to-end testing of the calibration pipeline (detect corners → bundle adjustment → `calibration.toml`).

**Config additions** (`synthetic_config.toml`):

```toml
[charuco]
squares_x = 6               # board squares along X
squares_y = 6               # board squares along Y
square_length = 24.33          # mm — side length of each chessboard square
marker_length = 17.0          # mm — side length of ArUco markers within squares
marker_bits = 5              # ArUco marker bit depth (4, 5, 6, or 7)
dict_size = 50               # ArUco dictionary size (50, 100, 250, 1000)
num_images = 12              # number of board poses to render per camera
board_image_resolution = 800 # px — resolution of the generated flat board image
background_color = [40, 40, 40]  # RGB background for calibration images

# Board placement region: the board is placed at random poses within this volume
[charuco.placement]
center = [0.0, 0.0, 0.0]    # center of the placement volume (world coords, cm)
extent = [15.0, 15.0, 10.0] # half-extents of the placement volume (cm)
tilt_range_deg = 30.0        # max tilt from facing-camera in any axis (degrees)
seed = 348294                    # RNG seed for reproducible board poses
```

**How it works**:

1. **Generate flat board image** — `cv2.aruco.CharucoBoard.generateImage()` produces a high-resolution image of the board
2. **Compute board poses** — For each of `num_images` poses: pick a random position within the placement volume and a random orientation (tilted up to `tilt_range_deg` from the mean camera viewing direction). Uses a seeded RNG for reproducibility
3. **Render into each camera** — For each (pose, camera) pair:
   - For every pixel in the output image, undistort to get the ideal ray direction, then intersect with the board plane (ray-plane intersection). This reverse-mapping approach correctly renders barrel distortion — board edges appear curved as they would in a real distorted image
   - Board texture lookup uses `cv2.remap` with bilinear interpolation for clean marker edges
   - **Horizontal flip**: the board image from `generateImage()` is drawn from the viewer's perspective, but the board's local X axis points opposite to the camera's view direction. The U coordinate is mirrored (`1 - u`) during texture mapping so that ArUco markers read correctly when viewed by the camera
   - Composite onto the background
4. **Write output** — Save images as `calibration/{cam_name}/board_{NN:03d}.png`

**Output structure**:

```
test_project/
├── calibration/
│   ├── cam1/
│   │   ├── board_000.png
│   │   ├── board_001.png
│   │   └── ...
│   ├── cam2/
│   │   └── ...
│   └── cam3/
│       └── ...
├── calibration.toml          # ground-truth (existing, from camera params)
└── ...                       # other existing outputs
```

**Verification** — After generating images, an optional sanity check runs the full aniposelib calibration pipeline on the generated images and compares the recovered camera parameters against the known ground truth:

1. Construct `CharucoBoard` with the same config params
2. `board.detect_image()` on each generated image per camera
3. Build `CameraGroup`, call `calibrate_rows()` with detected corners
4. Compare recovered K, R, t against ground-truth — report max parameter error
5. This validates both the image generator and the calibration pipeline

**Usage**:

```
uv run python generate_synthetic.py synthetic_config.toml            # all outputs (including charuco if [charuco] section present)
uv run python generate_synthetic.py synthetic_config.toml --no-charuco  # skip calibration image generation
```

The `[charuco]` section is optional — if absent, calibration images are not generated (backwards compatible with existing configs).

---

## Decisions

1. **Synchronization**: Common recording start (all cameras in sync), or implicit per-camera offset via the `_start` frame values in `trials.csv` — global frame `N` maps to video frame `_start + N` for each camera. No separate offset column; no timecode support. Future: UI for adjusting start frames interactively.

2. **Calibration convention**: OpenCV world-to-camera convention throughout. `rvec`/`tvec` in `calibration.toml` transform world points into camera space. Aniposelib uses the same convention.

3. **Number of cameras**: Minimum 2, no maximum. UI grid layout adapts to however many cameras are in `calibration.toml`.

4. **Playback**: Frame-step only — no video playback. Controls: `←`/`→`, `Ctrl+←`/`→`, `Ctrl+Alt+←`/`→`, `Home`/`End`.

5. **Resuming work**: Yes. On open, the app loads `points.csv` if it exists (existing 2D + 3D labels). Users can also import from `.slp` files via `SleapAdapter.load()` to seed labels from a prior SLEAP session.

6. **Distribution**: Python package only, using **uv** for dependency management (`pyproject.toml` + `uv.lock`). Entry point: `label3d` CLI command that launches the GUI.

7. **Point storage**: Raw pixel coordinates stored in `points.csv`. Undistortion is applied on the fly when computing epipolar lines and triangulating — never stored. This preserves fidelity to what the user clicked and allows recalculation if calibration is updated.

---

## Prompt Template for Initial Code Generation

Use this prompt to generate the first working scaffold:

```
Build a Python/PySide6 desktop app called "label3d" for labeling 3D points
across multiple synchronized calibrated videos. Use uv package manager for
handling dependencies.

Requirements:
- Load a project folder containing: trials.csv (maps trial names → per-camera
  video filenames + frame ranges), calibration.toml (aniposelib CameraGroup
  format: [cam_0], [cam_1], … sections with keys name, size, matrix,
  distortions, rotation, translation), and optionally existing points.csv.
- Display videos in a grid layout, one QGraphicsView panel per camera, all
  showing the same global frame (mapped to video frame via _start offset).
- User can step frames with arrow keys or buttons. No playback.
- Store labels in points.csv: a pandas MultiIndex CSV with row index
  (trial, frame, instance, bodypart) and column MultiIndex (view, coord).
  Each camera has coords {x, y, xr, yr, state}; the "3d" pseudo-camera has
  {x, y, z}. state ∈ {set, predicted, not_set, occluded}. NaN = missing.
- Click to place/move points in any camera panel. Full skeleton overlay
  (nodes + edges + name labels) on every panel. Points colored by
  reprojection error (green/yellow/red/grey).
- Selected bodypart shows white outline across all cameras and epipolar
  lines in other views (fundamental matrix from calibration).
- Auto-triangulate current frame on point edit (toggleable).
  CameraGroup.triangulate() for ≥2 views with state set/predicted.
- Import from / export to external pose files (SLEAP etc.) via pluggable
  PoseAdapter subclasses in label3d/adapters/.

Architecture: Model–Signal–View. Central ProjectState (QObject) emits
signals on state changes. QGraphicsView/QGraphicsScene per camera panel
for overlays (points, epipolar lines) and zoom/pan. pyqtgraph GL for 3D.

Start with Phase 1: project loader, video grid, frame stepping.
Then Phase 2: data model (skeleton, PointsStore, SLEAP adapter).
Then Phase 3: labeling overlays and interaction.
```

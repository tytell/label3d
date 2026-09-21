# label3d — Application Workflows

## Application Workflow

The app has three sequential top-level modes, accessible from a persistent toolbar or tab bar:

```
[ 1. Calibrate ]  [ 2. Label ]  [ 3. Export ]
```

---

## Calibration Workflow

### Overview

Uses `aniposelib.boards.CharucoBoard` for detection and `CameraGroup.calibrate_rows()` for bundle-adjusted calibration.

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

### Calibration input

```
experiments/{name}/
├── calibration/
│   ├── cam0/        # images or a video for cam0
│   ├── cam1/
│   └── …
├── calibration.toml # written here after calibration
```

For video input, `DetectionWorker` walks the video itself and runs `CharucoBoard.detect_image()` on exactly every Nth frame — the stride comes from the "Video frame step" spinbox in the calibration panel and persists as `frame_skip` in the `[calibration]` section of `project.toml` (default 20). Skipped frames are demuxed but never decoded. Image-based calibration is **deprecated** as of Phase 9: `CharucoBoard.detect_images(images)` is still present but guarded by an error, and a calibration folder holding images is rejected at setup and at detection. Calibration comes from videos matched into takes.

### Calibration UI

```
┌──────────────────────────────────────────────────────────────┐
│  Board: 6×8  sq=30mm  mk=22mm  4×50         [Run Calibration]│
├──────────────┬──────────────┬──────────────┬─────────────────┤
│   Camera 0   │   Camera 1   │   Camera 2   │   Camera 3      │
│  [image]     │  [image]     │  [image]     │  [image]        │
│  ● detected  │  ● detected  │  ○ missed    │  ● detected     │
│  err: 0.8px  │  err: 1.1px  │  —           │  err: 0.7px     │
└──────────────┴──────────────┴──────────────┴─────────────────┘
│  [◀ Prev]  Frame 12 / 47  [Next ▶]                           │
│  Overall reprojection error: 0.9 px                [Save]    │
└──────────────────────────────────────────────────────────────┘
```

**Overlays:** Green dots = detected corners, Red crosses = reprojected corners after calibration.

### Calibration pipeline

```python
all_rows = []
for cam_name in cam_names:
    rows = detect_rows(board, source, skip=skip)  # every Nth frame for video, all images for a folder
    all_rows.append(rows)

cameras = [Camera(name=n) for n in cam_names]
camera_group = CameraGroup(cameras)
camera_group.calibrate_rows(all_rows, board, init_intrinsics=True, init_extrinsics=True)
camera_group.dump("calibration.toml")
```

### Axis Alignment (Optional)

After calibration, the world coordinate origin and axes are defined arbitrarily by the board placement. Axis alignment lets the user re-orient to a meaningful physical reference.

#### When it applies

Separate optional step that runs **after** calibration. Result stored in `calibration.toml` as `[alignment]` section. Camera extrinsics are **not modified**.

#### Reference video source

Axis reference points are picked from **trial videos**, not from calibration images.

#### Axis Alignment UI

```
┌─────────────────────────────────────────────────────────────────┐
│  Axis Alignment                              Trial: [trial1 v]  │
│                                              Frame: [  42  ] ◀▶ │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│   Camera 0   │   Camera 1   │   Camera 2   │  Pick targets      │
│  (video)     │  (video)     │  (video)     │  [Origin]          │
│  ✦ origin    │  ✦ origin    │              │  [X near]  [X far] │
│  → x near    │  → x near    │  → x near    │  [Y near]  [Y far] │
│              │              │              │  [Z near]  [Z far] │
│              │              │              │                    │
│              │              │              │  Table:            │
│              │              │              │  Origin  ✓         │
│              │              │              │  X near  ✓         │
│              │              │              │  X far   –         │
│              │              │              │  …                 │
│              │              │              │                    │
│              │              │              │  Flip: [x] [y] [z] │
│              │              │              │  [Compute]         │
│              │              │              │  [Apply & Save]    │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

**Pick targets** (seven buttons): Origin, X near/far, Y near/far, Z near/far. At least **Origin + two axis points** from two different axes required to compute alignment.

**Flip checkboxes** — negate the corresponding axis vector before Gram-Schmidt.

#### Gram-Schmidt algorithm

1. Triangulate each clicked target via DLT.
2. Compute three raw axis vectors from triangulated points.
3. Negate any vectors whose flip checkbox is checked.
4. Rank by length: longest = primary, next = secondary, third derived.
5. Orthogonalize via Gram-Schmidt: `a_hat`, `b_hat = normalize(secondary - proj)`, `c_hat = a_hat × b_hat`
6. Re-assign `a_hat, b_hat, c_hat` back to `x_hat, y_hat, z_hat` per user's x/y/z labels.
7. Build `R_align` as `[x_hat | y_hat | z_hat]` (columns are basis vectors in old-world space).

#### Transform convention

- New → old: `p_old = R_align @ p_new + origin`
- Old → new: `p_new = R_align.T @ (p_old − origin)`

---

## Labeling Workflow (Step 2)

### Setup

Before labeling, user must either:
- **Import a skeleton** from an existing `.slp` file (`sleap_io.load_file()`)
- **Define a skeleton** manually: add node names, draw edges, mark symmetry pairs

### Label view layout

```
+------------------------------------------------------------------+
| [ Calibrate ] [> Label ] [ Export ]   Trial: [trial1 v]          |
| Instance: [* User  o Predicted]   Bodypart: head (3 of 13)        |
| Reprojection error: 1.3 px mean            [H] Hide  [3D] Toggle |
+--------------+--------------+--------------+--------------+------+
|   Camera 0   |   Camera 1   |   Camera 2   |   Camera 3   |  3D  |
|  (video)     |  (video)     |  (video)     |  (video)     | axes |
|  * user      |  o predicted |  - epi line  |              |  *   |
|  - epi line  |  - epi line  |              |              | [cam]|
+--------------+--------------+--------------+--------------+------+
| <<  <  [Frame 042 / 500]  >  >>   [Jump: ___]   --seekbar-----  |
+------------------------------------------------------------------+
```

### Menus

```
> File
  - New project...
  - Open project...
  - Import from SLP files...
  - Save                Ctrl+S
  - Export points...

> View
  - Cameras / Layout (Grid / Horizontal / Vertical)

> Instances
  > Add
    - Average / Random  Ctrl+I
    - Random            Ctrl+Shift+I
    - Reprojected       (greyed when unavailable)
  - Delete instance     Ctrl+Backspace
```

### Skeleton overlay

The **full skeleton is always drawn** on every camera panel for every frame:
- **Edges** (bones): lines between connected nodes
- **Nodes** (points): circles at each labeled position
- **Node name labels**: text beside each node (toggled with `Ctrl+Tab`)

### Point coloring

Points are **colored by reprojection error**:

| Color | Meaning |
|---|---|
| Green | Low error (< 1 px) |
| Yellow | Medium error (1–3 px) |
| Red | High error (> 3 px) |
| Grey | No reprojection error available |
| White outline | Currently selected point |

### Point hover tooltip

```
head  —  cam0
Clicked:     (342.1, 201.4)
Reprojected: (341.3, 202.0)
Error:        1.1 px
```

### Point selection and epipolar lines

- Clicking a point selects that bodypart globally (white outline in all cameras)
- **Epipolar lines** for the selected bodypart are drawn in all other cameras where it has been labeled
- Only shown for the **selected** bodypart to avoid visual clutter

### Instance types and point state

| State | Render style | Meaning |
|---|---|---|
| `set` | Solid filled circle | User-placed ground truth |
| `predicted` | Open circle (ring only) | Imported from external tracker |
| `not_set` | Dim placeholder or absent | Instance exists but point not placed |
| `occluded` | Hollow circle with X, grey | User marked as not visible |

### Keyboard shortcuts

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

**Bodypart and instance selection**

| Key | Action |
|---|---|
| `↑` / `↓` | Cycle active bodypart |
| `` ` `` | Cycle to next instance |
| `1`–`9` | Select instance by index |
| `Escape` | Deselect |

**Instance management**

| Key | Action |
|---|---|
| `Ctrl+I` | Add instance (average if possible, random otherwise) |
| `Ctrl+Shift+I` | Add instance: random unconditionally |
| `Ctrl+Backspace` | Delete selected instance |
| `Ctrl+C` / `Ctrl+V` | Copy / paste instance |

**Point manipulation**

| Action | How |
|---|---|
| Place / move point | Left-click in any camera panel |
| Delete point | `Backspace` while bodypart selected |
| Mark as occluded | Right-click on placed point |
| Move whole instance | `Alt`+drag any point |

**Triangulation**

| Key | Action |
|---|---|
| `T` | Triangulate current frame |
| `Ctrl+Shift+T` | Triangulate all frames in current trial |

**Display toggles**

| Key | Action |
|---|---|
| `H` | Toggle show all instances |
| `Ctrl+Tab` | Toggle node name labels |
| `Ctrl+Shift+Tab` | Toggle skeleton edges |

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
|  Output:  /path/to/project/export/          [Browse]      |
|                                                           |
|  Summary: Trials: 3  Cameras: 4  Frames: 1,240            |
|                                    [  Export  ]           |
+----------------------------------------------------------+
```

### SLEAP export (SleapAdapter.save)

For each `(trial, camera)` pair → one `.slp` file:
- `Skeleton` built from bodypart names and edges
- `Video` reference set to matching file from `trials.csv`
- One `LabeledFrame` per labeled frame
- `Instance` (user) or `PredictedInstance` (predicted) per instance; `NaN` → invisible
- Saved via `sleap_io.save_file(labels, path)`

### Iterative workflow

```
Label (manual user frames) → Export → .slp → Train SLEAP model
  ↓
Import SLEAP predictions → predicted instances in points.csv
  ↓
Review / correct in Label view → promoted to user instances
  ↓
Export again → retrain  (repeat)
```

---

## Key Features

### Video synchronization
- `trials.csv` stores per-camera `_start` / `_end` frame indices; global frame `N` maps to video frame `_start + N`
- No playback — frame-step only

### Triangulation
- Clicked points undistorted on the fly (`cv2.undistortPoints`) before triangulation — raw pixels stored, undistorted used for geometry
- `CameraGroup.triangulate(points, undistort=True)` (DLT, ≥2 views with state `set` or `predicted`)
- **Auto-triangulate** (default on): triangulates current frame after any point edit
- `Ctrl+Shift+T` triangulates all frames in background thread

### 3D view
- `pyqtgraph.opengl.GLViewWidget` panel in resizable splitter column
- Points colored by reprojection error; skeleton edges; camera frustums (toggled independently)
- Falls back gracefully if OpenGL unavailable

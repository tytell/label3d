# label3d — Application Workflows

## Application Workflow

The app has two top-level modes in the tab bar. Exporting is **File > Export points...** (see *Export workflow*), not a tab.

```
[ 1. Calibrate ]  [ 2. Label ]
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
├── calibrations/
│   ├── cam0/        # take videos and corners.csv for cam0
│   ├── cam1/
│   ├── …
│   └── *.toml       # registry: every solution saved here (see below)
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

### Saved calibrations (registry)

Every solution is kept in the experiment's calibration registry (`experiments/{exp}/calibrations/`; format in [data_model.md](data_model.md#calibration-outputs)).

- **Run Calibration** saves one `charuco` entry per take (`charuco-{take}`) and one for the joint solution (`charuco-all`), and makes the joint one active. If the active calibration is a refined one, it asks first. The refined entry is kept, so you can switch back to it.
- **Use selected take** makes that take's entry active. It doesn't write a file.
- **Detect Corners** writes each camera's `corners.csv`, replacing the rows of the takes it detected.

The **Saved calibrations** list shows every entry (label, date, method, error), with refinements under the calibration they came from. The active entry is bold, and the tooltip shows its provenance.

| Button | Effect |
|---|---|
| **Make active** | points the experiment at this entry |
| **Rename…** | changes the label; the id and file name stay the same |
| **Delete…** | deletes the entry and every entry derived from it, after a confirmation that lists them. The active entry, and anything the active entry derives from, can't be deleted |

Making a calibration active here doesn't re-triangulate. The Label tab then shows the stale-3D warning (see *Switching calibrations*).

### Refining with labelled points

**Refine with points…** under the registry list adjusts a calibration of the active experiment using bodypart points labelled or predicted in several cameras. It runs aniposelib's `bundle_adjust_iter` on those points, starting from a charuco solution.

| Control | Default | Meaning |
|---|---|---|
| **Calibrate charuco takes** | all takes | calibrate the checked takes jointly first (Stage 1). If a charuco entry already used exactly these takes and the same corners and ranges, **Reuse …** starts from it instead |
| **Existing calibration** | the active entry | start from a saved `charuco` or `imported` entry |
| **Sources** | label3d and SLEAP user labels | which points count. label3d reprojection guesses never count |
| **Prediction score ≥** | 0.8 | for SLEAP predictions only |
| **Trials** | all | trials whose points are used |
| **Extrinsics only** | on | keep each camera's intrinsics |
| **Hold-out fraction** | 0.2 | frames kept out of the fit, to measure the error |
| **Repeats** | 3 | bundle adjustment subsamples at random; the run with the lowest held-out error is kept. Disabled when no camera pair has more than 200 points, since every run would then use all the points and come out the same |
| **Label** | `refine` | the new entry's label |

The **Usable points** table shows, for each camera pair, how many selected points both cameras see. **Run** works with any number of points; few points in a pair means that pair's geometry is barely constrained.

**Reading the results.** The dialog shows each error before (the starting calibration) and after (the refinement):

- **Held-out points** is the one to judge by: the median and 90th-percentile reprojection error on frames the fit never saw. It should go down.
- **Fitted points** always go down, so they say little.
- **Charuco corners** should not get much worse. A refinement that fits the animal but breaks the board fits the labels' mistakes. These are left out when the starting calibration has no charuco takes, for example an imported one.
- **Change from the starting calibration** is the per camera-pair rotation and baseline change.
- **Scale correction.** Points alone have no metric scale, so the result is mapped back onto the starting calibration's scale and axes. The rig can drift in size during the fit at no cost, so a correction away from 1 is not a problem by itself.
- **Alignment residual** is what that mapping can't explain: the RMS camera-centre distance (cm and % of the mean baseline) and orientation difference. Above 1% or 0.5° the refinement changed the shape of the rig; check that the charuco corner error hasn't grown.
- **Run-to-run spread** shows how much the repeats differ; it's left out when the refinement ran once.

After **Save and make active**, the Calibration tab redraws the charuco corners with the new calibration's errors, so you can step through the board frames to check it.

**Save** writes the Stage 1 charuco entry if it is new, then the `refine` entry under it with a record of every point used. **Save and make active** also makes it active; like **Make active**, it doesn't re-triangulate. **Discard** saves nothing.

### Calibration pipeline

```python
all_rows = []
for cam_name in cam_names:
    rows = detect_rows(board, source, skip=skip)  # every Nth frame for video, all images for a folder
    all_rows.append(rows)

cameras = [Camera(name=n) for n in cam_names]
camera_group = CameraGroup(cameras)
camera_group.calibrate_rows(all_rows, board, init_intrinsics=True, init_extrinsics=True)
save_calibration(exp_dir, camera_group, "charuco-all", "charuco", charuco_takes=[...])
```

The refinement (`calibration_refine.py`) continues from such a group:

```python
p2ds, index = gather_points(points_df, cam_names, PointSelection(sources={"label3d", "sleap_user"}))
result = refine(start_cgroup, p2ds, index, RefineOptions(), charuco_p2ds=corners)
save_refinement(exp_dir, result, parent_entry, "refine", selection, options, points_df)
```

### Axis Alignment (Optional)

After calibration, the world coordinate origin and axes are defined arbitrarily by the board placement. Axis alignment lets the user re-orient to a meaningful physical reference.

#### When it applies

Separate optional step that runs **after** calibration. The result is saved as an `align` registry entry whose parent is the aligned calibration, with the rotation and origin under `[metadata.alignment]` (`geometry.save_aligned_calibration()`).

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
| [ Calibrate ] [> Label ]  Calibration: [charuco-all v]  Trial: [trial1 v]            |
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
  - Sync with Sleap...      Ctrl+S
  - Set up Sleap files...
  - Import predictions...   read .slp files from anywhere (e.g. sleap-nn track output)
  - Last sync summary...
  - Load skeleton...        assign a skeleton (.json or .slp) to all cameras or chosen ones
  - Remove points outside the skeletons...   delete points a camera's skeleton lacks
  - Export points...       wide-format CSV + SLP write; greyed while any experiment's 3D is stale

> Navigate
  - Previous / Next labeled frame   [ / ]
  - Previous / Next new frame       Shift+[ / Shift+]   (crosses trials)

> View
  - Color points by > Reprojection error / Source
  - Show legend
  - Cameras / Layout (Grid / Horizontal / Vertical) / Zoom

> Instances
  > Add
    - Average / Random  Ctrl+I
    - Random            Ctrl+Shift+I
    - Reprojected       (greyed when unavailable)
  - Accept point        A         clear the selected point's new mark
  - Accept frame        Shift+A   clear every new mark in the frame
  - Triangulate all experiments...
  - Delete instance     Ctrl+Backspace
```

### Switching calibrations

The **Calibration** combo in the top bar lists the experiment's registry as `label · YYYY-MM-DD HH:MM`, newest first, with refinements indented under their parent. The selected item is the active calibration; the tooltip shows method, parent, takes, point counts and errors.

Choosing another entry:
1. makes it active and redraws the epipolar lines;
2. re-triangulates the current frame at once;
3. re-triangulates every trial of the experiment, current trial first, behind a progress dialog. Editing waits until this finishes, because the triangulation writes into the same point store. **Cancel** stops it.

When it finishes, `[calibration.triangulated_with]` records the calibration used. If the 3D points come from another calibration than the active one (a cancelled run, a switch made in the Calibration tab, or a project from before the registry), the status bar shows **3D points are from <label>; re-triangulate**. The link re-triangulates the experiment. **Instances > Triangulate all experiments** does every experiment and also clears the warning.

### Skeletons per camera

Cameras can have different skeletons (rules in [data_model.md](data_model.md#skeletons-phase-14)).

- **Load skeleton...** opens a dialog: choose a `skeleton.json` or a `.slp` file, and the cameras (**All cameras** by default). Each camera shows its current skeleton and whether it has `.slp` files, which keep their own skeleton — the new one applies only to files not created yet. A conflict with the other skeletons disables OK. A mapping that would then put different skeletons in one file is a warning; Sync refuses until it is fixed in **Set up Sleap files**. Points that would no longer be in their camera's skeleton are counted, with an option to delete them now.
- **Set up Sleap files...** checks after every edit that each file holds one skeleton. When the project has more than one skeleton, each file is listed as `name.slp  [skeleton]` (or `[mixed]`, in red, which disables Apply), and the preset starts at *one file per camera per experiment*.
- **Remove points outside the skeletons...** lists the labelled points, by node and camera, whose camera's skeleton lacks the node (left over after a skeleton changed) and deletes them after confirmation. Until then they are hidden, not triangulated and not exported; the status bar mentions them on load and after a Sync.

### Skeleton overlay

The **skeleton is drawn** on every camera panel for every frame, except nodes the camera's skeleton lacks:
- **Edges** (bones): lines between connected nodes
- **Nodes** (points): circles at each labeled position
- **Node name labels**: text beside each node (toggled with `Ctrl+Tab`)

### Point coloring

**View > Color points by** chooses between reprojection error (the default) and source. The choice is saved with the session.

By **reprojection error**:

| Color | Meaning |
|---|---|
| Green | Low error (< 5 px) |
| Yellow | Medium error (5–15 px) |
| Red | High error (> 15 px) |
| Grey | No reprojection error available |
| White outline | Currently selected point |

By **source**:

| Color | Meaning |
|---|---|
| Magenta | Labeled (or guessed) in label3d |
| Orange | Labeled by a user in SLEAP |
| Light blue | Predicted by SLEAP |

In both modes, a **white square** around a point marks it as new since the latest sync. Correcting the point, or accepting it (`A`, `Shift+A`), removes the square.

### Legend

A legend at the bottom of the Label tab's left sidebar explains the markers. It covers shapes (point state), the colours for the current **Color points by** mode, the new and selected marks, and the frame-slider ticks. Turn it off with **View > Show legend**; the setting is saved with the session. The legend icons are drawn with the same code as the markers (`point_item.point_style`), so they always match.

### Point hover tooltip

```
head  —  cam0
Source: SLEAP prediction
Score: 0.93
New since import 4 (2026-10-05 14:22)
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
| `predicted` | Open circle (ring only) | Predicted by SLEAP, or a label3d guess |
| `not_set` | Dim placeholder or absent | Instance exists but point not placed |
| `not_visible` | Grey filled circle | User marked as not visible |

### Frame slider

Each camera has a row of ticks: black for frames with user labels, light blue for frames with only predictions. When the trial has points new since the latest sync, an extra row on top marks those frames: orange if any new point is a SLEAP user label, light blue if all are predictions.

### Reviewing new points

After every SLEAP read that brings in points, a summary dialog lists new SLEAP labels, new and updated predictions, and conflicts per trial. **Go to first new frame** jumps there. Step through new frames with `Shift+]` / `Shift+[`, then correct points or accept them. See [user_workflow.md](../user_workflow.md).

### Keyboard shortcuts

**Frame navigation**

| Key | Action |
|---|---|
| `→` / `←` | Next / prev frame |
| `Ctrl+→` / `Ctrl+←` | ±4 frames |
| `Ctrl+Alt+→` / `Ctrl+Alt+←` | ±100 frames |
| `Home` / `End` | First / last frame |
| `]` / `[` | Next / prev labeled frame |
| `Shift+]` / `Shift+[` | Next / prev frame with new points (crosses trials) |
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
| `A` | Accept selected point (clear its new mark in every camera) |
| `Shift+A` | Accept every point in the frame |
| `Ctrl+C` / `Ctrl+V` | Copy / paste instance |

**Point manipulation**

| Action | How |
|---|---|
| Place / move point | Left-click in any camera panel |
| Delete point | `Backspace` while bodypart selected |
| Mark as not visible | Right-click on placed point |
| Move whole instance | `Alt`+drag any point |

**Triangulation**

| Key | Action |
|---|---|
| `T` | Triangulate current frame |
| `Ctrl+Shift+T` | Triangulate the whole current trial |

**Display toggles**

| Key | Action |
|---|---|
| `H` | Toggle show all instances |
| `Ctrl+Tab` | Toggle node name labels |
| `Ctrl+Shift+Tab` | Toggle skeleton edges |

---

## Export Workflow (Step 3)

The user-facing steps and the column reference are in [user_workflow.md](../user_workflow.md#7-export). Code: `export.py` (no Qt), `widgets/export_dialog.py`, `workers/export_worker.py`, `export_cli.py`.

### File > Export points...

**Enabled** when `stale_experiments(project)` is empty, i.e. every experiment with an active calibration has `triangulated_with` equal to it. The state is refreshed when the File menu opens and from `_update_calibration_status`. Experiments with no active calibration don't block it.

**Dialog** (`ExportDialog`): experiment checklist (all checked), **Include predictions** (on), **Only points with 3D** (off), output path (default `exports/points_{YYYYmmdd-HHMMSS}.csv`). There is no score threshold: every point is triangulated and exported whatever its score, with its `score` column.

**On Export:**
1. `uncalibrated_experiments`: if any checked experiment has no active calibration, a warning lists them. **Export anyway** exports only their 2D columns; **Cancel** returns to the dialog.
2. `ExportWorker` (background): if the SLP mapping is complete (the condition that enables Sync), `pending_slp_changes` reads the mapped files like Sync (`read_slp_for_sync` + `compute_merge`). Any new or changed points, new scores or conflicts → a warning listing them per experiment; nothing is written.
3. Otherwise `export_points` writes the CSV, then `project.write_slp_files()` (the write half of a Sync; no sync id is recorded).
4. A summary (`format_summary`): rows written, camera points per experiment by state/source, 2D-only experiments, SLP files written.

### `uv run label3d-export`

```
uv run label3d-export <project> [-o out.csv] [--experiments a,b] [--no-predictions] [--only-3d] [--no-slp]
```

Same checks, without asking: stale experiments or pending SLP changes → message on stderr, exit 1, nothing written; experiments without a calibration → warning, 2D only. `--no-slp` skips the SLP check and write. It doesn't triangulate. It loads the project with `Project.load`, so the videos must be reachable.

### CSV format

Wide format, one row per (experiment, trial, frame, instance, bodypart), as in `points.csv`. Columns in the fixed order `export_columns(project.cameras)`: the five index columns, then for each camera

`{cam}_video_frame, {cam}_x.px, {cam}_y.px, {cam}_state, {cam}_source, {cam}_score, {cam}_xr.px, {cam}_yr.px, {cam}_error.px`

then `x.cm, y.cm, z.cm, n_cameras, mean_error.px, calibration`. Units follow a dot (`UNIT_2D`, `UNIT_3D` in `export.py`); columns without one are unitless.

- `video_frame = {cam}_start + frame`, from `trials.csv`.
- `error`, `n_cameras`, `mean_error` (`mean_reprojection_errors`) and `calibration` (`triangulated_with`) are computed at export; they are not stored in `points.csv`. `sync` is not exported.
- A camera's cells are blank where its state is `not_set` (or it's a prediction and **Include predictions** is off). The 3D columns are blank where x/y/z are missing; triangulation clears them for points seen by fewer than 2 cameras, so they are never stale. A row is left out only when no camera has a point and there is no 3D point.

### SLEAP files (SleapAdapter.save)

Written by **Sync** and by **Export points...**, one file per entry in the `[slp]` mapping:
- `Skeleton` built from bodypart names and edges
- `Video` reference set to matching file from `trials.csv`
- One `LabeledFrame` per labeled frame
- `Instance` (user) or `PredictedInstance` (predicted) per instance; `NaN` → invisible
- Saved via `sleap_io.save_file(labels, path)`

Exporting to other pose formats is Phase 17.1.

### Iterative workflow

```
Label in label3d → File > Sync with Sleap → train in SLEAP
  ↓
Predictions reach label3d: on open / Sync (SLEAP GUI saved them into slp/)
                       or: File > Import predictions... (sleap-nn track output)
  ↓
Sync summary → step through new frames (Shift+]) → correct or accept (A / Shift+A)
  ↓
Sync again → retrain  (repeat)
```

The full user-facing guide is [user_workflow.md](../user_workflow.md).

---

## Key Features

### Video synchronization
- `trials.csv` stores per-camera `_start` / `_end` frame indices; global frame `N` maps to video frame `_start + N`
- No playback — frame-step only

### Triangulation
- Clicked points undistorted on the fly (`cv2.undistortPoints`) before triangulation — raw pixels stored, undistorted used for geometry
- DLT on ≥2 views with state `set` or `predicted` (any score). `geometry.triangulate_rows` solves every row at once with a batched SVD, the same DLT as aniposelib's `CameraGroup.triangulate(points, undistort=True)` without its per-point loop
- A point seen by fewer than 2 cameras has its 3D point and every reprojection cleared
- **Auto-triangulate** (default on): triangulates current frame after any point edit
- `Ctrl+Shift+T` triangulates the whole trial in one batch (`triangulate_trial`) in a background thread; **Triangulate all experiments** does one batch per trial

### 3D view

**Hidden.** The panel is not shown in the Label tab. The code is kept in `widgets/view_3d.py`; set `SHOW_3D_VIEW = True` in `widgets/label_view.py` to bring it back.

- `pyqtgraph.opengl.GLViewWidget` panel in resizable splitter column
- Points colored by reprojection error; skeleton edges; camera frustums (toggled independently)
- Falls back gracefully if OpenGL unavailable

# Phases 1–7 (Completed — Archived)

All phases in this file are fully implemented. Kept for reference only.

---

## Phase 1 — Package scaffold and video display ✓

**Goal**: `uv run label3d /path/to/project` opens a window showing synchronized video frames in a grid, navigable with arrow keys.

### 1.1 Package setup
- [x] `pyproject.toml` with dependencies: `PySide6`, `opencv-python-headless`, `numpy`, `pandas`, `sleap-io`, `toml`
- [x] `__main__.py`: parse CLI args (project dir), call `app.run()`
- [x] `app.py`: create `QApplication`, instantiate `MainWindow`, enter event loop

### 1.2 Project model (`project.py`)
- [x] `Project` dataclass: `root: Path`, `cameras: list[str]`, `trials: list[TrialInfo]`, `camera_info: dict[str, CameraInfo]`, `has_calibration: bool`
- [x] `Project.load(path)`: Read `trials.csv`, compute trial lengths, load `calibration.toml` if present
- [x] `Project.video_path(trial, camera)` → resolved absolute path
- [x] `Project.video_start_frame(trial, camera)` → start frame offset

### 1.3 Video reader (`video.py`)
- [x] `VideoReader` class wrapping `cv2.VideoCapture`
- [x] LRU cache of last 8 frames per reader
- [x] `FrameProvider` holds `dict[str, VideoReader]`; `get_frame(camera, global_frame)`

### 1.4 Camera panel (`widgets/camera_panel.py`)
- [x] `CameraPanel(QGraphicsView)` with `QGraphicsPixmapItem` background
- [x] Zoom via `wheelEvent`; pan via `ScrollHandDrag`
- [x] `save_view_state()` / `restore_view_state()`

### 1.5 Camera grid (`widgets/camera_grid.py`)
- [x] `CameraGrid(QWidget)` with nested `QSplitter`s
- [x] Three layout modes: `LayoutMode.GRID` / `HORIZONTAL` / `VERTICAL`
- [x] Single-camera solo view toggle

### 1.6 Main window and frame controls
- [x] `MainWindow(QMainWindow)` with `QTabBar` (Calibrate / Label / Export)
- [x] Menu bar: Label3D (Quit), File (New/Open/Export stubs), View (Cameras, Layout)
- [x] `FrameControls(QWidget)`: buttons, seekbar, frame counter
- [x] Keyboard shortcuts: `←`/`→`, `Ctrl+←`/`→`, `Ctrl+Alt+←`/`→`, `Home`/`End`

### 1.7 Application state (`state.py`)
- [x] `ProjectState(QObject)` with signals: `frame_changed`, `trial_changed`, `selection_changed`, `points_changed`

---

## Phase 2 — Data layer: skeleton, points, adapters ✓

**Goal**: Solid data model that can be created, queried, mutated, and persisted. SLEAP import works.

### 2.1 Skeleton (`skeleton.py`)
- [x] `Skeleton` dataclass: `nodes: list[str]`, `edges: list[tuple[str, str]]`, `symmetries: list[tuple[str, str]]`
- [x] `Skeleton.from_sleap(slp_path)`, `from_json(path)`, `to_json(path)`

### 2.2 Points store (`points.py`)
- [x] `PointsStore` class wrapping `pd.DataFrame`
- [x] `create_empty(cameras, bodyparts)`, `from_csv(path)`, `to_csv(path)`
- [x] Point-level API: `get_point()`, `set_point()`, `set_not_visible()`, `delete_point()`
- [x] Frame-level API: `get_frame_data()`, `get_instances()`
- [x] Instance-level API: `add_instance()`, `delete_instance()`, `copy_instance()`
- [x] 3D API: `set_3d()`, `set_reprojected()`
- [x] Dirty tracking; `QObject` signal on mutations

### 2.3 Adapter framework (`adapters/`)
- [x] `base.py`: `PoseAdapter` ABC
- [x] `__init__.py`: `get_adapters() -> dict[str, PoseAdapter]`
- [x] `sleap_adapter.py`: `SleapAdapter` — load from `.slp` files (state=`predicted` for `PredictedInstance`, `set` for `Instance`), save to `.slp` files

---

## Phase 3 — Labeling: overlays, interaction, selection ✓

**Goal**: Full labeling UI — see skeleton, click to place/move points, select bodyparts, see epipolar lines.

### 3.1 Point item (`widgets/point_item.py`)
- [x] `PointItem(QGraphicsEllipseItem)`: render by state, color from reprojection error, draggable

### 3.2 Skeleton overlay (`widgets/skeleton_overlay.py`)
- [x] `SkeletonOverlay` manages `PointItem`s + `QGraphicsLineItem` edges; item pool for performance

### 3.3 Mouse interaction in CameraPanel
- [x] Left-click near point → select; on empty space → place point; right-click → toggle `not_visible`
- [x] Drag: commit to `PointsStore`; if was `predicted` → promote to `set`

### 3.4 Selection model
- [x] `↑`/`↓` cycle bodypart; `` ` `` cycle instance; `1`–`9` select by index; `Escape` clear
- [x] `selection_changed` signal drives all overlay updates

### 3.5 Epipolar lines
- [x] `compute_fundamental_matrix(cam_a, cam_b)` — cached per camera pair
- [x] `compute_epipolar_line(F, point_2d)` — clip to image bounds
- [x] `EpipolarOverlay`: lines shown only for selected bodypart, color-coded by source camera

### 3.6–3.7 Keyboard shortcuts + bodypart list sidebar
- [x] All shortcuts registered as `QShortcut`s in `label_view.py`
- [x] `QListWidget` sidebar showing skeleton node names with per-bodypart status

---

## Phase 4 — Calibration ✓

**Goal**: Load calibration media, detect corners, run bundle adjustment, visualize results.

### 4.1 Calibration view (`widgets/calibration_view.py`)
- [x] Board settings form; "Detect Corners" and "Run Calibration" buttons; per-camera error badge

### 4.2 Calibration worker (`workers/calibration_worker.py`)
- [x] `DetectionWorker(QThread)`: detect corners per camera
- [x] `CalibrationWorker(QThread)`: `CameraGroup.calibrate_rows()`, compute per-frame errors
- [x] `charuco_overlay.py`: green dots (detected), red crosses (reprojected)

### 4.3 Calibration input discovery
- [x] Scan `experiments/{active}/calibration/cam*/` — video files → `detect_video()`, images → `detect_images()`

---

## Phase 5 — Triangulation and 3D view ✓

**Goal**: Real-time triangulation of current frame, batch triangulation, 3D visualization.

### Known bugs fixed (2026-05-25)

1. **RecursionError on point click** — `_triangulate_current_frame()` ended with `self._state.points_changed.emit(...)` → infinite recursion. Fixed by removing the emit and calling update methods directly.

2. **Wrong coordinate space for DLT** — aniposelib's `triangulate_simple` expects **normalized camera coordinates**. Original code called `cv2.undistortPoints(..., P=self.matrix)` which outputs undistorted **pixel** coordinates. Fixed by passing raw pixel coordinates with `undistort=True`.

### 5.1 Triangulation engine (`geometry.py`)
- [x] `triangulate_frame()`: collect 2D points per bodypart, build sub-cgroup, triangulate, fill 3D + reprojected columns
- [x] Auto-triangulate default on

### 5.2 Batch triangulation (`workers/triangulation_worker.py`)
- [x] `TriangulationWorker(QThread)`: iterate all labeled frames; `Ctrl+Shift+T`

### 5.3 3D view (`widgets/view_3d.py`)
- [x] `View3D` wrapping `pyqtgraph.opengl.GLViewWidget`; scatter plot, skeleton edges, camera frustums, world axes
- [x] Falls back gracefully if OpenGL unavailable

---

## Phase 6 — Hierarchical (multi-experiment) projects ✓

**Goal**: Support projects where different groups of trials have different calibrations, structured as `experiments/{name}/` subfolders.

### 6.1 Synthetic data generator — hierarchical output ✅
- [x] Route all outputs to `experiments/{experiment_name}/`; global `trials.csv` with `experiment` column at project root

### 6.2 `setup_project_cli.py` — two-level interactive flow ✅
- [x] Add `prompt_optional_list()` helper
- [x] Insert experiment keyword prompt; always write `experiment_keywords` to config

### 6.3 `project.py` — hierarchical loading
- [x] `ExperimentInfo` dataclass; `project.experiments: dict[str, ExperimentInfo]`
- [x] `project.active_experiment: str` + `project.set_experiment(name)`
- [x] All paths resolved via `active_experiment`

### 6.4 `state.py` + `MainWindow` — experiment selector
- [x] `ProjectState.current_experiment` + `experiment_changed = Signal(str)`
- [x] Experiment `QComboBox` in top bar (hidden when only one experiment)

### 6.5 Fix calibration paths
- [x] `DetectionWorker`/`CalibrationWorker` use `active_experiment` for correct paths

### 6.6 `PointsStore` — 5-level index
- [x] Row index `(experiment, trial, frame, instance, bodypart)`; `self.experiment_name` stored
- [x] `INDEX_NAMES` updated; `create_empty()` accepts `experiment_name`

### 6.7 `setup_project.py` — two-level classification
- [x] `ExperimentAssignment` dataclass; `_extract_experiment_id()` helper
- [x] `classify_files()` groups by experiment; `execute_setup()` creates hierarchical layout

---

## Phase 7 — SLP Import ✓

**Goal**: Import predicted instances from `project/slp/` into `PointsStore`; expose re-import via File > Import from SLP files.

### 7.1 `PointsStore.merge_from_df(adapter_df)` (`points.py`)
Merge rule per `(experiment, trial, frame, instance, bodypart)`:
- Existing `set` or `not_visible` → **Skip** (user data always wins)
- All cameras `predicted`, incoming `predicted` → replace (newer prediction wins)
- `not_set` or absent → insert

### 7.2 `Project.import_from_slp(experiment)` (`project.py`)
1. Collect `*.slp` from `self.slp_dir`
2. `SleapAdapter().load(slp_files, self.trials_df, self.cameras)`
3. Filter rows to active experiment; call `self.points.merge_from_df(filtered_df)`

### 7.3 Auto-import on first open
If `points.csv` empty and `slp/` has `.slp` files: `QMessageBox.question()` → import on Yes.

### 7.4 `_do_import_slp()` in `MainWindow`
- Show `QProgressDialog`; run import in `QThread`; update status bar

### 7.5 Experiment switch — per-experiment import check
After switching `PointsStore`, apply same empty-CSV check.

### 7.6 Instance initialization strategies (`label3d/instance_init.py`)

**Random** — scatter points around a center pixel (uses `node_offsets_cm` if available).

```python
def init_random(skeleton, camera_name, cameras, center_px, frame_size, rng=None) -> dict[str, tuple[float, float]]
```

**Average** — use mean relative positions from labeled instances in this experiment/trial.

```python
def init_average(points_store, experiment, trial, camera_name, skeleton, center_px, frame_size) -> tuple[dict[str, tuple[float, float]], bool]
# Returns (coords, used_fallback)
```

**Reprojected** — fill one camera's view for an existing partially-labeled instance (needs ≥2 other cameras).

```python
def init_reprojected(points_store, experiment, trial, frame, instance, target_camera, cameras) -> dict[str, tuple[float, float]] | None
```

**Availability check:**
```python
def available_strategies(points_store, experiment, trial, frame, instance, camera_name) -> set[Literal["random", "average", "reprojected"]]
```

### 7.7 "Instances" menu and right-click UI
- Menu bar "Instances > Add" with Average/Random (`Ctrl+I`), Random (`Ctrl+Shift+I`), Reprojected (greyed when unavailable)
- Right-click on empty space in `CameraPanel` → context menu with same three options
- `CameraPanel` emits `add_instance_requested = Signal(str, str, float, float)` (camera, strategy, x, y)
- `LabelView` handles dispatch: re-checks availability, allocates instance, calls init function, writes via `set_point(state="set")`

### 7.8 Keyboard shortcut update
- `Ctrl+I` → average if ≥2 reference instances available, otherwise random (center = frame center)
- `Ctrl+Shift+I` → random unconditionally
- Old "blank instance" path removed

---

## Prompt Template (for initial code generation — historical)

```
Build a Python/PySide6 desktop app called "label3d" for labeling 3D points
across multiple synchronized calibrated videos. Use uv package manager.

Requirements:
- Load a project folder containing: trials.csv, calibration.toml (aniposelib
  CameraGroup format), and optionally existing points.csv.
- Display videos in a grid layout, one QGraphicsView panel per camera.
- Store labels in points.csv: pandas MultiIndex CSV with row index
  (experiment, trial, frame, instance, bodypart) and column MultiIndex (view, coord).
  state ∈ {set, predicted, not_set, occluded}. NaN = missing.
- Click to place/move points. Full skeleton overlay on every panel.
  Points colored by reprojection error (green/yellow/red/grey).
- Selected bodypart shows white outline + epipolar lines in other views.
- Auto-triangulate current frame on point edit (toggleable).
- Import/export via pluggable PoseAdapter subclasses in label3d/adapters/.

Architecture: Model–Signal–View. Central ProjectState (QObject) emits signals.
QGraphicsView/QGraphicsScene per camera panel. pyqtgraph GL for 3D.
```

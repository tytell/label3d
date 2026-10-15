# label3d — Architecture

## Pattern

The app follows a **Model–Signal–View** pattern: a central `ProjectState` object holds all mutable state and emits Qt signals on changes. Views subscribe to those signals and update themselves. No view reads from another view directly.

## Package layout

Files marked ✓ exist; unmarked are planned.

```
label3d/
├── __init__.py                  # ✓
├── __main__.py                  # ✓ CLI entry: `label3d [project_dir]`
├── app.py                       # ✓ QApplication, argument parsing, launch
├── state.py                     # ✓ ProjectState: all mutable state + Qt signals
├── project.py                   # ✓ Project folder I/O: trials.csv, calibration, project.toml, paths
├── setup_project.py             # ✓ Project setup: classify files, match trials, create structure
├── points.py                    # ✓ PointsStore: DataFrame wrapper, CRUD, save/load
├── skeleton.py                  # ✓ Skeleton: nodes, edges, symmetry pairs
├── geometry.py                  # ✓ CameraParams, fundamental matrix, epipolar lines, triangulation
├── calibration_registry.py      # ✓ Calibration registry: save/list/load/rename/delete solutions (Phase 11)
├── calibration_solve.py         # ✓ Charuco solves without Qt: takes, ranges, repeated calibrate_rows, metadata
├── calibration_refine.py        # ✓ Refinement with labelled points: selection, Stage 2, alignment, points record (Phase 12)
├── video.py                     # ✓ VideoReader: cv2.VideoCapture wrapper, frame cache
├── instance_init.py             # ✓ random / average / reprojected instance init strategies
├── export.py                    # ✓ Wide-format CSV export and pre-flight checks, no Qt (Phase 13)
├── export_cli.py                # ✓ label3d-export command line tool (Phase 13)
├── widgets/
│   ├── __init__.py              # ✓
│   ├── main_window.py           # ✓ QMainWindow, tab bar, menus, top toolbar
│   ├── camera_panel.py          # ✓ QGraphicsView/Scene for one camera view
│   ├── camera_grid.py           # ✓ Nested QSplitter grid of CameraPanels
│   ├── frame_controls.py        # ✓ Bottom bar: navigation, seekbar, frame counter
│   ├── point_item.py            # ✓ QGraphicsEllipseItem subclass: one labeled point
│   ├── skeleton_overlay.py      # ✓ Manages all point_items + edges for one camera
│   ├── epipolar_overlay.py      # ✓ Epipolar line items for one camera
│   ├── charuco_overlay.py       # ✓ Detected / reprojected corner overlays for calibration
│   ├── view_3d.py               # ✓ pyqtgraph GLViewWidget: 3D points, edges, frustums (hidden; SHOW_3D_VIEW in label_view.py)
│   ├── bodypart_list.py         # ✓ Sidebar: bodypart names, click to select
│   ├── calibration_view.py      # ✓ Calibration mode: settings panel + camera grid
│   ├── refine_dialog.py         # ✓ Refine with points dialog (Phase 12)
│   ├── axis_alignment_widget.py # Phase 14: Axis alignment panel (not yet created)
│   ├── label_view.py            # ✓ Labeling mode: grid + controls + sidebar
│   ├── export_dialog.py         # ✓ File > Export points... options (Phase 13)
│   └── export_view.py           # Phase 15: export to other pose formats (not yet created)
├── adapters/
│   ├── __init__.py              # ✓ Registry: discover PoseAdapter subclasses
│   ├── base.py                  # ✓ PoseAdapter ABC
│   └── sleap_adapter.py         # ✓ SleapAdapter
└── workers/
    ├── __init__.py
    ├── calibration_worker.py    # ✓ DetectionWorker, MultiTakeCalibrationWorker, RefineWorker QThreads
    ├── export_worker.py         # ✓ ExportWorker QThread: SLP check, CSV export, SLP write
    └── triangulation_worker.py  # ✓ TriangulationWorker / AllExperimentsTriangulationWorker: one batch per trial
```

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
            │   ★ MILESTONE: video grid with frame stepping works
            ▼
2.1 skeleton.py  +  2.2 points.py (PointsStore)
            │
            ▼
2.3 adapters/ (SLEAP import)
            │   ★ MILESTONE: can import SLEAP data into points.csv
            ▼
3.1–3.2 point_item + skeleton_overlay (rendering)
            ▼
3.3 mouse interaction (click to place/move)
            ▼
3.4 selection model  +  3.5 epipolar lines
            │   ★ MILESTONE: full labeling workflow
            ▼
5.1 triangulation engine → error coloring → tooltips
            ├──────────────────────┐
            ▼                      ▼
4.1–4.3 calibration view     5.2–5.3 batch triangulation + 3D view
            │                      │
            └──────────┬───────────┘
                       ▼
         6.1–6.7 hierarchical projects (experiments/ layout) ✅
                       ▼
         7.1–7.8 SLP import + instance init strategies ✅
                       │   ★ MILESTONE: predicted instances visible from project open
                       ▼
         8.1–8.3 axis alignment (calibration_view tab)
                       ▼
              9.1 export view
                       ▼
              10.1–10.4 SLP write-back (File > Save + closeEvent dirty prompt)
                       │   ★ MILESTONE: full SLP round-trip
                       ▼
                  11.* polish
```

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

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| QGraphicsScene rebuild slow with many items | Laggy frame stepping | Pool QGraphicsItems; update positions instead of recreating |
| cv2.VideoCapture random seek slow for compressed video | 100ms+ delay jumping frames | LRU frame cache (8 frames); pre-decode ahead |
| pandas MultiIndex CRUD slow for point-level edits | Laggy click-to-place | Cache current frame as dict-of-dicts; batch-write to DataFrame on frame change |
| pyqtgraph OpenGL unavailable on some systems | 3D view crashes | Make 3D panel optional; catch import error, show message |
| aniposelib API instability (research code) | Calibration breaks with updates | Pin version in pyproject.toml |
| Large projects (many trials × long videos) | Memory pressure | Lazy loading: only open VideoCapture for current trial; only load frame slice from DataFrame |

---

## Design Decisions

1. **Synchronization**: Implicit per-camera offset via `_start` values in `trials.csv` — global frame `N` maps to video frame `_start + N`. No separate offset column; no timecode support.

2. **Calibration convention**: OpenCV world-to-camera convention throughout. `rvec`/`tvec` transform world points into camera space.

3. **Number of cameras**: Minimum 2, no maximum. UI grid layout adapts.

4. **Playback**: Frame-step only — no video playback.

5. **Point storage**: Raw pixel coordinates stored in `points.csv`. Undistortion applied on the fly for geometry — never stored. Preserves fidelity and allows recalculation if calibration is updated.

6. **Always hierarchical layout**: All projects use `experiments/{name}/` subfolders. A single-experiment project has exactly one subfolder (`experiments/default/`). Each experiment has a calibration registry (`calibrations/`) with exactly one active entry. `.slp` files are global at project root. `points.csv` is per-experiment with 5-level row index.

7. **Trial names are local to each experiment**: The global `trials.csv` `experiment` column disambiguates — `Trial1` in `PS18` is distinct from `Trial1` in `PS19`.

8. **Distribution**: Python package only, using **uv** for dependency management. Entry point: `label3d` CLI command.

9. **Instance index gaps after deletion**: Allowed — instance index is an opaque ID, not a sequential counter.

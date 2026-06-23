# Phase 8 — Axis Alignment

**Goal**: Let the user re-orient the world coordinate system to a physical reference by clicking points in trial video frames.

**Status**: Partially implemented — geometry helpers and CameraPanel pick mode done; UI widget not yet created.

## Current state

- `geometry.py`: `triangulate_point()`, `compute_axis_alignment()`, `apply_alignment_to_cameras()`, `save_aligned_calibration()` — implemented; not yet connected to UI
- `widgets/camera_panel.py`: `raw_clicked = Signal(str, float, float)` signal and `set_pick_mode(bool)` method — implemented
- `widgets/axis_alignment_widget.py` — **not yet created**
- `widgets/calibration_view.py` — not yet modified to add alignment tab

## 8.1 `load_cameras()` fix (`geometry.py`)

- [ ] Check whether `load_cameras()` already skips `"alignment"` sections — the skip list may only include `"metadata"`; add `"alignment"` to the skip list if not already present
- [ ] Add `load_alignment(path) -> tuple[np.ndarray, np.ndarray] | None` — reads `R_align` and `origin` from `[alignment]` section; returns `None` if absent

## 8.2 Axis alignment widget (`widgets/axis_alignment_widget.py`)

- [ ] `AxisAlignmentWidget(QWidget)`:
  - Trial selector `QComboBox` + frame navigation (prev/next buttons + frame spinbox)
  - Reuses the existing `CameraGrid` + `FrameProvider` to display trial video frames
  - Right-side panel: 7 `QPushButton` pick targets (Origin, X near, X far, Y near, Y far, Z near, Z far), a `QTableWidget` showing pick status (✓ / –) per target, and three "Flip" `QCheckBox`es for x/y/z
  - "Compute" button — runs `compute_axis_alignment()`; shows preview (axis vectors + R_align) in a text area
  - "Apply & Save" button — calls `save_aligned_calibration()`, writes `[alignment]` section to `calibration.toml`
  - Pick workflow:
    1. User clicks a pick-target button → button stays pressed (checked), all `CameraPanel`s enter pick mode (`set_pick_mode(True)`)
    2. User clicks in any camera panel → `raw_clicked(cam, x, y)` received; stored in a `dict[target, dict[cam, (x, y)]]`
    3. Each target accumulates clicks across cameras (≥2 cameras needed for triangulation); table cell updated
    4. Clicking the same button again or pressing Escape cancels pick mode
  - Frame navigation updates `FrameProvider` and `CameraGrid` directly (no `ProjectState` shared state)
  - On "Compute":
    1. For each pick target with ≥2 camera picks: call `triangulate_point(cameras, picks)`
    2. Pass triangulated points to `compute_axis_alignment(..., flip_x, flip_y, flip_z)`
    3. Show resulting `R_align` and `origin` in a read-only text area
  - On "Apply & Save":
    1. Write `[alignment]` section to `calibration.toml` (`save_aligned_calibration()`)
    2. Emit `alignment_saved` signal so `CalibrationView` / `LabelView` can reload calibration

## 8.3 Integrate into calibration view (`widgets/calibration_view.py`)

- [ ] Add "Axis Alignment" as a second tab in `CalibrationView` (alongside the existing board detection / calibration tab)
- [ ] Instantiate `AxisAlignmentWidget` in the new tab
- [ ] Connect `AxisAlignmentWidget.alignment_saved` → reload cameras in `LabelView` (via `ProjectState` or a direct callback)

# Phase 19 — Multi-Instance Support and Cross-Camera Matching

**Goal**: Allow multiple animal instances per frame, with explicit cross-camera and cross-frame identity tracking.

**Status**: Not started. Earlier phases assume at most one instance per frame per camera (instance index always 0). This phase lifts that restriction.

---

## Background

The data model already supports multiple instances: the 5-level row index `(experiment, trial, frame, instance, bodypart)` uses an integer instance index, and rows for different cameras sit at the same `(frame, instance)` level — so instance 0 is implicitly "the same animal across all cameras." The bug is entirely in the UI: right-clicking in camera A and adding creates instance 0 (filled only in A), then doing the same in camera B creates instance 1 (filled only in B). Triangulation never fires because no bodypart has ≥2 cameras under the same instance index.

---

## 19.1 Data model additions (`points.py`)

### `merge_instances(trial, frame, keep, discard) -> None`

For each camera: if `keep` has state `not_set` and `discard` has real data (`set` / `predicted` / `not_visible`), copy the data from `discard` into `keep`. Then call `delete_instance(trial, frame, discard)`. Use this when two cameras accidentally got different instance indices for the same animal.

```python
def merge_instances(
    self, trial: str, frame: int, keep: int, discard: int
) -> None: ...
```

### `split_instance_camera(trial, frame, instance, camera) -> int`

Copy that camera's data for `instance` into a newly-minted instance index, then reset the original camera's entries back to `not_set`. Returns the new instance index. Use this when two different animals ended up under the same instance index.

```python
def split_instance_camera(
    self, trial: str, frame: int, instance: int, camera: str
) -> int: ...
```

---

## 19.2 Right-click context menu restructuring (`label_view.py`)

Replace the current "Add instance" submenu with two top-level items:

| Menu item | Behaviour |
|---|---|
| **Add new instance** | Mints a new instance index. Fills *only this camera* using random / average / reprojected (current single-camera logic). |
| **Add matching instance** | Enters **pick mode** (§19.3). After the user clicks an existing instance in any camera panel, initialises that instance's data in this camera using reprojected (if ≥2 other cameras have data) or random/average fallback. |

"Reprojected" moves out of the "Add" submenu: it becomes a fill action on an existing instance (Instances menu → Fill → Reprojected).

---

## 19.3 Pick mode (`state.py`, `camera_panel.py`, `label_view.py`)

Pick mode is a transient interaction state where clicking in any camera panel captures an instance index rather than placing a point.

### `AppState` additions

```python
# In state.py
def enter_pick_mode(self, callback: Callable[[int], None]) -> None:
    """Enter pick mode; callback receives the chosen instance index."""

def cancel_pick_mode(self) -> None:
    """Exit pick mode without invoking the callback."""

@property
def in_pick_mode(self) -> bool: ...
```

### `CameraPanel` behaviour in pick mode

- Cursor changes to crosshair.
- Existing instance dots are drawn with a larger highlight ring.
- Hovering over a dot shows its instance index.
- Clicking a dot fires `instance_picked = Signal(str, int)` (camera, instance) instead of the normal `point_clicked`.
- Clicking empty space is a no-op (does not place a point).
- `Escape` cancels pick mode.

### `LabelView` orchestration

- "Add matching instance" sets a callback: `lambda inst: self._fill_instance_in_camera(camera_name, inst)` then calls `state.enter_pick_mode(cb)`.
- `_fill_instance_in_camera(camera, instance)`: runs reprojected init if possible, otherwise random/average; writes points; emits `points_changed`.
- All panels are notified of pick mode via `state.in_pick_mode`; they poll it on `selection_changed` or via a dedicated `pick_mode_changed` signal.

---

## 19.4 Instances menu additions (`main_window.py`, `label_view.py`)

| Item | Behaviour |
|---|---|
| **Merge into selected** | Enters pick mode. User clicks any instance in any camera. Calls `merge_instances(keep=selected, discard=clicked)`. |
| **Split from this camera** | Calls `split_instance_camera(trial, frame, selected, active_camera)`. Selects the new instance. |

Enable/disable rules:
- **Merge**: available when ≥2 distinct instance indices exist in this frame.
- **Split**: available when the selected instance has data in ≥2 cameras (otherwise splitting is pointless).

---

## 19.5 Cross-frame instance tracking (future)

Each frame currently maintains instance indices independently. For sequences with many frames it would be useful to propagate identity forward (instance 0 stays the same animal across frames). Two approaches:

**Explicit assignment**: after labeling a frame, user maps instances by clicking "Match to previous frame" → pick mode shows previous frame's instance positions as ghost overlays; clicking one links current instance to it.

**Automatic tracking**: run a nearest-neighbour or Hungarian-algorithm match between consecutive frames based on 3D position; surface conflicts for manual resolution. Requires at least sparse 3D data from triangulation.

This sub-phase is deferred until the per-frame multi-instance UI (§19.2–19.4) is stable.

---

## 19.6 Migration from single-instance data

Existing `points.csv` files produced during Phase 1–11 always have instance index 0. No migration is needed — they load and display correctly under the new system.

---

## Implementation order

1. §19.1 — `PointsStore.merge_instances` + `split_instance_camera` + unit tests
2. §19.3 — Pick mode in `AppState` and `CameraPanel`
3. §19.2 — Right-click menu restructure
4. §19.4 — Instances menu additions
5. §19.5 — Cross-frame tracking (separate sub-phase)

# Phase 11 — Calibration Registry and Switching Calibrations

**Goal**: Keep every calibration solution an experiment has produced, together with how it was made. Point to the one in use, and switch between them in the Label tab, with epipolar lines and triangulated points updating.

**Status**: Not started.

**Compatibility**: Pointer only. After this phase nothing reads or writes `experiments/{name}/calibration.toml` or `calibration_takes/*.toml`. A one-time import (11.3) copies the existing files into the registry so existing calibrations aren't lost.

---

## 11.1 Storage

```
experiments/{name}/calibrations/
    {label}_{YYYYmmdd-HHMMSS}.toml
```

Each file is a plain aniposelib `CameraGroup.dump()`, so `CameraGroup.load()` reads it directly. Provenance goes in the `[metadata]` table, which aniposelib keeps on dump and load (`cgroup.metadata`).

Other top-level tables can't be used. `CameraGroup.load()` treats every top-level key except `metadata` as a camera. This also matters for the `[alignment]` section planned in Phase 14, which must move under `[metadata.alignment]`.

```toml
[metadata]
id = "refined-a_20261005-142210"      # = file stem; unique
label = "refined-a"                   # user-editable short name
created = "2026-10-05T14:22:10"
method = "charuco"                    # charuco | refine | align | imported
parent = ""                           # id this one was derived from (refine, align)
charuco_takes = ["PreCal", "PostCal"]
notes = ""

[metadata.points]                     # refine only (Phase 12)
sources = ["label3d", "sleap_user", "sleap_pred"]
score_min = 0.8
frames = { t1 = [10, 55, 90], t3 = "all" }
n_points = 1840

[metadata.options]
only_extrinsics = true
holdout = 0.2

[metadata.errors]                     # px
charuco = 0.41
points = 2.1
holdout = 2.6
```

**Labels and ids:**
- The label defaults to the method plus a short description, e.g. `charuco-PreCal`, `charuco-all`, `refine`. The user can edit it when saving.
- Label characters are restricted to `[A-Za-z0-9_-]` so the file name stays portable.
- The timestamp makes the id unique even if a label is reused.

## 11.2 Active pointer (`project.toml`, `project.py`)

```toml
[experiments.exp1]
active_calibration = "charuco-all_20260901-101500"
```

- `Project.calibration_path(experiment=None)` resolves through the pointer and returns `None` when the pointer is unset or the file is missing.
- `has_calibration` comes from the pointer.
- `_read_camera_info()` reads the active file.
- Callers that assume a path exists need to handle `None`: `label_view.py`, `main_window.py` (Triangulate all experiments) and the calibration view's reload.
- **New method:** `Project.set_active_calibration(experiment, id)`. It writes the pointer through `save_config_sections()` and calls `reload_calibration()`.

## 11.3 Registry module (`calibration_registry.py`, no Qt)

```python
@dataclass
class CalibrationEntry:
    id: str
    label: str
    created: datetime
    method: str
    parent: str
    path: Path
    metadata: dict

def list_calibrations(exp_dir: Path) -> list[CalibrationEntry]      # newest first
def load_calibration(entry) -> CameraGroup
def save_calibration(exp_dir, cgroup, label, method, **metadata) -> CalibrationEntry
def import_legacy(exp_dir) -> list[CalibrationEntry]
```

`import_legacy()` imports `calibration.toml` and `calibration_takes/*.toml` as `method="imported"`. It runs once, when the registry folder is missing. It leaves the old files where they are, and sets the pointer to the entry imported from `calibration.toml`.

## 11.4 Writers

- **Calibration tab, per-take solutions** ([calibration_view.py](../../src/label3d/widgets/calibration_view.py), `_save_take_solutions`): one `charuco` entry per take, with `charuco_takes = [take]`.
- **Calibration tab, final solution** (the save that currently writes `calibration.toml` and `corners.csv`): one `charuco` entry with the takes used. It becomes active.
  - `corners.csv` stays as is: corners are inputs, not solutions.
  - Per-take corners (`corners_{take}.csv`) are what Phase 12 reuses.
- **Repeat runs:** the best of the repeated runs is the one saved, and `metadata.errors` records its error and the run-to-run spread.
- **Later writers:** axis alignment (Phase 14) saves an `align` entry whose parent is the aligned calibration. Refinement (Phase 12) saves `refine` entries.

## 11.5 Switching in the Label tab (`label_view.py`, `main_window.py`)

**The combo box:** a **Calibration** combo in the Label tab toolbar lists the registry, newest first, as `label · YYYY-MM-DD HH:MM`, with the active one checked. The tooltip shows method, parent, takes, point counts and errors.

**What changing it does:**
1. `set_active_calibration()`, which writes the pointer.
2. Rebuild the `EpipolarCache` and redraw the overlays (the existing `reload_calibration()` path).
3. Re-triangulate the current frame, then the current trial in the background (`TriangulationWorker`), so `xr/yr`, error colours and the 3D view update.
4. Re-triangulate the experiment's other trials in the background, one worker at a time, as `_run_next_triangulation` does now. The status bar shows progress.

**Recording which calibration was used:**
- `project.toml` stores `triangulated_with = "<id>"` per experiment once a full re-triangulation finishes.
- If it differs from `active_calibration` (an interrupted run, or a switch made in the Calibration tab), the Label tab shows "3D points are from <label>; re-triangulate" in the status bar and offers **Instances > Triangulate all experiments**.

**The Calibration tab:** add a read-only registry list (label, date, method, error) with **Make active**, **Rename** and **Delete** (with a confirmation). The active entry can't be deleted.

## 11.6 Tests (`tests/test_calibration_registry.py`)

- **Registry:**
  - `save_calibration` → `list_calibrations` → `load_calibration` round trip, keeping the metadata.
  - `CameraGroup.load()` works on a saved file.
  - Ids stay unique when a label is saved twice in the same second; a counter suffix is added if needed.
- **Import and pointer:**
  - `import_legacy` on a project with `calibration.toml` and two take files sets the pointer and doesn't touch the originals.
  - `calibration_path()` returns `None` with no pointer.
- **Switching:** gives different `xr/yr` for a perturbed calibration. This reuses the two-camera project in `test_project_sync.py`.

## Implementation order

1. 11.3 registry module + 11.6 tests
2. 11.2 pointer + caller fixes for `None`
3. 11.4 writers in the Calibration tab
4. 11.5 Label tab combo box, re-triangulation and stale-3D warning

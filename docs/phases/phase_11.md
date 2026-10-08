# Phase 11 — Calibration Registry and Switching Calibrations

**Goal**: Keep every calibration solution an experiment has produced, together with how it was made. Point to the one in use, and switch between them in the Label tab, with epipolar lines and triangulated points updating.

**Status**: Not started.

**Compatibility**: Pointer only. After this phase nothing reads or writes `experiments/{name}/calibration.toml` or `calibration_takes/*.toml`. A one-time import (11.3) copies the existing files into the registry so existing calibrations aren't lost. There are few legacy projects, so the import is kept simple.

---

## 11.1 Storage

```
experiments/{name}/calibrations/
    {label}_{YYYYmmdd-HHMMSS}.toml            # charuco, imported, align
    refined/
        {label}_{YYYYmmdd-HHMMSS}.toml        # refine (Phase 12)
        {label}_{YYYYmmdd-HHMMSS}_points.csv  # the exact points that went into it
```

Each `.toml` is a plain aniposelib `CameraGroup.dump()`, so `CameraGroup.load()` reads it directly. Provenance goes in the `[metadata]` table, which aniposelib keeps on dump and load (`cgroup.metadata`).

Refined calibrations live in their own folder, so the original charuco solutions stay separate from those derived from them. Phase 12.2 describes the `_points.csv` companion file.

Other top-level tables can't be used. `CameraGroup.load()` treats every top-level key except `metadata` as a camera. Alignment data therefore goes under `[metadata.alignment]` (11.4).

```toml
[metadata]
id = "refined-a_20261005-142210"      # = file stem; unique
label = "refined-a"                   # user-editable short name
created = "2026-10-05T14:22:10"
method = "charuco"                    # charuco | refine | align | imported
parent = ""                           # id this one was derived from (refine, align)
charuco_takes = ["PreCal", "PostCal"]
notes = ""

[metadata.charuco]                    # charuco only: enough to redo the solve
board = { squares_x = 7, squares_y = 5, square_length = 2.0, marker_length = 1.5, marker_bits = 4, dict_size = 50 }   # [charuco] in project.toml
frame_skip = 5
take_ranges = { PreCal = [0, 1800], PostCal = [0, 1650] }
detected = "2026-10-05T14:20:03"      # corners detection the solve used (11.4)

[metadata.points]                     # refine only (Phase 12)
sources = ["label3d", "sleap_user", "sleap_pred"]
score_min = 0.8
frames = { t1 = [10, 55, 90], t3 = "all" }
n_points = 1840
points_file = "refined-a_20261005-142210_points.csv"

[metadata.options]
only_extrinsics = true
holdout = 0.2

[metadata.errors]                     # px
charuco = 0.41
charuco_spread = 0.03                 # run-to-run spread over repeats
points = 2.1
holdout = 2.6
```

**Labels and ids:**
- The label defaults to the method plus a short description, e.g. `charuco-PreCal`, `charuco-all`, `refine`. The user can edit it when saving.
- Label characters are restricted to `[A-Za-z0-9_-]` so the file name stays portable. Take names are cleaned to fit: spaces become `_` and other characters are dropped. The joint solution (`JOINT_TAKE`, "All takes (joint)") becomes `charuco-all`.
- The timestamp makes the id unique even if a label is reused.
- The id never changes after it's created. Renaming changes `metadata.label` only, so `parent` and pointer references stay valid.

## 11.2 Active pointer (`project.toml`, `project.py`)

`project.toml` has no per-experiment tables. `[project]`, `[charuco]`, `[calibration]` and `[slp]` are all project-wide. The pointers therefore go in a sub-table of the existing `[calibration]` section, keyed by experiment:

```toml
[calibration.active]
exp1 = "charuco-all_20260901-101500"
"20260619" = "refine_20261005-142210"    # all-digit names quoted

[calibration.triangulated_with]          # 11.5
exp1 = "charuco-all_20260901-101500"
```

- `Project.calibration_path(experiment=None)` resolves through the pointer and returns `None` when the pointer is unset or the file is missing.
- `has_calibration` comes from the pointer.
- `_read_camera_info()` reads the active file.
- Callers that assume a path exists need to handle `None`: `label_view.py`, `main_window.py` (Triangulate all experiments) and the calibration view's reload.
- **New method:** `Project.set_active_calibration(experiment, id)`. It writes the pointer and calls `reload_calibration()`.
- **New method:** `Project.set_triangulated_with(experiment, id)`.
- `save_config_sections()` replaces whole top-level sections. The pointer writers copy `[calibration]`, change their sub-table and save the copy, as `CalibrationView._save_settings()` already does, so `frame_skip`, `repeats` and the drift thresholds are kept.

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

def list_calibrations(exp_dir: Path) -> list[CalibrationEntry]      # newest first, both folders
def load_calibration(entry) -> CameraGroup
def save_calibration(exp_dir, cgroup, label, method, **metadata) -> CalibrationEntry
def rename_calibration(entry, label) -> CalibrationEntry
def delete_calibration(exp_dir, entry) -> list[CalibrationEntry]   # also deletes descendants
def import_legacy(exp_dir) -> list[CalibrationEntry]
```

- `save_calibration` writes `refine` entries to `refined/` and all others to the top folder. It gives the group a new `metadata` dict rather than editing the one it has. `CameraGroup.__init__` has a mutable `metadata={}` default, so editing it in place would leak into other groups.
- `delete_calibration` removes the entry and every entry descended from it through `parent`, along with their `_points.csv` files. It returns everything it removed, so the caller can confirm first.

**Legacy import:** `import_legacy()` imports `calibration.toml` and `calibration_takes/*.toml` as `method="imported"`.
- It runs once, when the registry folder is missing, and leaves the old files where they are.
- `created` is each file's modification time.
- It sets `[calibration.active]` to the entry imported from `calibration.toml`. It also sets `[calibration.triangulated_with]` to a placeholder (`legacy`), so the Label tab shows the re-triangulate warning (11.5) once.

## 11.4 Writers

**Calibration tab: calibrating** ([calibration_view.py](../../src/label3d/widgets/calibration_view.py), `_save_take_calibrations`)
- When a run finishes, one `charuco` entry is saved per result: one per take with `charuco_takes = [take]`, and the joint solution with all the takes used. Every run's entries are kept.
- The joint entry becomes active by default.
- If the active entry is a `refine` entry, Calibrate first warns that the new joint solution will replace it as the active calibration. On confirmation it proceeds. The refined entry stays in the registry.
- **Repeat runs:** each result is the best of its repeated runs, and `metadata.errors` records its error and the run-to-run spread.

**Calibration tab: Use take** (`_on_use_take`)
- Only sets the pointer to the chosen result's entry. It doesn't write a file.

**Corners** (`calibration_io.py`)
- One `corners.csv` per camera holds every take's detections. It replaces `corners.csv` plus `corners_{take}.csv`.
- **Columns:** `take, frame, corner_id, x, y, detected`. `take` and `frame` replace the `"{take}:{frame}"` `source_key`. `detected` is the timestamp of the detection run.
- Re-detecting a take replaces that take's rows only.
- `load_corners(cam_dir, takes=None)` filters by take. This replaces `rows_for_take()` on the file path.
- Every `charuco` entry records the takes, board, frame skip, take ranges and `detected` timestamp (11.1). Together with the corners file, that's enough to rerun the solve. If a take is re-detected later, a mismatched `detected` timestamp shows that the entry used older corners.
- On load, the Calibration tab computes reprojection errors from the corners of the active entry's `charuco_takes`. For a `refine` entry, the takes are its parent's.
- Existing `corners.csv` and `corners_{take}.csv` files are read once and rewritten in the new layout. Nothing else is migrated.

**Project setup** ([setup_project.py:878](../../src/label3d/setup_project.py#L878))
- When setup from SLP files finds an existing aniposelib calibration (`assignment.calibration_toml`), it currently copies it to `experiments/{exp}/calibration.toml`. Instead it saves an `imported` entry with `metadata.source_path` set to the original path, and sets the pointer.
- `setup_project_cli.py`'s help text and the summary lines in `setup_project.py` are updated to match.

**Synthetic data**
- `src/synthetic/writers.py` (`write_calibration_toml`) and `generate_synthetic.py` write the ground-truth calibration as a registry entry (`method = "imported"`, label `ground-truth`) and set the pointer. They also set `triangulated_with`, since the generator writes the true 3D points.
- `src/synthetic/checks.py` loads it through the pointer.

**Tests:** regenerate the synthetic test project (`projects/test_project`, gitignored; built by `generate_synthetic.py`). Update the tests that build or read `calibration.toml` or the corners files directly: `test_project_loading`, `test_project_sync`, `test_setup_from_slp` and `test_calibration_takes`.

**Alignment** (`geometry.save_aligned_calibration`): the rotation and origin move from a top-level `[alignment]` table to `[metadata.alignment]`. The function takes the parent id and saves through the registry as an `align` entry. Nothing calls it until Phase 14.

**Refinement:** Phase 12 saves `refine` entries.

## 11.5 Switching in the Label tab (`label_view.py`, `main_window.py`)

**The combo box:** a **Calibration** combo in the Label tab toolbar lists the registry, newest first, as `label · YYYY-MM-DD HH:MM`. The selected item is the active entry. Refined entries are indented under their parent, so you can switch between a charuco solution and its refinements, or between refinements that used different points. The tooltip shows method, parent, takes, point counts and errors. The combo is refilled when the experiment changes.

**What changing it does:**
1. `set_active_calibration()`, which writes the pointer.
2. Rebuild the `EpipolarCache` and redraw the overlays (the existing `reload_calibration()` path).
3. Re-triangulate the current frame, then the current trial in the background (`TriangulationWorker`), so `xr/yr` and the error colours update. Point edits are blocked until the current trial is done, because the worker writes into the live store.
4. Re-triangulate the experiment's other trials in the background, one worker at a time, as `_run_next_triangulation` does now. The status bar shows progress.

**Recording which calibration was used:**
- `set_triangulated_with()` stores the id per experiment once a full re-triangulation finishes. **Instances > Triangulate all experiments** does the same for each experiment it finishes. Its messages refer to the active calibration instead of `calibration.toml`.
- If `triangulated_with` differs from the experiment's entry in `[calibration.active]` (an interrupted run, a switch made in the Calibration tab, or a legacy import), the Label tab shows "3D points are from <label>; re-triangulate" in the status bar and offers **Instances > Triangulate all experiments**.

**The Calibration tab:** add a read-only registry list (label, date, method, error), with refined entries under their parent. It has three buttons:
- **Make active.**
- **Rename:** changes the label only.
- **Delete:** the confirmation lists the refined entries that will be deleted with it. The active entry, and any entry whose descendants include the active one, can't be deleted.

## 11.6 Tests (`tests/test_calibration_registry.py`)

- **Registry:**
  - `save_calibration` → `list_calibrations` → `load_calibration` round trip, keeping the metadata.
  - `CameraGroup.load()` works on a saved file.
  - `refine` entries go to `refined/` and are listed with the others.
  - Ids stay unique when a label is saved twice in the same second; a counter suffix is added if needed.
  - Rename keeps the id and file name.
  - Delete removes descendants and their `_points.csv`.
  - Saving one group doesn't change another group's `metadata`.
- **Import and pointer:**
  - `import_legacy` on a project with `calibration.toml` and two take files sets the pointer and doesn't touch the originals.
  - `calibration_path()` returns `None` with no pointer.
  - Writing a pointer keeps `frame_skip` and the drift thresholds.
- **Corners:** the single-file round trip, filtering by take, re-detecting one take, and reading the old per-take files.
- **Switching:** gives different `xr/yr` for a perturbed calibration. This reuses the two-camera project in `test_project_sync.py`.

## 11.7 Reference docs

- **`docs/data_model.md`:**
  - The registry layout (including `refined/`), the `[metadata]` fields, `[calibration.active]` and `[calibration.triangulated_with]`.
  - The new `corners.csv` columns.
  - Remove `calibration.toml`, `calibration_takes/` and `corners_{take}.csv`.
- **`docs/workflows.md`:** the Label tab calibration combo and the Calibration tab registry list.
- **CLAUDE.md:** the `calibration.toml` line under "Data files".
- **`docs/project_setup_from_sleap.md`:** calibration discovery now creates an `imported` registry entry.

## Implementation order

1. 11.3 registry module + 11.6 tests
2. 11.2 pointer + caller fixes for `None`
3. 11.4 corners file + tests
4. 11.4 writers: Calibration tab, project setup, synthetic generator, alignment, tests
5. 11.5 Label tab combo box, re-triangulation and stale-3D warning; Calibration tab registry list
6. 11.7 reference docs

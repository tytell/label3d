# Phase 8 — SLP Write-back

**Goal**: Write current labels from `PointsStore`/`points.csv` back to the original SLP files in `project/slp/` on explicit save; prompt on close if there are unsaved changes.

**Design principle**: SLP files are synchronized copies updated on demand. Auto-save (every 30 s) writes only `points.csv`. SLP write-back happens only on explicit **File > Save**. User-added instances (Phase 7.6–7.8) live in `points.csv` like any other user instance and are included automatically — no special handling needed.

## 8.1 `SleapAdapter.build_file_map()` (`adapters/sleap_adapter.py`)

New stateless method: given `files: list[Path]`, `trials_df`, and `cameras`, reads each SLP file's embedded video path, matches via the existing `_build_basename_lookup()` logic, and returns a `dict[tuple[str, str, str], Path]` mapping `(experiment, trial, camera) → file_path`. Called at save time — nothing needs to be tracked at load time.

Also update `SleapAdapter.save()`:
- Add `file_map: dict[tuple, Path] | None = None` parameter
- When provided, route each `(experiment, trial, camera)` group back to its original path instead of generating `{camera}_{trial_name}.slp` names
- Make `output_dir` optional (kept for Phase 10 Export, which will still generate new names)
- Return `int` (count of files written) instead of `None`

## 8.2 `PointsStore.load_all_experiments()` (`points.py`)

New classmethod:
```python
@classmethod
def load_all_experiments(cls, project: Project) -> list[PointsStore]:
```
Iterates `project.experiments`, loads each experiment's `points.csv` if it exists, returns the list of stores (skips experiments with no CSV).

## 8.3 `Project.sync_slp_files()` (`project.py`)

```python
def sync_slp_files(self) -> int:
    """Flush points.csv and write current labels back to the SLP files in project/slp/.

    Returns the number of SLP files written.
    """
```

Steps:
1. `self.points.save_if_dirty()` — flush active experiment's CSV first
2. `slp_files = list(self.slp_dir.glob("*.slp"))`
3. `file_map = SleapAdapter().build_file_map(slp_files, self.trials_df, self.cameras)`
4. `all_stores = PointsStore.load_all_experiments(self)`
5. `all_df = pd.concat([s.df for s in all_stores])`
6. `return SleapAdapter().save(all_df, cameras=self.cameras, skeleton=self.skeleton, trials_df=self.trials_df, file_map=file_map)`

**State mapping in `SleapAdapter.save()`:**

| Per-camera states | Exported SLEAP type |
|---|---|
| Any camera has `set` or `not_visible` | `Instance` (user-ground-truth) |
| All cameras have `predicted` | `PredictedInstance` |
| All cameras have `not_set` | Instance omitted |

- [ ] Add unit test: create `PointsStore` with user-added instance; call `SleapAdapter.save()`; reload via `sleap_io.load_file()`; assert appears as `Instance` with correct coordinates

## 8.4 File > Save (`main_window.py`)

Add **File > Save** menu action (Ctrl+S):
- Enabled only when `points_store` is not None and `slp_dir` contains `.slp` files
- Run `self._project.sync_slp_files()` in a `SaveWorker(QThread)` (mirrors existing `ImportWorker` pattern)
- Show a `QProgressDialog` during the write
- On success: status bar message "Saved — wrote {n} SLP files"

## 8.5 On close — unsaved-changes prompt (`main_window.py`)

```python
if self._state.points_store and self._state.points_store.changed:
    reply = QMessageBox.question(
        self, "Save before closing?",
        "You have unsaved changes. Save and update SLP files before closing?",
        QMessageBox.StandardButton.Save
        | QMessageBox.StandardButton.Discard
        | QMessageBox.StandardButton.Cancel,
    )
```

- **Save** → call `sync_slp_files()` on main thread (brief block acceptable on close), then close
- **Discard** → close without saving
- **Cancel** → `event.ignore()`

## Data flow

```
Project open
  ↓  points.csv empty AND slp/ has *.slp?
       → QMessageBox("Import SLP predictions?")
       → SleapAdapter.load() → PointsStore.merge_from_df()

[User labels frames]
  ↓  auto-save every 30 s (points.csv only)

File > Save (Ctrl+S)
  ↓  points_store.save_if_dirty() → points.csv
  ↓  SleapAdapter.build_file_map() → (experiment, trial, camera) → file_path
  ↓  SleapAdapter.save(file_map=...) → original slp/*.slp files

App close (unsaved changes)
  ↓  QMessageBox("Save before closing?")
```

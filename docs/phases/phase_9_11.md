# Phases 9–11

---

## Phase 9 — Export

**Goal**: Export labeled data back to external formats.

### 9.1 Export view (`widgets/export_view.py`)
- [ ] Format dropdown (populated from adapter registry)
- [ ] Trial selector (all / specific)
- [ ] Instance type checkboxes (user / predicted)
- [ ] Output directory browser
- [ ] Summary: trial count, camera count, labeled frames, point counts by state
- [ ] "Triangulate All Before Export" checkbox (default on)
- [ ] "Export" button → batch triangulation if needed → `adapter.save()`
- [ ] Progress bar during export
- [ ] Experiment selector (all / specific); "All" uses `PointsStore.load_all_experiments()` for cross-experiment export

---

## Phase 10 — SLP Write-back

**Goal**: Write current labels from `PointsStore`/`points.csv` back to SLP files in `project/slp/` on explicit save; prompt on close if dirty.

**Design principle**: SLP files are synchronized copies updated on demand. Auto-save (every 30 s) writes only `points.csv`. SLP write-back happens only on explicit **File > Save**.

### 10.1 `Project.sync_slp_files()` (`project.py`)

```python
def sync_slp_files(self) -> int:
    """Flush points.csv and write current labels back to project/slp/*.slp.

    Returns the number of SLP files written.
    """
```

Steps:
1. `self.points.save_if_dirty()` — flush active experiment's CSV first
2. Load all experiments' DataFrames via `PointsStore.load_all_experiments(self)`
3. `all_df = pd.concat([store.df for store in all_stores])` — aggregate
4. `SleapAdapter().save(all_df, self.slp_dir, self.cameras, skeleton=self.skeleton, trials_df=self.trials_df)`
5. Return count of files written

**File naming convention:** `{cam}_{trial_name}.slp`, written to `project/slp/`.

### 10.2 File > Save (`main_window.py`)

Enable the **File > Save** menu action (currently a disabled stub):
- Run `self._project.sync_slp_files()` in a `QThread`
- Show a `QProgressDialog` during the write
- Status bar on success: "Saved — wrote {n} SLP files to {slp_dir}"

### 10.3 On close — dirty prompt (`main_window.py`)

```python
if self._state.points_store and self._state.points_store.dirty:
    reply = QMessageBox.question(
        self, "Save before closing?",
        "You have unsaved changes. Save and update SLP files before closing?",
        QMessageBox.StandardButton.Save
        | QMessageBox.StandardButton.Discard
        | QMessageBox.StandardButton.Cancel,
    )
```

### 10.4 Export of user-added instances

Instances created via Phase 7.6–7.8 live in `points.csv` exactly like any other user instance — included automatically in `sync_slp_files()`.

**State mapping in `SleapAdapter.save()`:**

| Per-camera states | Exported SLEAP type |
|---|---|
| Any camera has `set` or `not_visible` | `Instance` (user-ground-truth) |
| All cameras have `predicted` | `PredictedInstance` |
| All cameras have `not_set` | Instance omitted |

- [ ] Add unit test: create `PointsStore` with user-added instance; call `SleapAdapter.save()`; reload via `sleap_io.load_file()`; assert appears as `Instance` with correct coordinates

### Data flow

```
Project open
  ↓  points.csv empty AND slp/ has *.slp?
       → QMessageBox("Import SLP predictions?")
       → SleapAdapter.load() → PointsStore.merge_from_df()

[User labels frames]
  ↓  auto-save every 30 s (points.csv only)

File > Save (Ctrl+S)
  ↓  points_store.save_if_dirty() → points.csv
  ↓  SleapAdapter.save() → project/slp/{cam}_{trial}.slp

App close (dirty)
  ↓  QMessageBox("Save before closing?")
```

---

## Phase 11 — Polish and Robustness

- [ ] **Undo/redo**: Command pattern — each point edit creates `UndoCommand(trial, frame, instance, bodypart, camera, old_value, new_value)` on `QUndoStack`; `Ctrl+Z`/`Ctrl+Shift+Z`
- [ ] **Auto-save**: `PointsStore` writes `points.csv` every 30s if dirty + on app close; backup to `.points.csv.bak` before overwriting
- [ ] **Seekbar coloring**: color each frame tick by worst reprojection error (green/yellow/red/grey)
- [ ] **Suggested frames**: uniformly spaced unlabeled frames shown as ticks; `Space`/`Shift+Space` navigate to them
- [ ] **Progress indicators**: "42% of frames labeled" per trial
- [ ] **Frame pre-caching**: on step to frame N, pre-decode N+1 through N+3 in a background thread
- [ ] **Settings persistence** (`QSettings`, INI format):
  - App-level (`~/.config/label3d/label3d.ini`): window geometry, last-opened project path
  - Project-level (`.label3d_session.ini` in project root, gitignored): current trial/frame, camera layout, per-camera zoom/pan
- [ ] **Calibration frame extraction**: `setup_project` extracts every Nth frame from calibration videos as image stills
- [ ] **Toolbar and zoom UX**:
  - `QToolBar` with standard actions: Open Project, Save, Zoom In/Out, Fit View
  - **Drag-box zoom**: rubber-band rectangle on right-click-drag → `fitInView()`
  - Keyboard: `Ctrl+=`/`Ctrl+-` zoom in/out; `F` fit to frame; `G` zoom all panels to selected instance; `P` zoom active panel to selected point

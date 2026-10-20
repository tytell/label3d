# Phases 16–17

---

## Phase 16 — Export

**Goal**: Export labeled data back to external formats.

### 16.1 Export to other pose formats (`widgets/export_view.py`)

There is no Export tab any more (removed in Phase 13). This would be a dialog reached from the File menu, like **Export points...**.

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

## Phase 17 — Polish and Robustness

- [ ] **Undo/redo**: Command pattern — each point edit creates `UndoCommand(trial, frame, instance, bodypart, camera, old_value, new_value)` on `QUndoStack`; `Ctrl+Z`/`Ctrl+Shift+Z`
- [ ] **Auto-save**: `PointsStore` writes `points.csv` every 30s if dirty + on app close; backup to `.points.csv.bak` before overwriting
- [ ] **Seekbar coloring**: color each frame tick by worst reprojection error (green/yellow/red/grey)
- [ ] **Suggested frames**: uniformly spaced unlabeled frames shown as ticks; `Space`/`Shift+Space` navigate to them
- [ ] **Progress indicators**: "42% of frames labeled" per trial
- [ ] **Frame pre-caching**: on step to frame N, pre-decode N+1 through N+3 in a background thread
- [x] **Settings persistence** (`QSettings`, INI format):
  - App-level (`~/.config/label3d/label3d.ini`): window geometry, last-opened project path
  - Project-level (`.label3d_session.ini` in project root, gitignored): current trial/frame, camera layout, per-camera zoom/pan
- [ ] **Toolbar and zoom UX**:
  - `QToolBar` with standard actions: Open Project, Save, Zoom In/Out, Fit View
  - **Drag-box zoom**: rubber-band rectangle on right-click-drag → `fitInView()`
  - Keyboard: `Ctrl+=`/`Ctrl+-` zoom in/out; `F` fit to frame; `G` zoom all panels to selected instance; `P` zoom active panel to selected point

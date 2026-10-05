# Phase 8 — SLP Write-back

**Goal**: Keep `points.csv` and the SLP files in `project/slp/` in sync. The user assigns (experiment, trial, camera) cells to SLP files with **File > Set up Sleap files...**, then runs **File > Sync with Sleap...** to read new labels/predictions from the SLP files and write the merged result back. Prompt on close if there are unsynced changes.

**Design principle**: SLP files are synchronized copies updated on demand. Auto-save (every 30 s) writes only `points.csv`. SLP files are read at startup and read + written on explicit **Sync**. User-added instances (Phase 7.6–7.8) live in `points.csv` like any other user instance and are included automatically — no special handling needed.

**Expected workflow**: label in label3d → Sync → close label3d → train and predict in SLEAP (writing into the same SLP files) → reopen label3d, which reads the SLP files on startup. Because the round trip is this simple, sync uses a plain comparison of SLP contents vs `points.csv` — no sync history or snapshots.

Steps 8.1–8.5 below are done (first pass, one SLP file found per video by scanning `slp/`). Steps 8.6–8.13 replace that scan with an explicit mapping and add two-way sync.

## 8.1 `SleapAdapter.build_file_map()` (`adapters/sleap_adapter.py`) ✓

Stateless method: given `files: list[Path]`, `trials_df`, and `cameras`, reads each SLP file's embedded video references via `labels.videos`, matches via `_build_basename_lookup()`, and returns `dict[tuple[str, str, str], Path]` mapping `(experiment, trial, camera) → file_path`. Called at save time — nothing tracked at load time.

`SleapAdapter.save()` also updated:
- `file_map: dict[tuple, Path] | None = None` parameter added
- `output_dir` made optional (kept for the Phase 15 export view)
- Returns `int` (count of files written) instead of `None`
- Instance state logic corrected: all-`not_set` instances omitted; `not_visible` treated as user ground-truth (replaces incorrect `occluded` check)

## 8.2 `PointsStore.load_all_experiments()` (`points.py`) ✓

New classmethod iterates `project.experiments`, loads each experiment's `points.csv` via `from_csv()` if it exists, skips experiments with no CSV. Also in this step:

- `dirty` renamed to `changed` throughout (`_dirty` → `_changed`, `save_if_dirty()` → `save_if_changed()`, `_mark_dirty()` → `_mark_changed()`)
- Qt signal renamed `changed` → `data_changed` to free the name for the property; `connect`/`disconnect` calls updated in `main_window.py` and `state.py`

## 8.3 `Project.sync_slp_files()` (`project.py`) ✓

```python
def sync_slp_files(self) -> int:
    """Flush points.csv and write current labels back to the SLP files in project/slp/.

    Returns the number of SLP files written.
    """
```

Steps:
1. `self.points.save_if_changed()` — flush active experiment's CSV first
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

## 8.4 File > Save (`main_window.py`) ✓

`SaveWorker(QThread)` in `workers/save_worker.py` calls `sync_slp_files()` and emits `finished(n)` or `error(msg)`. File > Save (Ctrl+S) in the menu launches the worker behind a `QProgressDialog`; status bar shows "Saved — wrote N SLP file(s)". Action enabled at startup only when `slp/` contains `.slp` files.

## 8.5 On close — unsaved-changes prompt (`main_window.py`) ✓

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

- **Save** → `sync_slp_files()` on main thread (brief block acceptable on close), then close
- **Discard** → close without saving
- **Cancel** → `event.ignore()`
- Falls back to `save_if_changed()` (points.csv only) if no SLP files exist

## 8.6 SLP file mapping in `project.toml` ✓

Each SLP file is made of one or more **blocks**. A block is one experiment × a list of trials × a list of cameras. With blocks, files can be split and joined while `project.toml` stays readable.

```toml
[[slp.files]]
path = "slp/cam0_cam1_early.slp"     # relative to project root
[[slp.files.blocks]]
experiment = "exp1"
trials = ["t1", "t2", "t3"]
cameras = ["cam0", "cam1"]

[[slp.files]]
path = "slp/cam0_cam1_late.slp"
[[slp.files.blocks]]
experiment = "exp1"
trials = ["t4", "t5"]
cameras = ["cam0", "cam1"]

[[slp.files]]
path = "slp/cam2.slp"
[[slp.files.blocks]]
experiment = "exp1"
trials = ["t1", "t2", "t3", "t4", "t5"]
cameras = ["cam2"]
```

New module `slp_sync.py` (no Qt):

```python
@dataclass
class SlpBlock:
    experiment: str
    trials: list[str]
    cameras: list[str]

@dataclass
class SlpFileSpec:
    path: Path                 # relative to project root
    blocks: list[SlpBlock]

class SlpMapping:
    files: list[SlpFileSpec]

    @classmethod
    def from_config(cls, config: dict) -> SlpMapping
    def to_config(self) -> dict
    def cell_map(self) -> dict[tuple[str, str, str], Path]   # replaces build_file_map() for sync
    def missing_cells(self, project) -> list[tuple[str, str, str]]  # cells in trials.csv with a video but no file
    def duplicate_cells(self) -> list[tuple[str, str, str]]         # cells assigned to >1 file
    def is_complete(self, project) -> bool                          # no missing, no duplicates
```

- `Project.slp_mapping` property reads `config["slp"]`; `Project.save_slp_mapping(mapping)` writes it back to `project.toml`, keeping all other sections unchanged.
- A cell exists only where `trials.csv` has a non-empty video for that camera. Trials never share a video (start/end frames only trim a video), so each cell corresponds to exactly one video file.
- `build_file_map()` (8.1) is kept only to create a first mapping from existing SLP files (see 8.9, "Adopt existing").

**As implemented:**
- `missing_cells()` and `is_complete()` take `(trials_df, cameras)` instead of `project`, so they can be tested without a `Project`.
- Added `video_cells(trials_df, cameras) -> {cell: (video filename, start frame)}`, the single source for which cells exist. It looks up start frames by `(experiment, trial)`.
- Added `compact_blocks(cells)` and `SlpMapping.from_cell_map(cell_map, cell_order)`. Trials with the same camera set share a block. "Adopt existing" (8.9) will use these.

## 8.7 `SleapAdapter` fixes for multi-video files ✓

The current `save()` and `load()` assume one SLP file per (trial, camera). Changes:

1. **Group writes by file.** `save(df, mapping=..., ...)` loops over `SlpFileSpec`. Each file becomes one `sio.Labels` with one `sio.Video` per cell, and all labeled frames for those videos. Today, a file shared by several cameras or trials gets overwritten once per group, so only the last group is kept.
2. **Keep the rest of an existing file.** If the file exists, load it and keep its videos (matched by basename), suggestions and provenance. Replace only `labeled_frames`, and add or remove videos to match the mapping. Frames for videos no longer in this file are dropped here; the new owning file writes them.
3. **Frame offset.** Write `frame_idx = frame + cam{i}_start`, which is the inverse of `load()`.
4. **Match experiment as well as trial.** In both `load()` and `save()`, look up the `trials_df` row by `(experiment, trial)`, not by `trial` alone.
5. **Video path.** Set `sio.Video.filename` to the full path `experiments/{exp}/videos/{file}`, so SLEAP can open the video. Matching on load still uses the basename.
6. **Empty files.** Write a file even when it has no labeled frames yet, so new files contain their videos and skeleton.
7. `load(mapping=...)` reads only the files in the mapping. It ignores videos in a file that aren't assigned to that file, and logs a warning.

**As implemented:**
- The signatures are `save(df, cameras, skeleton, trials_df, file_map=None, output_dir=None, mapping=None, root=None)` and `load(files, trials, cameras, mapping=None, root=None)`. `trials_df` is now required. The `file_map` and `output_dir` modes also write each file once.
- **Instance numbering.** For each camera, `save()` writes only the instances that have a finite `set` or `predicted` point there (`slp_sync.written_instances()`), in index order. `load()` skips instances with no finite points and numbers the rest 0, 1, … for each frame and camera. Merge (8.8) uses this to match instances back.
- **Missing points.** `not_visible` points are written as NaN, which is SLEAP's "not visible". Predicted nodes with no coordinates load as `not_set`.
- **Video paths.** A video already in an existing file keeps its stored path (matched by basename). Only newly added videos get the `experiments/{exp}/videos/` absolute path.
- **Atomic write.** Files are written to `{name}.tmp.slp`, then renamed over the original.
- `import sleap_io` moved to the top of the module.

## 8.8 Merge rules — `slp_sync.compute_merge()` ✓

Pure function, safe to run in a background thread:

```python
@dataclass
class MergeResult:
    df: pd.DataFrame           # store df with non-conflicting SLP data applied
    df_overwrite: pd.DataFrame # same, plus conflicting cells taken from SLP
    n_new: int                 # cells filled where the store had nothing
    n_predictions: int         # predicted cells replaced/added
    conflicts: list[tuple]     # (experiment, trial, frame, instance, bodypart, camera)

def compute_merge(store_df: pd.DataFrame, slp_df: pd.DataFrame) -> MergeResult
```

Merge is done **per camera cell** `(experiment, trial, frame, instance, bodypart, camera)`. The current `merge_from_df()` skips a whole bodypart row if any camera has user data, which would block a new SLEAP label in another camera.

| Store state | SLP state | Result |
|---|---|---|
| missing / `not_set` | any | take SLP (new) |
| `predicted` | `predicted` | take SLP (predictions replace older predictions) |
| `predicted` | `set` | take SLP (user beats prediction) |
| `set` / `not_visible` | `predicted` | keep store (user beats prediction) |
| `set` / `not_visible` | `not_set` | keep store (SLP can't store `not_visible`; deletions in SLEAP are not detected) |
| `set` / `predicted` | `set`, same x/y (±0.01 px) | no change (a predicted point in a user instance is saved as part of a user `Instance` and reloads as `set`) |
| `set` | `set`, different x/y | **conflict** |
| `not_visible` | `set` | **conflict** |

Frame-level rule: on a (frame, camera) where the store has any user instance, SLP *predicted* instances are ignored completely. Otherwise a prediction for an animal that's already labeled would come in as an extra instance.

**Instance matching.** Within each frame and camera, the n-th SLP instance maps to the n-th store instance that `save()` writes for that camera. SLP instances beyond those map next to store instances that have no data in that camera, sorted by index, and after that to new indices starting at the frame's max + 1. This way a round trip gives back the same instance numbers even when an instance is missing from one camera. A new SLEAP label in a camera that had no labels joins the existing animal, and new instances line up across cameras. Known limitation: if SLEAP reorders or deletes user instances, the result can be wrong.

After merging, re-run triangulation for the frames that changed. Use the same path as the current import. `MergeResult.changed_frames` and `conflict_frames` list those frames.

`PointsStore.merge_from_df()` is replaced by `PointsStore.replace_df(df)`, which swaps the data, clears caches and marks the store changed.

**As implemented:** `replace_df()` is added. `merge_from_df()` stays until 8.11 removes its last caller (the old import flow). `compute_merge()` converts index levels to str/int so trial names parsed as ints from `points.csv` still match.

## 8.9 File > Set up Sleap files... (`widgets/slp_setup_dialog.py`) ✓

A modal dialog that edits an `SlpMapping` in memory. Nothing is written until **Apply**.

**Layout**
- Experiment selector at the top (blocks belong to one experiment, but one file can have blocks from several experiments).
- **Assignment grid**: rows = trials, columns = cameras. Each cell shows the file it's assigned to, with one color per file. Unassigned cells are hatched. Cells can be multi-selected (click, drag, shift/ctrl, click a row or column header).
- **File list** on the right: file name (editable), cell count, color swatch.

**Actions**
| Action | Effect |
|---|---|
| Preset ▾ | Replaces the whole mapping: *one file per camera*, *one file per camera per experiment*, *one file per trial × camera*, *one file for everything*. Filename pattern editable, e.g. `{camera}.slp`, `{experiment}_{camera}.slp`. |
| Adopt existing | Builds a first mapping from the SLP files already in `slp/` using `build_file_map()`. Offered automatically when `slp/` has files but `project.toml` has no `[slp]` section. |
| Split by camera | For each selected file: one new file per camera. |
| Split by trial... | For the selected file: move the selected trials (grid rows) into a new file. |
| New file from selection | Move the selected grid cells into a new file. This is the general split. |
| Join | Merge the selected files into one; the first file's name is kept. |
| Assign to ▸ | Move the selected cells into an existing file. |

The user's example works like this: **Preset → one file per camera** (3 files). **Join** cam0 + cam1. Select the rows for trials 4–5 and **Split by trial...**, giving files for cam0+cam1 × t1–3, cam0+cam1 × t4–5, and cam2.

Blocks are compacted after each edit: cells are grouped into the fewest (trials × cameras) rectangles per experiment. A file whose shape isn't a rectangle just keeps several blocks.

**Validation.** Apply is disabled until every cell has exactly one file and file names are unique and end in `.slp`. The status line shows the problem, e.g. "4 cells unassigned".

**Apply** (runs in `SlpSetupWorker` behind a progress dialog):
1. `points.save_if_changed()`.
2. Copy every existing `slp/*.slp` to `slp/backup/{YYYYmmdd-HHMMSS}/`. Existing files are never deleted without a backup.
3. **Read before rewriting**: read the *old* mapping's files (or all `slp/*.slp` if there was no mapping) and merge them into `points.csv` using 8.8. Labels made in SLEAP but never synced are then not lost. Show the conflict warning from 8.10 if needed.
4. `Project.save_slp_mapping(new_mapping)`.
5. Write every file in the new mapping from `points.csv` for all experiments (8.7). Files in `slp/` that are no longer in the mapping are removed; their copies stay in the backup folder.
6. Status bar: "Set up N SLP file(s); backup in slp/backup/…". Update whether Sync is enabled.

**As implemented:**
- **Editing logic** lives in `slp_sync.SlpMappingEditor`, which has no Qt dependency: `apply_preset`, `move_cells`, `new_file`, `split_by_camera`, `split_by_trial`, `join`, `rename`, `unique_path`, `errors`. The dialog only calls these and redraws. Cells in `project.toml` whose video is no longer in `trials.csv` are dropped when the dialog opens.
- **Naming.** New files get automatic names: `new.slp` from a selection, `{stem}_2.slp` from split by trial, and `{stem}_{camera}.slp` from split by camera. Rename them in the file list. Names that differ only by case are reported, because macOS file systems ignore case.
- **Split by trial** moves only the trials selected in the grid for the experiment shown. Rows from other experiments stay where they are.
- **File list.** It shows names only, since the text is editable. The video count is in the tooltip.
- **Adopt existing** runs automatically when the dialog opens if `project.toml` has no `[slp]` section and `slp/` has files.
- **Apply order:**
  1. The read and merge run first (`SlpReadWorker`).
  2. The conflict warning appears.
  3. `apply_merge()` updates the points and `project.toml` is saved.
  4. `SlpWriteWorker` backs up `slp/*.slp`, writes the files, then removes unmapped files.

  The backup happens after the warning rather than first, so Cancel leaves nothing behind.
- **New `Project` methods, reused by 8.10:** `experiment_points_df()`, `read_slp_for_sync(mapping)`, `apply_merge(results, overwrite)`, `write_slp_files(mapping)`, `backup_slp_files()` and `remove_unmapped_slp_files(mapping)`. `PointsStore` gained `empty_df()` and `read_csv_df()`, so other experiments' points can be read without creating a QObject. Workers are in `workers/slp_sync_worker.py`; they use a `done` signal instead of overriding `QThread.finished`.
- **Save stays enabled** whenever `slp/` has files after setup. 8.10 replaces this with the mapping-complete rule.
- **Not done:** re-triangulating the merged frames. The current import path doesn't triangulate either, so this is left for 8.10/8.11.

## 8.10 File > Sync with Sleap... (`main_window.py`) ✓

Rename **File > Save** to **Sync with Sleap...** (keep Ctrl+S). Remove **Import from SLP files…**, since startup read and Sync now do the same job.

**Enabled** only when `project.slp_mapping.is_complete(project)`. Re-check after Set up, after switching experiments, and after reloading `trials.csv`. The disabled tooltip says why, e.g. "3 videos have no SLP file — use Set up Sleap files...".

**Sequence** (read, then write):
1. `points.save_if_changed()`.
2. `SyncReadWorker`: load all mapped files → `slp_df`; for each experiment, `compute_merge(store_df, slp_df_for_experiment)`. Emits `list[MergeResult]`.
3. Main thread: if any result has conflicts, show **one** warning:
   > "N points in M frames were changed in the SLEAP files and differ from label3d.
   > **Use SLEAP** — overwrite label3d points with the SLEAP versions.
   > **Keep label3d** — keep label3d points; SLEAP files will be overwritten.
   > **Cancel** — change nothing."
4. Apply `df` or `df_overwrite` for each experiment. The active experiment goes through `PointsStore.replace_df()`; other experiments are written directly to their `points.csv`. Re-triangulate the changed frames.
5. `SyncWriteWorker` (the renamed `SaveWorker`): write all mapped files from the merged data (8.7).
6. Status bar: "Synced — read N new, M predictions; wrote K SLP file(s)".

A mapped file that doesn't exist on disk is treated as empty on read and created on write, with a warning in the log.

`Project.sync_slp_files()` (8.3) is split into `read_slp_for_sync() -> list[MergeResult]` and `write_slp_files() -> int`, so the conflict prompt can run on the main thread between them.

**As implemented:**
- The read and write reuse `SlpReadWorker` / `SlpWriteWorker` from 8.9; `SaveWorker` and `ImportWorker` are deleted, along with `Project.sync_slp_files()`, `import_from_slp()`, `load_slp_adapter_df()`, `merge_slp_adapter_df()`, `slp_import_available` and `PointsStore.merge_from_df()`.
- **Re-triangulation** runs *after* the write, not before it, so the background write and the triangulation workers never touch the DataFrame at the same time.  3D coordinates are not written to SLP, so the file contents are unaffected.  Only the **active** experiment is re-triangulated (`_retriangulate`), because calibration is loaded for the active experiment alone; `_run_next_triangulation` walks one `TriangulationWorker` per trial.  Other experiments are re-triangulated with **Instances > Triangulate all experiments...**.
- `_update_sync_action()` sets the enabled state and the disabled tooltip; it runs at menu setup, after Set up, and on an experiment switch.
- `tests/test_import.py` becomes `tests/test_slp_load.py`: the `merge_from_df` and `slp_import_available` tests are gone, the adapter state tests stay, and the merge test now goes through `compute_merge()`.

## 8.11 Startup read (`main_window.py`) ✓

When a project opens (and when switching experiments), if the mapping is complete, run steps 1–4 of 8.10 without the write. Conflicts show the same single warning; **Keep label3d** leaves the SLP files as they are until the next Sync. If anything was merged, the status bar says "Loaded N new labels / M predictions from SLP files".

This replaces `_maybe_offer_slp_import()`, the "Import SLP predictions?" prompt. If the mapping is missing or incomplete and `slp/` has files, show a one-time message instead: "SLP files found but not assigned — open Set up Sleap files...?"

**As implemented:** `_read_slp_on_open()` runs after the session is restored and after an experiment switch.  It stays silent when nothing was merged — the status bar only reports a non-zero read.  `_offer_slp_setup()` asks once per session (`_offered_slp_setup`) and opens the Set up dialog on Yes.

## 8.12 On close — update 8.5 ✓

The prompt becomes "Sync with Sleap before closing?" with Sync / Discard / Cancel. **Sync** runs read + write on the main thread; a conflict still shows the warning. If the mapping is incomplete, fall back to `save_if_changed()`.

**As implemented:** `_sync_on_close()` does read → `_confirm_merge()` → `apply_merge()` → `write_slp_files()` synchronously and returns False when the conflict dialog is cancelled, which leaves the window open.  No re-triangulation on close.

## 8.13 Unit tests (`tests/test_sleap_adapter.py`, `tests/test_slp_sync.py`, `tests/test_project_sync.py`) ✓

**As implemented (8.13):** `tests/test_slp_sync.py` gained `TestSlpMappingEditor` — the preset → join → split-by-trial sequence that builds the 3-file example, split-by-camera, a refused split, and the unassigned-cells error.  New `tests/test_project_sync.py` builds a temp project (2 trials x 2 cameras, one file per camera) and covers: the mapping is complete, every mapped file is written, a mapped file that does not exist reads as empty, a write → read round trip reports no changes and no conflicts, a label only in the SLP file comes back as new, an edited point is a conflict, and Use SLEAP takes the SLP value.  It also runs `AllExperimentsTriangulationWorker` against a hand-written 2-camera calibration and checks the 3D columns reach points.csv.

Done with 8.6–8.8 (52 tests): the mapping round trip, missing and duplicate cells, block compaction (the test builds the example's final mapping directly; the per-camera → join → split steps get tested with 8.9), every row of the merge table, the frame rules, instance matching, adapter tests 1–7 below, and a save → load → merge round trip that must give no changes and no conflicts.

Mapping and merge tests (no Qt):
1. `SlpMapping` round-trips through `to_config()` / `from_config()`; `missing_cells` and `duplicate_cells` find gaps and overlaps.
2. Block compaction: the 3-camera example (per camera → join → split by trial) gives the expected blocks.
3. `compute_merge()`: one test per row of the 8.8 table, plus the rule that predictions are ignored on frames with user instances.

Adapter tests: the original plan follows.

Test the full round-trip through `SleapAdapter.save()` → `sleap_io.load_file()`.

**Fixtures needed:**
- A minimal `trials_df` with one trial and two cameras, referencing synthetic video filenames
- A pair of stub SLP files in a temp dir — each containing a `sio.Video` with the matching filename but no labeled frames (enough for `build_file_map()` to read)
- A `PointsStore` (or bare DataFrame) with one user-set instance (state `"set"`) and one predicted instance (state `"predicted"`) across a few bodyparts

**Test cases:**
1. **User instance exported as `Instance`** — row with any camera state `"set"` → reloaded instance is `sio.Instance`, not `sio.PredictedInstance`; coordinates match
2. **Predicted instance exported as `PredictedInstance`** — all cameras `"predicted"` → reloaded instance is `sio.PredictedInstance`
3. **All-`not_set` instance omitted** — frame with all states `"not_set"` produces no instance in the output file
4. **`build_file_map()` routes correctly** — given two stub SLP files (one per camera), `build_file_map()` returns the right `(experiment, trial, camera) → path` mapping; `save()` writes to those paths, not to generated names
5. **Shared file** — a file mapped to two cameras × two trials keeps all four videos and all their frames after `save()` (the overwrite bug from 8.7)
6. **Frame offset** — with `cam0_start = 10`, frame 5 is saved as `frame_idx` 15 and loads back as frame 5
7. **Existing file preserved** — suggestions in an existing SLP file are still there after `save()`

Each test calls `sio.save_file()` to write stub SLP files, then `SleapAdapter().save(df, cameras=..., mapping=...)`, then `sio.load_file()` to reload and assert.

## Data flow

```
Project open / switch experiment
  ↓  mapping complete?
       yes → read mapped SLP files → compute_merge() → conflicts? → one warning
             → PointsStore.replace_df() → re-triangulate changed frames
       no, but slp/ has files → "Open Set up Sleap files...?"

[User labels frames]
  ↓  auto-save every 30 s (points.csv only)

File > Set up Sleap files...
  ↓  edit mapping (presets, split, join) → Apply
  ↓  backup slp/*.slp → read old files + merge → save [slp] to project.toml
  ↓  write all files in the new mapping from points.csv

File > Sync with Sleap... (Ctrl+S; enabled only when mapping is complete)
  ↓  save_if_changed() → points.csv
  ↓  READ: mapped SLP files → compute_merge() per experiment
  ↓  conflicts? → "Use SLEAP / Keep label3d / Cancel"
  ↓  apply merge → points.csv
  ↓  WRITE: SleapAdapter.save(mapping=...) → slp/*.slp

App close (unsynced changes)
  ↓  "Sync with Sleap before closing?"

[Outside label3d: train + predict in SLEAP into the same files] → reopen → startup read
```

# Phase 14 — Per-Camera Skeletons

**Goal**: Allow the `.slp` files for different cameras to have different skeletons. A node that a camera never sees is often left out of that camera's SLEAP model. Nodes with the same name are matched across skeletons and triangulated from the cameras that have them. A node in only some skeletons can be labelled only in those cameras, and the points list shows which nodes those are.

**Status**: Planned.

---

## Background

Today there is one skeleton per project:

- `skeleton.json` at the project root is written at setup, from the setup file's `[skeleton]` table or from the `.slp` files. `Project.load` reads it.
- Setup (`setup_project.py`, two places) raises "SLP files have inconsistent skeletons" when the files' node sets differ.
- `SleapAdapter._write_file` keeps an existing file's skeleton only if its node names equal the project skeleton's; otherwise it **replaces** the file's skeleton with the project one. With per-camera skeletons, a Sync would overwrite every file with the union.
- `Project._merge_by_experiment` drops SLP points of nodes that are not in the project skeleton.
- **File > Load skeleton...** (`Project.set_skeleton`) replaces `skeleton.json` and deletes points of nodes that were removed.
- Triangulation (`geometry.triangulate_rows`) already uses only cameras whose state is `set` or `predicted`. A camera that never has a point for a node therefore drops out of that node's triangulation without any change.

## Design summary

- **The `.slp` files are the source of the skeleton.** `skeleton.json` is no longer written or read. The project skeleton is the **union** of the skeletons of the mapped `.slp` files, built when the project loads and again whenever the files may have changed.
- **Matching** is by exact node name.
- **Edges and symmetries** are the union across files. A conflict between files is an error (14.1).
- **Availability** is per cell, i.e. per (experiment, trial, camera): a node is available in a cell when it is in the skeleton of the `.slp` file mapped to that cell. A cell with no mapped file, or whose file does not exist yet, has every node.
- **Each `.slp` file keeps its own skeleton** when written. A file created by label3d gets the union.
- A node available in fewer than 2 cameras of a trial is never triangulated. It can still be labelled in 2D.

---

## 14.1 Skeletons from the `.slp` files (`skeleton.py`, `project.py`)

- [ ] `Skeleton.from_slp(path) -> Skeleton`, replacing `from_sleap`. It reads only the file's metadata via `sleap_io.io.slp.read_skeletons(path)`, not the labelled frames, so it is cheap enough to run on every load. Edge direction is kept as in the file.
- [ ] `merge_skeletons(skeletons: dict[Path, Skeleton]) -> Skeleton` builds the union:
  - **Node order:** files in mapping order. The first file's nodes come first, in its order. Each later file's new nodes are appended in that file's order.
  - **Edges and symmetries:** union; direction is ignored when comparing.
  - **Conflicts** raise `SkeletonConflict(ValueError)`, which lists every conflict with the files involved, not just the first:
    1. Two skeletons that both contain nodes `a` and `b` disagree on whether `a–b` is an edge.
    2. Two skeletons that both contain nodes `a` and `b` disagree on whether `a ↔ b` is a symmetry.
    3. In the union, a node has more than one symmetry partner.
- [ ] `ProjectSkeleton` holds the skeletons:
  - `union: Skeleton`
  - `by_file: dict[Path, Skeleton]` (paths relative to the project root, as in `SlpMapping`)
  - `nodes_for(cell) -> set[str]`: the nodes available in a cell (the mapped file's skeleton, or all union nodes)
  - `cameras_for(experiment, trial, node) -> list[str]`, in camera order
  - `partial_nodes(experiment, trial) -> dict[str, list[str]]`: the nodes not available in every camera of the trial, with the cameras that have them
- [ ] `Project.skeleton` stays the union `Skeleton`, so most callers don't change. A new `Project.skeletons: ProjectSkeleton` is added alongside it.
- [ ] `Project.load_skeletons()` reads the mapped files, or `slp/*.slp` when there is no mapping (as `read_slp_for_sync` does). A mapped file that does not exist is skipped, and its cells get every node. It is called:
  - in `Project.load`;
  - at the start of every Sync and import (`read_slp_for_sync`, `read_predictions_for_import`), because the user may have dropped new files into `slp/`;
  - after the **Set up Sleap files** dialog saves a new mapping.

  If the union changed, `LabelView.set_skeleton` refreshes the view. If there is a `SkeletonConflict`, the project still loads but with the last good skeleton, the status bar reports the conflict, and Sync is refused with a dialog listing the conflicts.
- [ ] **Projects with no `.slp` files** (for example ones set up from videos only): the skeleton comes from `[skeleton]` in `project.toml` (`nodes`, `edges`, `symmetries`), which setup writes from the setup file's `[skeleton]` table. It is used only while there are no `.slp` files. The first Sync creates the files with it, and from then on the files are the source.
- [ ] **`node_offsets_cm`** (used by `init_random`) moves to `[skeleton.node_offsets_cm]` in `project.toml`. Offsets for nodes not in the union are ignored with a log message, not raised as an error, because nodes can now disappear when a file changes.
- [ ] **Old projects:** if `skeleton.json` exists and `project.toml` has no `[skeleton]` table, `Project.load` copies its `node_offsets_cm` (and, for a project without `.slp` files, its nodes, edges and symmetries) into `project.toml` once. The file itself is left in place and ignored.
- [ ] Remove **File > Load skeleton...**, `Project.set_skeleton`, `Skeleton.to_json`/`from_json` and `Skeleton.same_topology` if nothing else uses them. Skeletons are now changed in SLEAP. Points that a changed skeleton leaves behind are handled in 14.6.

## 14.2 Reading and writing `.slp` files (`adapters/sleap_adapter.py`, `project.py`)

- [ ] **Writing:** `_write_file` always keeps an existing file's skeleton (`existing.skeletons[0]`) and never replaces it. A new file gets the union, built by `_make_skeleton(project.skeleton)` with edges **and** symmetries (symmetries are dropped today). `_cell_frames` already writes only the nodes of the skeleton it is given, so a file never gains nodes.
- [ ] **Reading mapped files:** no change. Rows come from each instance's own `skeleton.node_names`, so a cell only gets points for its file's nodes.
- [ ] **Importing predictions from elsewhere** (`read_predictions_for_import`): such a file may have a different skeleton from the mapped one. `_merge_by_experiment` drops, for each cell, SLP points of nodes not in `nodes_for(cell)`, as well as nodes outside the union, which it already drops. The number dropped, by node and camera, goes into `MergeResult` and is shown in the sync summary dialog, not just logged.

## 14.3 Triangulation (`geometry.py`, `workers/triangulation_worker.py`)

- [ ] `triangulate_rows`, `triangulate_trial` and `triangulate_frame` take `available: dict[str, set[str]] | None` (camera → nodes for this trial). A camera's point counts only if its state is `set` or `predicted` **and** the node is in that camera's set. With today's data this changes nothing. It keeps points left over in unavailable cells (14.6) out of the 3D point.
- [ ] Callers pass `project.skeletons` per trial. The workers and the export CLI already have the project.
- [ ] Reprojections (`xr`/`yr`) are still computed for every camera. The label view does not draw them in cameras where the node is unavailable (14.4). Export leaves those cells out (14.6).
- [ ] `instance_init`: `init_random`, `init_average` and `init_reprojected` take the camera's available nodes and return coordinates only for those. This covers Add instance (one camera and all cameras) and the reprojected fill.

## 14.4 Label view (`widgets/bodypart_list.py`, `skeleton_overlay.py`, `epipolar_overlay.py`, `label_view.py`, `state.py`)

- [ ] `ProjectState` gets `available_nodes(camera) -> set[str]` and `node_cameras(bodypart) -> list[str]` for the current experiment and trial, backed by `ProjectSkeleton`. It emits `skeleton_changed` when the project skeleton is reloaded.
- [ ] **Points list** (`BodypartList`):
  - The count becomes `[n/k]`, where `k` is the number of cameras that have the node in this trial, not all cameras. For a node in every camera nothing changes.
  - A node in some but not all cameras gets a ◐ after its name. A node in only one camera gets `2D`, because it is never triangulated.
  - Each marked row's tooltip names the cameras that have the node, for example "Only in cam0, cam2" or "Only in cam1 — not triangulated".
  - A one-line key under the list ("◐ some cameras · 2D one camera") appears only when the trial has partial nodes.
  - The list also refreshes on `trial_changed`, since availability is per trial.
- [ ] **Camera panels:**
  - `SkeletonOverlay.update` takes the panel's available nodes. It draws no points, reprojections or edges for unavailable nodes, so they can't be clicked, dragged or toggled to not visible.
  - `EpipolarOverlay` draws no lines in a panel that lacks the selected node. Points in cameras that lack the node are not used as line sources either; they can only exist as leftovers (14.6).
  - When the selected node is unavailable in a panel, the panel's line label reads "*tail_tip* not in this camera's skeleton" so the empty panel is explained.
- [ ] `LabelView._can_place(camera, bodypart)` is a single guard used by every path that writes a point for one camera: drag, the not-visible toggle and the add/fill instance actions. It is a safety net behind the overlay, and refused writes show a status message.
- [ ] Next/previous node (`Down` / `Up`) still steps through every node. Selection is shared across cameras, and a node unavailable in one camera is usually available in another.

## 14.5 Project setup (`setup_project.py`, `setup_project_cli.py`, `widgets/slp_setup_dialog.py`)

- [ ] Replace both "inconsistent skeletons" checks with `merge_skeletons`. A `SkeletonConflict` aborts setup with the full list of conflicts.
- [ ] `FileAssignment.skeleton` holds the union, plus the per-file skeletons for the summary.
- [ ] The setup summary reports the union, then one line per partial node, for example "Skeleton: 12 nodes; tail_tip only in cam0, cam2".
- [ ] When there are `.slp` files, a setup file `[skeleton]` must match the union, as now (`_apply_config_skeleton`). Its `node_offsets_cm` go to `project.toml`. Without `.slp` files, the whole `[skeleton]` table is written to `project.toml` (14.1).
- [ ] Stop writing `skeleton.json`.
- [ ] **Set up Sleap files** dialog: after a mapping edit, the skeleton is re-merged. A conflict shows inline and disables OK.

## 14.6 Points that no longer fit

A skeleton can change after labelling: a node is dropped or renamed in SLEAP, or a file is remapped. `points.csv` may then hold points of nodes outside the union, or points in cells whose skeleton lacks the node. These points are **not deleted automatically**:

- they are not drawn, not triangulated (14.3) and not written to `.slp` (14.2);
- the export leaves them out, and `ExportResult` counts them so the dialog and the CLI can report them;
- on load and after a Sync, the status bar says how many there are.

- [ ] **File > Remove points outside the skeletons...** lists them by node (and camera, for unavailable cells). After confirmation it deletes them in every experiment, backing up `points.csv` as `set_skeleton` does today. `Project.skeleton_removals` is generalised to count both kinds, and `set_skeleton`'s deletion loop moves here.

## 14.7 Synthetic data (`generate_synthetic.py`, `src/synthetic/`, `synthetic_config.toml`)

- [ ] Write `node_offsets_cm` to `project.toml` instead of `skeleton.json`.
- [ ] New per-camera key `missing_nodes = [...]` under `[cameras.camN]`. That camera's `.slp` file is written with a skeleton without those nodes and with no points for them. The default config sets it on one camera, so the synthetic project exercises this phase.

## 14.8 Tests

- [ ] `merge_skeletons`: node order; union of edges and symmetries; each of the three conflicts, with every conflict listed.
- [ ] `Skeleton.from_slp` matches the skeleton from `sio.load_file`.
- [ ] Sync round trip on a synthetic project where cam1's file lacks a node:
  - reading gives no cam1 points for that node;
  - writing leaves cam1's file skeleton unchanged and keeps the other cameras' points;
  - a file created by label3d gets the union.
- [ ] Import from an outside file with the union skeleton drops the points in cells that lack the node, and reports them.
- [ ] Triangulation with `available` ignores a leftover `set` point in an unavailable cell, and gives NaN for a node in one camera only.
- [ ] `init_random` / `init_average` return only available nodes.
- [ ] `Project.load` migrates `node_offsets_cm` from an old `skeleton.json`.
- [ ] Remove-outside-skeletons: counts and deletion across experiments.

## 14.9 Docs

- [ ] `CLAUDE.md` and `docs/data_model.md`: drop `skeleton.json`; describe `[skeleton]` in `project.toml`, the union, and per-cell availability.
- [ ] `docs/quick_setup_from_slp.md`, `docs/project_setup_from_sleap.md`: skeletons may differ; the conflict rules; no **Load skeleton...**.
- [ ] `docs/workflows.md`: remove **Load skeleton...**; add **Remove points outside the skeletons...**.
- [ ] `docs/user_workflow.md`: the points-list markers and what an unavailable node looks like in a panel.
- [ ] `docs/synthetic_data.md`: `missing_nodes`.

## Order of work

1. 14.1 `from_slp`, `merge_skeletons`, `ProjectSkeleton` and their tests.
2. 14.2 writing that keeps each file's skeleton (this is what currently overwrites files), plus import filtering.
3. 14.7 synthetic `missing_nodes`, to have test data for the rest.
4. 14.1 project wiring: load, `project.toml` `[skeleton]`, migration, removal of `skeleton.json` and **Load skeleton...**.
5. 14.3 triangulation and instance init.
6. 14.4 label view.
7. 14.5 setup, 14.6 cleanup action, 14.9 docs.

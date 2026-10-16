# Phase 14 — Per-Camera Skeletons

**Goal**: Allow different cameras to have different skeletons. A node that a camera never sees is often left out of that camera's SLEAP model. Nodes with the same name are matched across skeletons and triangulated from the cameras that have them. A node in only some skeletons can be labelled only in those cameras, and the points list shows which nodes those are.

There are two ways a project gets its skeletons:

- **From `.slp` files**: the project is set up from existing `.slp` files, and each file's skeleton is used.
- **Loaded before any `.slp` file exists**: the user loads a skeleton (**File > Load skeleton...**), for all cameras or for chosen ones, labels in label3d, and the first Sync creates the `.slp` files with those skeletons.

**Status**: Planned.

---

## Background

Today there is one skeleton per project:

- `skeleton.json` at the project root is written at setup, from the setup file's `[skeleton]` table or from the `.slp` files. `Project.load` reads it.
- Setup (`setup_project.py`, two places) raises "SLP files have inconsistent skeletons" when the files' node sets differ.
- `SleapAdapter._write_file` keeps an existing file's skeleton only if its node names equal the project skeleton's; otherwise it **replaces** the file's skeleton with the project one. With per-camera skeletons, a Sync would overwrite every file with the union.
- `Project._merge_by_experiment` drops SLP points of nodes that are not in the project skeleton.
- **File > Load skeleton...** (`Project.set_skeleton`) replaces `skeleton.json` for the whole project and deletes points of nodes that were removed.
- Triangulation (`geometry.triangulate_rows`) already uses only cameras whose state is `set` or `predicted`. A camera that never has a point for a node therefore drops out of that node's triangulation without any change.

## Design summary

- **Each cell has one skeleton.** A cell is an (experiment, trial, camera). Its skeleton is the first of these that applies:
  1. the skeleton of its mapped `.slp` file, if that file exists;
  2. the skeleton **assigned** to its camera in `project.toml` (by **Load skeleton...** or the setup file);
  3. the skeleton shared by every existing `.slp` file of its camera, if they all agree (inferred);
  4. the union.

  A node is **available** in a cell when it is in the cell's skeleton.
- **Existing `.slp` files win.** A file keeps its own skeleton when written. A new file is created with its cells' skeleton (rules 2–4). An assigned skeleton therefore affects only cells whose file does not exist yet.
- **One skeleton per file.** All the cells mapped to one `.slp` file must have the same skeleton. **Set up Sleap files** checks this, and so does Sync (14.6).
- **The project skeleton is the union** of the existing files' skeletons and the assigned ones. Nodes are matched by exact name. Edges and symmetries are the union, and a conflict is an error (14.1). `skeleton.json` is no longer written or read.
- A node available in fewer than 2 cameras of a trial is never triangulated. It can still be labelled in 2D.

---

## 14.1 Skeletons and the union (`skeleton.py`, `project.py`)

- [x] `Skeleton.from_slp(path) -> Skeleton`, replacing `from_sleap`. It reads only the file's metadata via `sleap_io.io.slp.read_skeletons(path)`, not the labelled frames, so it is cheap enough to run on every load. Edge direction is kept as in the file.
- [x] `Skeleton` gets a `name` (the SLEAP skeleton name, or the file stem for a loaded `.json`). It is shown in dialogs and is not used for matching.
- [x] `merge_skeletons(skeletons: dict[str, Skeleton]) -> Skeleton` builds the union. The keys are labels for messages: a file path, or "assigned to cam0, cam2".
  - **Node order:** existing files in mapping order, then assigned skeletons in `project.toml` order. The first skeleton's nodes come first, in its order. Each later skeleton's new nodes are appended in that skeleton's order.
  - **Edges and symmetries:** union; direction is ignored when comparing.
  - **Conflicts** raise `SkeletonConflict(ValueError)`, which lists every conflict with the skeletons involved, not just the first:
    1. Two skeletons that both contain nodes `a` and `b` disagree on whether `a–b` is an edge.
    2. Two skeletons that both contain nodes `a` and `b` disagree on whether `a ↔ b` is a symmetry.
    3. In the union, a node has more than one symmetry partner.
- [x] `ProjectSkeleton` holds the skeletons:
  - `union: Skeleton`
  - `by_file: dict[Path, Skeleton]`: existing mapped files (paths relative to the project root, as in `SlpMapping`)
  - `by_camera: dict[str, Skeleton]`: assigned skeletons
  - `skeleton_for(cell) -> Skeleton` and `nodes_for(cell) -> set[str]`, following rules 1–4 above
  - `cameras_for(experiment, trial, node) -> list[str]`, in camera order
  - `partial_nodes(experiment, trial) -> dict[str, list[str]]`: the nodes not available in every camera of the trial, with the cameras that have them
  - `mapping_problems(mapping) -> list[str]`: the one-skeleton-per-file check (14.6)
- [x] `Project.skeleton` stays the union `Skeleton`, so most callers don't change. A new `Project.skeletons: ProjectSkeleton` is added alongside it.
- [x] `Project.load_skeletons()` reads the existing mapped files, or `slp/*.slp` when there is no mapping (as `read_slp_for_sync` does), and the assigned skeletons from `project.toml`. It is called:
  - in `Project.load`;
  - at the start of every Sync and import (`read_slp_for_sync`, `read_predictions_for_import`), because the user may have dropped new files into `slp/`;
  - after a Sync that created files;
  - after **Set up Sleap files** saves a new mapping, and after **Load skeleton...**.

  If the union changed, `LabelView.set_skeleton` refreshes the view. If there is a `SkeletonConflict`, the project still loads but with the last good skeleton, the status bar reports the conflict, and Sync is refused with a dialog listing the conflicts.
- [x] **`project.toml` layout:**

  ```toml
  [skeleton.node_offsets_cm]      # used by init_random
  snout = [10.0, 0.0, 0.0]

  [[skeleton.assigned]]           # one per loaded skeleton
  name = "full"
  cameras = ["cam0", "cam2"]
  nodes = ["snout", "left_eye", "tail_tip"]
  edges = [["snout", "left_eye"]]
  symmetries = []
  ```

  A camera appears in at most one `assigned` entry. An entry left with no cameras is removed.
- [x] **Offsets:** offsets for nodes not in the union are ignored with a log message, not raised as an error, because nodes can now disappear when a file changes.
- [x] **Old projects:** if `skeleton.json` exists and `project.toml` has no `[skeleton]` table, `Project.load` copies it once into `project.toml`: its offsets to `node_offsets_cm`, and its nodes, edges and symmetries as an `assigned` entry for every camera. The file itself is left in place and ignored. For a project whose `.slp` files all exist, the assignment changes nothing (rule 1 wins), but it still applies to files created later.
- [ ] Remove `Project.set_skeleton` (replaced by 14.2). Until then it deletes points of removed nodes and assigns the skeleton to every camera. `Skeleton.to_json` is removed (done); `from_json` and `same_topology` stay.

## 14.2 Load skeleton for chosen cameras (`widgets/load_skeleton_dialog.py`, `project.py`, `main_window.py`)

**File > Load skeleton...** opens a dialog instead of going straight to a file picker:

- [ ] **Source:** a `skeleton.json` or a `.slp` file (whose skeleton is read with `from_slp`).
- [ ] **Cameras:** a checkbox per camera plus **All cameras**, which is on by default. Next to each camera it shows:
  - its current skeleton (assigned name, inferred, or none);
  - whether it already has `.slp` files, and if so whether their skeleton differs from the one being loaded. A differing file gets the note "existing .slp files keep their skeleton; this applies to new files only".
- [ ] **Checks** before OK, shown inline:
  - The new union must merge (14.1). A conflict disables OK.
  - Mapped files that don't exist yet and would then hold cells with different skeletons are listed as a warning: "slp/trial1.slp is mapped to cam0 and cam1, which would have different skeletons — change the mapping in Set up Sleap files before the next Sync". OK stays enabled, and Sync refuses until the mapping is fixed (14.6). This allows loading skeletons first and fixing the mapping afterwards.
  - Points that would end up outside their cell's skeleton (cells without an existing file, for the chosen cameras, in nodes not in the new skeleton) are counted by node. If there are any, a **Remove these points now** checkbox is shown, off by default. When it is off they become leftovers (14.7).
- [ ] `Project.assign_skeleton(skeleton, cameras)` updates `[[skeleton.assigned]]`: the chosen cameras are removed from any other entry, and an entry with the same topology is reused rather than duplicated. Then it calls `load_skeletons()`.
- [ ] The setup file's `[skeleton]` table (`setup_project.py`) becomes an assignment to all cameras. It takes an optional `cameras = [...]` key to assign it to some cameras only.

## 14.3 Reading and writing `.slp` files (`adapters/sleap_adapter.py`, `project.py`)

- [x] **Writing:** `_write_file` always keeps an existing file's skeleton (`existing.skeletons[0]`) and never replaces it. A new file is created with `skeleton_for` its cells (one skeleton, guaranteed by 14.6), built by `_make_skeleton` with edges **and** symmetries (symmetries are dropped today) and with the skeleton's name. `SleapAdapter.save` takes `skeletons: ProjectSkeleton` in place of `skeleton`. `_cell_frames` already writes only the nodes of the skeleton it is given, so a file never gains nodes.
- [x] **Reading mapped files:** no change. Rows come from each instance's own `skeleton.node_names`, so a cell only gets points for its file's nodes.
- [x] **Importing predictions from elsewhere** (`read_predictions_for_import`): such a file may have a different skeleton from the cell's. `_merge_by_experiment` drops, for each cell, SLP points of nodes not in `nodes_for(cell)`, as well as nodes outside the union, which it already drops. The number dropped, by node and camera, goes into `MergeResult` and is shown in the sync summary dialog, not just logged.

## 14.4 Triangulation and instance init (`geometry.py`, `workers/triangulation_worker.py`, `instance_init.py`)

- [ ] `triangulate_rows`, `triangulate_trial` and `triangulate_frame` take `available: dict[str, set[str]] | None` (camera → nodes for this trial). A camera's point counts only if its state is `set` or `predicted` **and** the node is in that camera's set. With today's data this changes nothing. It keeps leftover points (14.7) out of the 3D point.
- [ ] Callers pass `project.skeletons` per trial. The workers and the export CLI already have the project.
- [ ] Reprojections (`xr`/`yr`) are still computed for every camera. The label view does not draw them in cameras where the node is unavailable (14.5). Export leaves those cells out (14.7).
- [ ] `init_random`, `init_average` and `init_reprojected` take the camera's available nodes and return coordinates only for those. This covers Add instance (one camera and all cameras) and the reprojected fill.

## 14.5 Label view (`widgets/bodypart_list.py`, `skeleton_overlay.py`, `epipolar_overlay.py`, `label_view.py`, `state.py`)

- [ ] `ProjectState` gets `available_nodes(camera) -> set[str]` and `node_cameras(bodypart) -> list[str]` for the current experiment and trial, backed by `ProjectSkeleton`. It emits `skeleton_changed` when the project skeleton is reloaded.
- [ ] **Points list** (`BodypartList`):
  - The count becomes `[n/k]`, where `k` is the number of cameras that have the node in this trial, not all cameras. For a node in every camera nothing changes.
  - A node in some but not all cameras gets a ◐ after its name. A node in only one camera gets `2D`, because it is never triangulated.
  - Each marked row's tooltip names the cameras that have the node, for example "Only in cam0, cam2" or "Only in cam1 — not triangulated".
  - A one-line key under the list ("◐ some cameras · 2D one camera") appears only when the trial has partial nodes.
  - The list also refreshes on `trial_changed`, since availability is per trial.
- [ ] **Camera panels:**
  - `SkeletonOverlay.update` takes the panel's available nodes. It draws no points, reprojections or edges for unavailable nodes, so they can't be clicked, dragged or toggled to not visible.
  - `EpipolarOverlay` draws no lines in a panel that lacks the selected node. Points in cameras that lack the node are not used as line sources either; they can only exist as leftovers (14.7).
  - When the selected node is unavailable in a panel, the panel's line label reads "*tail_tip* not in this camera's skeleton" so the empty panel is explained.
- [ ] `LabelView._can_place(camera, bodypart)` is a single guard used by every path that writes a point for one camera: drag, the not-visible toggle and the add/fill instance actions. It is a safety net behind the overlay, and refused writes show a status message.
- [ ] Next/previous node (`Down` / `Up`) still steps through every node. Selection is shared across cameras, and a node unavailable in one camera is usually available in another.

## 14.6 One skeleton per `.slp` file (`slp_sync.py`, `widgets/slp_setup_dialog.py`, `project.py`)

`ProjectSkeleton.mapping_problems(mapping)` checks every file in a mapping:

- [ ] **A file that doesn't exist yet:** all its cells must have the same skeleton by rules 2–4 (same nodes in the same order, same edges and symmetries). Otherwise it is an **error**, naming the file and the cameras on each side, for example "slp/trial1.slp: cam0 (full) and cam1 (no_tail) have different skeletons".
- [ ] **An existing file:** its skeleton wins (rule 1).
  - A cell whose camera has an **assigned** skeleton that differs from the file's is an **error**: the assignment says the camera should have a different skeleton from the file's.
  - A cell with no assignment whose skeleton currently differs from the file's (for example one moved from another existing file) is a **warning**. It lists the nodes that would become unavailable and how many labelled points that leaves behind (14.7).

Where the check runs:

- [ ] **Set up Sleap files** dialog: the check runs after every edit, including **apply preset**, so a pattern like `{trial}.slp` that puts cameras with different skeletons into one file is caught at once. Each file row shows its skeleton name, or "mixed". Errors are listed inline and disable OK. Warnings are listed and need a confirmation on OK.
- [ ] The dialog's **default preset** for a project with more than one skeleton is one file per camera per experiment (`{experiment}_{camera}.slp`). That never mixes skeletons.
- [ ] **Sync** runs the check before reading or writing. With an error it is refused, with a dialog that lists the problems and offers to open **Set up Sleap files**. This catches a skeleton loaded after the mapping was made (14.2).
- [ ] **Setup from `.slp` files** (14.8): the mapping comes from the files, so each file has one skeleton by construction. Nothing extra is needed.

## 14.7 Points that no longer fit

A skeleton can change after labelling: a node is dropped or renamed in SLEAP, a file is remapped, or a different skeleton is loaded for a camera. `points.csv` may then hold points of nodes outside the union, or points in cells whose skeleton lacks the node. These points are **not deleted automatically** (except by the checkbox in 14.2):

- they are not drawn, not triangulated (14.4) and not written to `.slp` (14.3);
- the export leaves them out, and `ExportResult` counts them so the dialog and the CLI can report them;
- on load and after a Sync, the status bar says how many there are.

- [ ] **File > Remove points outside the skeletons...** lists them by node, and by camera for unavailable cells. After confirmation it deletes them in every experiment, backing up `points.csv` as `set_skeleton` does today. `Project.skeleton_removals` is generalised to count both kinds, and `set_skeleton`'s deletion loop moves here. The Load skeleton dialog (14.2) uses the same code.

## 14.8 Project setup (`setup_project.py`, `setup_project_cli.py`)

- [ ] Replace both "inconsistent skeletons" checks with `merge_skeletons`. A `SkeletonConflict` aborts setup with the full list of conflicts.
- [ ] `FileAssignment.skeleton` holds the union, plus the per-file skeletons for the summary.
- [ ] The setup summary reports the union, then one line per partial node, for example "Skeleton: 12 nodes; tail_tip only in cam0, cam2".
- [ ] A setup file `[skeleton]` is written as an assignment (14.2). When there are `.slp` files, it must merge with them without conflict, replacing today's exact-match check (`_apply_config_skeleton`). Its `node_offsets_cm` go to `project.toml`.
- [x] Stop writing `skeleton.json`.

## 14.9 Synthetic data (`generate_synthetic.py`, `src/synthetic/`, `synthetic_config.toml`)

- [x] Write `node_offsets_cm` to `project.toml` (`[skeleton.node_offsets_cm]`); `skeleton.json` is no longer written.
- [x] New per-camera key `missing_nodes = [...]` under `[cameras.camN]`. That camera's `.slp` file is written with a skeleton without those nodes (named `synthetic_without_<nodes>`) and with no points for them; `points.csv` leaves those cells `not_set`.
- [ ] Turn on `missing_nodes = ["dorsal"]` for cam3 in `synthetic_config.toml` (it is there, commented out) once loading (14.1) and setup (14.8) accept different skeletons, then regenerate `projects/test_project`.
- [x] A `--no-slp` option writes the project without `.slp` files and with the skeletons as `assigned` entries in `project.toml` (honouring `missing_nodes`). This gives test data for the load-skeleton-first workflow.

## 14.10 Tests

- [ ] `merge_skeletons`: node order; union of edges and symmetries; each of the three conflicts, with every conflict listed.
- [ ] `Skeleton.from_slp` matches the skeleton from `sio.load_file`.
- [ ] `ProjectSkeleton.skeleton_for`: each of rules 1–4, including an inferred camera skeleton and cameras whose files disagree (which fall back to the union).
- [ ] `Project.assign_skeleton`: a camera moves between entries; empty entries are removed; the same topology is reused.
- [ ] `mapping_problems`: a new file mixing skeletons (error); an existing file against a differing assignment (error); a cell moved into an existing file with another skeleton (warning, with the point count); one file per camera (no problems).
- [ ] Sync round trip on a synthetic project where cam1's file lacks a node:
  - reading gives no cam1 points for that node;
  - writing leaves cam1's file skeleton unchanged and keeps the other cameras' points.
- [ ] Load-skeleton-first workflow on a `--no-slp` synthetic project: the first Sync creates each file with its cameras' assigned skeleton; reloading gives the same availability from the files.
- [ ] Sync is refused when the mapping mixes skeletons.
- [ ] Import from an outside file with the union skeleton drops the points in cells that lack the node, and reports them.
- [ ] Triangulation with `available` ignores a leftover `set` point in an unavailable cell, and gives NaN for a node in one camera only.
- [ ] `init_random` / `init_average` return only available nodes.
- [ ] `Project.load` migrates an old `skeleton.json` into `project.toml`.
- [ ] Remove-outside-skeletons: counts and deletion across experiments.

## 14.11 Docs

- [ ] `CLAUDE.md` and `docs/data_model.md`: drop `skeleton.json`; describe `[skeleton]` in `project.toml`, the four rules for a cell's skeleton, and the union.
- [ ] `docs/quick_setup_from_slp.md`, `docs/project_setup_from_sleap.md`: skeletons may differ between files; the conflict rules; the optional `cameras` key in `[skeleton]`.
- [ ] `docs/workflows.md`: the new **Load skeleton...** dialog, the one-skeleton-per-file rule in **Set up Sleap files**, and **Remove points outside the skeletons...**.
- [ ] `docs/user_workflow.md`: the load-skeleton-first workflow, the points-list markers, and what an unavailable node looks like in a panel.
- [ ] `docs/synthetic_data.md`: `missing_nodes` and `--no-slp`.

## Order of work

1. 14.1 `from_slp`, `merge_skeletons`, `ProjectSkeleton` (with `skeleton_for` and `mapping_problems`) and their tests.
2. 14.3 writing that keeps each file's skeleton (this is what currently overwrites files) and creates new files with their cells' skeleton, plus import filtering.
3. 14.9 synthetic `missing_nodes` and `--no-slp`, to have test data for the rest.
4. 14.1 project wiring: load, the `project.toml` layout, migration, removal of `skeleton.json`.
5. 14.6 the mapping check in Sync and in **Set up Sleap files**.
6. 14.2 **Load skeleton...** dialog.
7. 14.4 triangulation and instance init.
8. 14.5 label view.
9. 14.7 cleanup action, 14.8 setup, 14.11 docs.

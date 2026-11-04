# Phase 16 — Axis Alignment

**Goal**: Re-orient an experiment's 3D output to a physical reference object with a clear origin and x, y, z axes. The user clicks points on the object in two or more cameras. label3d triangulates them with the active calibration and computes an **alignment**: a rotation and an origin that are applied to the 3D points coming out of the calibration. The calibration itself is not changed. Once an experiment has an alignment, every 3D point it outputs is aligned.

**Status**: In progress on branch `axis-alignment`. 16.1 (geometry), 16.2 (axis takes, `[alignment]` record), 16.3 (setup), 16.4 (adding axis takes later), 16.5 (saved clicks, computing from them), 16.6 (the Align tab), 16.7 (aligned output), 16.8 (recomputing) and 16.11 (docs) done; awaiting a check in the app. Left: the rest of 16.10 (remaining tests, synthetic axis object), deferred for now.

**Builds on**: Phase 9 (calibration takes), Phase 11 (registry, `[calibration.active]` / `[calibration.triangulated_with]`), Phase 12 (refinement aligns its result back onto the start), Phase 13 (batch triangulation, export), Phase 15 (Parquet working files), Phase 20 (versions and upgrades), and 1-based frame numbers wherever the user sees or types them (`frames.py`).

## What the user does

1. Records the reference object, usually on its own: one still image or a short video per camera. The files are found at project setup by an align keyword in their names (16.3), or added later in the app (16.4). Occasionally the object is in a calibration take or a trial instead: then one frame of it (the first by default) is extracted into an axis take of still images, at setup or later. `.slp` files never have the axis points; the user always clicks them.
2. After a good calibration is active, opens the **Align** tab (between Calibrate and Label). Picks the axis take and, for a video, the frame.
3. Clicks the reference points in at least two cameras each:
   - **4 points** (usual): origin, plus one point on each of x, y, z.
   - **7 points** (usual): origin, plus two points on each axis. The axis direction comes from its two points only (`far − near`); they are usually displaced from the origin, and the origin isn't used for direction.
   - **3 or 5 points** (less common): origin plus any two axes, with 1 or 2 points each. The third axis is their cross product, chosen to make the system right-handed (x × y = z, y × z = x, z × x = y).
4. Optionally enters known distances on the object, as a check of the calibration's scale.
5. Clicks **Compute**, checks the preview (axes drawn on every camera, errors, angles, scale), then **Apply**. Apply saves the alignment in `project.toml`; from then on, exported 3D points are aligned.
6. Can re-align at any time: click again, or recompute from the saved clicks when the calibration has changed (16.8).

## The transform

- The alignment is `R_align` (3×3, columns = the new x, y, z axes in calibration coordinates) and `origin` (cm, calibration coordinates).
- An aligned point is `p_aligned = R_align.T @ (p_cal − origin)`; the inverse is `p_cal = R_align @ p_aligned + origin`.
- It is a rigid transform (det +1, no scaling), so distances and reprojection errors are unchanged.
- **Where it applies:**
  - `points.parquet` keeps 3D points in the calibration frame. Triangulation, reprojection, epipolar lines, refinement (Phase 12) and re-triangulation all work there, as now.
  - The alignment is applied wherever 3D points leave label3d or are shown: export (GUI and `label3d-export`), and the hidden 3D view if it is turned on.
  - Changing or recomputing an alignment never needs re-triangulation.
- **Which calibration it belongs to:** an alignment records the calibration entry its clicks were triangulated with. It is **current** when the 3D points (`triangulated_with`) come from that entry, or from a refinement descended from it; refinement aligns its result back onto its start (`calibration_refine.align_to`), so the frame is kept. Otherwise it is **stale**, and should be recomputed (16.8).

## Current state

- `geometry.py`:
  - `triangulate_point()` and `compute_axis_alignment()` exist and are kept; `instance_init.py` also uses `triangulate_point()`.
  - `apply_alignment_to_cameras()` and `save_aligned_calibration()` follow the old design, which rotated the cameras and saved an `align` calibration entry. They are removed, together with `"align"` in `calibration_registry.METHODS` and its test in `test_calibration_registry.py`. No project has `align` entries.
- `compute_axis_alignment()` needs changes (16.1): it takes one vector per axis, always uses the longest axis as primary, and orthogonalizes a supplied third axis instead of deriving it. Given left-handed picks, that can return a matrix with det −1.
- `widgets/camera_panel.py`: `raw_clicked = Signal(str, float, float)` and `set_pick_mode(bool)` exist.
- The `"alignment"` skips in `project._read_camera_info` and `setup_project.py` (calibration files) are leftovers from an older design; remove them.
- The main window has two tabs (Calibrate = 0, Label = 1), and `_on_tab_changed` and the session settings hard-code index 1 for Label.
- Single images can't be read: `VideoReader` and `_frame_count` use `cv2.VideoCapture`, and `setup_project.classify_files` rejects calibration images (the Phase 9 deprecation of image-based **charuco** calibration).
- `SetupConfig.camera_aliases` is used during setup but not written to `project.toml`.

## 16.1 Geometry (`geometry.py`)

- [x] `axis_vector(points) -> np.ndarray`: given `{"origin", "near", "far"}` triangulated points for one axis, return `far − near` if both exist, otherwise `point − origin` for the single point. The positive direction is near → far (or origin → point)
- [x] Rework `compute_axis_alignment(origin, axes, primary, flips)`:
  - `axes: dict[str, np.ndarray]` holds 2 or 3 of `x`, `y`, `z`; `primary` is the axis kept exactly; flips are applied to the raw vectors first, as now
  - Gram-Schmidt: normalize the primary axis. Orthogonalize the second axis against it; when all three are clicked, that is the next one in `primary_axis_order`. **Always** derive the third from the right-hand rule
  - When the third axis was clicked too, use it only as a check: report the angle between the clicked and derived third axis. If it is more than 90°, the picks are left-handed: raise a clear error ("flip one axis"), never return det −1
  - Return an `AlignmentResult` dataclass: `R_align`, `origin`, the primary and secondary axes, the angles between each pair of raw axes (how far from orthogonal the clicks were), and how far each axis moved in Gram-Schmidt
- [x] `primary_axis_order(axes, errors, rule) -> list[str]`, with `rule` one of `"longest"` (default), `"lowest_error"`, or an explicit axis name chosen by the user:
  - `"longest"`: longest clicked distance first
  - `"lowest_error"`: smallest mean reprojection error of the axis's points first
  - An explicit axis puts that axis first, with the other order set by `"longest"`
  - A derived (cross-product) axis can't be primary
- [x] `triangulate_point()` also returns the per-camera reprojection error (or a new `triangulate_pick()`, if changing the return breaks `instance_init.py`)
- [x] `scale_check(points, known_lengths) -> dict`: for each axis with a known length, compare the measured distance with it (near–far, or origin–point for 1-point axes) and give the % difference. A check only; nothing is rescaled
- [x] `align_points(xyz, R_align, origin)` and `unalign_points(...)`: the transform and its inverse (see The transform), vectorized over (N, 3), NaN passed through
- [x] Remove `apply_alignment_to_cameras()` and `save_aligned_calibration()` (see Current state)

## 16.2 Axis takes, still images and the `[alignment]` record

An axis take is a row of type `axis` in `trials.csv` (`trial_types.py`; the `type` column replaced `is_calibration` in 0.4.0, see 16.9). Like a calibration take, its files are in `experiments/{name}/calibrations/{cam}/`. It may be a single still image per camera. Like calibrations, alignments belong to one experiment; a take is never shared between experiments.

- [x] `video.py`: `VideoReader` opens image files (`IMAGE_EXTENSIONS`, now defined only here: `.png .jpg .jpeg .tif .tiff .bmp`) with `cv2.imread` as a 1-frame video. `video.frame_count` replaces `project._frame_count`, so `check_trial_videos` and trial lengths measure an image as 1 frame (`_end` = 1 in `trials.csv`, 1-based). `_video_mtime` works unchanged
- [x] `project.toml` gets one `[alignment.{experiment}]` table per experiment (`alignment.py`). All writes go through `version.write_project_toml`. **Frame numbers here are 1-based**, as in `trials.csv`; they are converted with `shown_frame` / `stored_frame` when read and written

  ```toml
  [alignment.default.settings]      # last used on the Align tab
  points_per_axis = 2               # 1 or 2
  cross_product = ""                # "", "x", "y" or "z": the axis that is derived
  primary = "longest"               # "longest", "lowest_error", "x", "y", "z"
  flip = ["z"]                      # axes to reverse
  known_length_cm = { x = 10.0 }    # optional, per axis

  [alignment.default.transform]     # the alignment in use; absent until the first Apply
  R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]   # rows of R_align
  origin = [12.3, -4.5, 0.8]        # cm, calibration coordinates
  calibration = "charuco_20261102-093000"   # entry the clicks were triangulated with
  created = 2026-11-03T14:15:00
  primary = "x"                     # the axis actually kept exact
  derived = ""                      # the axis from the cross product, if any
  axis_angles_deg = { xy = 89.6, yz = 90.3, zx = 90.1 }   # between the clicked axes
  max_error_px = 0.8                # worst reprojection error of the clicks
  scale_error_pct = { x = 0.4 }     # where a known length was given

  [alignment.default.source]        # the frame the alignment was clicked on
  take = "Align"                    # always an axis take
  type = "axis"
  frame = 1                         # synchronized frame, 1-based

  [alignment.default.source.videos] # each camera's file and frame within that file, 1-based
  cam0 = { file = "Align_cam0.png", frame = 1 }
  cam1 = { file = "Align_cam1.png", frame = 1 }

  [[alignment.default.history]]     # earlier transforms, newest last; same keys as transform

  [alignment.default.extracted.Align]   # an axis take extracted from a frame of a take or trial
  take = "PreCal"                   # the calibration take or trial it came from
  type = "calibration"              # or "trial"
  frame = 1                         # 1-based
  [alignment.default.extracted.Align.videos]
  cam0 = { file = "PS18_PreCal_cam0.mp4", frame = 1 }
  ```

  - `transform` and `source` are written on Apply and on recompute (16.8). The one they replace is appended to `history`, so a re-alignment can be undone by hand
  - A camera's `frame` under `source.videos` is `cam_start + frame`, so it still names the right image if the take's offsets in `trials.csv` change later
- [x] `Project`:
  - `alignment`, `alignment_source`, `alignment_settings` and `alignment_history` read the table; `save_alignment` (old transform to `history`) and `save_alignment_settings` write it
  - `alignment_status(experiment)` returns `alignment.NONE`, `CURRENT` or `STALE` (see The transform), for `triangulated_with`, or the active calibration before anything is triangulated
  - `alignment_source_for(take_or_trial, frame)` builds the `source` record, adding each camera's start offset
  - `axis_takes` and `get_take` (calibration or axis take)
- [x] A take name must be unique among the experiment's trials and takes; checked by setup (16.3) and the add-take dialog (16.4)
- [x] ~~The Align tab also offers ordinary trials~~ Dropped: a frame of a calibration take or trial is extracted into an axis take (16.3, 16.4), and the Align tab lists axis takes only. `extracted` records where it came from (`Project.extracted_axis_takes`)
- [x] **Camera aliases**: setup writes `SetupConfig.camera_aliases` to a `[camera_aliases]` table (`LateralOrtho = ["Lateral"]`), and `Project` loads it. Projects without it have no aliases (`Project.camera_aliases`). The add-take dialog (16.4) uses it

## 16.3 Project setup (`setup_project.py`, `setup_project_cli.py`)

Axis files are found much like calibration videos. They're separate recordings with no points in them, never something read from `.slp` files.

- [x] `SetupConfig`: `align_keywords: list[str]` (e.g. `["Align"]`) and `align_regex: str`. Each keyword names one take, and a regex's `take` group (else group 1, else the whole match) supplies the name, exactly as for `calib_keywords` / `calib_regex`. Both empty (the default): no axis takes
- [x] Generalize the take matching: `_is_calibration` is now `_matches_take`, `_match_takes(config, files, keywords, regex)` does the grouping, and `_match_calibration_takes` keeps its signature as a wrapper. `SetupConfig.is_axis_file` / `is_calibration_file` test a name. If keywords can't separate takes, fall back to `calib_match_strategy` (or `match_strategy`)
- [x] Classification order for each file: align match → axis; else calibration match → calibration; else trial. A file that matches both an align and a calibration keyword goes in `assignment.errors`, naming the file (`_file_kind`)
- [x] **Still images**: in `classify_files`, image files that match an align keyword become axis files. The Phase 9 error for calibration images stays and is checked after axis images are taken out; other images stay unmatched, as now
- [x] Experiment and camera as for calibration: `_in_listed_experiment`, `_identify_experiment`, `_identify_camera` (with `camera_aliases`)
- [x] `FileAssignment.axis_takes: dict[str, dict[str, dict[str, Path]]]`, keyed experiment → take → camera → file, filled by both setup routes:
  - `classify_files` (folder setup): from `source_dir`, videos and images
  - `classify_slp_files` (setup from SLEAP files): `_find_axis_files(dirs, config, assignment)` searches the same folders as `_find_calibration_videos` (`_take_search_dirs`) (`calibration_dir` if given, else `source_dir` and the `.slp` folders), including images. Finding an existing `calibration.toml` doesn't stop the search, since a calibration and an alignment are separate
  - `_unreferenced_source_videos` leaves axis files out, so they never become trials
- [x] `_assign_axis_takes` groups them per experiment after the trials and calibration takes are known. Errors: a take name that is also a trial or calibration take of the experiment; files whose take can't be determined (two files for one camera with no other way to separate them). Warnings: a take with fewer than two cameras can't be triangulated; an axis take for an experiment with no trials. No warning when an experiment has no alignment, since alignment is optional
- [x] `execute_setup`:
  - Axis files go into `calibrations/{cam}/` and are always brought into the project, as calibration videos are: copied in reference mode, otherwise moved or copied by the video mode
  - A file name that clashes with another file in that folder is an error before anything moves (`_take_file_clashes`)
  - Rows go into `trials.csv` with `type = axis` and bare file names; `check_trial_videos` measures them
  - `write_project_toml` gets `camera_aliases` and writes `[camera_aliases]`
- [x] **Axis takes from a frame** of a calibration take or trial: `[setup.axis_frames]` in a setup file (`Align = { from = "PreCal", frame = 1 }`; frame from 1, default 1), `SetupConfig.axis_frames` (0-based), and an interactive prompt. `_assign_axis_frames` finds the source in every experiment that has it; errors for a source in no experiment, a name already used, or a frame a source video lacks. `execute_setup` extracts each camera's frame as `{video stem}-axis-fr{NNNN}.png` in `calibrations/{cam}/` (`setup_project.axis_still_name`; the frame within that video, from 1, four digits: `cam1_PreCal-axis-fr0001.png`) (`video.extract_frame`) before any video is moved, writes an `axis` row, and records `[alignment.{exp}.extracted.{take}]`. Setup leaves start offsets at 0, so the video frame is the frame given
- [x] `preview_assignment` lists axis takes per experiment, with per-camera files and whether each is an image or a video, next to the calibration takes
- [x] CLI: both interactive routes ask for axis take keywords (`prompt_axis_takes`), with the regex fallback as for calibration; `--from-slp` takes `--align-keywords`. Setup files take `[setup] align_keywords`, `align_regex`; `config_from_toml` reads them and `save_config_toml` round-trips them. Add them to the template in `setup_project_cli.py`'s docstring
- [x] Projects set up before this phase have no `[alignment]` or `[camera_aliases]` table and need nothing done; their takes are added later with the dialog (16.4)

## 16.4 Adding an axis take later (`axis_takes.py`, `widgets/add_axis_take_dialog.py`)

- [x] **File > Add axis take...** opens `AddAxisTakeDialog` for the active experiment, named in the dialog
- [x] An **Add take...** button on the Align tab opens it too, and the Align tab selects the new take (16.6); File > Add axis take... refreshes the Align tab's list
- [x] **Files** tab: choose several files, or a folder (its images and videos, `axis_takes.media_files`). `axis_takes.match_files` matches them to cameras as setup does (`setup_project.identify_camera`, with `[camera_aliases]`); a second file for a camera, or one naming no camera, is listed as unmatched. A combo per camera lets the user change or clear a match. **Add** is disabled until at least two cameras have a file
- [x] Take name: what the file names share once the camera's name is taken out (`PS18_Align_cam1.png` → `PS18_Align`, `suggest_take_name`), else `Align`; `Align2`, … when taken. The user can edit it; it must be unique among the experiment's trials and takes (`check_take_name`, checked as the user types)
- [x] `axis_takes.add_from_files`: the files are **copied** (never moved) into `calibrations/{cam}/` (`setup_project.copy_file`). The same file already there is fine; a different file of that name is refused before anything changes
- [x] **Frame of a take or trial** tab: pick a calibration take or trial and a frame (from 1, default 1; the spinbox stops at its length, `source_length`). `axis_takes.add_from_frame` reads each camera's video at its start offset plus the frame and writes `{video stem}-axis-fr{NNNN}.png`, numbered by the frame within that video (`video.extract_frame`), and records `[alignment.{exp}.extracted.{take}]`. A still already there is that same frame and is used as it is, so two takes can share it. A frame a camera's video lacks is refused before anything is written
- [x] Either way an `axis` row is added to `trials.csv` (with `trials.csv.bak`, and a `type` column if the file had none), measured with `check_trial_videos`, and `Project.refresh_takes` re-reads the takes, so no reload is needed
- [x] No GUI for removing a take in this phase
- [x] Tests: `tests/test_add_axis_take.py` (matching, names, both ways of adding, refusals that change nothing, the dialog)

## 16.5 Picks working file (`alignment_picks.py`)

- [x] `experiments/{name}/calibrations/alignment_picks.parquet` holds the 2D clicks, saved as they are made (Phase 15 Parquet conventions, `parquet_io.write_table`). Columns: `take, frame, target, camera, x, y`. `frame` is 0-based, like the other working files. `target` is one of `origin, x1, x2, y1, y2, z1, z2` (`TARGETS`); in 1-point mode only `x1, y1, z1` are used, and in 2-point mode the axis runs `x1` → `x2`. Raw pixels, as everywhere
- [x] `read_picks(exp_dir, take, frame)`, `write_picks(...)` (replaces that frame's rows, keeps the rest; empty picks remove the frame), `picked_frames(exp_dir, take)`, `targets_for(points_per_axis, cross_product)`
- [x] Settings are in `[alignment.{exp}.settings]` (16.2)
- [x] `alignment.compute_alignment(cameras, picks, settings)` triangulates every target the settings need, builds the axes (`axis_vector`, `compute_axis_alignment` with the primary rule, per-axis mean errors and flips) and the scale check, and returns a `Computation` (points, per-camera errors, axes, result, scale, max error; `.alignment(calibration)` makes the record). A target with fewer than two cameras is an `AlignmentError` naming it. Compute (16.6) and Recompute (16.8) both use it
- [x] Keeping the 2D clicks means the alignment can be **recomputed** with a new calibration (16.8) without clicking again
- [x] Tests: `tests/test_alignment_picks.py`

## 16.6 Align tab (`widgets/axis_alignment_view.py`)

- [x] Main window: tabs are **Calibrate | Align | Label** (`CALIBRATE_TAB`, `ALIGN_TAB`, `LABEL_TAB` = 0, 1, 2; stack in the same order). `_on_tab_changed` shows the frame controls only on Label; the window still starts on Label. Align is disabled, with a tooltip, while the experiment has no active calibration (`_update_align_tab`, on load, experiment switch and calibration change). The session file doesn't record the tab, so nothing there changed
- [x] `AxisAlignmentView(QWidget)` has its own `CameraGrid` and `FrameProvider` (an image is a 1-frame video). No `ProjectState` coupling. It loads when first shown, and reloads on experiment switch and when the active calibration changes; the frame is drawn once the tab is shown, since hidden panels skip frame updates
- [x] One panel on the right, top to bottom: status line; **Axis take** (take combo, **Add take…**, frame spinbox numbered from 1, hidden for 1-frame takes); **Axes** (clicks per axis, third axis, primary axis, flip X/Y/Z, known lengths in cm, — for none); the pick table (point, cameras clicked, mean error after Compute; clicking a row makes it the active point); **Next point**, **Clear clicks**, **Compute**, **Apply**; the results area
  - Status line: no alignment / in use (computed with *label*) / stale ("computed with *label*; the 3D points now come from *label*"). **Recompute** comes with 16.8
  - Settings are saved to `[alignment.{exp}.settings]` as they change. A derived axis can't stay the primary one
- [x] Picking:
  1. The active point puts every `CameraPanel` into pick mode; **Esc** stops (the panels pan again)
  2. A left click sets (or replaces) that camera's click for the active point (`raw_clicked`); a right click within 12 screen pixels of a click removes it (`empty_right_clicked`)
  3. Clicks are drawn as labelled markers (`O`, `X1`, `X2`, …, coloured by axis). The active point's clicks draw epipolar lines in the other cameras (`EpipolarCache`)
  4. Once the active point has a click in every camera the take has, the next unfinished point becomes active; **Next point** or **Enter** moves on by hand
  5. A point needs clicks in ≥ 2 cameras to be triangulated (`compute_alignment` says which don't)
  6. Clicks are saved as they are made (`write_picks`) and shown again when the take and frame are opened
- [x] **Compute** (`alignment.compute_alignment`, 16.5) with the active calibration. The results area shows per-point, per-camera reprojection errors; angles between the clicked axes and how far each moved; the primary axis and why; the scale check (a `!` beyond 2%); `R_align` and the origin; and, when an alignment is in use, the rotation angle and origin shift from it. The overlay adds hollow markers at the reprojected points and x/y/z arrows from the new origin (un-aligned with `unalign_points` and projected; length = the known length or the clicked distance). Errors appear in the results area
- [x] **Apply** saves `[alignment.{exp}.transform]` and `source` against the active calibration (`Project.save_alignment`, earlier transform to `history`) and emits `alignment_applied`. The main window then offers to re-triangulate when the 3D points come from another calibration (`_retriangulate_experiment`). The calibration and the points file are not touched
- [x] Tests: `tests/test_axis_alignment_view.py` (picking, saving, settings, Compute, Apply, the main window's tabs), driven through the view's methods; checked by rendering the tab offscreen, not yet in the app

## 16.7 Aligned output

Once an experiment has an alignment, un-aligned 3D points are never output.

- [x] `export.py`: compute reprojection errors from the stored calibration-frame points first (as now, at the `p3d` step), then replace the `3d` x, y, z columns with `align_points(...)`, in both wide and long layouts. `experiment_rows` takes the `Alignment`; the long layout is built from the aligned wide rows
- [x] A stale alignment blocks export of that experiment's 3D columns (`stale_alignments`; `export_points` raises unless the experiment is in `ExportOptions.two_d_only`, which also skips the stale-triangulation check for it):
  - GUI (`export_dialog.py`): says which experiments are stale and offers **Recompute alignment** or **Export 2D only** for them. Recompute closes the dialog with `recompute_requested`, and the main window opens the Align tab for that experiment; the recompute itself is 16.8
  - CLI (`label3d-export`): exits with an error naming the experiments, unless `--no-3d` is given. `--no-3d` exports every selected experiment 2D only
  - An experiment with no alignment exports as now
- [x] Export records the alignment: Parquet key-value metadata `label3d.alignment` holds JSON with each experiment's `R` (by rows), `origin` and `calibration` (`read_alignment_metadata`); CSV gets nothing extra. The columns stay as they are; the docs say that 3D is aligned when the experiment has an alignment, and the summary lists the aligned experiments
- [x] `view_3d.py` (hidden by `SHOW_3D_VIEW`): `set_alignment` draws points, edges and camera frustums aligned; `LabelView.reload_alignment` sets it on experiment change and after Apply
- [x] Main window status bar: next to the stale-triangulation warning, "Alignment is out of date; *recompute*" when stale. The link opens the Align tab until 16.8 adds the recompute

## 16.8 Recomputing and re-aligning

- [x] An experiment can be re-aligned at any time: new clicks (same or another take/frame) and **Apply** replace the transform, keeping the old one in `history` (`alignment.with_alignment`, since 16.2)
- [x] **Recompute** (Align tab button, status-bar link, export dialog): `AxisAlignmentView.recompute` shows the take and frame of `source` with their saved clicks from `alignment_picks.parquet`, and computes again with the current settings, which are the saved ones. The result is shown as after Compute, headed "Recomputed from the clicks saved for …", for the user to **Apply**. The button is greyed out without an alignment. It triangulates with the **active** calibration, as Compute does, not with `triangulated_with`: when the two differ, the 3D points are stale anyway, and Apply then offers to re-triangulate. The status-bar link and the export dialog's **Recompute alignment** open the Align tab (switching experiment if needed) and recompute
- [x] After a new charuco solve, or a switch to a calibration the alignment wasn't computed with, the status turns stale once the 3D points come from it (re-triangulation sets `triangulated_with`); the status bar and the Align tab's status say so. Nothing is recomputed silently
- [x] When the clicks no longer triangulate well with the new calibration (largest error above both 2 × the previous `max_error_px` and that + 1 px; `RECOMPUTE_ERROR_FACTOR`, `RECOMPUTE_ERROR_MIN_PX`), the results say so and suggest checking the drawn axes and clicking again. Missing clicks or a removed take are reported instead of computing

## 16.9 Versions

- [ ] New pieces: `[alignment]` and `[camera_aliases]` tables in `project.toml`, `alignment_picks.parquet`, still-image takes. Projects without them stay valid, so no upgrade step is needed
- [x] `LAYOUT_VERSION` is 0.4.0. The `type` column of `trials.csv` (`trial`, `calibration`, `axis`; any case, blank = trial) replaces `is_calibration`. The upgrade step `TRIAL_TYPES` (`layout.py`, `upgrade._add_type_column`) converts it in place, backing up `trials.csv`. A project that records 0.3.0 only gets steps from later layouts. `check_not_newer` also stops older label3d from opening the project. Tested on a hand-built 0.3.0 project in `test_upgrade.py`, not a fixture from a v0.3.0 worktree

## 16.10 Tests

- [ ] `tests/test_axis_alignment.py`, on a synthetic rig with a known rotation and origin; picks are made by projecting known 3D points:
  - 4-point and 7-point modes recover `R_align` and `origin`; 7-point is unaffected by the axis points being displaced from the origin
  - Each two-axis pair with the third from the cross product gives det +1 and the expected third axis
  - Left-handed picks raise; parallel axes raise; fewer than two axes raise
  - Primary-axis rules: longest, lowest error, explicit; a derived axis can't be primary
  - Noisy picks: the primary axis is kept exactly, the others move by the reported angles
  - Scale check flags a mis-scaled rig
  - `align_points` / `unalign_points` round-trip, keep distances, pass NaN through
- [ ] `[alignment]` record: written and read back, with 1-based frames in the file and 0-based in memory; per-camera frames include `cam_start` for a trial with a nonzero offset; re-apply moves the old transform to `history`
- [ ] `alignment_status`: current for the same entry and for a refine child of it, stale for another charuco entry, none without a transform
- [ ] Export: 3D columns aligned in wide and long layouts; reprojection errors unchanged by the alignment; stale alignment refused (GUI helper and CLI), `--no-3d` allowed; Parquet metadata holds the transform
- [x] Recompute: saved clicks re-triangulated with a new calibration rebuild the transform (`TestRecompute` in `test_axis_alignment_view.py`: the same rig in a moved frame gives `Q @ R_TRUE`, a shifted camera gives the warning, the status-bar link recomputes)
- [ ] Still-image takes: a 1-frame take loads, measures `_end` = 1, and is left out of charuco detection
- [ ] Setup (`test_setup_project.py`, `test_setup_from_slp.py`):
  - align keywords make axis takes from videos and images, in both routes
  - a file matching both an align and a calibration keyword is an error
  - calibration images still raise
  - alignment files never become trials
  - reference mode copies alignment files into the project
  - axis rows and `[camera_aliases]` are written
  - `save_config_toml` round-trips `align_keywords` / `align_regex`
- [ ] Add-take logic (`match_axis_files` and the function the dialog calls to add the take): camera matching with aliases, unique take names, files copied not moved, `trials.csv` row and `[alignment]` entry added
- [ ] Remove the `align` registry test from `test_calibration_registry.py`
- [ ] Synthetic data: an optional `[alignment_object]` in `synthetic_config.toml` renders a still image of axis markers per camera at a known pose. That gives an end-to-end check that alignment recovers the generator's world frame from the charuco-anchored calibration frame

## 16.11 Docs

- [x] `docs/user_workflow.md`: the Align step between Calibrate and Label (section 3; later sections renumbered, links updated); adding an axis take; re-aligning and recomputing; 3D export columns are aligned (16.7)
- [x] `docs/design/data_model.md`: axis takes, picks file, `[alignment]` and `[camera_aliases]` tables, where the transform applies, Parquet export metadata; the old `align` registry entry and `[metadata.alignment]` removed
- [x] `docs/design/project_setup_from_sleap.md`: alignment files are found beside the `.slp` files like calibration videos (done in 16.3)
- [x] `docs/design/workflows.md` (tab bar, the Axis Alignment section rewritten for the Align tab, export checks, 3D view), `docs/design/architecture.md` (modules), `DESIGN.md` (purpose, phase status table brought up to date), `CLAUDE.md` (workflow steps, data files, phase status)

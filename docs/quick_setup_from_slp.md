# Quick Setup: New Project from an Existing SLP File

How to install label3d and turn an existing `.slp` file plus a folder of calibration
videos into a label3d project, using a setup TOML file. This is the fast path for
the common case: you already labeled in SLEAP, the videos it references are still
where they were recorded, and you have separate calibration recordings (one or more
takes) sitting next to it. For all the options this supports, see
[project_setup_from_sleap.md](project_setup_from_sleap.md).

Worked example throughout: `projects/OMR_flat/`.

## 1. Install

Requires `uv` (already installed).

```bash
git clone <repo-url> label3d
cd label3d
uv sync
```

That creates `.venv/` and installs everything from `pyproject.toml` (PySide6,
aniposelib, sleap-io, etc.). Nothing further to build — `uv run` picks up the venv
automatically from here on.

## 2. Lay out your inputs

Setup doesn't require any particular folder structure for the inputs — it reads
paths recorded inside the `.slp` file to find the trial videos, wherever they
live, and separately looks for calibration videos in a folder you point it at.
`projects/OMR_flat/` keeps everything in one flat folder for convenience:

```
projects/OMR_flat/
├── OMR_setup.toml
├── zebrafish-single-LR.v001.slp
├── 20260619-cal01-left-GX010457_reencoded.mp4
├── 20260619-cal01-right-GX010391_reencoded.mp4
├── 20260721-cal01-left-GX010553_reencoded.mp4
├── 20260721-cal01-right-GX010486_reencoded.mp4
├── 20260721-cal03-left-GX010597_reencoded.mp4
├── 20260721-cal03-right-GX010530_reencoded.mp4
├── 20260807-cal01-left-GX010698_reencoded.mp4
├── 20260807-cal01-right-GX010630_reencoded.mp4
├── 20260807-cal02-left-GX010740_reencoded.mp4
└── 20260807-cal02-right-GX010672_reencoded.mp4
```

- The `.slp` file (`zebrafish-single-LR.v001.slp`) carries the skeleton, the labels,
  and the recorded path of every trial video — those videos are **not** in this
  folder; they stay wherever they were recorded. Setup only needs to be able to
  find them (see `video_search_dirs` below).
- The calibration videos **are** here, one per camera per take. The `-cal01-`,
  `-cal02-`, `-cal03-` tokens each name a separate calibration take (e.g. one
  recorded per experiment day), and `-left-`/`-right-` name the camera.

## 3. Write the setup TOML

`OMR_setup.toml`:

```toml
[setup]
slp_files = ["projects/OMR_flat"]          # .slp files or folders holding them
output_dir = "projects/OMR"
camera_names = ["left", "right"]
calib_regex = "cal\\d+"                 # alternative: regex with a "take" group
experiment_keywords = ["20260619", "20260721", "20260807"]   # omit or [] for single experiment named "default"
match_strategy = "keyword"   # "keyword" | "regex" | "number" | "datestamp" | "mtime"
trial_keyword = "trial"      # only for keyword strategy

# SLP-first setup: give slp_files instead of source_dir
video_mode = "reference"                 # "reference" | "copy" | "move"
video_search_dirs = ["/Volumes/TytellLab$/NewData/VerticalOMR/3_edited_fish_videos"]      # searched when a recorded path moved
calibration_dir = "projects/OMR_flat"   # or calibration_dir to search
import_labels = true                     # import SLP labels into points.csv
allow_missing_videos = true

[project]
name = "OMR"

[charuco]
squares_x = 8
squares_y = 10
square_length = 10
marker_length = 7
marker_bits = 6
dict_size = 50
```

Field by field:

- **`slp_files`** — a `.slp` file or a folder containing one or more; every `.slp`
  found must share the same skeleton.
- **`output_dir`** — where the new project is created. Must not already exist.
- **`camera_names`** — matched against each video's filename, then its parent
  folder name, to tell cameras apart. Here: `left`, `right`.
- **`experiment_keywords`** — one label3d "experiment" per acquisition day here
  (`20260619`, `20260721`, `20260807`), matched against filenames. Leave this out
  (or `[]`) if everything belongs to one experiment.
- **`match_strategy`** / **`trial_keyword`** — how *trial* videos are grouped
  into left/right pairs. This project's trial videos happen to be named with the
  word "trial" in them, so `keyword` + `trial_keyword = "trial"` is enough; other
  strategies (`regex`, `number`, `datestamp`, `mtime`) exist for other naming
  schemes.
- **`calib_regex`** — how *calibration* videos are told apart from trial videos
  and grouped into takes. `"cal\\d+"` matches `cal01`, `cal02`, `cal03` in the
  filenames above, and each distinct match becomes one calibration take.
- **`video_mode = "reference"`** — trial videos are **not** copied or moved; the
  project just records their real path. (Calibration videos are always copied
  into the project, since they're small and the project should own its
  calibration inputs.)
- **`video_search_dirs`** — extra folders to search when a path recorded inside
  the `.slp` no longer exists (e.g. the drive was remounted elsewhere). Setup
  tries to match as much of the recorded folder structure as it can under each
  search directory.
- **`calibration_dir`** — where to look for calibration videos (here, the same
  flat folder). Could instead be `calibration_toml` pointing straight at an
  existing aniposelib `calibration.toml`, if you already have one. It's imported
  into the project's calibration registry and made active.
- **`allow_missing_videos = true`** — proceed even if some trial videos can't be
  located; the affected camera is left blank in `trials.csv` rather than
  aborting setup.
- **`import_labels = true`** — after the project is created, import the `.slp`
  labels straight into `points.csv`, so the project opens already labeled.
- **`[charuco]`** — your calibration board's parameters (squares, marker size in
  cm, ArUco dictionary), used when you calibrate in the app.
- **`[skeleton]`** — optional. Either `file = "path/to/skeleton.json"` (relative
  to the working directory) or `nodes`, `edges`, `symmetries` and
  `node_offsets_cm` inline. It must match the `.slp` files' skeleton (same
  nodes, edges and symmetries) or setup stops with an error; a matching one wins,
  so its node order and offsets are kept. Needed for a flat folder with no
  `.slp` files, or add one later with **File > Load Skeleton…**.

## 4. Run it

```bash
uv run python setup_project_cli.py projects/OMR_flat/OMR_setup.toml
```

This prints a preview (trials found, calibration takes found, any warnings about
unresolved videos) and writes the project to `output_dir`. Add `-n`/`--dry-run`
first to see the preview without writing anything.

## 5. Open the project

```bash
uv run label3d projects/OMR
```

If any trial videos were left unresolved (reported in the preview / warnings),
re-run setup after widening `video_search_dirs`, or fix them later by editing
`trials.csv` directly.

## 6. Using the app

The app has three tabs, used in order: **Calibrate → Label → Export**.

### Calibrating

If the setup TOML already found a `calibration.toml` (as in `OMR`, via
`calibration_dir`), it was imported as the active calibration and you can skip
straight to Label. To calibrate a take from scratch:

1. Open the Calibrate tab; pick the experiment/take to calibrate.
2. The app steps through the calibration video for each camera, detecting
   Charuco board corners every Nth frame (Charuco parameters come from your
   `[charuco]` settings, or a settings panel if you didn't set them at setup).
   Green dots mark detected corners.
3. Click **Run Calibration** to bundle-adjust all cameras together
   (`aniposelib`). The panel reports per-camera and overall reprojection error;
   red crosses are the reprojected corners overlaid on the detections — tight
   overlap means a good calibration.
4. Every run is saved to the experiment's calibration registry
   (`experiments/{exp}/calibrations/`), one solution per take plus the joint
   one, and the joint solution becomes active. **Use selected take** switches to
   a single take's solution; the **Saved calibrations** list holds them all.
   Set **Repeat calibration** above 1 to keep the best of several runs if the
   error is high.
5. (Optional) **Axis alignment** — a separate step, run after calibration, to
   re-orient the arbitrary board-defined axes to something physically
   meaningful (e.g. tank walls). Pick an origin plus two axis points in the
   trial videos; doesn't touch the camera calibration itself.

### Labeling

Each camera's video appears in its own panel, with the full skeleton drawn on
top.

**Adding an instance** (one animal/object to label in the current frame):

- `Ctrl+I` — add an instance, positioned by averaging nearby frames when
  possible, otherwise at a random position.
- `Ctrl+Shift+I` — add an instance at a random position unconditionally.
- Once added, the instance's bodyparts are selectable in turn (`↑`/`↓`, or
  `` ` `` to cycle whole instances, `1`–`9` to jump to one by index); click in
  any camera panel to place the active bodypart there. Points already placed
  in other views show epipolar lines for the selected bodypart, so you can
  click the same physical point precisely.
- `Ctrl+Backspace` deletes the selected instance; `Ctrl+C`/`Ctrl+V` copies one
  to another frame.
- `T` triangulates the current frame (also happens automatically after each
  point edit); `Ctrl+Shift+T` triangulates the whole trial.

**Zooming and panning** (per camera panel):

- Scroll wheel zooms in/out, clamped so you can't zoom out past the full frame.
- Drag on empty space (or middle-click drag) to pan while zoomed in.
- Double-click resets the view to fit the whole frame.
- `Ctrl+=` to zoom to the instance
- `Ctrl+-` to zoom to a small area around the selected point.
- `Ctrl+0` to zoom to the whole frame.

**Other useful shortcuts:**

| Key | Action |
|---|---|
| `→` / `←` | Next / previous frame |
| `]` / `[` | Next / previous labeled frame |
| `Ctrl+→` / `Ctrl+←` | ±4 frames |
| `H` | Toggle showing all instances |
| `Ctrl+Tab` | Toggle node name labels |
| `Backspace` | Delete the selected point |
| Right-click a point | Mark it occluded |

Full list in [workflows.md](workflows.md).

### Exporting

The Export tab writes one `.slp` per (trial, camera) for SLEAP training. Pick
which trials and which point states (user / predicted) to include, then
**Export**. To close the loop: train on the exported `.slp` files in SLEAP, run
inference, drop the resulting predictions into `slp/`, and re-import (File >
Import from SLP files) to review and correct them as predicted instances.

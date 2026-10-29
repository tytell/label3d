# Setting up a project from SLEAP files

Builds a label3d project from existing `.slp` files. The skeletons and labels come from the `.slp` files. The trial videos stay where they are, and the project records their paths. Calibration comes from calibration videos or from an existing aniposelib `calibration.toml`.

After setup, continue with [user_workflow.md](user_workflow.md#2-calibrate).

## Install

```bash
git clone <repo-url> label3d
cd label3d
uv sync
```

## Steps

1. Copy a template next to your `.slp` files and edit it:
   - [templates/setup_one_slp.toml](templates/setup_one_slp.toml): one `.slp` file, with the calibration videos in the same folder.
   - [templates/setup_many_slp.toml](templates/setup_many_slp.toml): several `.slp` files with different skeletons, camera aliases, and calibration videos in another folder.
2. Check it with `inspect_slp_cli.py` ([below](#checking-with-inspect_slp_clipy)) until every `.slp` file has trials in each experiment and nothing unexpected is missing.
3. Preview without writing anything: `uv run python setup_project_cli.py -n setup.toml`
4. Create the project: `uv run python setup_project_cli.py setup.toml`
5. Open it: `uv run label3d <output_dir>`

Relative paths in the setup file are relative to the folder you run the command from.

## Setup file keys

All keys go under `[setup]` unless noted otherwise.

| Key | Meaning |
|---|---|
| `source_dir` | Every `.slp` file in the folder and its subfolders is used, and the videos there that no `.slp` file refers to become trials too. |
| `slp_files` | If your `.slp` files are in a different folder, or you only want to use some of the `.slp` files, you can list the filenames or folder names here. Folders are searched with their subfolders. |
| `output_dir` | The new project folder. It must not exist yet. |
| `camera_names` | Each video is matched to a camera by its filename, then by its parent folder's name. |
| `camera_aliases` | `{ camera = ["other name", ...] }`. Other names a camera has in filenames. The longest matching name or alias wins. |
| `experiment_keywords` | One experiment per keyword, matched in filenames. Videos that name none of them are left out. Leave this out to put everything in one experiment, `default`. |
| `match_strategy` | How videos are grouped into trials: `keyword` (with `trial_keyword`; the filename up to the keyword and its number, e.g. `PS21_Trial7`, names the trial), `regex` (with `trial_regex`; group 1 is the trial), `number`, `datestamp` (with `datestamp_window_sec`), or `mtime` (with `mtime_window_sec`). |
| `calib_keywords` | One keyword per calibration take, e.g. `["Pre-Cal", "Post-Cal"]`. |
| `calib_regex` | Alternative to `calib_keywords`: each distinct match is a take, named by the `take` group if there is one. |
| `video_mode` | `reference` (leave videos in place), `copy`, or `move`. Calibration videos are always copied into the project. |
| `video_search_dirs` | Folders to search when a recorded video path no longer exists. |
| `calibration_dir` | Folder to search for calibration videos or a `calibration.toml`. Without it, setup searches `source_dir` and the `.slp` files' folders. |
| `calibration_toml` | An existing aniposelib calibration. It is imported and made active, so you can skip calibrating. |
| `allow_missing_videos` | `true` sets the project up with the videos that were found, and leaves the missing ones blank in `trials.csv`. |
| `import_labels` | `true` (the default) imports the `.slp` labels into `points.parquet`. |
| `[project] name` | Defaults to the name of the output folder. |
| `[charuco]` | The calibration board: `squares_x`, `squares_y`, `square_length`, `marker_length`, `marker_bits`, `dict_size`. |
| `[skeleton]` | Optional: `file = "skeleton.json"`, or inline `nodes`, `edges`, `symmetries` and `node_offsets_cm`, plus optional `cameras = [...]`. It is only needed for cameras that have no `.slp` file yet. |

### How the videos are found

A `.slp` file records each video's path from when it was labelled. If that path no longer exists (another drive letter, another mount, Windows to Mac), setup looks under each `video_search_dirs` entry for:

1. A folder with the same name as the search directory somewhere in the recorded path. Everything below that folder is kept.
2. Failing that, the longest end of the recorded path that exists under the search directory.
3. Failing that, the same filename beside the `.slp` file or in a search directory.

If two search directories match equally well, the video is left unresolved. Narrow `video_search_dirs` to pick one of them.

### Calibration

For each experiment, setup uses the first of these it finds:

1. `calibration_toml`.
2. A `calibration.toml` in the searched folders whose camera names match the project's.
3. Calibration videos in the searched folders, found by `calib_keywords` or `calib_regex`. With `experiment_keywords` set, a calibration video's filename must also name its experiment, e.g. `Pre-Cal_PS21_Charuco_Ventral.mp4`.

An experiment with none of these is set up without a calibration.

### Skeletons

Each `.slp` file keeps its own skeleton, and the project skeleton is the union of the skeletons, matched by node name. A camera can't label a node its skeleton lacks. Setup stops only when two skeletons disagree about a symmetry between two nodes they share. With `experiment_keywords` set, each `.slp` file must have at least one video in a trial of a listed experiment.

## Checking with `inspect_slp_cli.py`

This command reads the `.slp` files and shows what setup would do. It writes nothing. It takes the same setup file as setup does (`output_dir` is optional):

```bash
uv run python inspect_slp_cli.py BGvPS_flat/BGvPS_setup.toml
```

```
SLP files (2):
  .../BGvPS_flat/BGPS_Ventral.v002.slp
    226 video(s), 226 found
    BG75: 26 trial(s)
    PS21: 30 trial(s)
    170 video(s) in an experiment not listed
  ...

Recorded video folders (4):
  D:/Olivia/BluegillVsPumpkinseed/VideoConversions/VENTRAL VIDEOS  [NOT FOUND]  226 video(s), 226 located
    -> /Volumes/Lab/.../VideoConversions/VENTRAL VIDEOS  (226 video(s))
  ...

------------------------------------------------------------
What setup would do:
Project, cameras, skeleton (and the nodes each .slp file lacks), strategy,
trials per experiment, calibration, and what was left out
```

The output has three sections:

- **SLP files**: for each file, the number of videos found, the trials per experiment, and a count of each problem: `not found`, `ambiguous`, `no camera match`, `in an experiment not listed`, or `not in a trial of a listed experiment`.
- **Recorded video folders**: each folder the `.slp` files recorded, whether it exists, and where its videos were found instead (`->`). This shows whether `video_search_dirs` points at the right tree.
- **What setup would do**: the same preview as `setup_project_cli.py -n`.

To see more, add `-v` for the trials that are missing cameras, `-vv` for each placed video and short problem lists, and `-vvv` for everything.

Flags override the file, so you can try other settings without editing it:

```bash
uv run python inspect_slp_cli.py setup.toml --match-strategy regex --trial-regex '(Trial\d+)'
uv run python inspect_slp_cli.py setup.toml --experiments PS21,BG75,PS37
uv run python inspect_slp_cli.py setup.toml --camera-alias LateralOrtho=Lateral
uv run python inspect_slp_cli.py setup.toml --video-search-dir /Volumes/Lab/NewData
```

You can also run it without a setup file, before writing one:

```bash
uv run python inspect_slp_cli.py BGvPS_flat/ --cameras LateralOrtho,LateralOffAxis,Ventral
```

The other flags are `--trial-keyword`, `--mtime-window`, `--datestamp-window` and `--calib-keywords`.

### Reading the problems

| Problem | Fix |
|---|---|
| `not found`; folder `[NOT FOUND]` with no `->` | Add the folder that now holds the videos, or a parent of it, to `video_search_dirs`. |
| `ambiguous` | Two search directories match. Remove one or make it more specific. |
| `no camera match` | Add the name used in the filename or folder to `camera_aliases`. |
| `in an experiment not listed` | Expected for experiments you are leaving out. Otherwise add the keyword to `experiment_keywords`. |
| `not in a trial of a listed experiment` | The match strategy didn't pair the video with the other cameras. Try another `match_strategy` or `trial_regex`, and use `-v` to see the trials that are missing cameras. |
| Calibration incomplete for an experiment | Check `calibration_dir` and the take keywords, and check that the calibration filenames name the experiment. |

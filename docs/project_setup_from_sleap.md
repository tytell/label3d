# Project Setup from SLEAP Files

**Goal**: Build a label3d project from one or more existing `.slp` files instead of a flat
folder of videos. The videos stay wherever they already live — the project references them by
path rather than copying them in. The skeleton and the labels come out of the `.slp` files;
calibration, which `.slp` files do not carry, is found next to them or supplied by the user.

**Status**: Implemented. `classify_slp_files()` / `import_labels()` in
`setup_project.py`, path resolution in `project.py`, `--from-slp` in
`setup_project_cli.py`, tests in `tests/test_video_paths.py` and
`tests/test_setup_from_slp.py`.

## Motivation

The existing setup utility ([data_model.md](data_model.md) § Project Setup Utility) assumes a
flat folder of videos that it moves or copies into `experiments/{name}/videos/`. A user who has
already labeled in SLEAP has a different starting point: a handful of `.slp` files that name
their videos by absolute path, videos scattered across one or more acquisition folders that may
be large or read-only, a skeleton already defined, and ground-truth labels worth importing.
Copying gigabytes of video to restructure a folder they already have is wasteful, so this path
references the videos in place.

## Overview

```
.slp files ──► resolve video paths ──► identify cameras ──► group into trials
                      │                                            │
                      ├──► skeletons ─► (union)                    ├──► trials.csv (path cells)
                      │                                            └──► project.toml [video_roots]
                      └──► labels ───────────────────────────────► points.csv
calibration.toml or calibration videos ────────────────────────────► experiments/{exp}/calibrations/ (registry)
```

Both entry paths end in the same on-disk project, and everything downstream (labeling,
triangulation, export, SLP sync) is unchanged.

---

## 1. External video paths

Today a `trials.csv` camera cell holds a bare filename resolved against
`experiments/{exp}/videos/`. Referenced projects need path-valued cells, in two forms chosen
automatically at setup time.

`project.toml` gains an optional table:

```toml
[video_roots]
default = "/Volumes/data/2026-03-14"
PS18 = "/Volumes/data/PS18"
```

Setup computes the common parent of all of an experiment's videos. If one exists, it writes that
parent as the experiment's `video_root` and relative cells; otherwise it writes absolute cells and
no root. Relative cells keep the project portable when the video drive remounts elsewhere — one
line to repoint — while absolute cells are the fallback for videos spread across unrelated trees.

Resolution rule, applied wherever a camera cell becomes a path:

1. Cell is absolute → use as-is.
2. Experiment has a `video_root`, or the cell contains a path separator → resolve against the
   root, falling back to the project root when there is no root.
3. Bare filename, no root → `experiments/{exp}/videos/{filename}`, exactly as today.

Existing projects have no `[video_roots]` and only bare filenames, so rule 3 preserves their
behavior byte for byte.

### Tasks (`project.py`)

- [x] `Project.load()`: read `[video_roots]` from `project.toml` into `ExperimentInfo` (new
      `video_root: Path | None` field).
- [x] New helper `_resolve_video(cell: str, video_root: Path | None, videos_dir: Path, root: Path) -> Path`
      implementing the three-way rule; unit-test each branch.
- [x] `Project.video_path()` ([project.py:178](../src/label3d/project.py#L178)) — use the helper.
      Calibration takes keep their `calibration/{camera}/` location unless their cell is
      path-valued, in which case the same helper applies.
- [x] `_resolve_trial_length()` ([project.py:435](../src/label3d/project.py#L435)) — use the
      helper; it currently takes `videos_dir` and joins the filename directly.
- [x] A referenced video that does not exist raises a clear load-time error naming the trial,
      camera, and attempted path, rather than failing later inside OpenCV.

---

## 2. Reading the SLP files

### Video references

`_read_slp_info()` ([setup_project.py:630](../src/label3d/setup_project.py#L630)) currently
returns basenames taken from `labels.labeled_frames`. It changes to return resolved full paths
taken from `labels.videos`, so a video that is present in the file but has no labeled frames
still becomes part of the project.

A `.slp` records the video path as it was when labeling happened, which is often stale — usually
because the whole tree moved: the drive letter and the folders above the collection changed, the
structure below it did not. Resolution therefore keeps as much of the recorded structure as it
can. The recorded path is first split into components with `_recorded_parts()`, which tolerates
backslashes and drops a leading drive letter, so a Windows-recorded path is comparable on a posix
host. Then, in order:

1. The recorded path, if it exists.
2. **Anchor** — a `video_search_dirs` entry whose own directory name occurs, matched exactly, as a
   directory component of the recorded path. Everything recorded below that component is joined
   onto the search directory. The deepest occurrence wins when a name repeats.
3. **Longest suffix** — the deepest trailing run of recorded components that exists under a search
   directory. This catches a collection folder that was renamed on the new mount, where no name
   anchors.
4. The same basename in the directory containing the `.slp` file, then in each `video_search_dirs`
   entry, in order — the historical behavior, last so it cannot beat a structurally correct match
   in the wrong camera folder.
5. Unresolved — collected into `FileAssignment.missing_videos` and shown in the preview.

For example, a recorded
`R:/NewData/VerticalOMR/3_edited_fish_videos/20260619_zebrafish_1_synced_trimmed/left/trial01.mp4`
with `video_search_dirs = ["/Volumes/TytellLab$/NewData/VerticalOMR/3_edited_fish_videos"]` anchors
on `3_edited_fish_videos` and resolves to
`/Volumes/TytellLab$/…/3_edited_fish_videos/20260619_zebrafish_1_synced_trimmed/left/trial01.mp4`.

When two search directories match equally well at the same step — two mounts of the same tree, say
— the reference is **left unresolved** with a warning naming the candidates, rather than resolved
to an arbitrary one of them. Narrow `video_search_dirs` to pick one.

`_resolve_video_reference()` returns a `ResolvedVideo` recording which rule fired and which
recorded prefix was replaced by which search directory. Those prefix replacements are tallied in
`FileAssignment.video_remaps` and the preview prints one line per remapped tree, so a wrong mount
is visible before anything is written.

Unresolved references are **not fatal** during classification; the preview lists them.
`execute_setup` refuses to run unless they are resolved or `allow_missing_videos` is set
(`--allow-missing-videos`, or `allow_missing_videos = true` in a setup TOML), which sets the
project up from the videos that *were* located: the affected cameras simply drop out of their
trials and their `trials.csv` cells are left blank, which `Project.load()` already tolerates.

### Skeleton

Nodes, edges and symmetries are read from each file's `labels.skeleton` into
`FileAssignment.slp_skeletons`. Since Phase 14 the files may have different skeletons;
`merge_skeletons` builds their union, and only a conflict (shared nodes that disagree on a
symmetry) is an error; edges may differ, since each camera draws its own skeleton's edges. Nothing is written for them: the project reads the
skeletons from the `.slp` files. A setup file `[skeleton]` must merge with them and is
written to `project.toml` as an assignment.

### Cameras and trials

Camera names come from the user's list, as in the folder-based path. Each resolved video path is
matched in two stages: the substring rule (`_identify_camera`) against the filename, then, if that
fails, against the parent directory name — the common layout where each camera writes into its own
folder and filenames carry no camera token.

Trial grouping reuses `_run_match_strategy()` untouched. It already operates on `(camera, Path)`
pairs and never assumes the paths share a directory, so `keyword`, `regex`, `number`, `datestamp`,
and `mtime` all work against external paths as-is. Experiment assignment likewise reuses
`_identify_experiment()` on the filename.

### Tasks (`setup_project.py`)

- [x] `SetupConfig`: add `slp_paths: list[Path]`, `video_mode: str` (`"move" | "copy" | "reference"`),
      `video_search_dirs: list[Path]`, `calibration_toml: Path | None`, `calibration_dir: Path | None`.
      `copy_files` stays as a deprecated alias mapping to `video_mode`.
- [x] `FileAssignment`: add `missing_videos: list[tuple[Path, str]]` (reference, source `.slp`) and
      `video_roots: dict[str, Path]`.
- [x] `_read_slp_info()` → return `(video_paths: list[Path], skeleton: dict)` using `labels.videos`.
- [x] `_resolve_video_reference(recorded: Path, slp_dir: Path, search_dirs: list[Path]) -> ResolvedVideo`
      — anchor, then longest suffix, then basename; ambiguous matches are refused.
- [x] `_identify_camera_from_path()` — filename, then parent directory name.
- [x] `classify_slp_files(config) -> FileAssignment` — the SLP-first counterpart to
      `classify_files()`, producing the same `FileAssignment` shape so the preview and execute
      steps are shared.
- [x] `_compute_video_roots(assignment) -> dict[str, Path]` — common parent per experiment, or
      absent when there is none.

---

## 3. Calibration

`.slp` files carry no calibration, so setup looks for it in this order, per experiment:

1. **An existing `calibration.toml`** — in `source_dir` and the directories containing the
   `.slp` files, or only in `calibration_dir` if given. Recognized by parsing it as an aniposelib `CameraGroup.dump()`
   and checking that its camera names match the project's. Saved as an `imported` entry in the
   experiment's calibration registry (`experiments/{exp}/calibrations/`), with
   `metadata.source_path` set to the original file, and made active. The calibrate step is then
   unnecessary.
2. **Calibration videos** — found by the usual `calib_keywords` / `calib_regex` rules in those
   same directories, matched into takes by the existing `_match_calibration_takes()`, and
   referenced in place like trial videos (rows in `trials.csv` with `is_calibration = True` and
   path-valued cells). The user calibrates in the app as usual.
3. **Prompt** — the CLI asks for a directory, re-running steps 1–2 against it. "Skip" is
   accepted; the project is created without calibration and the user adds it later.

Because calibration videos are referenced rather than moved, `Project.video_path()` for a
calibration take must fall through to the path-resolution rule when the cell is path-valued
(see § 1).

### Tasks

- [x] `_find_calibration_toml(dirs, camera_names) -> Path | None` — parse-and-validate, not just
      a filename match.
- [x] `_find_calibration_videos(dirs, config) -> list[tuple[str, Path]]` — reuse
      `_is_calibration()` and `_identify_camera_from_path()`.
- [x] `execute_setup()` — save a found `calibration.toml` as an `imported` registry entry and make it active (Phase 11).
- [x] Preview shows, per experiment, which of the three outcomes applies.

---

## 4. Importing the labels

Setup writes the project structure first, then imports labels through the machinery that
File > Sync with Sleap already uses, so that states (`set` vs `predicted`), instance ordering,
and multi-instance handling match what the in-app import produces. No merge logic is duplicated.

Order of operations in `execute_setup()`:

1. Write `trials.csv`, `project.toml` (including `[video_roots]`; `[skeleton]` only from a setup file).
2. Copy the `.slp` files into `slp/`. These are small, and the project format expects them
   there; only videos are referenced in place.
3. Build an `SlpMapping` from `assignment.slp_files` via `SlpMapping.from_cell_map()`, with paths
   relative to the project root, and write it as the `[slp]` section of `project.toml`.
4. Reopen the project with `Project.load()`, then for each experiment run
   `read_slp_for_sync()` → `apply_merge(results, overwrite=True)`. `overwrite=True` is correct
   here because `points.csv` starts empty, so there is nothing to conflict with.

A `--no-import` flag stops after step 3, leaving the `.slp` files mapped but unimported for a
user who wants to review the mapping in the app first.

### Tasks

- [x] `execute_setup()` — write the `[slp]` section when `assignment.slp_files` is non-empty
      (useful for the folder-based path too, which currently copies the files without mapping them).
- [x] `import_labels(project_dir) -> dict[str, int]` — reopen, sync per experiment, return
      instance counts for the summary.
- [x] Skip cleanly when a trial has no `.slp` coverage.

---

## 5. CLI

`setup_project_cli.py` gains an SLP-first branch:

```
uv run python setup_project_cli.py --from-slp labels/*.slp
uv run python setup_project_cli.py slp_config.toml
```

- `--from-slp FILE [FILE ...]` accepts files, globs, or a directory scanned for `.slp`.
- `--video-mode {move,copy,reference}`, defaulting to `reference` on the SLP path and `move` on
  the folder path (preserving today's behavior).
- `--video-search-dir DIR` (repeatable) for stale recorded paths.
- `--calibration FILE_OR_DIR`, `--no-import`.
- Interactive mode opens by asking whether to start from a folder of videos or from `.slp` files,
  then follows the matching prompt sequence. The SLP branch prompts for camera names, trial
  matching strategy, experiment keywords, and calibration location; it does not ask about the
  source folder layout.

Config-file mode grows `slp_files`, `video_mode`, `video_search_dirs`, `allow_missing_videos`, `calibration_toml`,
`calibration_dir` under `[setup]`, and `save_config_toml()` round-trips them. Without
`slp_files`, the `.slp` files in `source_dir` make the setup SLP-first (`config_from_toml`):
`source_dir` is also a video search directory, and its videos that no `.slp` file refers to
are added as trials (`SetupConfig.include_source_videos`).

`inspect_slp_cli.py` takes the same setup file (or `.slp` files and the matching flags) and
reports, without writing anything, each `.slp` file's video references (found where, by which
rule), the recorded video folders, and the setup preview.

### Preview additions

```
SLP files (3):
  cam1_labels.slp   4 videos, 182 labeled frames
  ...

Videos (referenced in place):
  [default]  root: /Volumes/data/2026-03-14
    trial_001  cam1: ok   cam2: ok   cam3: MISSING
  ...

Calibration:
  [default]  calibration found at /Volumes/data/2026-03-14/calibration.toml (imported into the calibration registry)
```

`execute_setup` refuses to run while any video is MISSING unless the user confirms, in which case
the affected camera cells are left blank.

---

## 6. Testing

- [x] Extend the synthetic generator ([synthetic_data.md](synthetic_data.md)) with a mode that
      writes `.slp` files referencing videos left **outside** the project tree.
- [x] `test_setup_from_slp.py` — resolution fallbacks (recorded path, `.slp` directory, search
      dirs, missing); camera identification by parent directory; `video_roots` computed and
      omitted; skeleton extraction and the inconsistency error.
- [x] `test_project.py` additions — the three-branch path resolution, including that existing
      bare-filename projects are unaffected.
- [x] One end-to-end test: generate → `--from-slp` → `Project.load()` → assert `points.csv`
      matches the labels in the source `.slp` files.

---

## Open questions

- Should a referenced project be convertible to a self-contained one later (a "localize videos"
  command that copies them in and rewrites the cells)? Not needed for the first pass.
- When two `.slp` files disagree about a frame's labels, `compute_merge` currently resolves
  conflicts against the store. During setup the store is empty, so the last file processed wins.
  Worth a warning if it happens in practice.

# Phase 20 — Project versions and upgrades

**Goal**: Give label3d a version number, record in each project which version its layout belongs to, and offer to upgrade a project made by an older version when it is opened. Upgrading restores the parts of the project's `.slp` files that older Syncs dropped, using the original `.slp` files the user points to.

**Status**: Complete. Tested on the fixture projects and on copies of the real BGvPS projects (20.12), and checked in the app. Tagged `v0.3.0`.

---

## 20.1 Versions

There was no version before this phase, so the versions are assigned after the fact to commits whose projects differ:

| Version | Commit | Project layout |
|---|---|---|
| 0.0.9 | `088c44f` | Last GitHub `main` before the Parquet merge. CSV working files (`points.csv`, `corners.csv`, `sync_history.csv`, `refined/{id}_points.csv`); `skeleton.json`; `.slp` files flat in `slp/`. Sync drops videos and frames outside the project |
| 0.1.0 | `8c09482` | Parquet working files; skeleton in `project.toml`; `.slp` files flat in `slp/`. Sync still drops videos and frames outside the project |
| 0.2.0 | `33d9095` | Same files as 0.1.0. Sync keeps videos and frames outside the project |
| 0.3.0 | this phase | `slp/{stem}/{name}.slp` with `models/` beside it; `px, py, pscore` point columns; `trials.csv` numbered from 1 (`[trials] frame_base = 1`); `[label3d]` in `project.toml` |

- `pyproject.toml` gets `version = "0.3.0"`. Later releases bump it as usual. The project layout version changes only when the layout does, so a 0.3.4 app can still write 0.3.0 projects.
- Git tags `v0.0.9`, `v0.1.0` and `v0.2.0` go on the commits above, and `v0.3.0` goes on the commit that finishes this phase. Tags are local until pushed.
- Projects older than 0.0.9 (pre-registry: `calibration.toml`, `calibration_takes/`, `calibration/{cam}/`) are still refused and set up again, as now.

## 20.2 The running version (`label3d/version.py`)

- `__version__` comes from `importlib.metadata.version("label3d")`.
- `commit()` is the short hash from `git rev-parse --short HEAD`, run in the package's folder, with `+dirty` if `git status --porcelain` lists changes to tracked files. Outside a git checkout, the hash comes from `_build_info.py`, which a small hatch build hook writes when a wheel is built (`_build_info.py` is gitignored). If neither is available, it is `"unknown"`.
- **Label3D > About label3d** shows both, with the project's layout version. `label3d --version` and `label3d-export --version` print them.

## 20.3 What `project.toml` records

```toml
# label3d 0.3.0 (b9b76dc)
[label3d]
project_version = "0.3.0"
written_by = "0.3.0 (b9b76dc)"
```

- `project_version` is the layout version. Only setup and upgrades write it.
- `written_by` and the comment line are rewritten every time label3d writes `project.toml` (`_write_config`, and the writer used in setup). The `toml` package drops comments, so `_write_config` puts the comment line back first.
- A project whose `project_version` is newer than the app's layout version is refused: "made with label3d X; this is Y, update label3d". Versions 0.3.0 and later check this. Older apps don't know the key and will open the project anyway; nothing can be done about that.

## 20.4 Working out an old project's version

A project without `[label3d]` is matched by its files, newest layout first (`detect_version(root)` in `upgrade.py`):

| Found | Version |
|---|---|
| `calibration.toml`, `calibration_takes/` or `calibration/{cam}/` in an experiment | older than 0.0.9: refused, as now |
| any CSV working file (`points.csv`, `corners.csv`, `sync_history.csv`, `refined/*_points.csv`) or `skeleton.json` | 0.0.9 |
| `.slp` files directly in `slp/` | 0.1.0 (0.2.0 looks the same; treated as 0.1.0, because it can't be known that Sync never dropped anything) |
| `.slp` files in `slp/{stem}/`, no `[label3d]` | 0.2.x development build (e.g. the current `projects/BGvPS`); only the steps whose condition holds run |
| no `.slp` files | no SLP restoration needed; the other steps by their conditions |

Each upgrade step has its own check (`applies(root) -> bool`), so detection only decides what the dialog says. A project half-way between layouts gets exactly the steps it needs.

## 20.5 Upgrade steps

`layout.plan_upgrade` lists the steps a project needs, each by its own check; `upgrade.run_upgrade` runs them in this order after the backup (20.6):

1. **Points** (0.0.9) — each `experiments/{name}/points.csv` → `points.parquet`. It is read as 0.0.9 did (`read_points_csv`: `header=[0, 1]`, `index_col=[0..4]`, experiment/trial/bodypart as strings, frame/instance as int; `normalize_columns`) and written with `write_points_file`. The `.csv` and `.csv.bak` files are removed (they are in the backup).
2. **Corners** (0.0.9) — `calibrations/{cam}/corners.csv` → `corners.parquet` (`take`, `detected` as strings).
3. **Sync history** (0.0.9) — `sync_history.csv` → `sync_history.parquet`.
4. **Refinement records** (0.0.9) — `refined/{id}_points.csv` → `{id}_points.parquet`, and `points_file` in the refinement's `.toml` renamed.
5. **Skeleton** (0.0.9) — `skeleton.json` → `[skeleton]` in `project.toml`, assigned to every camera (moved here from `Project.load`), and the file removed.
6. **SLP folders** — each `slp/{name}.slp` → `slp/{stem}/{name}.slp`, and `[slp]` paths in `project.toml` rewritten. `slp/backup/` stays where it is: `find_slp_files` skips it, and 20.7 reads the earliest backup from it.
7. **Frame numbering** — `trials.csv` renumbered from 1 and `[trials] frame_base = 1` (moved here from `Project.load`).
8. **Restore the `.slp` files** (from 0.0.9 and 0.1.0, when files are mapped) — 20.7. `project.toml` is written before this step, which loads the project with `Project.load(root, check_version=False)`.
9. **Stamp** — `project_version` written last, so an upgrade that stops part-way is found again on the next open.

`px, py, pscore` need no step: `normalize_columns` adds them, empty, when a points file is read.

An unversioned project that needs none of steps 1–8 (one set up between `b9b76dc` and this phase without flat `.slp` files) gets its version recorded by `Project.load`, with no dialog.

## 20.6 Backup

- Before any step, every file the upgrade will change, move or remove is copied to `upgrade_backup/{from_version}_{YYYYmmdd-HHMMSS}/` in the project, keeping its relative path: `project.toml`, `trials.csv` (and `.bak`), the CSV working files and their `.bak` files, `skeleton.json`, the refinement `.toml` files next to CSV records, each `points.parquet` (and `.bak`), `sync_history.parquet`, and the `.slp` files in `slp/` outside `models/` and `backup/`. Videos, calibration videos, `models/` and `slp/backup/` are not copied, since the upgrade doesn't change them.
- Files are copied with `shutil.copy2`. On APFS this is still a full copy; for BGvPS the `.slp` files are about 250 MB.
- If a step raises, files and folders that weren't there before are removed (a listing of the project is taken first), the backed-up files are copied back, and the error is raised again. The backup is kept.
- The backup folder is not inside `slp/`, so `find_slp_files` doesn't see it.
- A note in the user docs says the folder can be deleted once the upgraded project has been checked.

## 20.7 Restoring the `.slp` files

Before 0.2.0, Sync rebuilt each mapped `.slp` file from the project's part alone: videos not in `trials.csv`, and frames of trial videos outside `_start`–`_end`, were dropped, along with their suggestions. The project's points (the working file) are complete; the `.slp` copies in `slp/` may not be.

The upgrade dialog asks where the original `.slp` files are:

- **Search folder...** — searched recursively (`find_originals`, leaving out `models/` and `backup/`). Each project file `slp/{name}.slp` is matched by file name; every match is offered in that file's list, the first selected. **Choose file...** picks one file for the selected row.
- **Earliest Sync backup** (`earliest_backup`; the default) — offered when `slp/backup/{timestamp}/` holds a copy of the file. The earliest backup is the file as it was before the first Sync. It is the original if setup copied or moved it in and nothing was changed in SLEAP before that Sync.
- **Keep the project's copy** — per file, for files with no original, e.g. ones setup moved into the project. What older Syncs dropped stays lost, and the dialog asks before going ahead.

For each file with an original:

1. **Read in first.** Before anything is replaced, a Sync *read* runs against the project's current `.slp` copies, so labels or predictions added in SLEAP since the last Sync go into the points (`read_slp_for_sync`, `apply_merge`). Where a point differs, label3d's is kept; the result message counts new points and differences. New points carry the sync id, so they show as new in the app.
2. **Recover hidden predictions.** Before 0.3.0 there were no `px, py, pscore` columns: a prediction that was labelled over in label3d was replaced in the points, but the original file still has it. For each point with a user label (`source` `label3d` or `sleap_user`) and empty `px`, the predicted instance in the original at the same video, frame and node fills `px, py, pscore`. The project's `.slp` copy is read the same way, for predictions imported after setup.
3. **Replace.** The original is copied to `slp/{stem}/{name}.slp`. The `models/` folder beside it is copied too, unless that folder is beside other `.slp` files as well (setup refuses such a folder; the upgrade leaves it out and says so).
4. **Write.** A Sync *write* (`write_slp_files`) rewrites the project's part of the file from the points and keeps everything else, as 0.2.0 does. User instances are written with `from_predicted` linking the hidden prediction, as at HEAD.

The project's points override the original within the project's part, that is, trial videos between `_start` and `_end`. User labels in the original within those frames are replaced by the project's; predictions are kept, under the user instance (step 2). Videos and frames outside the project come from the original unchanged.

## 20.8 When it runs

- **In the app.** `Project.load` raises `ProjectNeedsUpgrade(plan)` before opening any video. `app.py` shows `UpgradeDialog`: the steps, the backup location and, for files to restore, a row each with its original (20.7), with **Upgrade** and **Cancel**. `upgrade_project` runs the upgrade in `UpgradeWorker` behind a busy progress dialog that shows each step, reports the result or the error, and the app loads the project again.
- **Command line.** `label3d-upgrade <project> [--slp-source DIR | --slp-from-backup | --keep-slp] [-n]` runs the same steps without the GUI. With files to restore and no `--slp` option it stops (exit 2); with `--slp-source`, a file with no match keeps the project's copy and one with several matches is an error. `-n` prints the version, the steps and the originals.
- `label3d-export` on an old project says to run `label3d-upgrade` first.

## 20.9 Code

| Module | Contents |
|---|---|
| `version.py` | `__version__`, `commit()`, `LAYOUT_VERSION`, `version_string()`, `project_version()`, `check_not_newer()`, `write_project_toml()` |
| `layout.py` | `detect_version`, `legacy_files`, `plan_upgrade`, `UpgradePlan`, `ProjectNeedsUpgrade`, step names and texts. It doesn't import `project.py`, which uses it |
| `upgrade.py` | `run_upgrade` (backup, steps, roll back, stamp), the step functions, `read_points_csv` (0.0.9's reader), `restore_slp_files`, `find_originals`, `earliest_backup` |
| `upgrade_cli.py` | `label3d-upgrade` |
| `widgets/upgrade_dialog.py` | `UpgradeDialog`, `upgrade_project` |
| `workers/upgrade_worker.py` | runs `run_upgrade` off the GUI thread |
| `project.py` | version check in `load` (`check_version=False` for the upgrade itself); `store_experiment_df`, shared with `apply_merge`; `_write_config` through `write_project_toml`; the two one-time conversions and the CSV refusal removed |
| `setup_project.py`, `synthetic/writers.py` | new projects get `[label3d]`; synthetic `trials.csv` numbered from 1 |
| `hatch_build.py` | writes `label3d/_build_info.py` with the commit into a wheel |

## 20.10 Testing

**Old versions as git worktrees.** Each old version is checked out in the scratchpad with its own environment, leaving the main checkout alone:

```
git worktree add $SCRATCH/label3d-0.0.9 v0.0.9
git worktree add $SCRATCH/label3d-0.1.0 v0.1.0
(cd $SCRATCH/label3d-0.0.9 && uv sync)
```

They are removed with `git worktree remove` when done. Each old version installs its own `aniposelib` and `sleap-io` from PyPI; the local clones aren't needed.

**Test projects.**

| Case | How it is made |
|---|---|
| Real 0.0.9 | a copy of `projects/old/BGvPS2` (CSV, `skeleton.json`, flat `slp/`, refinement) |
| Real 0.1.0 | `projects/BGvPS_flat` set up with the 0.1.0 worktree (`BGvPS_setup.toml`), some points labelled or imported, then a Sync from the 0.1.0 code so the copies lose videos. The originals are the `.slp` files in `BGvPS_flat/lateral` and `ventral` |
| 0.2.x development | a copy of the current `projects/BGvPS` (`slp/{stem}/`, no `frame_base`) |
| Unit fixtures | `tests/fixtures/old_projects/v0.0.9` and `v0.1.0` (about 300 kB each), made by `make_old_projects.py`: each version's `generate_synthetic.py` (2 cameras, 320×256, no charuco videos), originals with user frames, predictions, frames outside the trials, a video not in the project and suggestions, then `old_sync.py` run by that version (Sync read, two labels over predictions, corners saved, Sync write) |

Copies of real projects go in the scratchpad. Videos are left where they are, through `[video_roots]` or symlinks, and are never copied.

**Checks.**
- `detect_version` gives the right version for each case, and the right steps.
- After the upgrade, points are equal to the old points (in values, types and index) apart from the frame numbering in `trials.csv`.
- The restored `.slp` file holds every video and labelled frame of the original that is outside the project, unchanged, and the project's part equals the project's points.
- A point labelled over a prediction gets that prediction back in `px, py, pscore`.
- Stopping in each step (a raised error) leaves the project as it was.
- A second upgrade finds nothing to do.
- The upgraded project opens in HEAD.
- `project.toml` keeps the comment line after the app writes it again.

## 20.11 Notes

- Test projects that tests build by hand in the old style (`skeleton.json`, 0-based `trials.csv`) go through `tests.helpers.make_current`, which brings them to the current layout without the upgrade.
- The synthetic generator still writes `.slp` files flat in `slp/`. Flat files are valid in a versioned project (they are found recursively); only an unversioned project has them moved.
- A 0.0.9 `skeleton.json` with symmetries the `.slp` files' skeletons lack is reported as a skeleton conflict after the upgrade (Phase 14 rule), as it was when `Project.load` moved it.

## 20.12 Results on real projects

| Project | From | Result |
|---|---|---|
| `projects/old/BGvPS2` (copy) | 0.0.9 | Points (50 rows), corners and three refinement records converted, refinement `.toml` files renamed, skeleton moved; `.slp` copies kept (no originals). Loads, with the symmetry conflict above |
| `projects/BGvPS` (copy) | 0.2.x development build (`slp/{stem}/`, no `frame_base`) | Detected as 0.2.0; only the frame numbering. 2 s |
| BGvPS set up from `projects/BGvPS_flat` with the 0.1.0 worktree, synced by 0.1.0 (copy) | 0.1.0 | The 0.1.0 Sync had cut the Lateral file from 177 to 52 MB and Ventral from 69 to 17 MB. Restored from `BGvPS_flat/lateral` and `ventral`: 578 of 578 and 226 of 226 videos, all 229,682 and 135,834 labelled frames, both `models/` folders. Points unchanged; 41 predictions recovered under user labels (1 labelled over in label3d, 40 SLEAP user instances made from predictions). 166 s |

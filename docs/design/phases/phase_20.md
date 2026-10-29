# Phase 20 — Project versions and upgrades

**Goal**: Give label3d a version number, record in each project which version its layout belongs to, and offer to upgrade a project made by an older version when it is opened. Upgrading restores the parts of the project's `.slp` files that older Syncs dropped, using the original `.slp` files the user points to.

**Status**: Plan, awaiting review.

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
- **Help > About** shows both. `label3d --version` and `label3d-export --version` print them.

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

Run in this order. Each step changes files only inside the project; the backup is made first (20.6).

**From 0.0.9**

1. **Points** — each `experiments/{name}/points.csv` → `points.parquet`. It is read as 0.0.9 did (`header=[0, 1]`, `index_col=[0..4]`, experiment/trial/bodypart as strings, frame/instance as int; `normalize_columns`) and written with `write_points_file`. The `.csv` and `.csv.bak` files go to the backup.
2. **Corners** — `calibrations/{cam}/corners.csv` → `corners.parquet` (`take` as string).
3. **Sync history** — `sync_history.csv` → `sync_history.parquet`.
4. **Refinement records** — `refined/{id}_points.csv` → `{id}_points.parquet`, and `points_file` in that refinement's `.toml` renamed.
5. **Skeleton** — `skeleton.json` → `[skeleton]` in `project.toml` (the existing `_migrate_skeleton_json`, moved here from `Project.load`). Points on nodes outside the skeleton are dealt with as Phase 14 does on load.

**From 0.1.0 (and 0.0.9)**

6. **Restore the `.slp` files** — see 20.7.

**To 0.3.0 (all older projects)**

7. **SLP folders** — each `slp/{name}.slp` → `slp/{stem}/{name}.slp`, with the `models/` folder that sat beside its original copied in if one is found (as setup does, `fc4c488`), and `[slp]` paths in `project.toml` rewritten. Old `slp/backup/{timestamp}/` folders move to the upgrade backup.
8. **Frame numbering** — `trials.csv` renumbered from 1 and `[trials] frame_base = 1` (the existing one-time conversion in `Project.load`, moved here).
9. **Point columns** — `px, py, pscore` are added empty if missing. (`normalize_columns` may already do this; to be confirmed. If it does, this is a check, not a rewrite.)
10. **Stamp** — `[label3d]` written last, so an upgrade that stops part-way is detected again on the next open.

The one-time conversions now in `Project.load` (skeleton.json, frame base) are removed from there. A project that needs them goes through the upgrade instead.

## 20.6 Backup

- Before any step, every file the upgrade will change, move or remove is copied to `upgrade_backup/{from_version}_{YYYYmmdd-HHMMSS}/` in the project, keeping its relative path: `project.toml`, `trials.csv`, the CSV working files, `skeleton.json`, `slp/**/*.slp` and `slp/backup/`. Videos, calibration videos and `models/` folders are not copied, since the upgrade doesn't change them.
- Files are copied with `shutil.copy2`. On APFS this is still a full copy; for BGvPS the `.slp` files are about 250 MB.
- If a step raises, the files in the backup are copied back, files the upgrade created are removed, and the error is shown. The backup is kept.
- The backup folder is not inside `slp/`, so `find_slp_files` doesn't see it. It is also left out of setup's searches.
- A note in the user docs says the folder can be deleted once the upgraded project has been checked.

## 20.7 Restoring the `.slp` files

Before 0.2.0, Sync rebuilt each mapped `.slp` file from the project's part alone: videos not in `trials.csv`, and frames of trial videos outside `_start`–`_end`, were dropped, along with their suggestions. The project's points (the working file) are complete; the `.slp` copies in `slp/` may not be.

The upgrade dialog asks where the original `.slp` files are:

- **Choose folder...** — searched recursively, as setup does (leaving out `models/`, `backup/` and the output folder). Each project file `slp/{name}.slp` is matched by file name. The dialog lists every project file with its match: one found, several found (pick one), or none.
- **Use earliest Sync backup** — offered when `slp/backup/{timestamp}/` holds a copy of the file. The earliest backup is the file as it was before the first Sync. It is the original if setup copied or moved it in and nothing was changed in SLEAP before that Sync.
- **Keep the project's copy** — per file, for files with no original, e.g. ones setup moved into the project. What older Syncs dropped stays lost, and the dialog says so.

For each file with an original:

1. **Read in first.** Before anything is replaced, a Sync *read* runs against the project's current `.slp` copies, so labels or predictions added in SLEAP since the last Sync go into the points. This is the normal read step of Sync (`read_slp_for_sync`, merge) with the usual review of new points.
2. **Recover hidden predictions.** Before 0.3.0 there were no `px, py, pscore` columns: a prediction that was labelled over in label3d was replaced in the points, but the original file still has it. For each point with a user label (`source` `label3d` or `sleap_user`) and empty `px`, the predicted instance in the original at the same video, frame and node fills `px, py, pscore`. The project's `.slp` copy is read the same way, for predictions imported after setup.
3. **Replace.** The original is copied to `slp/{stem}/{name}.slp`, with its `models/` folder.
4. **Write.** A Sync *write* (`write_slp_files`) rewrites the project's part of the file from the points and keeps everything else, as 0.2.0 does. User instances are written with `from_predicted` linking the hidden prediction, as at HEAD.

The project's points override the original within the project's part, that is, trial videos between `_start` and `_end`. User labels in the original within those frames are replaced by the project's; predictions are kept, under the user instance (step 2). Videos and frames outside the project come from the original unchanged.

## 20.8 When it runs

- **In the app.** When `Project.load` finds an old version, it raises `ProjectNeedsUpgrade(from_version, steps)` before opening videos. The main window shows a dialog listing the steps and the backup location, with **Upgrade** and **Cancel**. Upgrade runs in a worker with progress (the points files of BGvPS take a while to convert), then shows the SLP dialog (20.7), then loads the project.
- **Command line.** `label3d-upgrade <project> [--slp-source DIR | --slp-from-backup | --keep-slp] [--dry-run]` runs the same steps without the GUI. `--dry-run` prints the detected version and the steps. It is used by the tests and for upgrading several projects at once.
- `label3d-export` on an old project fails with "run label3d-upgrade first".

## 20.9 Code

| Module | Contents |
|---|---|
| `version.py` | `__version__`, `commit()`, `LAYOUT_VERSION = "0.3.0"`, `version_line()` |
| `upgrade.py` | `detect_version`, `UpgradeStep` (name, `applies`, `files`, `run`), `STEPS`, `plan_upgrade`, `run_upgrade` (backup, steps, roll back, stamp), the 0.0.9 CSV readers |
| `upgrade_cli.py` | `label3d-upgrade` |
| `widgets/upgrade_dialog.py` | step list and SLP source chooser |
| `workers/upgrade_worker.py` | runs `run_upgrade` off the GUI thread |
| `project.py` | `ProjectNeedsUpgrade`; version check in `load`; `_write_config` writes the comment and `written_by`; the two one-time conversions removed |
| `setup_project.py` | new projects get `[label3d]` |

The 0.0.9 CSV readers are copied into `upgrade.py` from `088c44f` so they stay as they were, and are used nowhere else.

## 20.10 Testing

**Old versions as git worktrees.** Each old version is checked out in the scratchpad with its own environment, leaving the main checkout alone:

```
git worktree add $SCRATCH/label3d-0.0.9 v0.0.9
git worktree add $SCRATCH/label3d-0.1.0 v0.1.0
(cd $SCRATCH/label3d-0.0.9 && uv sync)
```

They are removed with `git worktree remove` when done. `sleap/` and `aniposelib/` are gitignored, so where an old script needs them they are symlinked into the worktree.

**Test projects.**

| Case | How it is made |
|---|---|
| Real 0.0.9 | a copy of `projects/old/BGvPS2` (CSV, `skeleton.json`, flat `slp/`, refinement) |
| Real 0.1.0 | `projects/BGvPS_flat` set up with the 0.1.0 worktree (`BGvPS_setup.toml`), some points labelled or imported, then a Sync from the 0.1.0 code so the copies lose videos. The originals are the `.slp` files in `BGvPS_flat/lateral` and `ventral` |
| 0.2.x development | a copy of the current `projects/BGvPS` (`slp/{stem}/`, no `frame_base`) |
| Unit fixtures | small synthetic projects made by `generate_synthetic.py` in the 0.0.9 and 0.1.0 worktrees, with a Sync run by each. They are committed under `tests/fixtures/projects/` (a few hundred kB, no videos beyond what the generator makes small) |

Copies of real projects go in the scratchpad. Videos are left where they are, through `[video_roots]` or symlinks, and are never copied.

**Checks.**
- `detect_version` gives the right version for each case, and the right steps.
- After the upgrade, points are equal to the old points (in values, types and index) apart from the frame numbering in `trials.csv`.
- The restored `.slp` file holds every video and labelled frame of the original that is outside the project, unchanged, and the project's part equals the project's points.
- A point labelled over a prediction gets that prediction back in `px, py, pscore`, and deleting the user instance in the app shows it.
- Stopping in each step (a raised error) leaves the project as it was.
- A second upgrade finds nothing to do.
- The upgraded project opens in HEAD; Sync and export work.
- `project.toml` keeps the comment line after the app writes it again.

## 20.11 Order of work

1. `version.py`, `pyproject.toml` version, tags, `[label3d]` written by setup and `_write_config`; refuse newer projects.
2. Worktrees and fixture projects for 0.0.9 and 0.1.0.
3. `upgrade.py`: detection, backup and roll back, steps 1–5 and 7–10, with tests.
4. SLP restoration (step 6) with tests on the fixtures, then on the BGvPS copies.
5. `label3d-upgrade`.
6. Upgrade dialog and worker in the app.
7. Docs: `data_model.md` (`[label3d]`, versions), `user_workflow.md` (upgrading, the backup folder), `CLAUDE.md`, `DESIGN.md` index.

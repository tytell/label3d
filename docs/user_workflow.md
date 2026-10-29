# label3d workflow guide

This guide walks through a label3d project from start to finish. You label points in several synchronized camera views, train a SLEAP model on those labels, review what the model predicts, and repeat until the model is good enough to run on everything.

## Overview

1. Set up a project.
2. Calibrate the cameras.
3. Label points in label3d, then sync them to the SLEAP files.
4. Train a model and run inference in SLEAP.
5. Bring the predictions back into label3d, review them, correct or add labels, and sync again. Repeat steps 4–5 until the model is good enough.
6. Run the final model on every frame, then triangulate and export.

label3d keeps your labels in each experiment's `points.parquet`. The `.slp` files in the project's `slp/` folder are copies of those labels that SLEAP can read. **Sync with Sleap** keeps the two in step.

Each `.slp` file has a folder of its own in `slp/`, so that the `models` folder SLEAP trains into sits beside it. Setup copies an existing `models` folder along with its `.slp` file. If a folder holds several `.slp` files and a `models` folder, setup stops: put each `.slp` file in its own folder with its models first.

A `.slp` file can hold more than the project: videos of other experiments, or frames outside a trial. Sync leaves those alone, so the file still has them when you open it in SLEAP.

## Opening a project from an older label3d

label3d records its version in each project's `project.toml`. A project made before that (label3d 0.2.0 or earlier) is upgraded the first time you open it: a dialog lists what changes, and **Upgrade** does it. Every file that changes is first copied to `upgrade_backup/` in the project; if anything goes wrong, the project is put back as it was. Delete `upgrade_backup/` once you have checked the upgraded project.

**The `.slp` files.** Before 0.2.0, **Sync** cut the project's `.slp` files down to the project: videos of other experiments and frames outside the trials were lost. The upgrade puts them back from the `.slp` files the project was set up from. For each file, choose its original:

- **Earliest Sync backup** (the default, when there is one): the copy in `slp/backup/` from before the first Sync. It is the original if setup copied the file into the project and nothing was changed in SLEAP before that Sync.
- **Search folder...** finds the originals by name in a folder and its subfolders; **Choose file...** picks one.
- **Keep the project's copy** when there is no original; what was dropped stays lost.

The project's labels go into the original's frames for the project's trials; everything else in it is kept, and its `models` folder is copied along. A label you placed over a SLEAP prediction gets that prediction back underneath it, as label3d does now.

From the command line: `uv run label3d-upgrade <project> --slp-source <folder>` (or `--slp-from-backup`, or `--keep-slp`); add `-n` to see what it would do.

Projects from before the calibration registry (with `calibration.toml` or `calibration_takes/` in an experiment) can't be upgraded; set them up again.

## 1. Set up a project

There are two ways to start:

- **From existing SLEAP files:** if you already have `.slp` files with labels, follow [setup_from_slp.md](setup_from_slp.md). The videos stay where they are.
- **From videos:** run `uv run python setup_project_cli.py` and answer the prompts. It groups your videos into experiments and trials and creates the project folder.

Open the project with `uv run label3d <project folder>`.

The first time you open it, assign videos to SLEAP files with **File > Set up Sleap files...**. The preset *one file per camera* works for most projects.

**Skeletons.** The skeletons come from the `.slp` files, and cameras may have different ones (a node a camera never sees can be left out of its model). Starting without `.slp` files, load one with **File > Load skeleton...** — a `skeleton.json` or any `.slp` file — for all cameras or, to give some cameras a smaller skeleton, for chosen ones. The first **Sync** then creates each `.slp` file with its cameras' skeleton. A file must hold cameras with the same skeleton, so with several skeletons use *one file per camera per experiment*.

## 2. Calibrate

In the **Calibrate** tab, detect the charuco board in each calibration take and run the calibration. If an experiment has a take before and after the trials, label3d compares them and warns you if a camera seems to have moved in between.

Every calibration you run is kept. Each run saves one solution per take and one for all takes together, and the joint one becomes active. To use a single take instead, pick it and click **Use selected take**. The **Saved calibrations** list shows them all; you can rename or delete them there.

### Switching calibrations

To try a different calibration while labelling, choose it from the **Calibration** box at the top of the window. label3d re-triangulates the experiment's points with it, so you can compare reprojection errors.

Making a calibration active in the **Calibrate** tab doesn't re-triangulate. The status bar then says **3D points are from <label>; re-triangulate**; click the link, or use **Instances > Triangulate all experiments...** to do every experiment.

### Refining with labelled points

Once you have points labelled in several cameras, they can improve the calibration. In the **Calibrate** tab, click **Refine with points…**. It starts from a charuco calibration and adjusts the cameras to fit your points.

- By default it uses your own labels (from label3d and SLEAP), not predictions, and keeps each camera's intrinsics.
- Judge the result by the **held-out points** error: it's measured on frames the fit never saw, and it should go down. The **charuco corner** error should not get much worse.
- **Save and make active** keeps it. As with any calibration made active in the Calibrate tab, re-triangulate afterwards.

The details of every option are in [workflows.md](design/workflows.md#refining-with-labelled-points).

## 3. Label

In the **Label** tab:

- **Pick a point:** click a bodypart in the list on the left, or use `↑` / `↓`. A point in only some cameras' skeletons is marked ◐ (hover to see which), and one in a single camera `2D` — it is never triangulated. Its count `[n/k]` is out of the cameras that have it.
- **Cameras without the point:** a panel whose skeleton lacks the selected point shows nothing for it and says "*point* not in this camera's skeleton"; you can't place it there.
- **Place it:** click in each camera view. After the first camera, epipolar lines show where the point should be in the others.
- **Not visible:** right-click a point to mark it not visible in that camera.
- **Navigate:** `→` / `←` step frames, and `]` / `[` jump between labelled frames.
- **Frame numbers start at 1** in the app, and in the `_start` / `_end` columns of `trials.csv`. Exported files and the `.slp` files count from 0, as SLEAP does, so frame 1 in the app is frame 0 there.

When you've labelled a set of frames, choose **File > Sync with Sleap** (`Ctrl+S`). This writes your labels into the `.slp` files.

## 4. Train and run inference in SLEAP

Use the `.slp` files in the project's `slp/` folder. There are two ways to train and predict.

**In the SLEAP GUI**
1. Open a file from `slp/`.
2. Train a model, then run inference.
3. Save the file in place.

The predictions are now in the file label3d syncs with. label3d reads them the next time you open the project or choose **Sync with Sleap**.

**On the command line**
1. Train with `sleap-nn train` (see the SLEAP documentation for the training configuration).
2. Predict with `sleap-nn track -i <labels .slp or video> -m <model folder>`. Run `sleap-nn track --help` for the options, including where to write the output.

The command line writes predictions to a **separate** `.slp` file, not into the files in `slp/`. Bring them in with **File > Import predictions...** (step 5). You can select several files at once; they don't need to be in the project folder.

## 5. Review the predictions

### What happens when predictions come in

Predictions come in when you reopen the project, choose **Sync with Sleap**, or choose **File > Import predictions...**. label3d then:

- **Merges them with your labels.** Predictions never replace your own labels. On a frame where a camera already has a user label, that camera's predictions are ignored.
- **Marks them new.** Every point brought in from SLEAP is marked *new*. This includes predictions and labels someone added in SLEAP.
- **Shows a summary.** For each trial, the summary lists new SLEAP labels, new predictions, and updated predictions (re-predicted points that moved by more than 0.5 px), plus how many frames have new points. **Go to first new frame** takes you there. You can reopen it later with **File > Last sync summary...**.

If a label you made in label3d was also changed in SLEAP, label3d asks which version to keep before anything changes.

### Finding new points

- **Frame slider:** the row on top marks frames with new points. Orange means a new label made in SLEAP; blue means new predictions only.
- **Next / previous new frame:** `Shift+]` and `Shift+[` jump between frames with new points, moving on to the next trial when you reach the end of one.
- **In the camera views:** new points have a white square around them.
- **View > Color points by > Source** colours every point by where it came from:
  - magenta: labelled in label3d
  - orange: labelled in SLEAP
  - light blue: predicted by SLEAP

  Hover over a point to see its source, score and which sync brought it in.
- **View > Color points by > Prediction score** colours predictions by SLEAP's score: green near 1, yellow around 0.5, red near 0. User labels have no score and are grey.
- **Legend:** the panel below the Points list explains every shape and colour, and switches along with **Color points by**. Hide it with **View > Show legend**.

### Correcting and accepting

- **Predictions are locked**, as in SLEAP. You can click a predicted point to select it and see its epipolar lines, but you can't move it.
- **Correct a prediction:** double-click it. That camera gets a user instance copied from the prediction, which you can then drag. Other cameras are not changed. The prediction is kept underneath: delete the user instance (`Ctrl+Backspace` with the mouse over that camera) and the prediction comes back. `Ctrl+Shift+Backspace` deletes the instance in every camera.
- **Correct a point:** drag it to the right place. It becomes a user label from label3d and loses its new mark.
- **Accept a point that's already right:** select it and press `A`. Press `Shift+A` to accept every point in the frame. Accepting only clears the new mark; a prediction stays a prediction.
- **Add labels where the model fails.** These frames are the most useful ones for the next round of training.

When you're done, **Sync with Sleap** again and go back to step 4.

## 6. Final pass

Once the model is good enough:

1. Run `sleap-nn track` on every video and bring the results in with **File > Import predictions...**.
2. Check the reprojection errors in the Label tab. Points with large errors (red when colouring by error) point to tracking or calibration problems.
3. Choose **File > Sync with Sleap**, so that the SLP files and label3d agree.
4. Choose **Instances > Triangulate all experiments...**. This triangulates every point of every experiment with its active calibration, including points brought into experiments you weren't viewing.
5. Choose **File > Export points...** (below).

## 7. Export

**File > Export points...** writes every point, from all experiments, to one file, as Parquet or CSV. It also writes the current points into the SLP files, as **Sync** does.

It is greyed out while any experiment's 3D points come from a calibration other than its active one. Hover over it to see which; then use **Instances > Triangulate all experiments...**.

The dialog has:

- **Experiments:** all are checked by default.
- **Include predictions:** on by default. Off leaves out SLEAP predictions. Every prediction is exported whatever its score; use the `score` column to filter them afterwards.
- **Only points with 3D:** off by default.
- **Format:** Parquet (the default) or CSV. Parquet files are much smaller and keep each column's type; R, Python and many viewers read them (see [Opening the files](#opening-the-files)). The format sets the file's extension.
- **Layout:** **Wide** (the default) has one row per point, with a block of columns for each camera. **Long** has one row per point in each camera, plus one row for its 3D point. See [Export columns](#export-columns).
- **Output file:** by default `exports/points_{date-time}.parquet` in the project folder.

When you click **Export**:

- **Experiments without an active calibration:** label3d warns you and, if you choose **Export anyway**, exports only their 2D points.
- **New points in the SLP files:** if the SLP files have points that label3d doesn't have yet (for example, predictions saved from the SLEAP GUI), label3d says so and exports nothing. Run **Sync with Sleap**, triangulate, then export again.
- Otherwise it writes the file and the SLP files, then shows how many rows it wrote and the points per experiment by state and source.

### From the command line

After a large inference job you can export without opening the app:

```
uv run label3d-export <project folder> [-o out] [--format parquet|csv] [--layout wide|long] [--experiments a,b] [--no-predictions] [--only-3d] [--no-slp]
```

It writes Parquet in the wide layout unless you ask otherwise. `-o` gets the extension of the format, so `-o out.csv` without `--format csv` writes `out.parquet`.

It runs the same checks. It stops with an error (exit code 1) if an experiment's 3D points are out of date or the SLP files have new points. It doesn't triangulate, so run **Triangulate all experiments** in label3d first. `--no-slp` skips the SLP files entirely.

### Export columns

#### Wide layout

One row per point: experiment, trial, frame, instance and bodypart, as in the project's points files. Frame numbers in exports count from 0, one less than the app shows. Each camera has its own set of columns, named with the camera as a prefix (for example `LateralOrtho_x.px`), followed by the 3D point. Units follow a dot: `.px` for pixels, `.cm` for the 3D coordinates. The columns are always in this order.

| Column | Meaning |
|---|---|
| `experiment`, `trial`, `frame`, `instance`, `bodypart` | which point; `frame` is the frame within the trial |
| `{cam}_video_frame` | the frame in that camera's video (the trial frame plus the camera's start offset) |
| `{cam}_x.px`, `{cam}_y.px` | the point in that camera, in pixels |
| `{cam}_state` | `set`, `predicted` or `not_visible` |
| `{cam}_source` | `label3d`, `sleap_user` (labelled in SLEAP) or `sleap_pred` (predicted by SLEAP) |
| `{cam}_score` | SLEAP's score, where there is one |
| `{cam}_xr.px`, `{cam}_yr.px`, `{cam}_error.px` | where the 3D point reprojects in that camera, and its distance from (`x`, `y`); empty without a 3D point |
| `x.cm`, `y.cm`, `z.cm` | the triangulated 3D point |
| `n_cameras` | the number of cameras used to triangulate (points `set` or `predicted`) |
| `mean_error.px` | the mean reprojection error over those cameras |
| `calibration` | the id of the calibration used |

A camera's columns are empty where it has no point. The 3D columns are empty for a point seen by fewer than two cameras, or in an experiment without a calibration. Points never placed in any camera are left out. **Include predictions** off empties the prediction cells but keeps the 3D point, which may have used them; `n_cameras` and `mean_error.px` always describe the triangulation.

#### Long layout

One row per point in each camera, and one more for its 3D point. The `view` column says which: a camera name, or `3d`. The columns are always in this order:

| Column | Camera rows | `3d` rows |
|---|---|---|
| `experiment`, `trial`, `frame`, `instance`, `bodypart` | which point | which point |
| `view` | the camera | `3d` |
| `video_frame` | the frame in that camera's video | empty |
| `x`, `y` | the point, in pixels | the 3D point, in cm |
| `z` | empty | the 3D point, in cm |
| `state`, `source`, `score` | as in the wide layout | empty |
| `xr`, `yr`, `error` | reprojection and its error, in pixels | empty |
| `n_cameras`, `mean_error`, `calibration` | empty | as in the wide layout; `mean_error` in pixels |

A camera has a row only where it has a point, and a point has a `3d` row only where it was triangulated. Each point's camera rows come first, then its `3d` row. The same rules as the wide layout decide what is left out.

### Opening the files

Parquet files open in most data tools:

- **Python:** `pandas.read_parquet("points_….parquet")`
- **R:** `arrow::read_parquet("points_….parquet")`, or `arrow::open_dataset()` to filter with dplyr before reading it all in
- **Viewers:** [Tad](https://www.tadviewer.com) or the DuckDB UI (`duckdb -ui`) let you browse and sort a file without writing code.

The project's own working files are Parquet too: each experiment's `points.parquet`, each camera's `corners.parquet`, the project's `sync_history.parquet`, and the record of points kept with each refined calibration. They open the same way, but change them only through label3d. In `points.parquet` each camera column is named `camera/coordinate`, for example `LateralOrtho/x`.

## Keyboard reference for review

| Key | Action |
|---|---|
| `Shift+]` / `Shift+[` | Next / previous frame with new points |
| `]` / `[` | Next / previous labelled frame |
| `A` | Accept the selected point |
| `Shift+A` | Accept every point in the frame |
| `Ctrl+S` | Sync with Sleap |
| `T` | Triangulate the current frame |

The full shortcut list is in [workflows.md](design/workflows.md#keyboard-shortcuts).

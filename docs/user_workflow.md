# label3d workflow guide

This guide walks through a label3d project from start to finish. You label points in several synchronized camera views, train a SLEAP model on those labels, review what the model predicts, and repeat until the model is good enough to run on everything.

> **Draft.** This covers setup, calibration, labelling and the review loop. The final steps (refining the calibration, comparing calibrations, and exporting 3D points) will be added when they're built.

## Overview

1. Set up a project.
2. Calibrate the cameras.
3. Label points in label3d, then sync them to the SLEAP files.
4. Train a model and run inference in SLEAP.
5. Bring the predictions back into label3d, review them, correct or add labels, and sync again. Repeat steps 4–5 until the model is good enough.
6. Run the final model on every frame, then triangulate and export.

label3d keeps your labels in each experiment's `points.csv`. The `.slp` files in the project's `slp/` folder are copies of those labels that SLEAP can read. **Sync with Sleap** keeps the two in step.

## 1. Set up a project

There are two ways to start:

- **From existing SLEAP files:** if you already have `.slp` files with labels, follow [quick_setup_from_slp.md](quick_setup_from_slp.md). The videos stay where they are.
- **From videos:** run `uv run python setup_project_cli.py` and answer the prompts. It groups your videos into experiments and trials and creates the project folder.

Open the project with `uv run label3d <project folder>`.

The first time you open it, assign videos to SLEAP files with **File > Set up Sleap files...**. The preset *one file per camera* works for most projects.

## 2. Calibrate

In the **Calibrate** tab, detect the charuco board in each calibration take and run the calibration. If an experiment has a take before and after the trials, label3d compares them and warns you if a camera seems to have moved in between.

Every calibration you run is kept. Each run saves one solution per take and one for all takes together, and the joint one becomes active. To use a single take instead, pick it and click **Use selected take**. The **Saved calibrations** list shows them all; you can rename or delete them there.

To try a different calibration while labelling, choose it from the **Calibration** box at the top of the window. label3d re-triangulates the experiment's points with it, so you can compare reprojection errors.

## 3. Label

In the **Label** tab:

- **Pick a point:** click a bodypart in the list on the left, or use `↑` / `↓`.
- **Place it:** click in each camera view. After the first camera, epipolar lines show where the point should be in the others.
- **Not visible:** right-click a point to mark it not visible in that camera.
- **Navigate:** `→` / `←` step frames, and `]` / `[` jump between labelled frames.

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
- **Legend:** the panel below the Points list explains every shape and colour, and switches along with **Color points by**. Hide it with **View > Show legend**.

### Correcting and accepting

- **Correct a point:** drag it to the right place. It becomes a user label from label3d and loses its new mark.
- **Accept a point that's already right:** select it and press `A`. Press `Shift+A` to accept every point in the frame. Accepting only clears the new mark; a prediction stays a prediction.
- **Add labels where the model fails.** These frames are the most useful ones for the next round of training.

When you're done, **Sync with Sleap** again and go back to step 4.

## 6. Final pass

Once the model is good enough:

1. Run `sleap-nn track` on every video and bring the results in with **File > Import predictions...**.
2. Check the reprojection errors in the Label tab. Points with large errors (red when colouring by error) point to tracking or calibration problems.
3. Batch-triangulate and export: *coming in a later version*, together with calibration refinement.

## Keyboard reference for review

| Key | Action |
|---|---|
| `Shift+]` / `Shift+[` | Next / previous frame with new points |
| `]` / `[` | Next / previous labelled frame |
| `A` | Accept the selected point |
| `Shift+A` | Accept every point in the frame |
| `Ctrl+S` | Sync with Sleap |
| `T` | Triangulate the current frame |

The full shortcut list is in [workflows.md](workflows.md#keyboard-shortcuts).

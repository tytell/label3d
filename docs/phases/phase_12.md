# Phase 12 — Refining the Calibration with Labelled Points

**Goal**: Refine a calibration using the bodypart points labelled or predicted in several cameras. A refinement always starts from a charuco calibration computed by aniposelib's `calibrate_rows` on the user's chosen takes. aniposelib's `bundle_adjust_iter` then adjusts it with the digitised points. Every result is saved to the registry (Phase 11) with a full record of its inputs. The charuco solution it started from is kept too.

**Status**: Not started. Depends on Phase 10 (scores, sources) and Phase 11 (registry).

---

## 12.1 Two stages

```
Stage 1  charuco corners from takes T  ──calibrate_rows──▶  cgroup_charuco   (saved, method="charuco")
Stage 2  cgroup_charuco + points P     ──bundle_adjust_iter──▶  cgroup_refined  (saved, method="refine", parent=cgroup_charuco)
```

**Stage 1**
- Corners for each chosen take come from `calibration/{cam}/corners_{take}.csv` through `load_corners()` and `rows_for_take()`.
- They're calibrated with the existing `calibrate_rows()` path in `workers/calibration_worker.py`, including repeated runs and keeping the best.
- If the registry already has a `charuco` entry with exactly the same takes, the dialog offers to start from it instead of recalibrating.

**Stage 2**
- `cgroup.bundle_adjust_iter(p2ds, extra=None, only_extrinsics=..., n_samp_full=..., n_samp_iter=...)` runs on a copy of the Stage 1 group.
- `p2ds` is a `C × N × 2` array of **raw pixel** coordinates, in the order of `cgroup.get_names()`. aniposelib projects with distortion, so no undistortion is needed. Missing cameras are NaN.
- There's no `extra`. The board constraint applies only to charuco points, and Stage 1 has already used it.

## 12.2 Point selection (`calibration_refine.py`, no Qt)

```python
@dataclass
class PointSelection:
    sources: set[str]                 # subset of {"label3d", "sleap_user", "sleap_pred"}
    score_min: float = 0.0            # applies to sleap_pred only
    trials: list[str] | None = None   # None = all trials of the experiment
    min_cameras: int = 2

def gather_points(points_df, cam_names, sel) -> tuple[np.ndarray, pd.MultiIndex]
    # -> (p2ds C×N×2, index of the (trial, frame, instance, bodypart) behind each column)
```

**Which observations count:**
- A camera observation counts if its `state` is `set` or `predicted`, its `source` is in `sel.sources`, its x/y are finite, and for `sleap_pred` its `score` is at least `score_min`.
- label3d reprojection guesses (`state=predicted`, `source=label3d`) are left out. They're derived from the calibration, so fitting to them would be circular.

**Which points count:** a column (one bodypart in one frame) is kept if it has at least `min_cameras` cameras that count.

**Counts for the dialog:** `count_points(...)` returns, per camera pair, the number of points seen by both. The dialog shows these live.

## 12.3 Holding out frames, fixing scale, validating

**Hold-out**
- About 20% of frames (configurable) are kept out of Stage 2, and the split is by frame, not by point.
- Reported error is measured on the held-out frames. The error on the frames used for fitting always goes down, so it says little.
- The split uses a fixed seed stored in the metadata, so a run can be repeated.

**Fixing scale and world frame**
- Bundle adjustment on animal points alone has no metric scale, and it can shift the whole rig.
- After Stage 2, apply a similarity transform (Umeyama, on camera centres) that maps the refined rig back onto `cgroup_charuco`. This keeps the charuco scale (cm) and world axes.
- The size of the transform's scale correction is reported. A large one is a warning sign.

**Reported errors:**
1. **Held-out points:** median and 90th-percentile reprojection error, before (Stage 1) and after (Stage 2).
2. **Charuco corners from the chosen takes:** before and after. Stage 2 should not make these much worse.
3. **Per camera-pair drift from Stage 1:** from `calibration_compare.compare_calibrations()`, as in the take comparison.

**Repeats:** `bundle_adjust_iter` subsamples at random, so Stage 2 runs `n_repeats` times (default 3). The run with the lowest held-out error is kept, and the run-to-run spread is reported, as in the recent charuco change.

## 12.4 Refine dialog (`widgets/refine_dialog.py`) and worker (`workers/calibration_worker.py`)

Opened from the Calibration tab with **Refine with points...**.

| Control | Default |
|---|---|
| Charuco takes (checklist), or *Start from existing entry* combo | all takes |
| Sources: label3d user / SLEAP user / SLEAP predictions | first two checked |
| Prediction score ≥ | 0.8 |
| Trials (checklist) | all |
| Extrinsics only | on |
| Hold-out fraction | 0.2 |
| Repeats | 3 |
| Label | `refine` |

**Running a refinement**
- A live table shows usable points per camera pair. **Run** is disabled while any pair has fewer than about 50 points; the threshold is a constant, revisited once there's real data.
- `RefineWorker(QThread)` runs Stage 1 if needed, then Stage 2, the alignment and the validation. It emits progress text and a `RefineResult`.

**Results view**
- The before and after error table, the per-pair drift, and the scale correction.
- **Save** writes the Stage 1 entry if it's new, then the `refine` entry. Both use `[metadata]` as in Phase 11.1, with `points.sources`, `score_min` and `frames` filled in.
- **Save and make active** also moves the pointer.
- **Discard** saves nothing.

## 12.5 Tests (`tests/test_calibration_refine.py`)

Uses the synthetic generator: a known rig, charuco takes and a moving animal.

- **Point selection:** `gather_points` respects sources, the score threshold and `min_cameras`, and leaves out label3d reprojection guesses.
- **Recovery:** after a small rotation and translation of one camera in the Stage 1 group, Stage 2 with the true animal points moves it back toward the truth. Held-out error goes down.
- **Alignment:** the similarity alignment keeps the baseline lengths of the Stage 1 group within tolerance.
- **Hold-out:** the split is by frame and is reproducible from its seed.
- **Metadata:** a saved `refine` entry has its `parent`, sources, score threshold and frame list.

## Implementation order

1. 12.2 `gather_points` / `count_points` + tests
2. 12.1 + 12.3 `refine()` pipeline (no Qt) + recovery test
3. 12.4 worker and dialog

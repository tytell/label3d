"""Find and remove duplicate instances in a SLEAP .slp file.

Duplicates arise when .slp files that share frames are merged with SLEAP's
"keep both" strategy: every shared frame gets a second copy of its instances.
A second user instance in a frame breaks single-instance training
("The size of tensor a (N) must match the size of tensor b (2N)").

Two instances are duplicates when they are the same kind (user or predicted),
use the same skeleton and track, and have the same coordinates within
--tolerance pixels (default 0.001; copies can differ by float32 rounding),
with the same points missing.

Usage:
    python remove_slp_duplicates.py labels.slp                       # inspect only
    python remove_slp_duplicates.py labels.slp --csv report.csv      # also list every problem
    python remove_slp_duplicates.py labels.slp --remove-duplicates   # write labels.dedup.slp
    python remove_slp_duplicates.py labels.slp --remove-duplicates -o cleaned.slp

The input file is never changed. Frame numbers are SLEAP frame indices (0-based).
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio


def is_duplicate(a, b, tol):
    """Same kind, skeleton and track, and every point equal within tol pixels (missing points must match)."""
    if type(a) is not type(b) or a.skeleton is not b.skeleton or a.track is not b.track:
        return False
    pa, pb = np.asarray(a.numpy(), dtype=np.float64), np.asarray(b.numpy(), dtype=np.float64)
    return pa.shape == pb.shape and np.allclose(pa, pb, rtol=0, atol=tol, equal_nan=True)


def split_duplicates(instances, tol):
    """Return (kept, removed) where removed maps each dropped instance to its kept twin.

    Within a group of identical user instances, the one linked to a prediction
    (from_predicted) is kept, so the hidden prediction stays attached.
    """
    groups = []
    for inst in instances:
        group = next((g for g in groups if is_duplicate(g[0], inst, tol)), None)
        if group is None:
            groups.append([inst])
        else:
            group.append(inst)
    kept, removed = [], {}
    for group in groups:
        keeper = next((i for i in group if getattr(i, "from_predicted", None) is not None), group[0])
        kept.append(keeper)
        removed.update({id(i): keeper for i in group if i is not keeper})
    # Preserve the original order of the instances that stay.
    kept_ids = {id(i) for i in kept}
    return [i for i in instances if id(i) in kept_ids], removed


def inspect(labels, tol):
    """Return a list of problem records and per-frame cleaned instance lists."""
    problems = []
    cleaned = {}
    for lf_idx, lf in enumerate(labels.labeled_frames):
        kept, removed = split_duplicates(list(lf.instances), tol)
        video = Path(str(lf.video.filename)).name
        for inst in lf.instances:
            if id(inst) in removed:
                kind = "user" if isinstance(inst, sio.Instance) and not isinstance(inst, sio.PredictedInstance) else "predicted"
                problems.append(dict(video=video, frame_idx=lf.frame_idx,
                                     problem=f"duplicate {kind} instance", action="remove"))
        n_user_left = sum(1 for i in kept if not isinstance(i, sio.PredictedInstance))
        if n_user_left > 1:
            problems.append(dict(video=video, frame_idx=lf.frame_idx,
                                 problem=f"{n_user_left} distinct user instances",
                                 action="none (fix by hand)"))
        if removed:
            cleaned[lf_idx] = (kept, removed)
    return problems, cleaned


def print_report(labels, problems):
    n_user = sum(len(lf.user_instances) for lf in labels.labeled_frames)
    n_pred = sum(len(lf.predicted_instances) for lf in labels.labeled_frames)
    print(f"{len(labels.videos)} videos, {len(labels.labeled_frames)} labeled frames, "
          f"{n_user} user instances, {n_pred} predicted instances, "
          f"{len(labels.skeletons)} skeleton(s)")

    by_problem = collections.Counter(p["problem"] for p in problems)
    if not by_problem:
        print("No duplicate instances and no frames with more than one user instance.")
        return
    print("\nProblems:")
    for problem, n in sorted(by_problem.items()):
        print(f"  {problem}: {n}")

    user_level = [p for p in problems if "user" in p["problem"]]
    if user_level:
        print("\nUser-instance problems (these break single-instance training):")
        for p in user_level:
            print(f"  {p['video']}  frame_idx {p['frame_idx']}: {p['problem']} -> {p['action']}")

    pred_by_video = collections.Counter(p["video"] for p in problems if p["problem"] == "duplicate predicted instance")
    if pred_by_video:
        print(f"\nDuplicate predicted instances by video ({len(pred_by_video)} videos; "
              f"use --csv for every frame):")
        for video, n in pred_by_video.most_common(10):
            print(f"  {n:6d}  {video}")
        if len(pred_by_video) > 10:
            print(f"  ... and {len(pred_by_video) - 10} more videos")


def write_csv(problems, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frame_idx", "problem", "action"])
        writer.writeheader()
        writer.writerows(problems)
    print(f"\nWrote {len(problems)} rows to {path}")


def remove_duplicates(labels, cleaned):
    n_removed = 0
    for lf_idx, (kept, removed) in cleaned.items():
        lf = labels.labeled_frames[lf_idx]
        # A user instance whose hidden prediction was dropped now points to the kept twin.
        for inst in kept:
            fp = getattr(inst, "from_predicted", None)
            if fp is not None and id(fp) in removed:
                inst.from_predicted = removed[id(fp)]
        n_removed += len(lf.instances) - len(kept)
        lf.instances = kept
    return n_removed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slp", type=Path, help="input .slp file (not modified)")
    parser.add_argument("--remove-duplicates", action="store_true",
                        help="write a copy with duplicate instances removed")
    parser.add_argument("-o", "--output", type=Path,
                        help="output file for --remove-duplicates (default: <input stem>.dedup.slp)")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="max coordinate difference in pixels for two instances to count as duplicates (default 0.001)")
    parser.add_argument("--csv", type=Path, help="write every problem found to this CSV file")
    args = parser.parse_args()

    print(f"Loading {args.slp} ...")
    labels = sio.load_slp(str(args.slp), open_videos=False)
    problems, cleaned = inspect(labels, args.tolerance)
    print_report(labels, problems)
    if args.csv:
        write_csv(problems, args.csv)

    if not args.remove_duplicates:
        if cleaned:
            print("\nInspection only. Rerun with --remove-duplicates to write a cleaned copy.")
        return

    output = args.output or args.slp.with_name(args.slp.stem + ".dedup.slp")
    if output.resolve() == args.slp.resolve():
        sys.exit("Refusing to overwrite the input file; choose a different --output.")
    if not cleaned:
        print("\nNothing to remove; no file written.")
        return
    n_removed = remove_duplicates(labels, cleaned)
    print(f"\nRemoved {n_removed} duplicate instances. Saving {output} ...")
    sio.save_slp(labels, str(output))

    remaining, _ = inspect(sio.load_slp(str(output), open_videos=False), args.tolerance)
    left = [p for p in remaining if p["action"] == "remove"]
    print(f"Saved. Check of the saved file: {len(left)} duplicates remain.")
    hand = [p for p in remaining if p["action"] != "remove"]
    if hand:
        print(f"{len(hand)} frame(s) still have more than one distinct user instance; fix those in SLEAP.")


if __name__ == "__main__":
    main()

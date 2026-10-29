"""Make the old-version projects that the upgrade tests start from (Phase 20).

Each old version runs from its own git worktree, so the files are written by
that version's code:

    git worktree add <dir>/label3d-0.0.9 v0.0.9
    git worktree add <dir>/label3d-0.1.0 v0.1.0
    (cd <dir>/label3d-0.0.9 && uv sync)    # and the same for 0.1.0
    uv run python tests/fixtures/old_projects/make_old_projects.py <dir>

For each version, writes v{version}/project (the project as that version left
it) and v{version}/originals (the .slp files it was set up from):

1. generate_synthetic.py of that version makes a small project (no charuco
   videos). Its points file is removed, so the points come from the .slp files.
2. The trials end at frame 6, so frames 7-9 are outside the project.
3. Originals are made from the generated .slp files: user instances on frames
   0-1 and 7-9, predictions on 2-6, one more video that isn't in trials.csv,
   and suggestions. They are copied into slp/ and mapped in project.toml, as
   setup did.
4. old_sync.py, run by that version, reads them in, labels over two
   predictions and writes the .slp files back, dropping what was outside the
   project.

The .slp files refer to videos by absolute paths in the build folder; label3d
matches them by basename.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sleap_io as sio
import toml

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "synthetic_small.toml"
VERSIONS = ["0.0.9", "0.1.0"]
LAST_TRIAL_FRAME = 6  # 0-based, as trials.csv was before 0.3.0
USER_FRAMES = {0, 1, 7, 8, 9}
EXTRA_VIDEO = "{camera}_not_in_project.mp4"


def run_old(worktree: Path, args: list[str], cwd: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(["uv", "run", "--project", str(worktree), "python", *args],
                   cwd=cwd, env=env, check=True)


def make_original(labels: sio.Labels, camera: str) -> sio.Labels:
    """The .slp file a user would have set the project up from."""
    frames = []
    for lf in labels.labeled_frames:
        instances = []
        for inst in lf.instances:
            pts = inst.numpy()
            if lf.frame_idx in USER_FRAMES:
                instances.append(sio.Instance.from_numpy(pts, skeleton=inst.skeleton))
            else:
                instances.append(sio.PredictedInstance.from_numpy(
                    pts, skeleton=inst.skeleton, point_scores=np.full(len(pts), 0.8),
                    score=0.9,
                ))
        frames.append(sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx,
                                       instances=instances))

    # A video the project doesn't have, labelled and suggested
    first = labels.labeled_frames[0]
    extra = sio.Video(filename=str(first.video.filename).replace(
        Path(first.video.filename).name, EXTRA_VIDEO.format(camera=camera)
    ), open_backend=False)
    extra_pts = first.instances[0].numpy() + 5.0
    frames.append(sio.LabeledFrame(video=extra, frame_idx=3, instances=[
        sio.Instance.from_numpy(extra_pts, skeleton=first.instances[0].skeleton)
    ]))
    suggestions = [sio.SuggestionFrame(video=extra, frame_idx=5),
                   sio.SuggestionFrame(video=first.video, frame_idx=8)]
    return sio.Labels(labeled_frames=frames, videos=[*labels.videos, extra],
                      skeletons=labels.skeletons, suggestions=suggestions)


def shorten_trials(path: Path) -> None:
    df = pd.read_csv(path)
    for col in df.columns:
        if col.endswith("_end"):
            df[col] = LAST_TRIAL_FRAME
    df.to_csv(path, index=False)


def map_slp_files(project: Path, cameras: list[str], trials: pd.DataFrame) -> None:
    """[slp] in project.toml: one file per camera, all experiments (written as old setup did)."""
    config = toml.load(project / "project.toml")
    files = []
    for camera in cameras:
        blocks = [
            {"experiment": exp, "trials": sorted(rows["trial"].astype(str)), "cameras": [camera]}
            for exp, rows in trials.groupby("experiment")
        ]
        files.append({"path": f"slp/{camera}.slp", "blocks": blocks})
    config["slp"] = {"files": files}
    (project / "project.toml").write_text(toml.dumps(config))


def build(version: str, worktree: Path, out: Path) -> None:
    work = Path(tempfile.mkdtemp(prefix=f"label3d-{version}-"))
    run_old(worktree, [str(worktree / "generate_synthetic.py"), str(CONFIG), "--no-charuco"],
            cwd=work)
    project = work / "project"
    for points in project.glob("experiments/*/points.*"):
        points.unlink()
    shorten_trials(project / "trials.csv")

    originals = work / "originals"
    originals.mkdir()
    cameras = []
    for slp in sorted((project / "slp").glob("*.slp")):
        cameras.append(slp.stem)
        sio.save_slp(make_original(sio.load_slp(str(slp)), slp.stem), str(originals / slp.name))
        shutil.copy2(originals / slp.name, slp)
    map_slp_files(project, cameras, pd.read_csv(project / "trials.csv"))

    run_old(worktree, [str(HERE / "old_sync.py"), str(project), version], cwd=work)

    dest = out / f"v{version}"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    shutil.copytree(project, dest / "project")
    shutil.copytree(originals, dest / "originals")
    shutil.rmtree(work)

    for slp in sorted((dest / "project" / "slp").glob("*.slp")):
        kept, original = sio.load_slp(str(slp)), sio.load_slp(str(dest / "originals" / slp.name))
        print(f"  {slp.name}: {len(kept.videos)} of {len(original.videos)} videos, "
              f"{len(kept.labeled_frames)} of {len(original.labeled_frames)} frames kept")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("worktrees", type=Path,
                        help="folder holding label3d-0.0.9/ and label3d-0.1.0/")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    for version in VERSIONS:
        print(f"label3d {version}")
        build(version, args.worktrees / f"label3d-{version}", args.out)


if __name__ == "__main__":
    main()

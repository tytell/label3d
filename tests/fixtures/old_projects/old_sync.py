"""Run a Sync in an old label3d version, as a user would have (Phase 20 fixtures).

Run with the old version's interpreter, from make_old_projects.py:

    uv run --project <worktree> python old_sync.py <project> <version>

Reads the mapped .slp files into the points, labels over two predicted points
in label3d, saves a corners file per camera in that version's format, then
writes the .slp files back. Before 0.2.0 that write drops the videos and frames
outside the project.
"""

import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QCoreApplication

from label3d.project import Project

# (trial, frame, instance, bodypart, camera, dx, dy): user labels placed over predictions
RELABELLED = [
    ("trial1", 2, 0, "snout", "cam1", 3.0, -2.0),
    ("trial1", 4, 0, "dorsal", "cam2", -1.5, 2.5),
]


def main() -> None:
    root, version = Path(sys.argv[1]), sys.argv[2]
    app = QCoreApplication.instance() or QCoreApplication([])

    project = Project.load(root)
    project.apply_merge(project.read_slp_for_sync(), overwrite=True)

    store = project.points
    for trial, frame, inst, bodypart, camera, dx, dy in RELABELLED:
        idx = (project.active_experiment, trial, frame, inst, bodypart)
        row = store.df.loc[idx]
        assert row[(camera, "state")] == "predicted", (idx, camera, row[(camera, "state")])
        store.set_point(trial, frame, inst, bodypart, camera,
                        float(row[(camera, "x")]) + dx, float(row[(camera, "y")]) + dy)
    store.save_if_changed()

    for name, exp in project.experiments.items():
        for camera in project.cameras:
            cam_dir = exp.root / "calibrations" / camera
            cam_dir.mkdir(parents=True, exist_ok=True)
            corners = pd.DataFrame({
                "take": ["PreCal"] * 3, "frame": [0, 0, 5], "corner_id": [0, 1, 0],
                "x": [10.5, 20.25, 11.0], "y": [30.0, 31.5, 29.75], "detected": [True, True, False],
            })
            if version == "0.0.9":
                corners.to_csv(cam_dir / "corners.csv", index=False)
            else:
                corners.to_parquet(cam_dir / "corners.parquet", index=False)

    backup = project.backup_slp_files()
    written = project.write_slp_files()
    print(f"  {version}: wrote {written} .slp files; backup in {backup}")
    del app


if __name__ == "__main__":
    main()

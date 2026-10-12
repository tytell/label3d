# label3d

A desktop app for labelling 3D points across synchronized, calibrated camera views, working with SLEAP.

- **Using it:** [docs/user_workflow.md](docs/user_workflow.md). It covers setup, calibration, labelling, the SLEAP train → review loop, and the final export.
- **Setting up from existing SLEAP files:** [docs/quick_setup_from_slp.md](docs/quick_setup_from_slp.md).
- **Design and developer docs:** [DESIGN.md](DESIGN.md).

Launch with `uv run label3d <project folder>`.

Export every point to one CSV without opening the app with `uv run label3d-export <project folder>` (options: `uv run label3d-export --help`). The columns are listed in [the workflow guide](docs/user_workflow.md#export-columns).

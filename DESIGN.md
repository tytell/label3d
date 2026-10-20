# label3d — Design Document

## Purpose

A cross-platform desktop app for:
1. **Calibrating** multi-camera rigs using Charuco boards (via aniposelib)
2. **Labeling** 3D points by clicking corresponding points across synchronized calibrated video streams
3. **Importing / exporting** 2D pose data from external trackers (SLEAP, etc.) and producing triangulated 3D coordinates

---

## Document Index

| Document | Contents |
|---|---|
| [docs/user_workflow.md](docs/user_workflow.md) | **User guide**: the label → train → review loop, step by step (draft) |
| [docs/design/data_model.md](docs/design/data_model.md) | Project layout, `points.csv` schema, `project.toml`, `trials.csv`, calibration registry, adapter interface, project setup utility |
| [docs/design/workflows.md](docs/design/workflows.md) | Calibration UI, axis alignment, labeling UI, export UI, keyboard shortcuts, key features |
| [docs/design/architecture.md](docs/design/architecture.md) | Package layout, build order, technology stack, risks, design decisions |
| [docs/design/testing.md](docs/design/testing.md) | Test modules and testing conventions |
| [docs/design/project_setup_from_sleap.md](docs/design/project_setup_from_sleap.md) | Building a project from existing `.slp` files: referenced (not copied) videos, skeleton and label import, calibration discovery (original design; outdated, see [docs/setup_from_slp.md](docs/setup_from_slp.md)) |
| [docs/design/synthetic_data.md](docs/design/synthetic_data.md) | Synthetic data generator: module layout, output structure, charuco image generation |
| [docs/design/phases/phase_8.md](docs/design/phases/phase_8.md) | Phase 8 — SLP write-back (completed) |
| [docs/design/phases/phase_9.md](docs/design/phases/phase_9.md) | Phase 9 — Calibration Takes (completed) |
| [docs/design/phases/phase_10.md](docs/design/phases/phase_10.md) | Phase 10 — Point provenance and reviewing new points (completed) |
| [docs/design/phases/phase_11.md](docs/design/phases/phase_11.md) | Phase 11 — Calibration registry and switching calibrations (completed) |
| [docs/design/phases/phase_12.md](docs/design/phases/phase_12.md) | Phase 12 — Refining the calibration with labelled points (complete) |
| [docs/design/phases/phase_13.md](docs/design/phases/phase_13.md) | Phase 13 — Batch triangulation and export |
| [docs/design/phases/phase_14.md](docs/design/phases/phase_14.md) | Phase 14 — Per-camera skeletons (implemented; to be checked in the app) |
| [docs/design/phases/phase_15.md](docs/design/phases/phase_15.md) | Phase 15 — Parquet working files (planned) |
| [docs/design/phases/phase_16.md](docs/design/phases/phase_16.md) | Phase 16 — Axis alignment (deferred; partially implemented) |
| [docs/design/phases/phase_17_18.md](docs/design/phases/phase_17_18.md) | Phases 17–18: Export to other pose formats, Polish and robustness (deferred) |
| [docs/design/phases/phase_19.md](docs/design/phases/phase_19.md) | Phase 19 — Multi-instance support and cross-camera matching (deferred) |
| [docs/design/phases/archive/phases_1_7.md](docs/design/phases/archive/phases_1_7.md) | Phases 1–7 (completed; archived for reference) |

---

## Implementation Status

| Phase | Description | Status |
|---|---|---|
| 1 | Package scaffold and video display | ✓ Complete |
| 2 | Data layer: skeleton, points, adapters | ✓ Complete |
| 3 | Labeling: overlays, interaction, selection | ✓ Complete |
| 4 | Calibration | ✓ Complete |
| 5 | Triangulation and 3D view | ✓ Complete |
| 6 | Hierarchical (multi-experiment) projects | ✓ Complete |
| 7 | SLP import + instance init strategies | ✓ Complete |
| 8 | SLP write-back (File > Sync with Sleap) | ✓ Complete |
| 9 | Calibration takes | ✓ Complete |
| 10 | Point provenance and reviewing new points | ✓ Complete |
| 11 | Calibration registry and switching | ✓ Complete |
| 12 | Calibration refinement with labelled points | ✓ Complete |
| 13 | Batch triangulation and export | In progress — code, tests and docs done; awaiting review in the app |
| 14 | Axis alignment | Deferred; partial — geometry helpers done, UI widget not created |
| 15 | Export to other pose formats | Deferred |
| 16 | Polish and robustness | Deferred |
| 17 | Multi-instance support | Deferred |

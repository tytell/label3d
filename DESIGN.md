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
| [docs/data_model.md](docs/data_model.md) | Project layout, `points.csv` schema, `project.toml`, `trials.csv`, `calibration.toml`, adapter interface, project setup utility |
| [docs/workflows.md](docs/workflows.md) | Calibration UI, axis alignment, labeling UI, export UI, keyboard shortcuts, key features |
| [docs/architecture.md](docs/architecture.md) | Package layout, build order, technology stack, risks, design decisions |
| [docs/testing.md](docs/testing.md) | Test modules and testing conventions |
| [docs/project_setup_from_sleap.md](docs/project_setup_from_sleap.md) | Building a project from existing `.slp` files: referenced (not copied) videos, skeleton and label import, calibration discovery |
| [docs/synthetic_data.md](docs/synthetic_data.md) | Synthetic data generator: module layout, output structure, charuco image generation |
| [docs/phases/phase_8.md](docs/phases/phase_8.md) | Phase 8 — SLP write-back (completed) |
| [docs/phases/phase_9.md](docs/phases/phase_9.md) | Phase 9 — Calibration Takes (completed) |
| [docs/phases/phase_10.md](docs/phases/phase_10.md) | **Phase 10 — Point provenance and reviewing new points** (next) |
| [docs/phases/phase_11.md](docs/phases/phase_11.md) | Phase 11 — Calibration registry and switching calibrations |
| [docs/phases/phase_12.md](docs/phases/phase_12.md) | Phase 12 — Refining the calibration with labelled points |
| [docs/phases/phase_13.md](docs/phases/phase_13.md) | Phase 13 — Batch triangulation and export |
| [docs/phases/phase_14.md](docs/phases/phase_14.md) | Phase 14 — Axis alignment (deferred; partially implemented) |
| [docs/phases/phase_15_16.md](docs/phases/phase_15_16.md) | Phases 15–16: Export view, Polish and robustness (deferred) |
| [docs/phases/phase_17.md](docs/phases/phase_17.md) | Phase 17 — Multi-instance support and cross-camera matching (deferred) |
| [docs/phases/archive/phases_1_7.md](docs/phases/archive/phases_1_7.md) | Phases 1–7 (completed; archived for reference) |

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
| 10 | Point provenance and reviewing new points | Not started (next) |
| 11 | Calibration registry and switching | Not started |
| 12 | Calibration refinement with labelled points | Not started |
| 13 | Batch triangulation and export | Not started |
| 14 | Axis alignment | Deferred; partial — geometry helpers done, UI widget not created |
| 15 | Export view | Deferred |
| 16 | Polish and robustness | Deferred |
| 17 | Multi-instance support | Deferred |

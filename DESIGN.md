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
| [docs/synthetic_data.md](docs/synthetic_data.md) | Synthetic data generator: module layout, output structure, charuco image generation |
| [docs/phases/phase_8.md](docs/phases/phase_8.md) | **Phase 8 — Axis Alignment** (current; partially implemented) |
| [docs/phases/phase_9_11.md](docs/phases/phase_9_11.md) | Phases 9–11: Export, SLP write-back, Polish |
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
| 8 | Axis Alignment | Partial — geometry helpers done, UI widget not created |
| 9 | Export view | Not started |
| 10 | SLP write-back (File > Save) | Not started |
| 11 | Polish and robustness | Not started |

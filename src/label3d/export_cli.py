"""label3d-export: export all points to one file without opening the GUI (Phase 13.3).

    uv run label3d-export <project> [-o out] [--format parquet|csv]
        [--layout wide|long] [--experiments a,b] [--no-predictions] [--only-3d]
        [--no-slp] [--no-3d]

The format sets the file extension; -o gets it if it has another one. 3D
points of an experiment with an axis alignment are written aligned.

Exit code 0 when the export is written; 1 when an experiment's 3D points or
its alignment are stale (unless --no-3d), the SLP files hold points label3d
doesn't have yet, or loading fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication

from label3d.export import (
    FORMATS,
    LAYOUTS,
    ExportOptions,
    describe_slp_changes,
    export_points,
    format_summary,
    pending_slp_changes,
    selected_experiments,
    slp_mapping_complete,
    stale_alignments,
    stale_experiments,
    uncalibrated_experiments,
)
from label3d.layout import ProjectNeedsUpgrade
from label3d.project import Project
from label3d.version import version_string


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="label3d-export",
        description="Export every point of a label3d project to one Parquet or CSV file "
        "and write the mapped SLP files.",
    )
    parser.add_argument("--version", action="version",
                        version=f"label3d-export {version_string()}")
    parser.add_argument("project", type=Path, help="project folder")
    parser.add_argument("-o", "--output", type=Path,
                        help="output file; its extension is set by --format "
                        "(default: <project>/exports/points_<timestamp>.<format>)")
    parser.add_argument("--format", choices=list(FORMATS), default="parquet",
                        help="file format (default: parquet)")
    parser.add_argument("--layout", choices=list(LAYOUTS), default="wide",
                        help="wide: one row per point; long: one row per point and "
                        "view (default: wide)")
    parser.add_argument("--experiments", help="comma-separated experiments (default: all)")
    parser.add_argument("--no-predictions", action="store_true",
                        help="leave out SLEAP predictions")
    parser.add_argument("--only-3d", action="store_true",
                        help="only points that have a 3D point")
    parser.add_argument("--no-slp", action="store_true",
                        help="don't check or write the SLP files")
    parser.add_argument("--no-3d", action="store_true",
                        help="export the 2D points only, leaving the 3D and reprojection "
                        "columns blank; allowed when an alignment is out of date")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # PointsStore is a QObject with an auto-save timer; it needs an application.
    _app = QCoreApplication.instance() or QCoreApplication([])
    try:
        project = Project.load(args.project)
    except ProjectNeedsUpgrade as exc:
        print(f"{exc}\nRun label3d-upgrade {args.project} first.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Could not load {args.project}: {exc}", file=sys.stderr)
        return 1

    experiments = args.experiments.split(",") if args.experiments else None
    options = ExportOptions(
        experiments=experiments,
        include_predictions=not args.no_predictions,
        only_3d=args.only_3d,
        write_slp=not args.no_slp,
        output=args.output,
        format=args.format,
        layout=args.layout,
    )
    try:
        if args.no_3d:
            options.two_d_only = selected_experiments(project, experiments)
        stale = [] if args.no_3d else stale_experiments(project, experiments)
        if stale:
            print(
                f"3D points are not from the active calibration in: {', '.join(stale)}. "
                "Triangulate all experiments in label3d first, or use --no-3d.",
                file=sys.stderr,
            )
            return 1
        stale = [] if args.no_3d else stale_alignments(project, experiments)
        if stale:
            print(
                "The axis alignment was computed with another calibration than the 3D "
                f"points in: {', '.join(stale)}. Recompute it on the Align tab in label3d, "
                "or use --no-3d to export the 2D points only.",
                file=sys.stderr,
            )
            return 1
        uncalibrated = uncalibrated_experiments(project, experiments)
        if uncalibrated:
            print(
                "Warning: no active calibration, exporting 2D points only: "
                + ", ".join(uncalibrated),
                file=sys.stderr,
            )
        if options.write_slp and slp_mapping_complete(project):
            pending = pending_slp_changes(project)
            if pending:
                print(
                    "The SLP files have points label3d doesn't have yet; nothing written.\n"
                    + describe_slp_changes(pending)
                    + "\nRun Sync with Sleap in label3d, triangulate, then export again.",
                    file=sys.stderr,
                )
                return 1
        result = export_points(project, options)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""label3d-upgrade: upgrade a project made by an older label3d, without the GUI (Phase 20).

    uv run label3d-upgrade <project> [--slp-source DIR | --slp-from-backup | --keep-slp]
        [--dry-run]

A project from before 0.2.0 whose .slp files are mapped needs one of the
--slp options: older Syncs dropped the videos and frames outside the project
from those files, and the originals put them back.

  --slp-source DIR    originals found under DIR by file name (a file with no
                      match keeps the project's copy; several matches is an error)
  --slp-from-backup   the copies in the earliest slp/backup/{timestamp}/ folder
  --keep-slp          keep the project's copies as they are

Exit code 0 when the project is upgraded or needs nothing, 1 on an error, 2
when an --slp option is needed or a file has several originals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication

from label3d.layout import STEP_TEXT, UpgradePlan, plan_upgrade
from label3d.upgrade import earliest_backup, find_originals, run_upgrade
from label3d.version import LAYOUT_VERSION, version_string


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="label3d-upgrade",
        description=f"Upgrade a project made by an older label3d to layout {LAYOUT_VERSION}.",
    )
    parser.add_argument("--version", action="version",
                        version=f"label3d-upgrade {version_string()}")
    parser.add_argument("project", type=Path, help="project folder")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--slp-source", type=Path, metavar="DIR",
                        help="folder holding the original .slp files (searched recursively)")
    source.add_argument("--slp-from-backup", action="store_true",
                        help="restore from the earliest Sync backup in slp/backup/")
    source.add_argument("--keep-slp", action="store_true",
                        help="keep the project's .slp files as they are")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="show the version, the steps and the originals; change nothing")
    return parser.parse_args(argv)


def choose_sources(
    plan: UpgradePlan, args: argparse.Namespace
) -> tuple[dict[Path, Path | None], list[str]]:
    """The original for each file to restore, and the problems that stop the upgrade."""
    if not plan.restore or args.keep_slp:
        return {rel: None for rel in plan.restore}, []
    if args.slp_from_backup:
        return {rel: earliest_backup(plan.root, rel.name) for rel in plan.restore}, []
    if args.slp_source is None:
        return {}, ["The .slp files may have lost videos and frames in older Syncs; give "
                    "--slp-source DIR, --slp-from-backup or --keep-slp."]
    found = find_originals([rel.name for rel in plan.restore], args.slp_source)
    problems = [
        f"{rel.as_posix()}: several originals:\n    "
        + "\n    ".join(str(p) for p in found[rel.name])
        for rel in plan.restore if len(found[rel.name]) > 1
    ]
    return {rel: (found[rel.name] or [None])[0] for rel in plan.restore}, problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # PointsStore is a QObject with an auto-save timer; it needs an application.
    _app = QCoreApplication.instance() or QCoreApplication([])
    try:
        plan = plan_upgrade(args.project)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not read {args.project}: {exc}", file=sys.stderr)
        return 1
    if plan is None:
        print(f"{args.project} already records its layout version; nothing to do.")
        return 0

    print(f"{args.project}: made by label3d {plan.from_version} or similar")
    for step in plan.steps:
        print(f"  {STEP_TEXT[step]}")
    sources, problems = choose_sources(plan, args)
    for rel, original in sources.items():
        print(f"  {rel.as_posix()}: " + (f"from {original}" if original else "keep the project's copy"))
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    try:
        result = run_upgrade(args.project, sources, progress=lambda text: print(f"... {text}"))
    except Exception as exc:
        print(f"Upgrade failed; the project is as it was: {exc}", file=sys.stderr)
        return 1
    print("\n".join(result.messages))
    if result.backup_dir is not None:
        print(f"The files as they were are in {result.backup_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

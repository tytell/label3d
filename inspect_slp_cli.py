#!/usr/bin/env python3
"""Inspect .slp files before setting up a project from them.

Lists the videos each .slp file refers to, whether they can be found, and
which camera, experiment and trial setup would give them under one matching
strategy. Reads only; nothing is written.

Usage:
    # the same setup TOML as setup_project_cli.py (output_dir is not needed)
    uv run python inspect_slp_cli.py setup.toml

    # more detail: -v (or --verbose=1), everything: -vv (or --verbose=2)
    uv run python inspect_slp_cli.py -vv setup.toml

    # flags override the TOML, or stand alone with .slp files or folders
    uv run python inspect_slp_cli.py setup.toml --match-strategy number
    uv run python inspect_slp_cli.py labels/ --cameras left,right,bottom \\
        --experiments PS,BG --match-strategy regex --trial-regex '([A-Z]+\\d+_\\d+)' \\
        --video-search-dir /Volumes/data
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections import Counter
from pathlib import Path

from label3d.setup_project import (
    CHARUCO_DEFAULTS,
    FileAssignment,
    SetupConfig,
    classify_slp_files,
    config_from_toml,
    parse_camera_aliases,
    preview_assignment,
)

MATCH_STRATEGIES = ["keyword", "regex", "number", "datestamp", "mtime"]


def split_list(raw: str | None) -> list[str]:
    """Comma-separated values, blanks dropped."""
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def build_config(args) -> SetupConfig:
    """SetupConfig from a setup TOML and/or the command-line flags."""
    inputs = [Path(p).expanduser().resolve() for p in args.inputs]
    tomls = [p for p in inputs if p.suffix == ".toml"]
    if tomls:
        if len(inputs) > 1:
            sys.exit("Give either one setup TOML or .slp files/folders, not both")
        config = config_from_toml(tomls[0], require_output=False)
        if not config.slp_paths:
            sys.exit(
                f"{tomls[0]} has no slp_files in [setup] and no .slp files in its source_dir"
            )
    else:
        if not args.cameras:
            sys.exit("--cameras is required without a setup TOML")
        config = SetupConfig(
            source_dir=inputs[0].parent,
            output_dir=Path("(none)"),
            camera_names=split_list(args.cameras),
            calib_keyword="calib",
            match_strategy="keyword",
            slp_paths=inputs,
            video_mode="reference",
            charuco=CHARUCO_DEFAULTS,
        )

    overrides: dict = {}
    if args.cameras:
        overrides["camera_names"] = split_list(args.cameras)
    if args.camera_alias:
        overrides["camera_aliases"] = {
            **config.camera_aliases, **parse_camera_aliases(args.camera_alias)
        }
    if args.experiments is not None:
        overrides["experiment_keywords"] = split_list(args.experiments)
    if args.match_strategy:
        overrides["match_strategy"] = args.match_strategy
    if args.trial_keyword:
        overrides["trial_keyword"] = args.trial_keyword
    if args.trial_regex:
        overrides["trial_regex"] = args.trial_regex
    if args.mtime_window is not None:
        overrides["mtime_window_sec"] = args.mtime_window
    if args.datestamp_window is not None:
        overrides["datestamp_window_sec"] = args.datestamp_window
    if args.calib_keywords is not None:
        keywords = split_list(args.calib_keywords)
        overrides["calib_keywords"] = keywords
        overrides["calib_keyword"] = keywords[0] if keywords else "calib"
    if args.video_search_dir:
        overrides["video_search_dirs"] = [
            *config.video_search_dirs,
            *(Path(d).expanduser().resolve() for d in args.video_search_dir),
        ]
    return dataclasses.replace(config, **overrides)


def video_placement(assignment: FileAssignment) -> dict[Path, tuple[str, str]]:
    """Resolved video -> (experiment, trial) it was grouped into."""
    placement = {}
    for trial, cam_files in assignment.trials.items():
        experiment = assignment.trial_experiment.get(trial, "default")
        for video in cam_files.values():
            placement[video] = (experiment, trial)
    return placement


# Without --verbose, a problem list longer than this is given as a count
MAX_LISTED = 10


def _problem(ref, placement, excluded) -> str | None:
    """Why a reference did not end up in a trial, or None if it did."""
    outcome = ref.resolution
    if outcome.path is None:
        return "ambiguous" if outcome.rule == "ambiguous" else "not found"
    if outcome.path in excluded:
        return "in an experiment not listed"
    if ref.camera is None:
        return "no camera match"
    if outcome.path not in placement:
        return "not in a trial of a listed experiment"
    return None


def _ref_lines(ref, placement) -> list[str]:
    """One reference: its recorded path, then where it was found and placed."""
    outcome = ref.resolution
    if outcome.path is None:
        status = "AMBIGUOUS" if outcome.rule == "ambiguous" else "MISSING"
        lines = [f"    {status:9}  {ref.recorded}"]
        lines += [f"               could be {c}" for c in outcome.candidates]
        return lines
    lines = [f"    ok         {ref.recorded}"]
    if outcome.rule != "recorded":
        lines.append(f"               found at {outcome.path}  ({outcome.rule})")
    camera = ref.camera or "(no camera match)"
    exp, trial = placement.get(outcome.path, ("-", "(no trial)"))
    lines.append(f"               camera {camera}, experiment {exp}, trial {trial}")
    return lines


def report_slp_files(assignment: FileAssignment, verbosity: int = 0) -> list[str]:
    """Each .slp file's video references: found where, which camera and trial.

    Verbosity 0 gives the trials per experiment and counts of each kind of
    problem; 1 lists the references placed in a trial, and each kind of problem
    with at most MAX_LISTED references; 2 lists every reference.
    """
    placement = video_placement(assignment)
    excluded = set(assignment.excluded)
    by_slp: dict[Path, list] = {}
    for ref in assignment.video_refs:
        by_slp.setdefault(ref.slp_path, []).append(ref)

    lines = [f"SLP files ({len(by_slp)}):"]
    for slp_path, refs in by_slp.items():
        found = sum(r.resolution.path is not None for r in refs)
        trials = sorted(
            {placement[r.resolution.path] for r in refs if r.resolution.path in placement}
        )
        lines.append(f"  {slp_path}")
        lines.append(f"    {len(refs)} video(s), {found} found")
        for exp, count in sorted(Counter(exp for exp, _ in trials).items()):
            lines.append(f"    {exp}: {count} trial(s)")

        problems: dict[str, list] = {}
        for ref in refs:
            kind = _problem(ref, placement, excluded)
            if kind is not None:
                problems.setdefault(kind, []).append(ref)
            if verbosity >= 2 or (verbosity == 1 and kind is None):
                lines += _ref_lines(ref, placement)
        if verbosity < 2:
            for kind, kind_refs in problems.items():
                if verbosity == 1 and len(kind_refs) <= MAX_LISTED:
                    for ref in kind_refs:
                        lines += _ref_lines(ref, placement)
                else:
                    lines.append(f"    {len(kind_refs)} video(s) {kind}")
        lines.append("")
    return lines


def report_folders(assignment: FileAssignment) -> list[str]:
    """The distinct folders the .slp files record videos in, and where they are now.

    A folder whose videos were found elsewhere is followed by each folder they
    were found in, with how many.
    """
    counts = Counter(str(ref.recorded.parent) for ref in assignment.video_refs)
    found = Counter(
        str(ref.recorded.parent)
        for ref in assignment.video_refs
        if ref.resolution.path is not None
    )
    moved: dict[str, Counter] = {}
    for ref in assignment.video_refs:
        outcome = ref.resolution
        if outcome.path is not None and outcome.rule != "recorded":
            moved.setdefault(str(ref.recorded.parent), Counter())[
                str(outcome.path.parent)
            ] += 1
    lines = [f"Recorded video folders ({len(counts)}):"]
    for folder in sorted(counts):
        exists = "exists" if Path(folder).is_dir() else "NOT FOUND"
        lines.append(
            f"  {folder}  [{exists}]  {counts[folder]} video(s), {found[folder]} located"
        )
        for new_folder, count in sorted(moved.get(folder, Counter()).items()):
            lines.append(f"    -> {new_folder}  ({count} video(s))")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List the videos in .slp files and test how setup would "
        "match them into experiments and trials. Reads only."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="PATH",
        help="A setup TOML (as for setup_project_cli.py), or .slp files / folders",
    )
    parser.add_argument("--cameras", help="Comma-separated camera names")
    parser.add_argument(
        "--camera-alias",
        action="append",
        metavar="CAM=ALIAS[,ALIAS]",
        help="Other names a camera goes by in filenames (repeatable; adds to the TOML's)",
    )
    parser.add_argument(
        "--experiments", help="Comma-separated experiment keywords ('' for none)"
    )
    parser.add_argument("--match-strategy", choices=MATCH_STRATEGIES)
    parser.add_argument("--trial-keyword", help="For --match-strategy keyword")
    parser.add_argument("--trial-regex", help="For --match-strategy regex")
    parser.add_argument("--mtime-window", type=float, help="For --match-strategy mtime (s)")
    parser.add_argument(
        "--datestamp-window", type=float, help="For --match-strategy datestamp (s)"
    )
    parser.add_argument("--calib-keywords", help="Comma-separated calibration take keywords")
    parser.add_argument(
        "--video-search-dir",
        action="append",
        metavar="DIR",
        help="Extra folder to search when a recorded video path has moved (repeatable)",
    )
    parser.add_argument(
        "-v", action="count", default=0, dest="verbosity",
        help="More detail: -v lists the videos placed in trials and short problem "
        "lists, -vv lists everything",
    )
    parser.add_argument(
        "--verbose", type=int, dest="verbosity", default=argparse.SUPPRESS,
        metavar="LEVEL", help="Detail level 0-2 (same as no -v, -v, -vv)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    config = build_config(args)
    try:
        assignment = classify_slp_files(config)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    print("\n".join(report_slp_files(assignment, verbosity=args.verbosity)))
    print("\n".join(report_folders(assignment)))
    print("-" * 60)
    print("What setup would do:")
    print()
    print(preview_assignment(config, assignment, verbosity=args.verbosity))


if __name__ == "__main__":
    main()

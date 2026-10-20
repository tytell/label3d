#!/usr/bin/env python3
"""Inspect .slp files before setting up a project from them.

Lists the videos each .slp file refers to, whether they can be found, and
which camera, experiment and trial setup would give them under one matching
strategy. Reads only; nothing is written.

Usage:
    # the same setup TOML as setup_project_cli.py (output_dir is not needed)
    uv run python inspect_slp_cli.py setup.toml

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


def report_slp_files(assignment: FileAssignment) -> list[str]:
    """Each .slp file's video references: found where, which camera and trial."""
    placement = video_placement(assignment)
    by_slp: dict[Path, list] = {}
    for ref in assignment.video_refs:
        by_slp.setdefault(ref.slp_path, []).append(ref)

    lines = [f"SLP files ({len(by_slp)}):"]
    for slp_path, refs in by_slp.items():
        found = sum(r.resolution.path is not None for r in refs)
        trials = sorted(
            {placement[r.resolution.path] for r in refs if r.resolution.path in placement}
        )
        experiments = sorted({exp for exp, _ in trials})
        lines.append(f"  {slp_path}")
        lines.append(
            f"    {len(refs)} video(s), {found} found; "
            f"experiments: {', '.join(experiments) or '-'}; "
            f"trials: {', '.join(t for _, t in trials) or '-'}"
        )
        for ref in refs:
            outcome = ref.resolution
            if outcome.path is None:
                status = "AMBIGUOUS" if outcome.rule == "ambiguous" else "MISSING"
                lines.append(f"    {status:9}  {ref.recorded}")
                for candidate in outcome.candidates:
                    lines.append(f"               could be {candidate}")
                continue
            camera = ref.camera or "(no camera match)"
            exp, trial = placement.get(outcome.path, ("-", "(no trial)"))
            lines.append(f"    ok         {ref.recorded}")
            if outcome.rule != "recorded":
                lines.append(f"               found at {outcome.path}  ({outcome.rule})")
            lines.append(f"               camera {camera}, experiment {exp}, trial {trial}")
        lines.append("")
    return lines


def report_folders(assignment: FileAssignment) -> list[str]:
    """The distinct folders the .slp files record videos in, and whether they exist."""
    counts = Counter(str(ref.recorded.parent) for ref in assignment.video_refs)
    found = Counter(
        str(ref.recorded.parent)
        for ref in assignment.video_refs
        if ref.resolution.path is not None
    )
    lines = [f"Recorded video folders ({len(counts)}):"]
    for folder in sorted(counts):
        exists = "exists" if Path(folder).is_dir() else "NOT FOUND"
        lines.append(
            f"  {folder}  [{exists}]  {counts[folder]} video(s), {found[folder]} located"
        )
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    config = build_config(args)
    try:
        assignment = classify_slp_files(config)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    print("\n".join(report_slp_files(assignment)))
    print("\n".join(report_folders(assignment)))
    print("-" * 60)
    print("What setup would do:")
    print()
    print(preview_assignment(config, assignment))


if __name__ == "__main__":
    main()

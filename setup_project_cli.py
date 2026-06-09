#!/usr/bin/env python3
"""Interactive CLI for setting up a label3d project from a flat folder of videos.

Usage:
    uv run python setup_project_cli.py                  # interactive mode
    uv run python setup_project_cli.py config.toml      # config file mode

Config file format (TOML):
    [setup]
    source_dir = "/path/to/flat/folder"
    output_dir = "/path/to/new/project"
    camera_names = ["left", "right", "bottom"]
    calib_keyword = "calib"
    experiment_keywords = ["PS", "BG"]   # omit or [] for single experiment named "default"
    match_strategy = "keyword"   # "keyword" | "regex" | "number" | "datestamp" | "mtime"
    trial_keyword = "Trial"      # only for keyword strategy
    trial_regex = "([A-Z]+\\d+_\\d+)"  # only for regex strategy
    mtime_window_sec = 60.0      # only for mtime strategy

    [project]
    name = "my_experiment"

    [charuco]
    squares_x = 6
    squares_y = 8
    square_length = 30.0
    marker_length = 22.0
    marker_bits = 4
    dict_size = 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import toml

from label3d.setup_project import (
    SetupConfig,
    classify_files,
    execute_setup,
    preview_assignment,
)

# Default charuco board parameters (from DESIGN.md)
CHARUCO_DEFAULTS = {
    "squares_x": 6,
    "squares_y": 8,
    "square_length": 30.0,
    "marker_length": 22.0,
    "marker_bits": 4,
    "dict_size": 50,
}


def prompt_path(label: str, must_exist: bool = False) -> Path:
    """Prompt user for a file path."""
    while True:
        raw = input(f"{label}: ").strip()
        if not raw:
            print("  Please enter a path.")
            continue
        p = Path(raw).expanduser().resolve()
        if must_exist and not p.exists():
            print(f"  Path does not exist: {p}")
            continue
        return p


def prompt_list(label: str) -> list[str]:
    """Prompt user for a comma-separated list."""
    while True:
        raw = input(f"{label}: ").strip()
        items = [s.strip() for s in raw.split(",") if s.strip()]
        if not items:
            print("  Please enter at least one value.")
            continue
        return items


def prompt_optional_list(label: str, hint: str = "") -> list[str]:
    """Prompt for comma-separated values; blank entry returns empty list."""
    if hint:
        print(f"  {hint}")
    raw = input(f"{label}: ").strip()
    return [s.strip() for s in raw.split(",") if s.strip()]


def prompt_choice(label: str, choices: list[str]) -> str:
    """Prompt user to choose from a list."""
    choices_str = " / ".join(choices)
    while True:
        raw = input(f"{label} [{choices_str}]: ").strip().lower()
        if raw in choices:
            return raw
        print(f"  Please choose one of: {choices_str}")


def prompt_float(label: str, default: float) -> float:
    """Prompt for a float with a default value."""
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid number, using default: {default}")
        return default


def prompt_int(label: str, default: int) -> int:
    """Prompt for an int with a default value."""
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Invalid number, using default: {default}")
        return default


def prompt_charuco() -> dict:
    """Prompt user for charuco board parameters."""
    print("\nCharuco board parameters (press Enter for defaults):")
    return {
        "squares_x": prompt_int("  Squares X", CHARUCO_DEFAULTS["squares_x"]),
        "squares_y": prompt_int("  Squares Y", CHARUCO_DEFAULTS["squares_y"]),
        "square_length": prompt_float(
            "  Square length (mm)", CHARUCO_DEFAULTS["square_length"]
        ),
        "marker_length": prompt_float(
            "  Marker length (mm)", CHARUCO_DEFAULTS["marker_length"]
        ),
        "marker_bits": prompt_int(
            "  ArUco bits (4/5/6/7)", CHARUCO_DEFAULTS["marker_bits"]
        ),
        "dict_size": prompt_int(
            "  Dictionary size (50/100/250/1000)", CHARUCO_DEFAULTS["dict_size"]
        ),
    }


def save_config_toml(config: SetupConfig, path: Path) -> None:
    """Write a TOML config file capturing all settings from a SetupConfig."""
    data: dict = {
        "setup": {
            "source_dir": str(config.source_dir),
            "output_dir": str(config.output_dir),
            "camera_names": config.camera_names,
            "calib_keyword": config.calib_keyword,
            "experiment_keywords": config.experiment_keywords,
            "match_strategy": config.match_strategy,
        },
        "project": {"name": config.project_name},
    }
    setup = data["setup"]
    if config.match_strategy == "keyword":
        setup["trial_keyword"] = config.trial_keyword
    elif config.match_strategy == "regex":
        setup["trial_regex"] = config.trial_regex
    elif config.match_strategy == "mtime":
        setup["mtime_window_sec"] = config.mtime_window_sec
    elif config.match_strategy == "datestamp":
        setup["datestamp_window_sec"] = config.datestamp_window_sec
    if config.charuco:
        data["charuco"] = config.charuco
    with open(path, "w") as f:
        toml.dump(data, f)


def run_interactive(copy_files: bool = False, dry_run: bool = False) -> None:
    """Run the setup utility in interactive mode."""
    print("=" * 60)
    print("label3d — Project Setup")
    print("=" * 60)
    print()

    source_dir = prompt_path("Source folder (flat folder with videos)", must_exist=True)
    output_dir = prompt_path("Output project folder")
    project_name = input("Project name [{}]: ".format(output_dir.name)).strip()
    if not project_name:
        project_name = output_dir.name

    print()
    camera_names = prompt_list("Camera names (comma-separated, e.g. left,right,bottom)")
    calib_keyword = input("Calibration keyword (e.g. calib, charuco): ").strip()

    print()
    print("--- Experiment grouping ---")
    experiment_keywords = prompt_optional_list(
        "Experiment keywords, e.g. PS,BG (leave blank for single experiment [default])",
        hint="Files matching PS\\d+ → experiment PS18, BG\\d+ → BG03, etc.",
    )
    if experiment_keywords:
        print(f"  Experiments: {', '.join(experiment_keywords)}")

    print()
    print("--- Trial matching (within each experiment) ---")
    match_strategy = prompt_choice(
        "Trial matching strategy", ["keyword", "regex", "number", "datestamp", "mtime"]
    )
    mtime_window = 60.0
    datestamp_window = 0.0
    trial_keyword = "Trial"
    trial_regex = ""
    if match_strategy == "keyword":
        trial_keyword = input("  Trial keyword (e.g. Trial, trial) [Trial]: ").strip() or "Trial"
    elif match_strategy == "regex":
        print("  Regex matched against filename stem; named group 'trial' or group 1 becomes the trial key.")
        print("  Example: ([A-Z]+\\d+_\\d+) matches 'PS18_20' from 'PS18_20_Ventral.mp4'")
        while not trial_regex:
            trial_regex = input("  Pattern: ").strip()
            if not trial_regex:
                print("  Please enter a pattern.")
    elif match_strategy == "mtime":
        mtime_window = prompt_float("  Clustering window (seconds)", 60.0)
    elif match_strategy == "datestamp":
        datestamp_window = prompt_float(
            "  Timestamp tolerance (seconds, 0=exact match)", 0.0
        )

    charuco = prompt_charuco()

    config = SetupConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        camera_names=camera_names,
        calib_keyword=calib_keyword,
        experiment_keywords=experiment_keywords,
        match_strategy=match_strategy,
        mtime_window_sec=mtime_window,
        datestamp_window_sec=datestamp_window,
        trial_keyword=trial_keyword,
        trial_regex=trial_regex,
        charuco=charuco,
        project_name=project_name,
        copy_files=copy_files,
    )

    # Classify files
    print("\nScanning files...")
    assignment = classify_files(config)

    # Show preview
    print()
    print("-" * 60)
    print(preview_assignment(config, assignment))
    print("-" * 60)

    if dry_run:
        print("\nDry run — no files will be moved or copied.")
        save_path_str = input("Save config to TOML file? Enter path (or press Enter to skip): ").strip()
        if save_path_str:
            save_path = Path(save_path_str).expanduser().resolve()
            save_config_toml(config, save_path)
            print(f"Config saved to: {save_path}")
            print(f"Run with: uv run python setup_project_cli.py {save_path}")
        return

    # Confirm
    action = "COPIED" if copy_files else "MOVED"
    confirm = input(f"\nProceed? Files will be {action} to the project folder. [y/N]: ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return

    # Execute
    result_dir = execute_setup(config, assignment)
    print(f"\nProject created at: {result_dir}")
    print("You can now open it with: uv run label3d", result_dir)


def run_from_config(config_path: Path, copy_files: bool = False, dry_run: bool = False) -> None:
    """Run the setup utility from a TOML config file."""
    raw = toml.load(config_path)
    setup = raw.get("setup", {})

    source_dir = Path(setup["source_dir"]).expanduser().resolve()
    output_dir = Path(setup["output_dir"]).expanduser().resolve()

    config = SetupConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        camera_names=setup["camera_names"],
        calib_keyword=setup.get("calib_keyword", "calib"),
        experiment_keywords=setup.get("experiment_keywords", []),
        match_strategy=setup.get("match_strategy", "keyword"),
        trial_keyword=setup.get("trial_keyword", "Trial"),
        trial_regex=setup.get("trial_regex", ""),
        mtime_window_sec=setup.get("mtime_window_sec", 60.0),
        datestamp_window_sec=setup.get("datestamp_window_sec", 0.0),
        charuco=raw.get("charuco", CHARUCO_DEFAULTS),
        project_name=raw.get("project", {}).get("name", output_dir.name),
        copy_files=copy_files,
    )

    print("Scanning files...")
    assignment = classify_files(config)

    print()
    print(preview_assignment(config, assignment))
    print()

    if dry_run:
        print("Dry run — no files will be moved or copied.")
        return

    result_dir = execute_setup(config, assignment)
    print(f"Project created at: {result_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up a label3d project from a flat folder of videos."
    )
    parser.add_argument(
        "config",
        nargs="?",
        metavar="CONFIG",
        help="TOML config file (omit for interactive mode)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Copy files instead of moving them (useful for testing)",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be done without moving or copying any files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        run_from_config(config_path, copy_files=args.copy, dry_run=args.dry_run)
    else:
        run_interactive(copy_files=args.copy, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

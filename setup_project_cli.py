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
    match_strategy = "number"    # "number" | "datestamp" | "mtime"
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


def run_interactive() -> None:
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
    match_strategy = prompt_choice(
        "Trial matching strategy", ["number", "datestamp", "mtime"]
    )
    mtime_window = 60.0
    datestamp_window = 0.0
    if match_strategy == "mtime":
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
        match_strategy=match_strategy,
        mtime_window_sec=mtime_window,
        datestamp_window_sec=datestamp_window,
        charuco=charuco,
        project_name=project_name,
    )

    # Classify files
    print("\nScanning files...")
    assignment = classify_files(config)

    # Show preview
    print()
    print("-" * 60)
    print(preview_assignment(config, assignment))
    print("-" * 60)

    # Confirm
    confirm = input("\nProceed? Files will be MOVED to the project folder. [y/N]: ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return

    # Execute
    result_dir = execute_setup(config, assignment)
    print(f"\nProject created at: {result_dir}")
    print("You can now open it with: uv run label3d", result_dir)


def run_from_config(config_path: Path) -> None:
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
        match_strategy=setup.get("match_strategy", "number"),
        mtime_window_sec=setup.get("mtime_window_sec", 60.0),
        datestamp_window_sec=setup.get("datestamp_window_sec", 0.0),
        charuco=raw.get("charuco", CHARUCO_DEFAULTS),
        project_name=raw.get("project", {}).get("name", output_dir.name),
    )

    print("Scanning files...")
    assignment = classify_files(config)

    print()
    print(preview_assignment(config, assignment))
    print()

    result_dir = execute_setup(config, assignment)
    print(f"Project created at: {result_dir}")


def main() -> None:
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        if not config_path.exists():
            print(f"Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        run_from_config(config_path)
    else:
        run_interactive()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive CLI for setting up a label3d project from a flat folder of videos.

Usage:
    uv run python setup_project_cli.py                  # interactive mode
    uv run python setup_project_cli.py config.toml      # config file mode

    # from existing .slp files, videos referenced where they already live
    # (see docs/project_setup_from_sleap.md)
    uv run python setup_project_cli.py --from-slp labels/ \
        --output /path/to/project --cameras left,right,bottom

Config file format (TOML):
    [setup]
    source_dir = "/path/to/flat/folder"
    output_dir = "/path/to/new/project"
    camera_names = ["left", "right", "bottom"]
    calib_keyword = "calib"          # legacy single keyword
    calib_keywords = ["PreCal", "PostCal"]   # one keyword per calibration take
    calib_regex = ""                 # alternative: regex naming each take, e.g. "cal\\d+"
    experiment_keywords = ["PS", "BG"]   # omit or [] for single experiment named "default"
    match_strategy = "keyword"   # "keyword" | "regex" | "number" | "datestamp" | "mtime"
    trial_keyword = "Trial"      # only for keyword strategy
    trial_regex = "([A-Z]+\\d+_\\d+)"  # only for regex strategy
    mtime_window_sec = 60.0      # only for mtime strategy

    # SLP-first setup: give slp_files instead of source_dir
    slp_files = ["/path/to/labels"]          # .slp files or folders holding them
    video_mode = "reference"                 # "reference" | "copy" | "move"
    video_search_dirs = ["/mnt/videos"]      # searched when a recorded path moved
    allow_missing_videos = false             # true: proceed with the videos that were found
    calibration_toml = "/path/calibration.toml"   # aniposelib file, imported into the registry; or calibration_dir
    import_labels = true                     # import SLP labels into points.csv

    [project]
    name = "my_experiment"

    # Optional; must not conflict with the .slp files' skeletons (shared nodes
    # must agree on edges and symmetries).
    # Either a skeleton.json (relative paths from the working directory) …
    [skeleton]
    file = "/path/to/skeleton.json"
    # cameras = ["cam1", "cam2"]   # only these cameras (default: all)
    # … or the skeleton inline:
    # nodes = ["snout", "left_eye", "right_eye", "tail"]
    # edges = [["snout", "left_eye"], ["snout", "right_eye"], ["snout", "tail"]]
    # symmetries = [["left_eye", "right_eye"]]
    # node_offsets_cm = { snout = [2.0, 0.0, 0.0], tail = [-3.0, 0.0, 0.0] }

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
    classify_slp_files,
    execute_setup,
    import_labels,
    preview_assignment,
    skeleton_from_setup,
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
            "calib_keywords": config.calib_keywords,
            "calib_regex": config.calib_regex,
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
    if config.slp_paths:
        setup["slp_files"] = [str(p) for p in config.slp_paths]
        setup["video_mode"] = config.effective_video_mode()
        if config.video_search_dirs:
            setup["video_search_dirs"] = [str(p) for p in config.video_search_dirs]
        if config.allow_missing_videos:
            setup["allow_missing_videos"] = True
        if config.calibration_toml:
            setup["calibration_toml"] = str(config.calibration_toml)
        if config.calibration_dir:
            setup["calibration_dir"] = str(config.calibration_dir)
    if config.charuco:
        data["charuco"] = config.charuco
    with open(path, "w") as f:
        toml.dump(data, f)


def prompt_trial_matching() -> dict:
    """Prompt for the trial-matching strategy and its parameters."""
    match_strategy = prompt_choice(
        "Trial matching strategy", ["keyword", "regex", "number", "datestamp", "mtime"]
    )
    settings = {
        "match_strategy": match_strategy,
        "trial_keyword": "Trial",
        "trial_regex": "",
        "mtime_window_sec": 60.0,
        "datestamp_window_sec": 0.0,
    }
    if match_strategy == "keyword":
        settings["trial_keyword"] = (
            input("  Trial keyword (e.g. Trial, trial) [Trial]: ").strip() or "Trial"
        )
    elif match_strategy == "regex":
        print("  Regex matched against filename stem; named group 'trial' or group 1 becomes the trial key.")
        print("  Example: ([A-Z]+\\d+_\\d+) matches 'PS18_20' from 'PS18_20_Ventral.mp4'")
        while not settings["trial_regex"]:
            settings["trial_regex"] = input("  Pattern: ").strip()
            if not settings["trial_regex"]:
                print("  Please enter a pattern.")
    elif match_strategy == "mtime":
        settings["mtime_window_sec"] = prompt_float("  Clustering window (seconds)", 60.0)
    elif match_strategy == "datestamp":
        settings["datestamp_window_sec"] = prompt_float(
            "  Timestamp tolerance (seconds, 0=exact match)", 0.0
        )
    return settings


def run_interactive_slp(dry_run: bool = False, do_import: bool = True) -> None:
    """Interactive setup from existing .slp files, referencing videos in place."""
    print()
    print("--- Source SLP files ---")
    print("  Give a folder of .slp files, or one .slp file. The videos they")
    print("  reference stay where they are; only the .slp files are copied in.")
    slp_source = prompt_path("SLP file or folder", must_exist=True)
    output_dir = prompt_path("Output project folder")
    project_name = input(f"Project name [{output_dir.name}]: ").strip() or output_dir.name

    print()
    camera_names = prompt_list("Camera names (comma-separated, e.g. left,right,bottom)")
    print("  Matched against each video's filename, then its parent folder name.")

    print()
    print("--- Video search ---")
    print("  Extra folders to search when a .slp records a path the video has moved from.")
    search_raw = input("Search folders (comma-separated, blank for none): ").strip()
    video_search_dirs = [
        Path(s.strip()).expanduser().resolve() for s in search_raw.split(",") if s.strip()
    ]

    allow_missing = input(
        "Continue if some videos cannot be found? [y/N]: "
    ).strip().lower() == "y"

    print()
    print("--- Experiment grouping ---")
    experiment_keywords = prompt_optional_list(
        "Experiment keywords, e.g. PS,BG (leave blank for single experiment [default])"
    )

    print()
    print("--- Trial matching ---")
    matching = prompt_trial_matching()

    print()
    print("--- Calibration ---")
    print("  An aniposelib calibration .toml or calibration videos beside the .slp")
    print("  files are found automatically. A .toml is imported into the project's")
    print("  calibration registry. Give another folder here, or leave blank.")
    calib_raw = input("Calibration file or folder (blank to auto-detect): ").strip()
    calibration_toml = None
    calibration_dir = None
    if calib_raw:
        calib_path = Path(calib_raw).expanduser().resolve()
        if calib_path.is_dir():
            calibration_dir = calib_path
        else:
            calibration_toml = calib_path
    calib_keywords = prompt_optional_list(
        "Calibration keywords for calibration videos [calib]"
    )

    charuco = prompt_charuco()

    config = SetupConfig(
        source_dir=slp_source if slp_source.is_dir() else slp_source.parent,
        output_dir=output_dir,
        camera_names=camera_names,
        calib_keyword=calib_keywords[0] if calib_keywords else "calib",
        calib_keywords=calib_keywords,
        experiment_keywords=experiment_keywords,
        slp_paths=[slp_source],
        video_mode="reference",
        video_search_dirs=video_search_dirs,
        allow_missing_videos=allow_missing,
        calibration_toml=calibration_toml,
        calibration_dir=calibration_dir,
        charuco=charuco,
        project_name=project_name,
        **matching,
    )

    print("\nReading SLP files...")
    assignment = classify_slp_files(config)

    print()
    print("-" * 60)
    print(preview_assignment(config, assignment))
    print("-" * 60)

    if dry_run:
        print("\nDry run — nothing was written.")
        return

    if assignment.missing_videos and not config.allow_missing_videos:
        print(
            f"\n{len(assignment.missing_videos)} video reference(s) could not be "
            "resolved. Re-run with extra search folders, or with "
            "--allow-missing-videos to use the videos that were found."
        )
        return
    if assignment.missing_videos:
        print(
            f"\n{len(assignment.missing_videos)} video reference(s) could not be "
            "resolved; those cameras will be left blank in trials.csv."
        )

    confirm = input("\nProceed? Videos stay where they are. [y/N]: ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return

    result_dir = execute_setup(config, assignment)
    print(f"\nProject created at: {result_dir}")
    if do_import:
        counts = import_labels(result_dir)
        total = sum(counts.values())
        print(f"Imported labels for {total} frame(s) from the SLP files.")
    print("You can now open it with: uv run label3d", result_dir)


def run_interactive(copy_files: bool = False, dry_run: bool = False) -> None:
    """Run the setup utility in interactive mode (the banner is printed by main)."""
    print()

    source_dir = prompt_path("Source folder (flat folder with videos)", must_exist=True)
    output_dir = prompt_path("Output project folder")
    project_name = input("Project name [{}]: ".format(output_dir.name)).strip()
    if not project_name:
        project_name = output_dir.name

    print()
    camera_names = prompt_list("Camera names (comma-separated, e.g. left,right,bottom)")

    print()
    print("--- Calibration takes ---")
    print("  Each keyword names one calibration take, e.g. PreCal,PostCal for a")
    print("  recording before and after the trials. A single keyword (calib) means")
    print("  one take per experiment.")
    calib_keywords = prompt_optional_list("Calibration keywords")
    calib_regex = ""
    if not calib_keywords:
        print("  No keywords given — supply a regex naming each take; its 'take' group,")
        print("  else group 1, else the whole match becomes the take name,")
        print("  e.g. _(?P<take>(?:Pre|Post)Cal)_  (leave blank to use 'calib')")
        calib_regex = input("Calibration regex: ").strip()
    calib_keyword = calib_keywords[0] if calib_keywords else "calib"

    allow_missing = input(
        "Continue if some videos cannot be found? [y/N]: "
    ).strip().lower() == "y"

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
        calib_keywords=calib_keywords,
        calib_regex=calib_regex,
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


def run_slp_setup(
    config: SetupConfig, dry_run: bool = False, do_import: bool = True
) -> None:
    """Classify, preview and execute an SLP-first setup non-interactively."""
    print("Reading SLP files...")
    assignment = classify_slp_files(config)

    print()
    print(preview_assignment(config, assignment))
    print()

    if dry_run:
        print("Dry run — nothing was written.")
        return

    result_dir = execute_setup(config, assignment)
    print(f"Project created at: {result_dir}")
    if do_import:
        counts = import_labels(result_dir)
        print(f"Imported labels for {sum(counts.values())} frame(s).")


def slp_config_from_args(args) -> SetupConfig:
    """Build a SetupConfig from the --from-slp command-line flags."""
    if not args.output:
        print("--from-slp requires --output", file=sys.stderr)
        sys.exit(1)
    if not args.cameras:
        print("--from-slp requires --cameras", file=sys.stderr)
        sys.exit(1)

    slp_paths = [Path(p).expanduser().resolve() for p in args.from_slp]
    output_dir = Path(args.output).expanduser().resolve()
    calibration_toml = None
    calibration_dir = None
    if args.calibration:
        calib_path = Path(args.calibration).expanduser().resolve()
        if calib_path.is_dir():
            calibration_dir = calib_path
        else:
            calibration_toml = calib_path

    calib_keywords = [s.strip() for s in args.calib_keywords.split(",") if s.strip()]
    return SetupConfig(
        source_dir=slp_paths[0].parent,
        output_dir=output_dir,
        camera_names=[s.strip() for s in args.cameras.split(",") if s.strip()],
        calib_keyword=calib_keywords[0] if calib_keywords else "calib",
        calib_keywords=calib_keywords,
        experiment_keywords=[
            s.strip() for s in (args.experiments or "").split(",") if s.strip()
        ],
        match_strategy=args.match_strategy,
        trial_keyword=args.trial_keyword,
        trial_regex=args.trial_regex,
        slp_paths=slp_paths,
        video_mode=args.video_mode,
        video_search_dirs=[
            Path(d).expanduser().resolve() for d in (args.video_search_dir or [])
        ],
        allow_missing_videos=args.allow_missing_videos,
        calibration_toml=calibration_toml,
        calibration_dir=calibration_dir,
        charuco=CHARUCO_DEFAULTS,
        project_name=args.name or output_dir.name,
    )


def run_from_config(config_path: Path, copy_files: bool = False, dry_run: bool = False) -> None:
    """Run the setup utility from a TOML config file."""
    raw = toml.load(config_path)
    setup = raw.get("setup", {})

    slp_files = [Path(p).expanduser().resolve() for p in setup.get("slp_files", [])]
    output_dir = Path(setup["output_dir"]).expanduser().resolve()
    source_dir = (
        Path(setup["source_dir"]).expanduser().resolve()
        if setup.get("source_dir")
        else slp_files[0].parent
    )

    config = SetupConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        camera_names=setup["camera_names"],
        calib_keyword=setup.get("calib_keyword", "calib"),
        calib_keywords=setup.get("calib_keywords", []),
        calib_regex=setup.get("calib_regex", ""),
        calib_match_strategy=setup.get("calib_match_strategy", ""),
        experiment_keywords=setup.get("experiment_keywords", []),
        match_strategy=setup.get("match_strategy", "keyword"),
        trial_keyword=setup.get("trial_keyword", "Trial"),
        trial_regex=setup.get("trial_regex", ""),
        mtime_window_sec=setup.get("mtime_window_sec", 60.0),
        datestamp_window_sec=setup.get("datestamp_window_sec", 0.0),
        charuco=raw.get("charuco", CHARUCO_DEFAULTS),
        project_name=raw.get("project", {}).get("name", output_dir.name),
        copy_files=copy_files,
        slp_paths=slp_files,
        video_mode=setup.get("video_mode", "reference" if slp_files else ""),
        video_search_dirs=[
            Path(d).expanduser().resolve() for d in setup.get("video_search_dirs", [])
        ],
        allow_missing_videos=setup.get("allow_missing_videos", False),
        calibration_toml=(
            Path(setup["calibration_toml"]).expanduser().resolve()
            if setup.get("calibration_toml")
            else None
        ),
        calibration_dir=(
            Path(setup["calibration_dir"]).expanduser().resolve()
            if setup.get("calibration_dir")
            else None
        ),
        skeleton=(
            skeleton_from_setup(raw["skeleton"], base_dir=Path.cwd())
            if "skeleton" in raw
            else None
        ),
        skeleton_cameras=(
            [str(c) for c in raw["skeleton"]["cameras"]]
            if "cameras" in raw.get("skeleton", {})
            else None
        ),
    )

    if slp_files:
        run_slp_setup(config, dry_run=dry_run, do_import=setup.get("import_labels", True))
        return

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
    slp = parser.add_argument_group(
        "setup from SLP files",
        "Build a project from existing .slp files, referencing their videos in place.",
    )
    slp.add_argument(
        "--from-slp",
        nargs="+",
        metavar="PATH",
        help=".slp files, or folders containing them",
    )
    slp.add_argument("--output", help="Output project folder (with --from-slp)")
    slp.add_argument("--cameras", help="Comma-separated camera names (with --from-slp)")
    slp.add_argument("--name", help="Project name (default: output folder name)")
    slp.add_argument("--experiments", help="Comma-separated experiment keywords")
    slp.add_argument(
        "--match-strategy",
        default="keyword",
        choices=["keyword", "regex", "number", "datestamp", "mtime"],
        help="How to group videos into trials (default: keyword)",
    )
    slp.add_argument("--trial-keyword", default="Trial", help="For --match-strategy keyword")
    slp.add_argument("--trial-regex", default="", help="For --match-strategy regex")
    slp.add_argument(
        "--calib-keywords",
        default="calib",
        help="Comma-separated calibration take keywords (default: calib)",
    )
    slp.add_argument(
        "--video-mode",
        default="reference",
        choices=["move", "copy", "reference"],
        help="What to do with the videos (default: reference — leave them in place)",
    )
    slp.add_argument(
        "--video-search-dir",
        action="append",
        metavar="DIR",
        help="Extra folder to search when a recorded video path has moved (repeatable)",
    )
    slp.add_argument(
        "--allow-missing-videos",
        action="store_true",
        default=False,
        help="Set up the project from the videos that were found, leaving the "
        "cameras whose videos are missing blank in trials.csv",
    )
    slp.add_argument(
        "--calibration",
        help="aniposelib calibration .toml (imported into the registry), or a folder to search",
    )
    slp.add_argument(
        "--no-import",
        action="store_true",
        default=False,
        help="Do not import the SLP labels into points.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    if args.from_slp:
        run_slp_setup(
            slp_config_from_args(args),
            dry_run=args.dry_run,
            do_import=not args.no_import,
        )
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        run_from_config(config_path, copy_files=args.copy, dry_run=args.dry_run)
    else:
        print("=" * 60)
        print("label3d — Project Setup")
        print("=" * 60)
        source_kind = prompt_choice("Start from", ["videos", "slp"])
        if source_kind == "slp":
            run_interactive_slp(dry_run=args.dry_run, do_import=not args.no_import)
        else:
            run_interactive(copy_files=args.copy, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

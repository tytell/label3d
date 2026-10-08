"""Project setup utility: organize a flat folder of videos/slp files into a label3d project."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import sleap_io as sio
import toml
from aniposelib.cameras import CameraGroup

from label3d.calibration_registry import REFINED_DIR, camera_dir, save_calibration
from label3d.project import Project, check_trial_videos, write_trials_csv
from label3d.skeleton import Skeleton
from label3d.slp_sync import SlpMapping

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".cine"}
SLP_EXTENSION = ".slp"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class SetupConfig:
    """Configuration for project setup."""

    source_dir: Path
    output_dir: Path
    camera_names: list[str]
    calib_keyword: str
    match_strategy: str  # "number" | "datestamp" | "mtime"
    mtime_window_sec: float = 60.0
    datestamp_window_sec: float = 0.0  # tolerance for datestamp matching (0 = exact)
    charuco: dict | None = None
    project_name: str = ""
    copy_files: bool = False
    trial_keyword: str = "Trial"  # used by "keyword" match strategy
    trial_regex: str = ""         # used by "regex" match strategy
    experiment_keywords: list[str] = field(default_factory=list)  # empty → single "default" experiment
    # Calibration take identification (Phase 9). Each keyword names one take,
    # e.g. ["PreCal", "PostCal"]; calib_regex is an alternative whose "take"
    # group, else group 1, else whole match supplies the name.
    calib_keywords: list[str] = field(default_factory=list)
    calib_regex: str = ""
    # Strategy used to separate takes when keywords/regex cannot; "" → match_strategy
    calib_match_strategy: str = ""
    # --- SLP-first setup (see docs/project_setup_from_sleap.md) ---
    # Source .slp files; when set, classify_slp_files() replaces classify_files()
    # and source_dir is unused.
    slp_paths: list[Path] = field(default_factory=list)
    # "move" | "copy" | "reference". "reference" leaves videos where they are and
    # records their paths in trials.csv. "" → derived from the legacy copy_files.
    video_mode: str = ""
    # Extra directories searched when a .slp records a video path that has moved
    video_search_dirs: list[Path] = field(default_factory=list)
    # Pre-existing aniposelib calibration, or a directory to search for one
    calibration_toml: Path | None = None
    calibration_dir: Path | None = None
    # Set up the project from the videos that could be located, leaving the
    # cameras whose videos are missing blank in trials.csv
    allow_missing_videos: bool = False
    # Skeleton given in the setup file (see skeleton_from_setup); must match
    # the skeleton of any .slp files
    skeleton: Skeleton | None = None

    def __post_init__(self):
        # Camera folders share calibrations/ with the refinements folder
        if REFINED_DIR in self.camera_names:
            raise ValueError(f"'{REFINED_DIR}' can't be used as a camera name")

    def effective_video_mode(self) -> str:
        """Video handling mode, falling back to the legacy copy_files flag."""
        if self.video_mode:
            return self.video_mode
        return "copy" if self.copy_files else "move"

    def effective_calib_keywords(self) -> list[str]:
        """Take keywords, falling back to the legacy single calib_keyword."""
        if self.calib_keywords:
            return list(self.calib_keywords)
        return [self.calib_keyword] if self.calib_keyword else []


@dataclass
class FileAssignment:
    """Result of classifying and matching files."""

    # trial_name -> {camera_name: filepath}
    trials: dict[str, dict[str, Path]] = field(default_factory=dict)
    # camera_name -> [calib files]
    calibration: dict[str, list[Path]] = field(default_factory=dict)
    # trial_name -> {camera_name: slp_path}
    slp_files: dict[str, dict[str, Path]] = field(default_factory=dict)
    # trial_name -> experiment_name (populated when experiment_keywords are set)
    trial_experiment: dict[str, str] = field(default_factory=dict)
    # experiment_name -> {camera_name: [calib files]} — always populated by classify_files
    exp_calibration: dict[str, dict[str, list[Path]]] = field(default_factory=dict)
    # experiment_name -> take_name -> {camera_name: calib video} (Phase 9)
    calibration_takes: dict[str, dict[str, dict[str, Path]]] = field(default_factory=dict)
    # files that couldn't be assigned
    unmatched: list[Path] = field(default_factory=list)
    # warnings generated during classification
    warnings: list[str] = field(default_factory=list)
    # skeleton extracted from .slp files (nodes/edges/symmetries); None if no SLP files present
    skeleton: dict | None = None
    # split detection: True when multiple calibration groups (prefixes) are found
    needs_split: bool = False
    # prefix -> {camera_name: [calib files]}
    calibration_groups: dict[str, dict[str, list[Path]]] = field(default_factory=dict)
    # trial_name -> calibration-group prefix
    trial_prefix: dict[str, str] = field(default_factory=dict)
    # SLP-first setup: video references that could not be resolved on disk,
    # as (recorded path, source .slp name)
    missing_videos: list[tuple[Path, str]] = field(default_factory=list)
    # experiment_name -> common parent of that experiment's videos, when one
    # exists; written to the [video_roots] section of project.toml
    video_roots: dict[str, Path] = field(default_factory=dict)
    # experiment_name -> pre-existing calibration.toml found next to the inputs
    calibration_toml: dict[str, Path] = field(default_factory=dict)
    # (recorded prefix, replacement root) -> number of videos remapped that way
    video_remaps: dict[tuple[str, str], int] = field(default_factory=dict)


def classify_files(config: SetupConfig) -> FileAssignment:
    """Scan source_dir, classify files by camera and type, match trials."""
    assignment = FileAssignment()

    source = config.source_dir
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    # Collect all relevant files
    all_files = sorted(source.iterdir())
    video_files: list[Path] = []
    slp_paths: list[Path] = []
    image_files: list[Path] = []

    for f in all_files:
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            video_files.append(f)
        elif ext == SLP_EXTENSION:
            slp_paths.append(f)
        elif ext in IMAGE_EXTENSIONS:
            image_files.append(f)

    # Classify each file by camera
    calib_kws = config.effective_calib_keywords()

    # Separate calibration vs data files
    calib_videos: list[tuple[str, Path]] = []  # (camera, path)
    calib_images: list[tuple[str, Path]] = []
    data_videos: list[tuple[str, Path]] = []

    for f in video_files:
        camera = _identify_camera(f.name, config.camera_names)
        if camera is None:
            assignment.unmatched.append(f)
            assignment.warnings.append(
                f"Video '{f.name}' does not match any camera name"
            )
            continue
        if _is_calibration(f.name, calib_kws, config.calib_regex):
            calib_videos.append((camera, f))
        else:
            data_videos.append((camera, f))

    for f in image_files:
        camera = _identify_camera(f.name, config.camera_names)
        if camera is None:
            assignment.unmatched.append(f)
            continue
        if _is_calibration(f.name, calib_kws, config.calib_regex):
            calib_images.append((camera, f))
        else:
            assignment.unmatched.append(f)

    if calib_images:
        names = ", ".join(sorted(f.name for _, f in calib_images)[:5])
        raise ValueError(
            f"Image-based calibration is deprecated: found {len(calib_images)} "
            f"calibration image(s) in {source} ({names}"
            f"{', ...' if len(calib_images) > 5 else ''}). "
            "Calibration must come from videos, which are matched into takes. "
            "Remove the images or record calibration video instead."
        )

    # Build calibration dict
    for camera, f in calib_videos:
        assignment.calibration.setdefault(camera, []).append(f)
    for camera, f in calib_images:
        assignment.calibration.setdefault(camera, []).append(f)

    # Match data videos into trials
    assignment.trials = _run_match_strategy(
        config.match_strategy, data_videos, config
    )

    # Match SLP files to trials via their embedded video references
    if slp_paths:
        # Build lookup: video_basename -> (trial_name, camera_name)
        video_to_trial_cam: dict[str, tuple[str, str]] = {}
        for trial_name, cam_files in assignment.trials.items():
            for cam, fp in cam_files.items():
                video_to_trial_cam[fp.name] = (trial_name, cam)

        covered: set[tuple[str, str]] = set()
        skeleton_dicts: list[dict] = []

        logger.debug("Trial video lookup: %s", sorted(video_to_trial_cam))

        for slp_path in slp_paths:
            try:
                video_basenames, skeleton_dict = _read_slp_info(slp_path)
            except Exception as e:
                assignment.warnings.append(
                    f"Could not read SLP file '{slp_path.name}': {e}"
                )
                continue

            logger.debug(
                "SLP '%s' contains videos: %s", slp_path.name, sorted(video_basenames)
            )

            skeleton_dicts.append(skeleton_dict)

            if not video_basenames:
                assignment.warnings.append(
                    f"SLP file '{slp_path.name}' has no labeled frames"
                )
                continue

            for vb in video_basenames:
                if vb in video_to_trial_cam:
                    trial_name, cam = video_to_trial_cam[vb]
                    assignment.slp_files.setdefault(trial_name, {})[cam] = slp_path
                    covered.add((trial_name, cam))
                else:
                    assignment.warnings.append(
                        f"SLP file '{slp_path.name}' references video '{vb}'"
                        " not found in source folder"
                    )

        # Warn about videos not covered by any SLP file
        for trial_name, cam_files in assignment.trials.items():
            for cam, fp in cam_files.items():
                if (trial_name, cam) not in covered:
                    assignment.warnings.append(
                        f"No SLP file covers video '{fp.name}'"
                        f" (trial '{trial_name}', camera '{cam}')"
                    )

        # Check skeleton consistency; error on mismatch
        if skeleton_dicts:
            first = set(skeleton_dicts[0]['nodes'])
            for sd in skeleton_dicts[1:]:
                if set(sd['nodes']) != first:
                    raise ValueError(
                        "SLP files have inconsistent skeletons — "
                        "all .slp files must share the same skeleton."
                    )
            assignment.skeleton = skeleton_dicts[0]

    # Warn about incomplete trials
    for trial_name, cam_files in assignment.trials.items():
        missing = set(config.camera_names) - set(cam_files.keys())
        if missing:
            assignment.warnings.append(
                f"Trial '{trial_name}' is missing cameras: {', '.join(sorted(missing))}"
            )

    # Assign trials to experiments using experiment_keywords
    if config.experiment_keywords:
        for trial_name, cam_files in assignment.trials.items():
            if cam_files:
                first_file = next(iter(cam_files.values()))
                assignment.trial_experiment[trial_name] = _identify_experiment(
                    first_file.name, config.experiment_keywords
                )
            else:
                assignment.trial_experiment[trial_name] = "default"

    # Build per-experiment calibration mapping
    all_calib = calib_videos + calib_images
    if config.experiment_keywords:
        for camera, filepath in all_calib:
            exp = _identify_experiment(filepath.name, config.experiment_keywords)
            assignment.exp_calibration.setdefault(exp, {}).setdefault(camera, []).append(filepath)
    else:
        for camera, filepath in all_calib:
            assignment.exp_calibration.setdefault("default", {}).setdefault(camera, []).append(filepath)

    # Group each experiment's calibration videos into takes (Phase 9).
    # Multiple takes per experiment are normal — typically one before and one
    # after the trials — so this no longer triggers a project split.
    calib_by_exp: dict[str, list[tuple[str, Path]]] = {}
    for camera, filepath in calib_videos:
        exp = (
            _identify_experiment(filepath.name, config.experiment_keywords)
            if config.experiment_keywords
            else "default"
        )
        calib_by_exp.setdefault(exp, []).append((camera, filepath))

    unresolved_all: list[Path] = []
    for exp, videos in calib_by_exp.items():
        takes, unresolved = _match_calibration_takes(config, videos)
        unresolved_all.extend(unresolved)
        if takes:
            assignment.calibration_takes[exp] = takes

    if unresolved_all:
        names = ", ".join(sorted(fp.name for fp in unresolved_all))
        raise ValueError(
            "Could not determine the calibration take for: "
            f"{names}. Give each take its own keyword via calib_keywords "
            "(e.g. [\"PreCal\", \"PostCal\"]), or set calib_regex to a pattern "
            "that matches each take's name."
        )

    # Warn about takes that are missing cameras
    for exp, takes in assignment.calibration_takes.items():
        for take, cam_files in takes.items():
            missing = [c for c in config.camera_names if c not in cam_files]
            if missing:
                assignment.warnings.append(
                    f"Calibration take '{take}' in experiment '{exp}' is missing "
                    f"cameras: {', '.join(sorted(missing))}"
                )

    # Warn for experiments that are missing calibration for one or more cameras
    experiments_with_trials = (
        set(assignment.trial_experiment.values()) if assignment.trial_experiment else {"default"}
    )
    for exp in experiments_with_trials:
        calib_for_exp = assignment.exp_calibration.get(exp, {})
        missing = [cam for cam in config.camera_names if not calib_for_exp.get(cam)]
        if missing:
            assignment.warnings.append(
                f"Experiment '{exp}' has no calibration video for: {', '.join(missing)}"
            )

    _apply_config_skeleton(config, assignment)
    return assignment


def classify_slp_files(config: SetupConfig) -> FileAssignment:
    """Build an assignment from .slp files, referencing their videos in place.

    The SLP-first counterpart to classify_files(): videos come from the paths
    recorded inside the .slp files rather than from a folder scan, and the
    skeleton comes from the files themselves. Trial grouping, experiment
    assignment and calibration-take matching are shared with the folder path.
    """
    assignment = FileAssignment()

    slp_paths = _expand_slp_paths(config.slp_paths)
    if not slp_paths:
        raise FileNotFoundError("No .slp files found in the given paths")

    search_dirs = [Path(d) for d in config.video_search_dirs]
    data_videos: list[tuple[str, Path]] = []
    video_sources: dict[Path, list[Path]] = {}  # resolved video -> .slp files
    skeleton_dicts: list[dict] = []
    seen: set[Path] = set()

    for slp_path in slp_paths:
        try:
            recorded, skeleton_dict, n_frames = _read_slp_videos(slp_path)
        except Exception as e:
            assignment.warnings.append(
                f"Could not read SLP file '{slp_path.name}': {e}"
            )
            continue

        skeleton_dicts.append(skeleton_dict)
        if n_frames == 0:
            assignment.warnings.append(
                f"SLP file '{slp_path.name}' has no labeled frames"
            )

        for ref in recorded:
            outcome = _resolve_video_reference(ref, slp_path.parent, search_dirs)
            if outcome.path is None:
                assignment.missing_videos.append((ref, slp_path.name))
                if outcome.rule == "ambiguous":
                    where = ", ".join(str(c) for c in outcome.candidates)
                    assignment.warnings.append(
                        f"Video reference '{ref}' matches more than one search "
                        f"directory ({where}); left unresolved — narrow "
                        "video_search_dirs to pick one."
                    )
                continue
            resolved = outcome.path
            if outcome.new_root is not None:
                key = (outcome.old_root, str(outcome.new_root))
                assignment.video_remaps[key] = assignment.video_remaps.get(key, 0) + 1

            video_sources.setdefault(resolved, []).append(slp_path)
            if resolved in seen:
                continue
            seen.add(resolved)

            camera = _identify_camera_from_path(resolved, config.camera_names)
            if camera is None:
                assignment.unmatched.append(resolved)
                assignment.warnings.append(
                    f"Video '{resolved.name}' does not match any camera name"
                )
                continue
            data_videos.append((camera, resolved))

    if skeleton_dicts:
        first = set(skeleton_dicts[0]["nodes"])
        for sd in skeleton_dicts[1:]:
            if set(sd["nodes"]) != first:
                raise ValueError(
                    "SLP files have inconsistent skeletons — "
                    "all .slp files must share the same skeleton."
                )
        assignment.skeleton = skeleton_dicts[0]

    assignment.trials = _run_match_strategy(
        config.match_strategy, data_videos, config
    )

    for trial_name, cam_files in assignment.trials.items():
        first_file = next(iter(cam_files.values()), None)
        assignment.trial_experiment[trial_name] = (
            _identify_experiment(first_file.name, config.experiment_keywords)
            if config.experiment_keywords and first_file is not None
            else "default"
        )
        for cam, video in cam_files.items():
            for slp_path in video_sources.get(video, []):
                assignment.slp_files.setdefault(trial_name, {})[cam] = slp_path

        missing = set(config.camera_names) - set(cam_files)
        if missing:
            assignment.warnings.append(
                f"Trial '{trial_name}' is missing cameras: {', '.join(sorted(missing))}"
            )

    assignment.video_roots = _compute_video_roots(
        assignment.trials, assignment.trial_experiment
    )

    for ref, slp_name in assignment.missing_videos:
        assignment.warnings.append(
            f"SLP file '{slp_name}' references video '{ref}' which was not found"
        )

    _assign_calibration(config, assignment, slp_paths)
    _apply_config_skeleton(config, assignment)
    return assignment


def skeleton_from_setup(table: dict, base_dir: Path | None = None) -> Skeleton:
    """The skeleton named by a setup file's [skeleton] table.

    The table either names a skeleton.json (``file``, relative to base_dir
    when not absolute) or gives ``nodes``, ``edges``, ``symmetries`` and
    ``node_offsets_cm`` inline, not both.
    """
    if "file" in table:
        if set(table) - {"file"}:
            raise ValueError("[skeleton] takes either 'file' or inline nodes, not both")
        path = Path(table["file"]).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Skeleton file not found: {path}")
        return Skeleton.from_json(path)
    return Skeleton.from_dict(table)


def _apply_config_skeleton(config: SetupConfig, assignment: FileAssignment) -> None:
    """Use the setup file's skeleton, after checking it against the .slp one."""
    if config.skeleton is None:
        return
    if assignment.skeleton is not None:
        from_slp = Skeleton.from_dict(assignment.skeleton)
        if not config.skeleton.same_topology(from_slp):
            raise ValueError(
                "The skeleton in the setup file does not match the .slp files' skeleton "
                f"(setup: {', '.join(config.skeleton.nodes)}; "
                f".slp: {', '.join(from_slp.nodes)})"
            )
    assignment.skeleton = config.skeleton.to_dict()


def _expand_slp_paths(paths: list[Path]) -> list[Path]:
    """Expand directories into the .slp files they contain; keep files as given."""
    expanded: list[Path] = []
    for path in paths:
        path = Path(path).expanduser()
        if path.is_dir():
            expanded.extend(sorted(path.glob(f"*{SLP_EXTENSION}")))
        elif path.exists():
            expanded.append(path)
        else:
            raise FileNotFoundError(f"SLP path not found: {path}")
    return expanded


def _find_calibration_toml(dirs: list[Path], camera_names: list[str]) -> Path | None:
    """Find an aniposelib calibration.toml among dirs whose cameras match.

    Validated by parsing rather than by filename, so an unrelated .toml sitting
    beside the .slp files is not mistaken for calibration.
    """
    wanted = {c.lower() for c in camera_names}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.toml")):
            try:
                data = toml.load(candidate)
            except Exception:
                continue
            names = {
                str(section.get("name", key)).lower()
                for key, section in data.items()
                if isinstance(section, dict) and key not in ("metadata", "alignment")
            }
            if names and wanted <= names:
                return candidate
    return None


def _find_calibration_videos(
    dirs: list[Path], config: SetupConfig
) -> list[tuple[str, Path]]:
    """Calibration videos found in dirs, as (camera, path) pairs."""
    keywords = config.effective_calib_keywords()
    found: list[tuple[str, Path]] = []
    for directory in dirs:
        if not directory.is_dir():
            continue
        for f in sorted(directory.iterdir()):
            if not f.is_file() or f.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if not _is_calibration(f.name, keywords, config.calib_regex):
                continue
            camera = _identify_camera_from_path(f, config.camera_names)
            if camera is not None:
                found.append((camera, f.resolve()))
    return found


def _assign_calibration(
    config: SetupConfig, assignment: FileAssignment, slp_paths: list[Path]
) -> None:
    """Attach calibration to each experiment: an existing toml, else videos.

    Searches the directory holding the .slp files, or calibration_dir when one
    was given. An experiment with neither is left uncalibrated and warned about;
    the user can calibrate in the app later.
    """
    experiments = sorted(set(assignment.trial_experiment.values())) or ["default"]

    if config.calibration_toml is not None:
        for exp in experiments:
            assignment.calibration_toml[exp] = Path(config.calibration_toml)
        return

    dirs = (
        [Path(config.calibration_dir)]
        if config.calibration_dir is not None
        else sorted({p.parent for p in slp_paths})
    )

    found_toml = _find_calibration_toml(dirs, config.camera_names)
    if found_toml is not None:
        for exp in experiments:
            assignment.calibration_toml[exp] = found_toml
        return

    calib_videos = _find_calibration_videos(dirs, config)
    if not calib_videos:
        for exp in experiments:
            assignment.warnings.append(
                f"Experiment '{exp}' has no calibration — "
                "supply a calibration.toml or calibration videos"
            )
        return

    by_exp: dict[str, list[tuple[str, Path]]] = {}
    for camera, filepath in calib_videos:
        exp = (
            _identify_experiment(filepath.name, config.experiment_keywords)
            if config.experiment_keywords
            else "default"
        )
        by_exp.setdefault(exp, []).append((camera, filepath))

    for exp, videos in by_exp.items():
        takes, unresolved = _match_calibration_takes(config, videos)
        if takes:
            assignment.calibration_takes[exp] = takes
        for camera, filepath in videos:
            assignment.exp_calibration.setdefault(exp, {}).setdefault(
                camera, []
            ).append(filepath)
        if unresolved:
            names = ", ".join(sorted(fp.name for fp in unresolved))
            assignment.warnings.append(
                f"Could not determine the calibration take for: {names}"
            )


def preview_assignment(config: SetupConfig, assignment: FileAssignment) -> str:
    """Return a human-readable summary of the file assignment."""
    lines = []

    lines.append(f"Project: {config.project_name or config.output_dir.name}")
    lines.append(f"Output:  {config.output_dir}")
    lines.append(f"Experiments: {', '.join(config.experiment_keywords) if config.experiment_keywords else '(default)'}")
    lines.append(f"Cameras: {', '.join(config.camera_names)}")
    if assignment.skeleton:
        source = "setup file" if config.skeleton is not None else ".slp files"
        lines.append(f"Skeleton: {', '.join(assignment.skeleton['nodes'])} (from {source})")
    else:
        lines.append("Skeleton: none (add one later with File > Load skeleton...)")
    strategy_str = config.match_strategy
    if config.match_strategy == "keyword":
        strategy_str += f" (keyword={config.trial_keyword!r})"
    elif config.match_strategy == "regex":
        strategy_str += f" (pattern={config.trial_regex!r})"
    lines.append(f"Strategy: {strategy_str}")
    lines.append("")

    # Trials — grouped by experiment when experiment_keywords are set
    if assignment.trials:
        if assignment.trial_experiment:
            exp_groups: dict[str, list[str]] = {}
            for tn, exp in assignment.trial_experiment.items():
                exp_groups.setdefault(exp, []).append(tn)
            lines.append(
                f"Trials ({len(assignment.trials)}) across {len(exp_groups)} experiment(s):"
            )
            for exp_name in sorted(exp_groups):
                lines.append(f"  [{exp_name}]")
                for trial_name in sorted(exp_groups[exp_name]):
                    cam_files = assignment.trials[trial_name]
                    lines.append(f"    {trial_name}:")
                    for cam in config.camera_names:
                        if cam in cam_files:
                            lines.append(f"      {cam}: {cam_files[cam].name}")
                        else:
                            lines.append(f"      {cam}: (missing)")
        else:
            lines.append(f"Trials ({len(assignment.trials)}):")
            for trial_name in sorted(assignment.trials):
                cam_files = assignment.trials[trial_name]
                lines.append(f"  {trial_name}:")
                for cam in config.camera_names:
                    if cam in cam_files:
                        lines.append(f"    {cam}: {cam_files[cam].name}")
                    else:
                        lines.append(f"    {cam}: (missing)")
    else:
        lines.append("Trials: none found")
    lines.append("")

    # Calibration — per experiment, broken out by take
    if assignment.calibration_takes:
        n_takes = sum(len(t) for t in assignment.calibration_takes.values())
        lines.append(f"Calibration ({n_takes} takes):")
        for exp_name in sorted(assignment.calibration_takes):
            takes = assignment.calibration_takes[exp_name]
            lines.append(f"  [{exp_name}]")
            for take in sorted(takes):
                cam_files = takes[take]
                parts = []
                for cam in config.camera_names:
                    present = cam in cam_files
                    parts.append(f"{cam}: {'ok' if present else '-- !'}")
                lines.append(f"    {take} — " + "  ".join(parts))
    elif assignment.exp_calibration:
        total = sum(
            len(files)
            for cam_dict in assignment.exp_calibration.values()
            for files in cam_dict.values()
        )
        lines.append(f"Calibration ({total} files):")
        for exp_name in sorted(assignment.exp_calibration):
            cam_dict = assignment.exp_calibration[exp_name]
            prefix = f"  [{exp_name}]  " if assignment.trial_experiment else "  "
            parts = []
            for cam in config.camera_names:
                n = len(cam_dict.get(cam, []))
                flag = "" if n else " !"
                parts.append(f"{cam}: {n}{flag}")
            lines.append(prefix + "  ".join(parts))
    else:
        lines.append("Calibration: none found")
    lines.append("")

    # Multi-group split
    if assignment.needs_split:
        n = len(assignment.calibration_groups)
        lines.append(f"Multiple calibration groups detected ({n}):")
        for prefix in sorted(assignment.calibration_groups):
            cam_dict = assignment.calibration_groups[prefix]
            parts = []
            for cam in config.camera_names:
                n_cam = len(cam_dict.get(cam, []))
                flag = "" if n_cam else " !"
                parts.append(f"{cam}: {n_cam}{flag}")
            group_trials = sorted(
                tn for tn, pre in assignment.trial_prefix.items() if pre == prefix
            )
            lines.append(f"  [{prefix}]  calibration — " + "  ".join(parts))
            if group_trials:
                lines.append(f"    trials: {', '.join(group_trials)}")
            else:
                lines.append("    trials: (none)")
        unassigned = sorted(
            tn for tn, pre in assignment.trial_prefix.items()
            if pre not in assignment.calibration_groups
        )
        if unassigned:
            lines.append(f"  trials with no matching group: {', '.join(unassigned)}")
        lines.append("")

    # SLP files
    if assignment.slp_files:
        lines.append(f"SLP files ({len(assignment.slp_files)} trials):")
        for trial_name in sorted(assignment.slp_files):
            cam_files = assignment.slp_files[trial_name]
            for cam, f in sorted(cam_files.items()):
                lines.append(f"  {trial_name}/{cam}: {f.name}")
    lines.append("")

    # Referenced videos (SLP-first setup) — where they live and whether they exist
    if config.effective_video_mode() == "reference":
        lines.append("Videos (referenced in place):")
        for exp_name in sorted(set(assignment.trial_experiment.values()) or {"default"}):
            root = assignment.video_roots.get(exp_name)
            lines.append(
                f"  [{exp_name}]  root: {root}" if root else f"  [{exp_name}]  (absolute paths)"
            )
            exp_trials = [
                t
                for t in sorted(assignment.trials)
                if assignment.trial_experiment.get(t, "default") == exp_name
            ]
            for trial_name in exp_trials:
                cam_files = assignment.trials[trial_name]
                parts = []
                for cam in config.camera_names:
                    path = cam_files.get(cam)
                    if path is None:
                        parts.append(f"{cam}: --")
                    else:
                        parts.append(f"{cam}: {'ok' if path.exists() else 'MISSING'}")
                lines.append(f"    {trial_name}  " + "  ".join(parts))
        if assignment.calibration_toml:
            for exp_name, path in sorted(assignment.calibration_toml.items()):
                lines.append(
                    f"  [{exp_name}]  calibration found at {path} "
                    "(imported into the calibration registry)"
                )
        lines.append("")

    if assignment.video_remaps:
        lines.append("Video paths remapped:")
        for (old_root, new_root), count in sorted(assignment.video_remaps.items()):
            source = old_root or "(basename only)"
            lines.append(f"  {source}  ->  {new_root}   ({count} video(s))")
        lines.append("")

    if assignment.missing_videos:
        lines.append(f"Unresolved video references ({len(assignment.missing_videos)}):")
        for ref, slp_name in assignment.missing_videos:
            lines.append(f"  {ref}  (from {slp_name})")
        lines.append("")

    # Unmatched
    if assignment.unmatched:
        lines.append(f"Unmatched files ({len(assignment.unmatched)}):")
        for f in assignment.unmatched:
            lines.append(f"  {f.name}")
    lines.append("")

    # Warnings
    if assignment.warnings:
        lines.append("Warnings:")
        for w in assignment.warnings:
            lines.append(f"  ! {w}")

    return "\n".join(lines)


def _blank_row(
    experiment: str, name: str, is_calibration: bool, cameras: list[str]
) -> dict:
    """An empty trials.csv row; cells are filled in by _fill_cell."""
    row: dict[str, str | int | bool] = {
        "experiment": experiment,
        "trial": name,
        "is_calibration": is_calibration,
    }
    for cam in cameras:
        row[cam] = ""
        row[f"{cam}_start"] = ""
        row[f"{cam}_end"] = ""
        row[f"{cam}_mtime"] = ""
    return row


def _video_cell(src: Path, video_root: Path | None) -> str:
    """The trials.csv cell for a referenced video.

    Relative to the experiment's video_root when it lies under it — so the
    project survives the videos being remounted elsewhere — and absolute
    otherwise. Project._resolve_video reads both forms.
    """
    if video_root is not None:
        try:
            return src.relative_to(video_root).as_posix()
        except ValueError:
            pass
    return src.as_posix()


def _slp_mapping_config(assignment: FileAssignment) -> dict | None:
    """The [slp] section of project.toml, mapping cells to files in slp/.

    Built from the trial/camera assignment so the project opens with its SLP
    files already mapped, ready for File > Sync with Sleap.
    """
    if not assignment.slp_files:
        return None
    cell_map: dict[tuple[str, str, str], Path] = {}
    for trial_name in sorted(assignment.slp_files):
        exp = assignment.trial_experiment.get(trial_name, "default")
        for cam in sorted(assignment.slp_files[trial_name]):
            slp_path = assignment.slp_files[trial_name][cam]
            cell_map[(exp, trial_name, cam)] = Path("slp") / slp_path.name
    return SlpMapping.from_cell_map(cell_map).to_config()


def _fill_cell(row: dict, camera: str, filename: str) -> None:
    """Record one camera's video in a trials.csv row."""
    row[camera] = filename
    row[f"{camera}_start"] = 0
    row[f"{camera}_end"] = ""


def execute_setup(config: SetupConfig, assignment: FileAssignment) -> Path:
    """Create project folder structure, move or copy files, write trials.csv and project.toml.

    Output layout:
      project/
        project.toml
        skeleton.json            <- from the setup file or the SLP files
        trials.csv               <- experiment, trial, is_calibration columns
        slp/
        experiments/
          default/
            videos/
            calibrations/{cam}/

    Returns the output directory path.
    """
    out = config.output_dir
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")

    if assignment.missing_videos and not config.allow_missing_videos:
        names = ", ".join(str(ref) for ref, _ in assignment.missing_videos[:5])
        raise FileNotFoundError(
            f"{len(assignment.missing_videos)} video reference(s) could not be "
            f"resolved ({names}). Supply --video-search-dir, or pass "
            "--allow-missing-videos to set up the project from the videos that "
            "were found."
        )
    if assignment.missing_videos:
        logger.warning(
            "Proceeding without %d unresolved video reference(s); the affected "
            "cameras are left blank in trials.csv",
            len(assignment.missing_videos),
        )

    mode = config.effective_video_mode()
    reference = mode == "reference"
    transfer = shutil.copy2 if mode == "copy" else shutil.move

    # Group trials by experiment
    if assignment.trial_experiment:
        exp_trials: dict[str, list[str]] = {}
        for trial_name, exp in assignment.trial_experiment.items():
            exp_trials.setdefault(exp, []).append(trial_name)
    else:
        exp_trials = {"default": sorted(assignment.trials)}

    # Create directory structure and move/copy files per experiment.
    # Calibration takes get rows too, flagged with is_calibration.
    all_experiments = sorted(set(exp_trials) | set(assignment.calibration_takes))
    trials_rows: list[dict] = []
    active_calibrations: dict[str, str] = {}
    for exp_name in all_experiments:
        exp_dir = out / "experiments" / exp_name
        video_root = assignment.video_roots.get(exp_name)
        videos_dir = exp_dir / "videos"
        if reference:
            exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            videos_dir.mkdir(parents=True, exist_ok=True)

        # Calibration videos always come into the project, even in reference
        # mode: they are small, they come from calibration_dir rather than
        # from the SLP references, and the project then owns its calibration
        # inputs.  Referenced trial videos are still left where they are.
        calib_transfer = shutil.copy2 if reference else transfer
        for cam, files in assignment.exp_calibration.get(exp_name, {}).items():
            cam_dir = camera_dir(exp_dir, cam)
            cam_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                calib_transfer(str(f), str(cam_dir / f.name))

        # An existing aniposelib calibration becomes an "imported" registry
        # entry, and the active one.
        src_toml = assignment.calibration_toml.get(exp_name)
        if src_toml is not None:
            exp_dir.mkdir(parents=True, exist_ok=True)
            entry = save_calibration(
                exp_dir, CameraGroup.load(str(src_toml)), "imported", "imported",
                source_path=str(src_toml),
            )
            active_calibrations[exp_name] = entry.id

        for trial_name in sorted(exp_trials.get(exp_name, [])):
            cam_files = assignment.trials[trial_name]
            row = _blank_row(exp_name, trial_name, False, config.camera_names)
            for cam in config.camera_names:
                if cam in cam_files:
                    src = cam_files[cam]
                    if reference:
                        _fill_cell(row, cam, _video_cell(src, video_root))
                    else:
                        transfer(str(src), str(videos_dir / src.name))
                        _fill_cell(row, cam, src.name)
            trials_rows.append(row)

        # Calibration files have already been copied above, so only record
        # them — always by bare filename, since they now live in the project's
        # calibrations/{cam}/ regardless of the video mode.
        for take in sorted(assignment.calibration_takes.get(exp_name, {})):
            cam_files = assignment.calibration_takes[exp_name][take]
            row = _blank_row(exp_name, take, True, config.camera_names)
            for cam, src in cam_files.items():
                _fill_cell(row, cam, src.name)
            trials_rows.append(row)

    # Write trials.csv at project root with experiment column
    if trials_rows:
        columns = ["experiment", "trial", "is_calibration"]
        for cam in config.camera_names:
            columns.extend([cam, f"{cam}_start", f"{cam}_end", f"{cam}_mtime"])
        df = pd.DataFrame(trials_rows, columns=columns)
        # Record each video's last frame and modification time now, so the
        # first open doesn't have to measure them.
        roots = {
            exp: Path(root) for exp, root in (assignment.video_roots if reference else {}).items()
        }
        check_trial_videos(df, out, roots, config.camera_names)
        write_trials_csv(out / "trials.csv", df, backup=False)

    # Move/copy SLP files (deduplicate: one slp_path may cover multiple trial+camera pairs).
    # In reference mode the .slp files are copied, not moved: the source folder
    # is the user's own, and only videos are left in place.
    if assignment.slp_files:
        slp_dir = out / "slp"
        slp_dir.mkdir(parents=True, exist_ok=True)
        slp_transfer = shutil.copy2 if reference else transfer
        seen: set[Path] = set()
        for trial_name, cam_files in assignment.slp_files.items():
            for cam, f in cam_files.items():
                if f not in seen:
                    seen.add(f)
                    slp_transfer(str(f), str(slp_dir / f.name))

    # Write project.toml
    write_project_toml(
        out,
        config.project_name or out.name,
        config.charuco,
        video_roots=assignment.video_roots if reference else None,
        slp_mapping=_slp_mapping_config(assignment),
        active_calibrations=active_calibrations,
    )

    # Write skeleton.json from the setup file or the SLP files
    if assignment.skeleton:
        with open(out / "skeleton.json", "w") as fh:
            json.dump(assignment.skeleton, fh, indent=2)

    return out


def import_labels(project_dir: Path) -> dict[str, int]:
    """Import the labels of a freshly created project's .slp files into points.csv.

    Runs the same read/merge path as File > Sync with Sleap, so imported states,
    instance ordering and multi-instance handling match what the app produces.
    overwrite=True is safe here because points.csv starts empty.

    Returns experiment name -> number of imported frames.
    """
    project = Project.load(project_dir)
    if not project.slp_mapping.files:
        return {}

    counts: dict[str, int] = {}
    for exp_name in sorted(project.experiments):
        project.set_experiment(exp_name)
        try:
            results = project.read_slp_for_sync(sync=False)
        except Exception as e:
            logger.warning("Could not read SLP files for '%s': %s", exp_name, e)
            continue
        if not results:
            continue
        project.apply_merge(results, overwrite=True)
        counts[exp_name] = sum(len(r.changed_frames) for r in results.values())

    return counts


def split_project_groups(
    config: SetupConfig, assignment: FileAssignment
) -> dict[str, tuple[SetupConfig, FileAssignment]]:
    """Split a multi-calibration-group assignment into per-prefix sub-configs/assignments.

    Returns a dict mapping each calibration prefix to a (SetupConfig, FileAssignment) pair
    ready to pass to execute_setup.  Returns an empty dict if needs_split is False.
    """
    if not assignment.needs_split:
        return {}

    result: dict[str, tuple[SetupConfig, FileAssignment]] = {}

    for prefix, cam_calib_dict in assignment.calibration_groups.items():
        sub_config = SetupConfig(
            source_dir=config.source_dir,
            output_dir=config.output_dir.parent / prefix,
            camera_names=config.camera_names,
            calib_keyword=config.calib_keyword,
            match_strategy=config.match_strategy,
            mtime_window_sec=config.mtime_window_sec,
            datestamp_window_sec=config.datestamp_window_sec,
            charuco=config.charuco,
            project_name=prefix,
            copy_files=config.copy_files,
            trial_keyword=config.trial_keyword,
            trial_regex=config.trial_regex,
            experiment_keywords=config.experiment_keywords,
            skeleton=config.skeleton,
        )

        sub_trials = {
            t: cams
            for t, cams in assignment.trials.items()
            if assignment.trial_prefix.get(t) == prefix
        }
        sub_calibration = {cam: list(files) for cam, files in cam_calib_dict.items()}
        sub_assignment = FileAssignment(
            trials=sub_trials,
            calibration=sub_calibration,
            exp_calibration={"default": sub_calibration},
            unmatched=list(assignment.unmatched),
            skeleton=assignment.skeleton,
        )

        result[prefix] = (sub_config, sub_assignment)

    return result


def write_project_toml(
    project_dir: Path,
    project_name: str,
    charuco: dict | None = None,
    video_roots: dict[str, Path] | None = None,
    slp_mapping: dict | None = None,
    active_calibrations: dict[str, str] | None = None,
) -> Path:
    """Write project.toml to the project directory.

    active_calibrations (experiment -> registry id) becomes [calibration.active].
    """
    config_data: dict = {
        "project": {"name": project_name},
    }
    if charuco:
        config_data["charuco"] = charuco
    if video_roots:
        config_data["video_roots"] = {
            exp: str(root) for exp, root in sorted(video_roots.items())
        }
    if slp_mapping:
        config_data["slp"] = slp_mapping
    if active_calibrations:
        config_data["calibration"] = {"active": dict(sorted(active_calibrations.items()))}

    path = project_dir / "project.toml"
    with open(path, "w") as f:
        toml.dump(config_data, f)
    return path


# --- Internal helpers ---


def _skeleton_dict(labels) -> dict:
    """Extract nodes / edges / symmetries from a loaded Labels object."""
    sk = labels.skeleton
    return {
        "nodes": [n.name for n in sk.nodes],
        "edges": [(e.source.name, e.destination.name) for e in sk.edges],
        "symmetries": [(s[0].name, s[1].name) for s in sk.symmetries],
    }


def _read_slp_info(slp_path: Path) -> tuple[set[str], dict]:
    """Open a .slp file; return (video_basenames, skeleton_dict).

    skeleton_dict has keys: nodes (list[str]), edges (list[tuple]),
    symmetries (list[tuple]).
    """
    labels = sio.load_file(str(slp_path))
    video_basenames = {Path(lf.video.filename).name for lf in labels.labeled_frames}
    return video_basenames, _skeleton_dict(labels)


def _video_filename(video) -> Path:
    """The path a sleap_io Video records, as a single Path.

    Image-sequence videos carry a list of filenames; the first one stands in for
    the sequence, since only its directory and stem are used for matching.
    """
    filename = video.filename
    if isinstance(filename, (list, tuple)):
        if not filename:
            raise ValueError("Video has no filename")
        filename = filename[0]
    return Path(filename)


def _read_slp_videos(slp_path: Path) -> tuple[list[Path], dict, int]:
    """Open a .slp file; return (recorded video paths, skeleton, labeled frames).

    Reads labels.videos rather than the labeled frames, so a video that is part
    of the file but carries no labels still becomes part of the project.
    """
    labels = sio.load_file(str(slp_path))
    paths = [_video_filename(v) for v in labels.videos]
    return paths, _skeleton_dict(labels), len(labels.labeled_frames)


DRIVE_LETTER_RE = re.compile(r"[A-Za-z]:")


@dataclass
class ResolvedVideo:
    """Outcome of locating one video reference from a .slp file."""

    path: Path | None
    # "recorded" | "anchor" | "suffix" | "beside_slp" | "basename"
    # | "ambiguous" | "unresolved"
    rule: str = "unresolved"
    # The recorded prefix that was replaced and what replaced it, for reporting
    # one remap line per tree instead of one per video.
    old_root: str = ""
    new_root: Path | None = None
    # Populated when several search directories match equally well; the
    # reference is then left unresolved rather than guessed at.
    candidates: list[Path] = field(default_factory=list)


def _recorded_parts(recorded: Path | str) -> list[str]:
    """Split a recorded video path into components, tolerating Windows spellings.

    A .slp written on Windows records e.g. ``R:/NewData/...`` or
    ``R:\\NewData\\...``; on a posix host neither splits correctly through
    Path. Both separators are honoured and a leading drive letter is dropped,
    leaving the components that can be matched against a search directory.
    """
    text = str(recorded).replace("\\", "/")
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if parts and DRIVE_LETTER_RE.fullmatch(parts[0]):
        parts = parts[1:]
    return parts


def _unique_existing(candidates: list[tuple[Path, Path]]) -> tuple[list[Path], list[Path]]:
    """Filter (candidate, search_dir) pairs to those on disk, deduplicated.

    Returns (resolved paths, the search dirs they came from), both in order.
    """
    paths: list[Path] = []
    dirs: list[Path] = []
    for candidate, directory in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved in paths:
            continue
        paths.append(resolved)
        dirs.append(directory)
    return paths, dirs


def _resolve_video_reference(
    recorded: Path, slp_dir: Path, search_dirs: list[Path]
) -> ResolvedVideo:
    """Locate a video a .slp file refers to, allowing for a moved recording.

    A recorded path usually moved as a whole tree — the drive letter and the
    folders above the collection changed, the structure below it did not. So
    the search keeps as much of the recorded structure as it can, trying in
    order:

    1. The recorded path itself.
    2. **Anchor** — a search directory whose own name appears as a directory
       component of the recorded path. Everything recorded below that
       component is joined onto the search directory.
    3. **Longest suffix** — the deepest trailing run of recorded components
       that exists under a search directory, so a search directory renamed on
       the new mount still matches.
    4. The basename beside the .slp file, then the basename in each search
       directory — the historical behaviour, kept last so it can no longer
       beat a structurally correct match in the wrong camera folder.

    When two search directories match equally well at the same step the
    reference is reported ambiguous and left unresolved, rather than resolved
    to an arbitrary one of them.
    """
    if recorded.exists():
        return ResolvedVideo(recorded.resolve(), "recorded")

    parts = _recorded_parts(recorded)
    if not parts:
        return ResolvedVideo(None)

    # 2. Anchor on a search directory's own name, matched exactly. Only
    #    directory components can anchor, so parts[:-1].
    anchored: list[tuple[Path, Path]] = []
    depths: dict[Path, int] = {}
    for directory in search_dirs:
        indices = [i for i, part in enumerate(parts[:-1]) if part == directory.name]
        if not indices:
            continue
        index = indices[-1]  # deepest occurrence when a name repeats
        anchored.append((directory.joinpath(*parts[index + 1:]), directory))
        depths[directory] = index
    paths, dirs = _unique_existing(anchored)
    if len(paths) > 1:
        return ResolvedVideo(None, "ambiguous", candidates=paths)
    if paths:
        index = depths[dirs[0]]
        return ResolvedVideo(
            paths[0], "anchor", "/".join(parts[: index + 1]), dirs[0]
        )

    # 3. Deepest trailing run of recorded components that exists.
    for k in range(len(parts) - 1, 1, -1):
        suffix = parts[-k:]
        paths, dirs = _unique_existing(
            [(d.joinpath(*suffix), d) for d in search_dirs]
        )
        if len(paths) > 1:
            return ResolvedVideo(None, "ambiguous", candidates=paths)
        if paths:
            return ResolvedVideo(
                paths[0], "suffix", "/".join(parts[:-k]), dirs[0]
            )

    # 4. Basename only, beside the .slp file first.
    candidate = slp_dir / parts[-1]
    if candidate.exists():
        return ResolvedVideo(candidate.resolve(), "beside_slp")
    paths, dirs = _unique_existing([(d / parts[-1], d) for d in search_dirs])
    if len(paths) > 1:
        return ResolvedVideo(None, "ambiguous", candidates=paths)
    if paths:
        return ResolvedVideo(paths[0], "basename", "/".join(parts[:-1]), dirs[0])

    return ResolvedVideo(None)


def _identify_camera_from_path(path: Path, camera_names: list[str]) -> str | None:
    """Identify a video's camera from its filename, then its parent directory.

    The parent directory is the fallback for rigs that write each camera into
    its own folder with filenames that carry no camera token.
    """
    camera = _identify_camera(path.name, camera_names)
    if camera is None:
        camera = _identify_camera(path.parent.name, camera_names)
    return camera


def _compute_video_roots(
    trials: dict[str, dict[str, Path]], trial_experiment: dict[str, str]
) -> dict[str, Path]:
    """Common parent directory of each experiment's videos, where one exists.

    A root lets trials.csv hold relative cells, so the project survives the
    videos being remounted elsewhere. Experiments whose videos span unrelated
    trees get no root and are recorded with absolute cells instead.
    """
    by_exp: dict[str, list[Path]] = {}
    for trial_name, cam_files in trials.items():
        exp = trial_experiment.get(trial_name, "default")
        by_exp.setdefault(exp, []).extend(cam_files.values())

    roots: dict[str, Path] = {}
    for exp, paths in by_exp.items():
        if not paths:
            continue
        parents = [p.parent.parts for p in paths]
        common: list[str] = []
        for parts in zip(*parents):
            if len(set(parts)) != 1:
                break
            common.append(parts[0])
        # A root of "/" alone says nothing useful; treat that as no root.
        if len(common) > 1:
            roots[exp] = Path(*common)
    return roots


def _run_match_strategy(
    strategy: str, videos: list[tuple[str, Path]], config: SetupConfig
) -> dict[str, dict[str, Path]]:
    """Group (camera, path) pairs into named sets using one matching strategy."""
    if strategy == "number":
        return _match_by_number(videos, config.camera_names)
    if strategy == "keyword":
        return _match_by_keyword(videos, config.trial_keyword)
    if strategy == "regex":
        return _match_by_regex(videos, config.trial_regex)
    if strategy == "datestamp":
        return _match_by_datestamp(
            videos, config.camera_names, config.datestamp_window_sec
        )
    if strategy == "mtime":
        return _match_by_mtime(videos, config.camera_names, config.mtime_window_sec)
    raise ValueError(f"Unknown match strategy: {strategy}")


def _is_calibration(filename: str, keywords: list[str], regex: str) -> bool:
    """Stage 1: does this file look like calibration at all?"""
    low = filename.lower()
    if any(kw.lower() in low for kw in keywords):
        return True
    if regex and re.search(regex, Path(filename).stem, re.IGNORECASE):
        return True
    return False


def _identify_take(filename: str, keywords: list[str], regex: str) -> str | None:
    """Stage 2: which calibration take does this file belong to?

    The matched keyword *is* the take name. With a regex, a named group "take"
    (else group 1, else the whole match) supplies it. Returns None when the
    take is undetermined.
    """
    if regex:
        m = re.search(regex, Path(filename).stem, re.IGNORECASE)
        if m:
            if "take" in (m.groupdict() or {}) and m.group("take"):
                return m.group("take")
            if m.groups() and m.group(1):
                return m.group(1)
            if not m.groups():
                # No capture group: the match itself names the take, so
                # cal\d+ gives cal01, cal02, ...  _is_calibration accepts a
                # group-less regex, so this stage must too or the files are
                # found and then dropped as ambiguous.
                return m.group(0)

    low = filename.lower()
    matches = [kw for kw in keywords if kw.lower() in low]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer the longest, so "PostCal" wins over a bare "Cal".
        matches.sort(key=len, reverse=True)
        if matches[1].lower() in matches[0].lower():
            return matches[0]
        return None  # genuinely ambiguous
    return None


def _match_calibration_takes(
    config: SetupConfig, calib_videos: list[tuple[str, Path]]
) -> tuple[dict[str, dict[str, Path]], list[Path]]:
    """Group one experiment's calibration videos into takes.

    Tries keyword/regex first. If that leaves a take undetermined — or puts two
    files from the same camera in one take — falls back to the configured
    match strategy. Returns (takes, unresolved_files).
    """
    keywords = config.effective_calib_keywords()
    takes: dict[str, dict[str, Path]] = {}
    ambiguous = False

    for cam, fp in calib_videos:
        take = _identify_take(fp.name, keywords, config.calib_regex)
        if take is None or cam in takes.get(take, {}):
            ambiguous = True
            break
        takes.setdefault(take, {})[cam] = fp

    if not ambiguous:
        return takes, []

    strategy = config.calib_match_strategy or config.match_strategy
    takes = _run_match_strategy(strategy, calib_videos, config)

    assigned = {fp for cam_files in takes.values() for fp in cam_files.values()}
    unresolved = [fp for _, fp in calib_videos if fp not in assigned]
    return takes, unresolved


def _identify_experiment(filename: str, experiment_keywords: list[str]) -> str:
    """Return the first experiment keyword found in filename (case-insensitive), or 'default'."""
    filename_lower = filename.lower()
    for kw in experiment_keywords:
        if kw.lower() in filename_lower:
            return kw
    return "default"


def _identify_camera(filename: str, camera_names: list[str]) -> str | None:
    """Find which camera a filename belongs to by substring match.

    Returns the camera name, or None if no match / ambiguous match.
    """
    filename_lower = filename.lower()
    matches = [cam for cam in camera_names if cam.lower() in filename_lower]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Try to disambiguate: prefer longer camera name matches
        matches.sort(key=len, reverse=True)
        # If the longest match contains a shorter match as a substring,
        # both would match — keep only the longest
        # e.g. camera names "cam1" and "cam10": "cam10_trial1.mp4" matches both
        # but "cam10" is the better match
        best = matches[0]
        # Check if the second match is a substring of the first
        if matches[1].lower() in best.lower():
            return best
        # Truly ambiguous
        return None
    return None


def _extract_calib_prefix(
    filename: str, camera_names: list[str], calib_keyword: str
) -> str:
    """Extract the experiment prefix from a calibration filename.

    Removes the camera name and calibration keyword from the stem, then
    returns the cleaned-up leading portion as the prefix.
    """
    stem = Path(filename).stem
    for cam in camera_names:
        stem = re.sub(re.escape(cam), "", stem, flags=re.IGNORECASE)
    if calib_keyword:
        stem = re.sub(re.escape(calib_keyword), "", stem, flags=re.IGNORECASE)
    return stem.strip("_- ")


def _extract_trial_prefix(
    filename: str,
    camera_names: list[str],
    match_strategy: str,
    trial_keyword: str = "",
    trial_regex: str = "",
) -> str:
    """Extract the experiment prefix from a trial filename.

    Removes the camera name and the trial-identifying portion (number,
    keyword+number, datestamp, etc.) and returns what remains as the prefix.
    """
    stem = Path(filename).stem
    for cam in camera_names:
        stem = re.sub(re.escape(cam), "", stem, flags=re.IGNORECASE)
    stem = stem.strip("_- ")

    if match_strategy == "number":
        m = re.search(r"\d+", stem)
        if m:
            return stem[: m.start()].strip("_- ")
        return stem

    if match_strategy == "keyword":
        if trial_keyword:
            m = re.search(re.escape(trial_keyword) + r"\d*", stem, re.IGNORECASE)
            if m:
                return stem[: m.start()].strip("_- ")
        return stem

    if match_strategy == "datestamp":
        for regex, _, _ in _DATESTAMP_PATTERNS:
            m = regex.search(stem)
            if m:
                return stem[: m.start()].strip("_- ")
        return stem

    # mtime / regex / unknown — fall back to first underscore-delimited token
    parts = re.split(r"[_\-\s]+", stem)
    return parts[0] if parts else ""


def _extract_trial_key_number(
    filename: str, camera_names: list[str]
) -> str | None:
    """Extract a numeric trial key from a filename.

    Removes the camera name and extension, then finds the remaining
    numeric portion(s) to use as a trial key.
    """
    stem = Path(filename).stem
    # Remove camera name (case-insensitive)
    for cam in camera_names:
        pattern = re.compile(re.escape(cam), re.IGNORECASE)
        stem = pattern.sub("", stem)

    # Find all numeric sequences
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    # Use the first numeric sequence as the trial key
    return numbers[0]


@dataclass
class _DatestampResult:
    """Parsed datestamp from a filename."""

    label: str  # human-readable label for trial naming
    timestamp: float  # seconds since midnight on the date (for range matching)
    date_str: str  # YYYYMMDD portion


# Each pattern returns (match_func) that extracts a _DatestampResult from regex groups
_DATESTAMP_PATTERNS: list[tuple[re.Pattern, bool, bool]] = [
    # YYYYYMMDDHHHMMSSS.fractional — e.g. Y20260116H142107.519576000
    (re.compile(r"Y(\d{4})(\d{2})(\d{2})H(\d{2})(\d{2})(\d{2})(?:\.(\d+))?"), True, True),
    # YYYYMMDD_HHMMSS (with optional fractional)
    (re.compile(r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})(?:\.(\d+))?"), True, False),
    # YYYY-MM-DD_HH-MM-SS (with optional fractional)
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[_T](\d{2})-(\d{2})-(\d{2})(?:\.(\d+))?"), True, False),
    # YYYY-MM-DD_HHMMSS (with optional fractional)
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[_T](\d{2})(\d{2})(\d{2})(?:\.(\d+))?"), True, False),
    # YYYYMMDD only
    (re.compile(r"(\d{4})(\d{2})(\d{2})"), False, False),
    # YYYY-MM-DD only
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), False, False),
]


def _extract_trial_key_datestamp(
    filename: str, camera_names: list[str]
) -> _DatestampResult | None:
    """Extract a date/datetime stamp from a filename.

    Returns a _DatestampResult with a label string and a numeric timestamp
    (seconds since midnight on the date) for range-based matching.
    """
    stem = Path(filename).stem
    # Remove camera name
    for cam in camera_names:
        pattern = re.compile(re.escape(cam), re.IGNORECASE)
        stem = pattern.sub("", stem)

    for regex, has_time, _is_yh_format in _DATESTAMP_PATTERNS:
        m = regex.search(stem)
        if m:
            groups = list(m.groups())
            date_str = f"{groups[0]}{groups[1]}{groups[2]}"
            if has_time:
                hh, mm, ss = int(groups[3]), int(groups[4]), int(groups[5])
                frac_str = groups[6] if len(groups) > 6 and groups[6] else "0"
                frac = float(f"0.{frac_str}")
                timestamp = hh * 3600 + mm * 60 + ss + frac
                label = f"{date_str}_{groups[3]}{groups[4]}{groups[5]}"
                return _DatestampResult(
                    label=label, timestamp=timestamp, date_str=date_str
                )
            else:
                return _DatestampResult(
                    label=date_str, timestamp=0.0, date_str=date_str
                )
    return None


def _match_by_keyword(
    files: list[tuple[str, Path]],
    trial_keyword: str,
) -> dict[str, dict[str, Path]]:
    """Match files into trials by a keyword+number pattern in the filename.

    The trial key is the portion of the stem up to and including the
    keyword+number (e.g. 'PS20_Trial8'), so experiment prefixes are
    preserved and different experiments with the same trial number are kept
    separate.  The camera name is stripped from the stem first so that a
    camera prefix (e.g. 'cam1_PS18_Trial1.mp4') does not contaminate the key.
    """
    key_groups: dict[str, dict[str, Path]] = {}
    for camera, filepath in files:
        stem = re.sub(re.escape(camera), "", filepath.stem, flags=re.IGNORECASE).strip("_- ")
        pattern = re.compile(re.escape(trial_keyword) + r"\d+", re.IGNORECASE)
        m = pattern.search(stem)
        if m is None:
            continue
        key = stem[: m.end()].strip("_- ")
        if key:
            key_groups.setdefault(key, {})[camera] = filepath
    return key_groups


def _extract_trial_key_regex(filename: str, pattern: re.Pattern) -> str | None:
    """Extract a trial key by applying a compiled regex to the filename stem.

    Uses the named group 'trial' if present, otherwise group 1, otherwise
    the full match.
    """
    m = pattern.search(Path(filename).stem)
    if m is None:
        return None
    try:
        return m.group("trial")
    except IndexError:
        pass
    if m.lastindex:
        return m.group(1)
    return m.group(0)


def _match_by_regex(
    files: list[tuple[str, Path]],
    trial_regex: str,
) -> dict[str, dict[str, Path]]:
    """Match files into trials by a user-supplied regex pattern.

    The pattern is matched against each filename stem. The trial key is taken
    from the named group 'trial' if present, else group 1, else the full match.
    Raises ValueError for an invalid pattern.
    """
    try:
        pattern = re.compile(trial_regex, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid trial_regex pattern '{trial_regex}': {exc}") from exc

    key_groups: dict[str, dict[str, Path]] = {}
    for camera, filepath in files:
        key = _extract_trial_key_regex(filepath.name, pattern)
        if key is None:
            continue
        key_groups.setdefault(key, {})[camera] = filepath
    return key_groups


def _match_by_number(
    files: list[tuple[str, Path]], camera_names: list[str]
) -> dict[str, dict[str, Path]]:
    """Match files into trials by numeric substring."""
    key_groups: dict[str, dict[str, Path]] = {}

    for camera, filepath in files:
        key = _extract_trial_key_number(filepath.name, camera_names)
        if key is None:
            continue
        trial_name = f"trial_{key}"
        key_groups.setdefault(trial_name, {})[camera] = filepath

    return key_groups


def _match_by_datestamp(
    files: list[tuple[str, Path]],
    camera_names: list[str],
    window_sec: float = 0.0,
) -> dict[str, dict[str, Path]]:
    """Match files into trials by date/datetime stamp in filename.

    If window_sec > 0, timestamps within the same date that differ by at most
    window_sec seconds are grouped into the same trial (useful when cameras
    embed slightly different sub-second timestamps). When window_sec == 0,
    only exact label matches are grouped.
    """
    if window_sec <= 0:
        # Exact matching — group by label string
        key_groups: dict[str, dict[str, Path]] = {}
        for camera, filepath in files:
            result = _extract_trial_key_datestamp(filepath.name, camera_names)
            if result is None:
                continue
            trial_name = f"trial_{result.label}"
            key_groups.setdefault(trial_name, {})[camera] = filepath
        return key_groups

    # Windowed matching — cluster by (date, timestamp) proximity
    parsed: list[tuple[str, Path, _DatestampResult]] = []
    for camera, filepath in files:
        result = _extract_trial_key_datestamp(filepath.name, camera_names)
        if result is None:
            continue
        parsed.append((camera, filepath, result))

    if not parsed:
        return {}

    # Sort by (date, timestamp) so clustering works
    parsed.sort(key=lambda x: (x[2].date_str, x[2].timestamp))

    # Cluster: files on the same date within window_sec of the cluster start
    clusters: list[list[tuple[str, Path, _DatestampResult]]] = []
    current: list[tuple[str, Path, _DatestampResult]] = [parsed[0]]

    for i in range(1, len(parsed)):
        cam, fp, res = parsed[i]
        cur_res = current[0][2]
        if (
            res.date_str == cur_res.date_str
            and abs(res.timestamp - cur_res.timestamp) <= window_sec
        ):
            current.append((cam, fp, res))
        else:
            clusters.append(current)
            current = [(cam, fp, res)]
    clusters.append(current)

    # Convert clusters to trial dict, using the first file's label
    result_dict: dict[str, dict[str, Path]] = {}
    for cluster in clusters:
        label = cluster[0][2].label
        trial_name = f"trial_{label}"
        # Deduplicate trial names
        base = trial_name
        suffix = 2
        while trial_name in result_dict:
            trial_name = f"{base}_{suffix}"
            suffix += 1
        cam_files: dict[str, Path] = {}
        for camera, filepath, _ in cluster:
            cam_files[camera] = filepath
        result_dict[trial_name] = cam_files

    return result_dict


def _match_by_mtime(
    files: list[tuple[str, Path]],
    camera_names: list[str],
    window_sec: float,
) -> dict[str, dict[str, Path]]:
    """Match files into trials by modification time clustering.

    Files within `window_sec` seconds of each other are grouped together.
    Within each cluster, at most one file per camera is expected.
    """
    if not files:
        return {}

    # Sort by modification time
    timed = [(camera, filepath, filepath.stat().st_mtime) for camera, filepath in files]
    timed.sort(key=lambda x: x[2])

    clusters: list[list[tuple[str, Path, float]]] = []
    current_cluster: list[tuple[str, Path, float]] = [timed[0]]

    for i in range(1, len(timed)):
        cam, fp, mtime = timed[i]
        # Check if within window of any file in the current cluster
        cluster_min_time = min(t for _, _, t in current_cluster)
        if mtime - cluster_min_time <= window_sec:
            current_cluster.append((cam, fp, mtime))
        else:
            clusters.append(current_cluster)
            current_cluster = [(cam, fp, mtime)]
    clusters.append(current_cluster)

    # Convert clusters to trial dict
    result: dict[str, dict[str, Path]] = {}
    for i, cluster in enumerate(clusters):
        trial_name = f"trial_{i + 1:03d}"
        cam_files: dict[str, Path] = {}
        for camera, filepath, _ in cluster:
            cam_files[camera] = filepath
        result[trial_name] = cam_files

    return result

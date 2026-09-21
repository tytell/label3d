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
    calib_keyword_lower = config.calib_keyword.lower()

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
        if calib_keyword_lower and calib_keyword_lower in f.name.lower():
            calib_videos.append((camera, f))
        else:
            data_videos.append((camera, f))

    for f in image_files:
        camera = _identify_camera(f.name, config.camera_names)
        if camera is None:
            assignment.unmatched.append(f)
            continue
        if calib_keyword_lower and calib_keyword_lower in f.name.lower():
            calib_images.append((camera, f))
        else:
            assignment.unmatched.append(f)

    # Build calibration dict
    for camera, f in calib_videos:
        assignment.calibration.setdefault(camera, []).append(f)
    for camera, f in calib_images:
        assignment.calibration.setdefault(camera, []).append(f)

    # Match data videos into trials
    if config.match_strategy == "number":
        assignment.trials = _match_by_number(data_videos, config.camera_names)
    elif config.match_strategy == "keyword":
        assignment.trials = _match_by_keyword(
            data_videos, config.trial_keyword
        )
    elif config.match_strategy == "regex":
        assignment.trials = _match_by_regex(data_videos, config.trial_regex)
    elif config.match_strategy == "datestamp":
        assignment.trials = _match_by_datestamp(
            data_videos, config.camera_names, config.datestamp_window_sec
        )
    elif config.match_strategy == "mtime":
        assignment.trials = _match_by_mtime(
            data_videos, config.camera_names, config.mtime_window_sec
        )
    else:
        raise ValueError(f"Unknown match strategy: {config.match_strategy}")

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

    # Detect calibration groups for multi-group split detection
    prefix_to_calib: dict[str, dict[str, list[Path]]] = {}
    for cam, fp in all_calib:
        prefix = _extract_calib_prefix(fp.name, config.camera_names, config.calib_keyword)
        prefix_to_calib.setdefault(prefix, {}).setdefault(cam, []).append(fp)

    if len(prefix_to_calib) > 1:
        assignment.needs_split = True
        assignment.calibration_groups = prefix_to_calib
        for trial_name, cam_files in assignment.trials.items():
            if cam_files:
                first_file = next(iter(cam_files.values()))
                assignment.trial_prefix[trial_name] = _extract_trial_prefix(
                    first_file.name,
                    config.camera_names,
                    config.match_strategy,
                    trial_keyword=config.trial_keyword,
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
                f"Experiment '{exp}' has no calibration images for: {', '.join(missing)}"
            )

    return assignment


def preview_assignment(config: SetupConfig, assignment: FileAssignment) -> str:
    """Return a human-readable summary of the file assignment."""
    lines = []

    lines.append(f"Project: {config.project_name or config.output_dir.name}")
    lines.append(f"Output:  {config.output_dir}")
    lines.append(f"Experiments: {', '.join(config.experiment_keywords) if config.experiment_keywords else '(default)'}")
    lines.append(f"Cameras: {', '.join(config.camera_names)}")
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

    # Calibration — per experiment
    if assignment.exp_calibration:
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


def execute_setup(config: SetupConfig, assignment: FileAssignment) -> Path:
    """Create project folder structure, move or copy files, write trials.csv and project.toml.

    Output layout:
      project/
        project.toml
        skeleton.json            <- written if SLP files provided a skeleton
        trials.csv               <- experiment + trial columns
        slp/
        experiments/
          default/
            videos/
            calibration/{cam}/

    Returns the output directory path.
    """
    out = config.output_dir
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")

    transfer = shutil.copy2 if config.copy_files else shutil.move

    # Group trials by experiment
    if assignment.trial_experiment:
        exp_trials: dict[str, list[str]] = {}
        for trial_name, exp in assignment.trial_experiment.items():
            exp_trials.setdefault(exp, []).append(trial_name)
    else:
        exp_trials = {"default": sorted(assignment.trials)}

    # Create directory structure and move/copy files per experiment
    trials_rows: list[dict] = []
    for exp_name in sorted(exp_trials):
        exp_dir = out / "experiments" / exp_name
        videos_dir = exp_dir / "videos"
        videos_dir.mkdir(parents=True)

        for cam, files in assignment.exp_calibration.get(exp_name, {}).items():
            cam_dir = exp_dir / "calibration" / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                transfer(str(f), str(cam_dir / f.name))

        for trial_name in sorted(exp_trials[exp_name]):
            cam_files = assignment.trials[trial_name]
            row: dict[str, str | int] = {"experiment": exp_name, "trial": trial_name}
            for cam in config.camera_names:
                if cam in cam_files:
                    src = cam_files[cam]
                    dst = videos_dir / src.name
                    transfer(str(src), str(dst))
                    row[cam] = src.name
                    row[f"{cam}_start"] = 0
                    row[f"{cam}_end"] = ""
                else:
                    row[cam] = ""
                    row[f"{cam}_start"] = ""
                    row[f"{cam}_end"] = ""
            trials_rows.append(row)

    # Write trials.csv at project root with experiment column
    if trials_rows:
        columns = ["experiment", "trial"]
        for cam in config.camera_names:
            columns.extend([cam, f"{cam}_start", f"{cam}_end"])
        df = pd.DataFrame(trials_rows, columns=columns)
        df.to_csv(out / "trials.csv", index=False)

    # Move/copy SLP files (deduplicate: one slp_path may cover multiple trial+camera pairs)
    if assignment.slp_files:
        slp_dir = out / "slp"
        slp_dir.mkdir(parents=True, exist_ok=True)
        seen: set[Path] = set()
        for trial_name, cam_files in assignment.slp_files.items():
            for cam, f in cam_files.items():
                if f not in seen:
                    seen.add(f)
                    transfer(str(f), str(slp_dir / f.name))

    # Write project.toml
    write_project_toml(out, config.project_name or out.name, config.charuco)

    # Write skeleton.json if a skeleton was extracted from SLP files
    if assignment.skeleton:
        with open(out / "skeleton.json", "w") as fh:
            json.dump(assignment.skeleton, fh, indent=2)

    return out


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
) -> Path:
    """Write project.toml to the project directory."""
    config_data: dict = {
        "project": {"name": project_name},
    }
    if charuco:
        config_data["charuco"] = charuco

    path = project_dir / "project.toml"
    with open(path, "w") as f:
        toml.dump(config_data, f)
    return path


# --- Internal helpers ---


def _read_slp_info(slp_path: Path) -> tuple[set[str], dict]:
    """Open a .slp file; return (video_basenames, skeleton_dict).

    skeleton_dict has keys: nodes (list[str]), edges (list[tuple]),
    symmetries (list[tuple]).
    """
    labels = sio.load_file(str(slp_path))
    video_basenames = {Path(lf.video.filename).name for lf in labels.labeled_frames}
    sk = labels.skeleton
    nodes = [n.name for n in sk.nodes]
    edges = [(e.source.name, e.destination.name) for e in sk.edges]
    symmetries = [(s[0].name, s[1].name) for s in sk.symmetries]
    return video_basenames, {"nodes": nodes, "edges": edges, "symmetries": symmetries}


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

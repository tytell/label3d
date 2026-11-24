"""Discover and register additional SLEAP videos in an existing project.

Discovery is read-only. Application stages all outputs, backs up each replaced
file, and rolls back if a write fails. SLEAP files are never rewritten here.
Qt objects and the live project are updated by the caller on the main thread.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from sleap_io.io.slp import read_videos

from label3d.adapters.sleap_adapter import SleapAdapter
from label3d.drives import read_video_roots, resolve_root, to_drive_form
from label3d.points import POINTS_FILE, write_points_file
from label3d.project import Project, check_trial_videos, write_trials_csv
from label3d.setup_project import (
    VIDEO_EXTENSIONS, SetupConfig, _identify_camera_from_path, _identify_take,
    _natural_key, _resolve_video_reference, _run_match_strategy, copy_file,
)
from label3d.slp_sync import SlpMapping, video_cells
from label3d.trial_types import CALIBRATION, TRIAL, TYPE_COLUMN, row_types
from label3d.version import write_project_toml
from label3d.video_matching import AmbiguousVideoError, VideoMatcher, path_key, path_parts


class AddVideosError(ValueError):
    pass


@dataclass
class DiscoveryOptions:
    match_strategy: str = "keyword"
    trial_keyword: str = "trial"
    trial_regex: str = ""
    calib_regex: str = r"cal\d+"
    search_dirs: list[Path] = field(default_factory=list)

    @classmethod
    def from_project(cls, project: Project) -> DiscoveryOptions:
        saved = project.config.get("discovery", {})
        if not saved:
            names = [str(value) for cam in project.cameras for value in project.trials_df[cam].dropna()]
            if any(re.search(r"trial[_ -]+\d+", name, re.IGNORECASE) for name in names):
                return cls(match_strategy="regex", trial_regex=r"(?P<trial>trial[_ -]*\d+)")
        return cls(
            **{k: saved[k] for k in ("match_strategy", "trial_keyword", "trial_regex", "calib_regex") if k in saved},
            search_dirs=[Path(resolve_root(p, project.config)) for p in saved.get("search_dirs", [])],
        )

    def to_config(self, project: Project) -> dict:
        return {
            **project.config.get("discovery", {}),
            "match_strategy": self.match_strategy, "trial_keyword": self.trial_keyword,
            "trial_regex": self.trial_regex, "calib_regex": self.calib_regex,
            "search_dirs": [to_drive_form(str(p), project.config) or str(p) for p in self.search_dirs],
        }


@dataclass
class Candidate:
    source: Path
    recorded: str
    path: Path | None
    experiment: str
    trial: str
    camera: str = ""
    issue: str = ""


@dataclass
class CalibrationVideo:
    path: Path
    take: str
    camera: str


@dataclass
class Discovery:
    candidates: list[Candidate]
    options: DiscoveryOptions
    fingerprints: dict[Path, tuple[int, int]]
    registered: int = 0


@dataclass
class AdditionResult:
    config: dict
    trials_df: pd.DataFrame
    backup_dir: Path
    n_trials: int
    n_videos: int
    n_experiments: int
    n_points: int

    def refresh(self, project: Project) -> None:
        """Adopt committed metadata and points, on the main thread."""
        active = project.active_experiment
        project.config = self.config
        project.refresh_trials(self.trials_df)
        project.load_skeletons()
        project.set_experiment(active)


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def search_directories(project: Project, options: DiscoveryOptions) -> list[Path]:
    """Known roots and their common collection folder, without walking a share."""
    roots = list(read_video_roots(project.config).values())
    candidates = [*options.search_dirs, *roots, *(p.parent for p in roots)]
    if roots:
        try:
            common = Path(os.path.commonpath([str(p) for p in roots]))
            if len(common.parts) > 2:
                candidates.append(common)
        except ValueError:
            pass
    return list(dict.fromkeys(p.expanduser().resolve() for p in candidates if p.is_dir()))


def _setup(project: Project, options: DiscoveryOptions) -> SetupConfig:
    for label, pattern in (("trial", options.trial_regex), ("calibration", options.calib_regex)):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise AddVideosError(f"Invalid {label} regex: {exc}") from exc
    if options.match_strategy == "keyword" and not options.trial_keyword.strip():
        raise AddVideosError("Enter a trial keyword before scanning.")
    if options.match_strategy == "regex" and not options.trial_regex.strip():
        raise AddVideosError("Enter a trial regex before scanning.")
    return SetupConfig(
        source_dir=project.root, output_dir=project.root,
        camera_names=project.cameras, camera_aliases=project.camera_aliases,
        calib_keyword="", calib_regex=options.calib_regex,
        match_strategy=options.match_strategy, trial_keyword=options.trial_keyword,
        trial_regex=options.trial_regex,
        mtime_window_sec=project.config.get("discovery", {}).get("mtime_window_sec", 60.0),
        datestamp_window_sec=project.config.get("discovery", {}).get("datestamp_window_sec", 0.0),
    )


def _suggest_experiment(path: Path, project: Project) -> str:
    for name, root in read_video_roots(project.config).items():
        if path.parent == root:
            return name
    # A directory is more reliable than an unrelated date in a video filename.
    for value in (*(p.name for p in list(path.parents)[:3]), path.name):
        match = re.search(r"(?<!\d)(20\d{6})(?!\d)", value)
        if match:
            return match.group(1)
    return path.parent.name


def discover_videos(project: Project, options: DiscoveryOptions | None = None,
                    progress=None) -> Discovery:
    options = options or DiscoveryOptions.from_project(project)
    config = _setup(project, options)
    directories = search_directories(project, options)
    matcher = VideoMatcher(project.trials_df, project.cameras, project.root, read_video_roots(project.config))
    registered_paths = {path_key(p.resolve()) for p in matcher.cell_paths.values()}
    sources = list(dict.fromkeys(project.root / f.path for f in project.slp_mapping.files))
    if not sources:
        sources = project.slp_files()
    if not sources:
        raise AddVideosError("No SLEAP files found. Set up the project's SLEAP files first.")
    stamps = {p: _stamp(p) for p in [project.root / "project.toml", project.trials_csv_path]}
    candidates, known, seen = [], 0, set()
    for index, source in enumerate(sources):
        if not source.is_file():
            raise AddVideosError(f"SLEAP file not found: {source}")
        stamps[source] = _stamp(source)
        if progress:
            progress(index, len(sources), f"Examining {source.name}…")
        for video in read_videos(str(source), open_backend=False):
            if not isinstance(video.filename, str):
                continue  # image sequences are not trial videos
            recorded = video.filename
            aliases = project.config.get("slp_video_paths", {}).get(source.relative_to(project.root).as_posix(), {})
            resolved_name = resolve_root(aliases.get(recorded, recorded), project.config)
            outcome = _resolve_video_reference(Path(resolved_name), source.parent, directories)
            path = outcome.path
            if path is not None and path_key(path.resolve()) in registered_paths:
                try:
                    matches = matcher.matches(resolved_name)
                except AmbiguousVideoError:
                    matches = []
                if outcome.rule in ("basename", "beside_slp") and not matches:
                    # A missing new recording may have an old sibling with
                    # the same basename. That is not a valid relocation.
                    path = None
                else:
                    known += 1
                    continue
            if path is None:
                try:
                    matches = matcher.matches(recorded)
                    if matches and (len(path_parts(recorded)) == 1 or any(
                        path_parts(recorded)[-2:] == path_parts(matcher.cell_paths[cell])[-2:]
                        for cell in matches
                    )):
                        known += 1
                        continue
                except AmbiguousVideoError:
                    pass
            identity = (source, path_key(path) if path else recorded)
            if identity in seen:
                continue
            seen.add(identity)
            hint = path or Path(recorded.replace("\\", "/"))
            issue = ""
            if path is None:
                issue = "Several files match; locate the video" if outcome.candidates else "Video not found; locate the video"
            elif path.suffix.lower() not in VIDEO_EXTENSIONS:
                issue = "Unsupported trial video type"
            candidates.append(Candidate(
                source, recorded, path, _suggest_experiment(hint, project), "",
                _identify_camera_from_path(hint, project.cameras, project.camera_aliases) or "", issue,
            ))
    # Match each experiment independently. Match one file at a time for the
    # name-based strategies, so duplicate camera/trial matches cannot overwrite.
    by_exp = defaultdict(list)
    for candidate in candidates:
        if candidate.path is not None and candidate.camera:
            by_exp[candidate.experiment].append(candidate)
    for items in by_exp.values():
        if options.match_strategy in ("keyword", "regex", "number"):
            for item in items:
                groups = _run_match_strategy(options.match_strategy, [(item.camera, item.path)], config)
                item.trial = next(iter(groups), "")
        else:
            groups = _run_match_strategy(options.match_strategy, [(v.camera, v.path) for v in items], config)
            for name, videos in groups.items():
                for item in items:
                    if item.path in videos.values():
                        item.trial = name
    candidates.sort(key=lambda v: (v.experiment, _natural_key(v.trial), v.camera, str(v.source)))
    return Discovery(candidates, options, stamps, known)


def calibration_candidates(project: Project, files: list[Path], regex: str) -> list[CalibrationVideo]:
    """The user selects the experiment; filenames need no experiment keyword."""
    result = []
    for path in sorted(set(p.resolve() for p in files)):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        camera = _identify_camera_from_path(path, project.cameras, project.camera_aliases) or ""
        keywords = project.config.get("discovery", {}).get("calib_keywords", [])
        take = _identify_take(path.name, keywords, regex) or ""
        result.append(CalibrationVideo(path, take, camera))
    return result


def candidate_problems(candidates: list[Candidate], cameras: list[str]) -> list[str]:
    counts = Counter((v.experiment, v.trial, v.camera) for v in candidates)
    problems = []
    for v in candidates:
        if v.issue:
            problems.append(f"{v.recorded}: {v.issue}")
        if v.path is None or not v.path.is_file():
            problems.append(f"Video not found: {v.recorded}")
        if v.camera not in cameras:
            problems.append(f"Choose a camera for {v.recorded}")
        for kind, value in (("experiment", v.experiment), ("trial", v.trial)):
            if (not value or value in (".", "..") or value != value.strip()
                    or any(c in value for c in '/\\\0<>:"|?*') or value.endswith('.')):
                problems.append(f"Invalid {kind} name: {value!r}")
        if counts[(v.experiment, v.trial, v.camera)] > 1:
            problems.append(f"Several videos assigned to {v.experiment}/{v.trial}/{v.camera}")
    identities = Counter(path_key(v.path.resolve()) for v in candidates if v.path is not None)
    if any(n > 1 for n in identities.values()):
        problems.append("The same video was selected more than once (possibly from several SLEAP files).")
    return list(dict.fromkeys(problems))


def incomplete_trials(candidates: list[Candidate], cameras: list[str],
                      existing: pd.DataFrame | None = None) -> list[str]:
    groups = defaultdict(set)
    for v in candidates:
        groups[(v.experiment, v.trial)].add(v.camera)
    if existing is not None:
        for exp, trial, cam in video_cells(existing, cameras):
            if (exp, trial) in groups:
                groups[(exp, trial)].add(cam)
    return [f"{exp}/{trial}: missing {', '.join(sorted(set(cameras) - found))}"
            for (exp, trial), found in groups.items() if set(cameras) - found]


def apply_additions(project: Project, discovery: Discovery, selected: list[Candidate],
                    calibrations: dict[str, list[CalibrationVideo]], *,
                    allow_incomplete: bool = False, progress=None) -> AdditionResult:
    """Stage and commit confirmed additions, leaving existing video assignments intact.

    Each new experiment must have an entry in calibrations; an empty list is an
    explicit choice to calibrate later. Call AdditionResult.refresh on the main
    thread after success. The caller saves the active store before starting.
    """
    if not selected:
        raise AddVideosError("Select at least one video to add.")
    if project.points is not None and project.points.changed:
        raise AddVideosError("Save the current labels before adding videos.")
    for path, stamp in discovery.fingerprints.items():
        if not path.exists() or _stamp(path) != stamp:
            raise AddVideosError(f"{path.name} changed since discovery. Scan again before adding videos.")
    output_stamps = {p: _stamp(p) if p.exists() else None for p in (
        project.sync_history_path,
        *(project.root / "experiments" / v.experiment / POINTS_FILE for v in selected),
    )}
    problems = candidate_problems(selected, project.cameras)
    if not allow_incomplete:
        problems += incomplete_trials(selected, project.cameras, project.trials_df)
    if problems:
        raise AddVideosError("\n".join(problems))
    config = copy.deepcopy(project.config)
    trials = project.trials_df.copy(deep=True)
    if "experiment" not in trials:
        trials.insert(0, "experiment", "default")
    if TYPE_COLUMN not in trials:
        trials[TYPE_COLUMN] = row_types(trials)
    roots = read_video_roots(config)
    matcher = VideoMatcher(trials, project.cameras, project.root, roots)
    registered = {path_key(p.resolve()) for p in matcher.cell_paths.values()}
    cell_map = project.slp_mapping.cell_map()
    # Older projects may have SLP files but no assignments yet. Preserve their
    # existing assignments by adopting video references, without rewriting files.
    if not cell_map:
        for source in project.slp_files():
            for video in read_videos(str(source), open_backend=False):
                if isinstance(video.filename, str):
                    for cell in matcher.matches(video.filename):
                        rel = source.relative_to(project.root)
                        if cell in cell_map and cell_map[cell] != rel:
                            raise AddVideosError(f"Several SLEAP files supply {cell}; set up their assignments first.")
                        cell_map[cell] = rel
    new_exps = sorted({v.experiment for v in selected} - set(project.experiments))
    for exp in new_exps:
        if exp not in calibrations:
            raise AddVideosError(f"Choose calibration videos for {exp}, or explicitly add calibration later.")
        paths = [str(v.path.parent) for v in selected if v.experiment == exp]
        root = Path(os.path.commonpath(paths))
        if len(root.parts) < 2:
            raise AddVideosError(f"Videos for {exp} need a common parent folder.")
        roots[exp] = root
        config.setdefault("video_roots", {})[exp] = to_drive_form(str(root), config) or str(root)
    new_cells, changed_rows = {}, []
    existing_keys = {(str(r['experiment']), str(r['trial'])): i for i, r in trials.iterrows()}
    for v in selected:
        if v.source not in discovery.fingerprints or not v.source.is_relative_to(project.root):
            raise AddVideosError("Only the project's scanned SLEAP files can supply new trials.")
        if path_key(v.path.resolve()) in registered:
            raise AddVideosError(f"Video already registered: {v.path}")
        cell = (v.experiment, v.trial, v.camera)
        key = cell[:2]
        if key not in existing_keys:
            existing_keys[key] = len(trials)
            row = {"experiment": v.experiment, "trial": v.trial, TYPE_COLUMN: TRIAL}
            for cam in project.cameras:
                row.update({cam: "", f"{cam}_start": pd.NA, f"{cam}_end": pd.NA, f"{cam}_mtime": ""})
            trials = pd.concat([trials, pd.DataFrame([row])], ignore_index=True)
            changed_rows.append(existing_keys[key])
        i = existing_keys[key]
        if row_types(trials).iloc[i] != TRIAL:
            raise AddVideosError(f"{v.experiment}/{v.trial} is already a calibration or axis take.")
        previous = trials.at[i, v.camera]
        if pd.notna(previous) and str(previous).strip():
            raise AddVideosError(f"{v.experiment}/{v.trial}/{v.camera} already has a video.")
        try:
            filename = v.path.relative_to(roots[v.experiment]).as_posix()
        except (KeyError, ValueError):
            filename = str(v.path)
        trials.at[i, v.camera] = filename
        trials.at[i, f"{v.camera}_start"] = 0
        cell_map[cell] = v.source.relative_to(project.root)
        new_cells[cell] = cell_map[cell]
        registered.add(path_key(v.path.resolve()))
    config["slp"] = SlpMapping.from_cell_map(cell_map).to_config()
    config["discovery"] = discovery.options.to_config(project)
    # Remember explicit relocations only when structural matching cannot find
    # the accepted cell. Scope them to the SLEAP file to permit duplicate names.
    new_matcher = VideoMatcher(trials, project.cameras, project.root, roots)
    for v in selected:
        try:
            matched = new_matcher.matches(v.recorded)
        except AmbiguousVideoError:
            matched = []
        if (v.experiment, v.trial, v.camera) not in matched:
            config.setdefault("slp_video_paths", {}).setdefault(v.source.relative_to(project.root).as_posix(), {})[v.recorded] = (
                to_drive_form(str(v.path), config) or str(v.path)
            )
    calibration_files = {}
    for exp, videos in calibrations.items():
        if exp not in new_exps and videos:
            raise AddVideosError("This workflow adds calibration recordings only for new experiments.")
        by_take = defaultdict(dict)
        for video in videos:
            if (video.camera not in project.cameras or not video.take.strip()
                    or video.take in (".", "..") or any(c in video.take for c in '/\\\0<>:"|?*')
                    or video.take != video.take.strip() or video.take.endswith('.')
                    or not video.path.is_file()):
                raise AddVideosError(f"Choose a valid take, camera and file for {video.path}")
            if video.camera in by_take[video.take]:
                raise AddVideosError(f"Several calibration files for {exp}/{video.take}/{video.camera}")
            if (exp, video.take) in existing_keys:
                raise AddVideosError(f"{exp}/{video.take} is already a trial or take.")
            by_take[video.take][video.camera] = video.path
        for take, cameras in by_take.items():
            row = {"experiment": exp, "trial": take, TYPE_COLUMN: CALIBRATION}
            for cam in project.cameras:
                row.update({cam: "", f"{cam}_start": pd.NA, f"{cam}_end": pd.NA, f"{cam}_mtime": ""})
            for cam, source in cameras.items():
                target = Path("experiments") / exp / "calibrations" / cam / source.name
                if target in calibration_files or (project.root / target).exists():
                    raise AddVideosError(f"Calibration destination already exists: {target}")
                calibration_files[target] = source
                # Measure the original now; use its copied local filename after staging.
                row[cam], row[f"{cam}_start"] = str(source), 0
            trials = pd.concat([trials, pd.DataFrame([row])], ignore_index=True)
    # Measure only new/extended rows; preserve existing ranges and timestamps.
    touched = sorted(set(changed_rows) | {existing_keys[c[:2]] for c in new_cells}
                     | set(range(len(project.trials_df) + len(changed_rows), len(trials))))
    measured = trials.loc[touched].copy().reset_index(drop=True)
    check = check_trial_videos(measured, project.root, roots, project.cameras, progress)
    if check.missing:
        raise AddVideosError("\n".join(check.missing))
    for _, row in measured.iterrows():
        for cam in project.cameras:
            if pd.notna(row[cam]) and row[cam]:
                end, start = row[f"{cam}_end"], row[f"{cam}_start"]
                if pd.isna(end) or end == "" or int(end) < int(start):
                    raise AddVideosError(f"Cannot read a valid frame range from {row[cam]}")
    for cam in project.cameras:
        for suffix in ("_end", "_mtime"):
            trials.loc[touched, cam + suffix] = measured[cam + suffix].to_numpy()
    for i in touched:
        if trials.at[i, TYPE_COLUMN] == CALIBRATION:
            for cam in project.cameras:
                value = trials.at[i, cam]
                if pd.notna(value) and value:
                    trials.at[i, cam] = Path(value).name
    candidate_project = copy.copy(project)
    candidate_project.config = config
    candidate_project.refresh_trials(trials)
    candidate_project.load_skeletons()
    problems = candidate_project.slp_write_problems()
    if problems:
        raise AddVideosError("\n".join(problems))
    incoming = SleapAdapter().load(
        None, trials, project.cameras, mapping=SlpMapping.from_cell_map(new_cells),
        root=project.root, video_roots=roots, progress=progress,
        video_aliases=config.get("slp_video_paths", {}),
    )
    results = candidate_project._merge_by_experiment(incoming, project.next_sync_id(), progress)
    if any(r.conflicts for r in results.values()):
        raise AddVideosError("The new videos conflict with existing labels. No additions were written.")
    backup = project.root / "import_backup" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with tempfile.TemporaryDirectory(prefix=".add-videos-", dir=project.root) as temp:
        stage = Path(temp)
        write_trials_csv(stage / "trials.csv", trials, backup=False)
        write_project_toml(stage / "project.toml", config)
        for relative, source in calibration_files.items():
            dest = stage / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            copy_file(str(source), str(dest))
        for exp, result in results.items():
            if result.has_changes or exp in new_exps:
                dest = stage / "experiments" / exp / POINTS_FILE
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_points_file(result.df, dest)
        if project.sync_history_path.exists():
            shutil.copyfile(project.sync_history_path, stage / project.sync_history_path.name)
        history_project = copy.copy(project)
        history_project.root = stage
        history_project._record_sync(results, False, "add")
        targets = sorted(p.relative_to(stage) for p in stage.rglob("*") if p.is_file())
        backup.mkdir(parents=True)
        existed = set()
        for relative in targets:
            dest = project.root / relative
            if dest.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dest, saved)
                existed.add(relative)
        (backup / "manifest.json").write_text(json.dumps({
            "added_videos": [str(v.path) for v in selected],
            "replaced_files": [str(p) for p in sorted(existed)],
            "created_files": [str(p) for p in targets if p not in existed],
        }, indent=2))
        # Reject concurrent edits even if they happened while labels were read.
        for path, stamp in discovery.fingerprints.items():
            if not path.exists() or _stamp(path) != stamp:
                raise AddVideosError(f"{path.name} changed while preparing additions; scan again.")
        for path, stamp in output_stamps.items():
            if (_stamp(path) if path.exists() else None) != stamp:
                raise AddVideosError(f"{path} changed while preparing additions; scan again.")
        committed = []
        try:
            for relative in targets:
                dest = project.root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                committed.append(relative)
                os.replace(stage / relative, dest)
        except Exception:
            for relative in reversed(committed):
                dest = project.root / relative
                if relative in existed:
                    shutil.copyfile(backup / relative, dest)
                else:
                    dest.unlink(missing_ok=True)
            raise
    return AdditionResult(
        config, trials, backup, len(changed_rows), len(selected), len(new_exps),
        sum(r.n_new + r.n_predictions for r in results.values()),
    )

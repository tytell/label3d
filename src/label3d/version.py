"""The running label3d version, and the project layout version it writes.

__version__ is the package version from pyproject.toml.  LAYOUT_VERSION is
the version of the project layout this code reads and writes; it changes only
when the layout does, so later releases can keep writing the same layout.

project.toml records both (Phase 20):

    # label3d 0.3.0 (b9b76dc)
    [label3d]
    project_version = "0.3.0"
    written_by = "0.3.0 (b9b76dc)"

project_version is written by setup and by upgrades only.  written_by and the
comment line are rewritten whenever label3d writes project.toml.
"""

from __future__ import annotations

import subprocess
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import toml

try:
    from label3d._build_info import COMMIT as BUILD_COMMIT
except ImportError:
    BUILD_COMMIT = "unknown"

try:
    __version__ = version("label3d")
except PackageNotFoundError:
    __version__ = "unknown"

LAYOUT_VERSION = "0.3.0"

VERSION_TABLE = "label3d"


@cache
def commit() -> str:
    """The short git hash of the running code, "+dirty" if tracked files changed.

    Read from git when the package runs from a checkout, otherwise from
    _build_info.py, which the build hook writes into a wheel.
    """
    package_dir = Path(__file__).resolve().parent
    try:
        top = _git(package_dir, "rev-parse", "--show-toplevel")
        # Only this repository's checkout counts, not one the package is installed inside
        if (Path(top) / "src" / "label3d").resolve() == package_dir:
            head = _git(package_dir, "rev-parse", "--short", "HEAD")
            dirty = _git(package_dir, "status", "--porcelain", "--untracked-files=no")
            return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return BUILD_COMMIT


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=5
    )
    return result.stdout.strip()


def version_string() -> str:
    """"0.3.0 (b9b76dc)"."""
    return f"{__version__} ({commit()})"


def parse_version(text: str) -> tuple[int, ...]:
    """"0.3.0" -> (0, 3, 0)."""
    return tuple(int(part) for part in str(text).split("."))


def project_version(config: dict) -> str | None:
    """The layout version project.toml records, or None for a project from before 0.3.0."""
    return config.get(VERSION_TABLE, {}).get("project_version")


def check_not_newer(config: dict) -> None:
    """Raise ValueError if the project's layout is newer than this code's."""
    recorded = project_version(config)
    if recorded is not None and parse_version(recorded) > parse_version(LAYOUT_VERSION):
        written_by = config.get(VERSION_TABLE, {}).get("written_by", recorded)
        raise ValueError(
            f"This project was made with label3d {written_by}, which is newer than "
            f"this label3d ({version_string()}). Update label3d to open it."
        )


def write_project_toml(path: Path, config: dict, new_project: bool = False) -> None:
    """Write project.toml with the version comment and [label3d] first.

    new_project also records LAYOUT_VERSION as the project's version; otherwise
    project_version is kept as it is, or left out for an old project.
    """
    table = dict(config.get(VERSION_TABLE, {}))
    if new_project:
        table["project_version"] = LAYOUT_VERSION
    table["written_by"] = version_string()
    ordered = {VERSION_TABLE: table}
    ordered.update((k, v) for k, v in config.items() if k != VERSION_TABLE)
    config.clear()
    config.update(ordered)
    with open(path, "w") as f:
        f.write(f"# label3d {version_string()}\n")
        toml.dump(config, f)

"""Named network drives, so [video_roots] works on Mac, Windows and Linux.

The same share is mounted at a different path on each system, so project.toml
names it once per system and the video roots refer to it by name:

    [drives.mac]
    tytelllab = "/Volumes/TytellLab$"

    [drives.windows]
    tytelllab = '\\\\rstore.it.tufts.edu\\TytellLab$'

    [video_roots]
    20260619 = "{tytelllab}/NewData/VerticalOMR/..."

A root without a {drive} prefix is a plain path, used as it is. A plain root
under a known drive is rewritten in {drive} form when the project loads
(convert_video_roots), so a project set up on one system opens on the other.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

DRIVES_TABLE = "drives"
PLATFORMS = ("mac", "windows", "linux")

# "{name}", "{name}/rest" or "{name}\rest"
_DRIVE_REF = re.compile(r"^\{([^{}]+)\}(?:[/\\](.*))?$")


class DriveError(FileNotFoundError):
    """A video root names a drive that has no root on this system."""


def current_platform() -> str:
    """The key of this system in [drives]: mac, windows or linux."""
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def _pure_path(platform: str) -> type[PurePath]:
    return PureWindowsPath if platform == "windows" else PurePosixPath


def _case_insensitive(platform: str) -> bool:
    # Mac and Windows file systems ignore case by default
    return platform != "linux"


def drive_roots(config: dict, platform: str) -> dict[str, str]:
    """The drives of one system in project.toml, name -> root."""
    table = config.get(DRIVES_TABLE, {}).get(platform, {})
    return {str(name): str(root) for name, root in table.items()}


def resolve_root(value: str, config: dict, platform: str | None = None) -> str:
    """A [video_roots] value as a path on this system.

    "{name}/rest" becomes the root of drive `name` in [drives.{platform}]
    followed by rest; any other value is returned unchanged. Raises
    DriveError if the drive has no root for this system.
    """
    platform = platform or current_platform()
    match = _DRIVE_REF.match(value)
    if match is None:
        return value
    name, rest = match.group(1), match.group(2) or ""
    roots = drive_roots(config, platform)
    if name not in roots:
        raise DriveError(
            f"Video root {value!r} uses drive {{{name}}}, which has no root for this "
            f"system. Add {name} = \"...\" to [{DRIVES_TABLE}.{platform}] in project.toml."
        )
    pure = _pure_path(platform)
    parts = [p for p in re.split(r"[/\\]", rest) if p]
    return str(pure(roots[name], *parts))


def _relative_parts(path: str, root: str, platform: str) -> list[str] | None:
    """Parts of path below root, or None if path is not under root."""
    pure = _pure_path(platform)
    path_parts = list(pure(path).parts)
    root_parts = list(pure(root).parts)
    if len(path_parts) < len(root_parts):
        return None
    head = path_parts[: len(root_parts)]
    if _case_insensitive(platform):
        head = [p.casefold() for p in head]
        root_parts = [p.casefold() for p in root_parts]
    if head != root_parts:
        return None
    return path_parts[len(root_parts):]


def to_drive_form(path: str, config: dict, platform: str | None = None) -> str | None:
    """A plain path in "{name}/rest" form, or None if it is under no known drive.

    The drives of this system are tried first, then those of the others, so a
    root written on Windows is converted on a Mac too. The longest matching
    root wins.
    """
    platform = platform or current_platform()
    if _DRIVE_REF.match(path):
        return None
    order = [platform] + [p for p in PLATFORMS if p != platform]
    for system in order:
        best: tuple[int, str, list[str]] | None = None
        for name, root in drive_roots(config, system).items():
            rest = _relative_parts(path, root, system)
            if rest is None:
                continue
            depth = len(_pure_path(system)(root).parts)
            if best is None or depth > best[0]:
                best = (depth, name, rest)
        if best is not None:
            _, name, rest = best
            return "/".join([f"{{{name}}}", *rest])
    return None


def convert_video_roots(config: dict, platform: str | None = None) -> bool:
    """Rewrite plain [video_roots] under a known drive in {drive} form (in place).

    Returns True if any root changed.
    """
    roots = config.get("video_roots", {})
    changed = False
    for exp, value in roots.items():
        converted = to_drive_form(str(value), config, platform)
        if converted is not None and converted != value:
            roots[exp] = converted
            changed = True
    return changed


def read_video_roots(config: dict, platform: str | None = None) -> dict[str, Path]:
    """[video_roots] of project.toml as paths on this system, experiment -> root."""
    return {
        str(exp): Path(resolve_root(str(value), config, platform)).expanduser()
        for exp, value in config.get("video_roots", {}).items()
    }


def unmounted_drives(config: dict, platform: str | None = None) -> list[str]:
    """Drives used by [video_roots] whose root does not exist on this system."""
    platform = platform or current_platform()
    roots = drive_roots(config, platform)
    used = {
        m.group(1)
        for value in config.get("video_roots", {}).values()
        if (m := _DRIVE_REF.match(str(value)))
    }
    return [
        f"{{{name}}} = {roots[name]}"
        for name in sorted(used)
        if name in roots and not Path(roots[name]).exists()
    ]

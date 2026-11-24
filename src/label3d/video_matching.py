"""Match SLEAP video references to registered videos without basename collisions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from label3d.slp_sync import Cell, video_cells


def path_parts(value: str | Path) -> tuple[str, ...]:
    """Components of either a Windows or POSIX reference, on either host."""
    parts = tuple(p for p in str(value).replace("\\", "/").split("/") if p not in ("", "."))
    return tuple(p.casefold() for p in parts) if sys.platform != "linux" else parts


def path_key(value: str | Path) -> str:
    return "/".join(path_parts(value))


class AmbiguousVideoError(ValueError):
    pass


class VideoMatcher:
    """Prefer full paths, then longest path suffix; require unique basename fallback.

    Several trials can deliberately reference different ranges of the same video.
    Those cells are returned together. Distinct files with an equally good match
    are an error, rather than importing one video's labels into several trials.
    """

    def __init__(self, trials: pd.DataFrame, cameras: list[str], root: Path | None = None,
                 video_roots: dict[str, Path] | None = None, aliases: dict[str, str] | None = None):
        self.cell_paths: dict[Cell, Path] = {}
        self._external: set[Cell] = set()
        self._cache: dict[str, list[Cell]] = {}
        self._aliases = {path_key(k): v for k, v in (aliases or {}).items()}
        roots = video_roots or {}
        for cell, (filename, _) in video_cells(trials, cameras).items():
            path = Path(filename)
            if path.is_absolute() or cell[0] in roots or len(path.parts) > 1:
                self._external.add(cell)
            if not path.is_absolute():
                base = roots.get(cell[0])
                if base is None and root is not None:
                    base = root / "experiments" / cell[0] / "videos"
                if base is not None:
                    path = base / path
            self.cell_paths[cell] = path
        self._by_name: dict[str, list[Cell]] = {}
        for cell, path in self.cell_paths.items():
            parts = path_parts(path)
            if parts:
                self._by_name.setdefault(parts[-1], []).append(cell)

    def matches(self, filename: str | Path) -> list[Cell]:
        key = str(filename)
        if key not in self._cache:
            self._cache[key] = self._matches(self._aliases.get(path_key(filename), filename))
        return list(self._cache[key])

    def _matches(self, filename: str | Path) -> list[Cell]:
        parts = path_parts(filename)
        if not parts:
            return []
        candidates = self._by_name.get(parts[-1], [])
        scored = []
        for cell in candidates:
            expected = path_parts(self.cell_paths[cell])
            score = 0
            for a, b in zip(reversed(parts), reversed(expected)):
                if a != b:
                    break
                score += 1
            # An exact path wins over a shorter suffix with the same score.
            scored.append(((score, parts == expected), cell))
        if not scored:
            return []
        best = max(score for score, _ in scored)
        found = [cell for score, cell in scored if score == best]
        if best[0] == 1 and len(parts) > 1:
            # A new sibling experiment can use exactly the same basenames as
            # an old one. Shared collection folders give evidence that these
            # references are different files, even across Windows/Mac mounts.
            # Do not import the new video's points into the old experiment.
            kept = []
            for cell in found:
                expected = path_parts(self.cell_paths[cell])
                different = False
                if cell in self._external:
                    for anchor in set(parts[:-1]) & set(expected[:-1]):
                        a = len(parts) - 1 - parts[::-1].index(anchor)
                        b = len(expected) - 1 - expected[::-1].index(anchor)
                        if parts[a + 1:] != expected[b + 1:]:
                            different = True
                            break
                    if not different and Path(filename).is_file() and self.cell_paths[cell].is_file():
                        different = Path(filename).resolve() != self.cell_paths[cell].resolve()
                if not different:
                    kept.append(cell)
            found = kept
        identities = {path_key(self.cell_paths[cell]) for cell in found}
        if len(identities) > 1:
            raise AmbiguousVideoError(
                f"Video '{filename}' matches several registered files: "
                + ", ".join(sorted(str(self.cell_paths[c]) for c in found))
                + ". Keep experiment folders in the SLEAP video paths to disambiguate them."
            )
        return found

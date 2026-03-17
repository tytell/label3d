"""Base class for pose format adapters (import/export)."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class PoseAdapter(ABC):
    """Convert between an external pose format and the canonical points DataFrame."""

    @abstractmethod
    def load(
        self,
        files: list[Path],
        trials: pd.DataFrame,
        cameras: list[str],
    ) -> pd.DataFrame:
        """Read external files and return a points-compatible DataFrame.

        Parameters
        ----------
        files : list[Path]
            Format-specific files (e.g. one .slp per network).
        trials : pd.DataFrame
            Contents of trials.csv — used to resolve file→camera→trial mapping.
        cameras : list[str]
            Canonical camera list from calibration.toml.

        Returns
        -------
        pd.DataFrame
            MultiIndex(trial, frame, instance, bodypart) rows,
            MultiIndex(view, coord) columns.  Missing points are NaN.
            Reprojected coords (xr, yr) and 3d columns are omitted.
        """

    @abstractmethod
    def save(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        cameras: list[str],
        **kwargs,
    ) -> None:
        """Write clicked 2D points to the external format."""

    @classmethod
    def name(cls) -> str:
        """Short identifier shown in the UI (e.g. 'sleap', 'deeplabcut')."""
        return cls.__name__

    @classmethod
    def file_filter(cls) -> str:
        """Qt file dialog filter string."""
        return "All files (*)"

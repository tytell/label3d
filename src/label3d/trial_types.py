"""What each row of trials.csv is: a trial, a calibration take or an axis take.

The type column holds "trial", "calibration" or "axis", in any case; a blank
cell is a trial. Calibration takes are charuco recordings; axis takes are
recordings of an object with marked axes, for axis alignment (Phase 16). Both
kinds of take keep their files in calibrations/{camera}/.

Before 0.4.0 trials.csv had an is_calibration column (True/False) instead;
the upgrade replaces it (layout.TRIAL_TYPES).
"""

from __future__ import annotations

import pandas as pd

TYPE_COLUMN = "type"
TRIAL = "trial"
CALIBRATION = "calibration"
AXIS = "axis"
ROW_TYPES = (TRIAL, CALIBRATION, AXIS)

LEGACY_CALIBRATION_COLUMN = "is_calibration"


def legacy_calibration_mask(df: pd.DataFrame) -> pd.Series:
    """The rows an is_calibration column marks as calibration takes."""
    return df[LEGACY_CALIBRATION_COLUMN].astype(str).str.lower().isin(("true", "1", "yes"))


def row_types(df: pd.DataFrame) -> pd.Series:
    """Each row's type, lower case. Blank cells, and a file with no type column, are trials.

    A file not yet upgraded is read from its is_calibration column. Raises
    ValueError for a type that isn't one of ROW_TYPES.
    """
    if TYPE_COLUMN in df.columns:
        types = df[TYPE_COLUMN].fillna("").astype(str).str.strip().str.lower()
        types = types.where(types != "", TRIAL)
        unknown = sorted(set(types) - set(ROW_TYPES))
        if unknown:
            raise ValueError(
                f"trials.csv has unknown row types: {', '.join(unknown)}. "
                f"The {TYPE_COLUMN} column takes {', '.join(ROW_TYPES)}."
            )
        return types
    if LEGACY_CALIBRATION_COLUMN in df.columns:
        return legacy_calibration_mask(df).map({True: CALIBRATION, False: TRIAL})
    return pd.Series(TRIAL, index=df.index)


def with_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """df with its is_calibration column replaced by a type column in the same place."""
    if LEGACY_CALIBRATION_COLUMN not in df.columns:
        return df
    out = df.copy()
    position = out.columns.get_loc(LEGACY_CALIBRATION_COLUMN)
    types = row_types(out)
    out = out.drop(columns=LEGACY_CALIBRATION_COLUMN)
    if TYPE_COLUMN in out.columns:
        out[TYPE_COLUMN] = types
    else:
        out.insert(position, TYPE_COLUMN, types)
    return out

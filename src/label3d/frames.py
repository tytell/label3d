"""Frame numbers as the user sees them.

Frames are 0-based everywhere in memory, in the working files and in exports
(as in SLEAP's .slp files).  Everything the user reads or types — the GUI and
the _start/_end columns of trials.csv — numbers them from FRAME_BASE.
"""

FRAME_BASE = 1


def shown_frame(frame: int) -> int:
    """The number the user sees for a 0-based frame."""
    return frame + FRAME_BASE


def stored_frame(shown: int) -> int:
    """The 0-based frame for a number the user typed or read."""
    return shown - FRAME_BASE

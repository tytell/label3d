"""Shared fixtures for the data layer test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import BODYPARTS, CAMERAS

# ---------------------------------------------------------------------------
# Qt application — must exist before any QObject/Signal/QTimer is created
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication instance required by PointsStore."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Paths to synthetic test project
# ---------------------------------------------------------------------------

TEST_PROJECT = Path(__file__).parent.parent / "test_project"


@pytest.fixture
def test_project_path() -> Path:
    return TEST_PROJECT


@pytest.fixture
def skeleton_json_path() -> Path:
    return TEST_PROJECT / "skeleton.json"


@pytest.fixture
def points_csv_path() -> Path:
    return TEST_PROJECT / "points.csv"


@pytest.fixture
def trials_csv_path() -> Path:
    return TEST_PROJECT / "videos" / "trials.csv"


@pytest.fixture
def slp_dir() -> Path:
    return TEST_PROJECT / "slp"


# ---------------------------------------------------------------------------
# Skeleton fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_skeleton():
    """Minimal 3-node skeleton for unit tests (no file I/O needed)."""
    from label3d.skeleton import Skeleton
    return Skeleton(
        nodes=["snout", "left_eye", "dorsal"],
        edges=[("snout", "left_eye"), ("snout", "dorsal")],
        symmetries=[],
    )


@pytest.fixture
def skeleton_with_spaces():
    """Skeleton whose node names contain spaces and special characters."""
    from label3d.skeleton import Skeleton
    return Skeleton(
        nodes=["left front paw", "right front paw", "eye (left)", "joint:elbow"],
        edges=[("left front paw", "right front paw")],
        symmetries=[("left front paw", "right front paw")],
    )


# ---------------------------------------------------------------------------
# PointsStore fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_store(qapp):
    """Empty PointsStore with the standard 3-camera, 3-bodypart structure."""
    from label3d.points import PointsStore
    return PointsStore.create_empty(CAMERAS, BODYPARTS)


@pytest.fixture
def populated_store(qapp, points_csv_path):
    """PointsStore loaded from the synthetic test project's points.csv."""
    from label3d.points import PointsStore
    return PointsStore.from_csv(points_csv_path)


@pytest.fixture
def store_with_one_point(qapp):
    """PointsStore with a single set point for easy assertions."""
    from label3d.points import PointsStore
    store = PointsStore.create_empty(CAMERAS, BODYPARTS)
    store.set_point("trial1", 0, 0, "snout", "cam1", 100.0, 200.0)
    return store


# ---------------------------------------------------------------------------
# Adapter / trials fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trials_df(trials_csv_path) -> pd.DataFrame:
    return pd.read_csv(trials_csv_path)


@pytest.fixture
def slp_files(slp_dir) -> list[Path]:
    return sorted(slp_dir.glob("cam[123]_trial1.slp"))

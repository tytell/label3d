"""label3d-export command line tool (Phase 13.3)."""

from __future__ import annotations

import shutil

import pandas as pd
import pytest
import toml

from label3d.export import EXPORT_COLUMNS
from label3d.export_cli import main
from tests.conftest import TEST_PROJECT, requires_test_project


@pytest.fixture
def project_copy(tmp_path, qapp):
    root = tmp_path / "proj"
    shutil.copytree(TEST_PROJECT, root)
    return root


@requires_test_project
class TestCli:
    def test_exports_synthetic_project(self, project_copy, capsys):
        out = project_copy / "out.csv"
        assert main([str(project_copy), "-o", str(out)]) == 0
        df = pd.read_csv(out)
        assert list(df.columns) == EXPORT_COLUMNS
        assert set(df["experiment"]) == {"exp1", "exp2"}
        assert "Wrote" in capsys.readouterr().out

    def test_experiments_option(self, project_copy):
        out = project_copy / "out.csv"
        assert main([str(project_copy), "-o", str(out), "--experiments", "exp2"]) == 0
        assert set(pd.read_csv(out)["experiment"]) == {"exp2"}

    def test_stale_experiment_fails(self, project_copy, capsys):
        config_path = project_copy / "project.toml"
        config = toml.load(config_path)
        config["calibration"]["triangulated_with"]["exp1"] = "something-else"
        config_path.write_text(toml.dumps(config))
        out = project_copy / "out.csv"
        assert main([str(project_copy), "-o", str(out)]) == 1
        assert "exp1" in capsys.readouterr().err
        assert not out.exists()

    def test_missing_project_fails(self, tmp_path):
        assert main([str(tmp_path / "nope")]) == 1

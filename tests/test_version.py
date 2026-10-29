"""Tests for the label3d version and its record in project.toml (Phase 20)."""

from __future__ import annotations

import shutil

import pytest
import toml

from label3d import version
from label3d.project import Project
from label3d.setup_project import write_project_toml as setup_project_toml
from label3d.version import (
    LAYOUT_VERSION,
    check_not_newer,
    parse_version,
    project_version,
    version_string,
    write_project_toml,
)
from tests.conftest import TEST_PROJECT, requires_test_project


def test_commit_from_this_checkout():
    """Run from the git checkout, the commit is a short hash."""
    commit = version.commit().removesuffix("+dirty")
    assert len(commit) >= 7 and all(c in "0123456789abcdef" for c in commit)


def test_version_string_names_the_package_version_and_commit():
    assert version_string() == f"{version.__version__} ({version.commit()})"


def test_parse_version_compares_numerically():
    assert parse_version("0.10.0") > parse_version("0.9.1")


def test_written_file_starts_with_the_version_comment(tmp_path):
    path = tmp_path / "project.toml"
    write_project_toml(path, {"project": {"name": "p"}})
    assert path.read_text().splitlines()[0] == f"# label3d {version_string()}"


def test_label3d_table_comes_first(tmp_path):
    path = tmp_path / "project.toml"
    write_project_toml(path, {"project": {"name": "p"}, "label3d": {"project_version": "0.3.0"}})
    assert list(toml.load(path)) == ["label3d", "project"]


def test_new_project_records_the_layout_version(tmp_path):
    path = tmp_path / "project.toml"
    write_project_toml(path, {"project": {"name": "p"}}, new_project=True)
    table = toml.load(path)["label3d"]
    assert table == {"project_version": LAYOUT_VERSION, "written_by": version_string()}


def test_old_project_gets_no_project_version(tmp_path):
    """Only setup and upgrades record project_version; a rewrite only stamps written_by."""
    path = tmp_path / "project.toml"
    write_project_toml(path, {"project": {"name": "p"}})
    config = toml.load(path)
    assert project_version(config) is None
    assert config["label3d"]["written_by"] == version_string()


def test_rewrite_keeps_the_recorded_project_version(tmp_path):
    path = tmp_path / "project.toml"
    config = {"label3d": {"project_version": "0.3.0", "written_by": "0.3.0 (0000000)"}}
    write_project_toml(path, config)
    assert toml.load(path)["label3d"] == {
        "project_version": "0.3.0", "written_by": version_string()
    }


def test_newer_project_is_refused():
    config = {"label3d": {"project_version": "99.0.0", "written_by": "99.0.0 (abcdef0)"}}
    with pytest.raises(ValueError, match="99.0.0 \\(abcdef0\\).*Update label3d"):
        check_not_newer(config)


@pytest.mark.parametrize("config", [{}, {"label3d": {"project_version": LAYOUT_VERSION}}])
def test_same_or_unversioned_project_is_accepted(config):
    check_not_newer(config)


def test_setup_records_the_layout_version(tmp_path):
    path = setup_project_toml(tmp_path, "test", None)
    assert path.read_text().startswith("# label3d ")
    assert project_version(toml.load(path)) == LAYOUT_VERSION


@requires_test_project
def test_load_refuses_a_newer_project(tmp_path):
    shutil.copytree(TEST_PROJECT, tmp_path / "p")
    config = toml.load(tmp_path / "p" / "project.toml")
    config["label3d"] = {"project_version": "99.0.0"}
    (tmp_path / "p" / "project.toml").write_text(toml.dumps(config))
    with pytest.raises(ValueError, match="newer than this label3d"):
        Project.load(tmp_path / "p")


@requires_test_project
def test_saving_config_keeps_the_comment(tmp_path):
    shutil.copytree(TEST_PROJECT, tmp_path / "p")
    proj = Project.load(tmp_path / "p")
    proj.save_config_sections({"video_roots": {}})
    text = (tmp_path / "p" / "project.toml").read_text()
    assert text.startswith(f"# label3d {version_string()}\n")
    assert toml.loads(text)["label3d"]["written_by"] == version_string()

"""The Align tab (Phase 16.6), driven through its methods on a synthetic rig."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import toml

from label3d import alignment as alignment_record
from label3d.alignment_picks import read_picks
from label3d.calibration_registry import save_calibration
from label3d.geometry import CameraParams, build_camera_group
from label3d.project import Project
from label3d.state import ProjectState
from label3d.widgets.axis_alignment_view import AxisAlignmentView, target_label
from label3d.widgets.main_window import ALIGN_TAB, LABEL_TAB, MainWindow
from tests.helpers import make_current
from tests.test_axis_alignment import O_TRUE, R_TRUE, look_at, to_cal

CAMERAS = ["cam1", "cam2", "cam3"]
ONE_POINT = {"origin": [0, 0, 0], "x1": [8.0, 0, 0], "y1": [0, 6.0, 0], "z1": [0, 0, 5.0]}


def rig() -> dict[str, CameraParams]:
    positions = {"cam1": [80.0, 0.0, 30.0], "cam2": [0.0, 80.0, 40.0], "cam3": [-60.0, -50.0, 50.0]}
    return {cam: look_at(cam, pos) for cam, pos in positions.items()}


def pixel(cameras, cam: str, p_obj) -> tuple[float, float]:
    x, y = cameras[cam].project(to_cal(p_obj)[np.newaxis])[0]
    return float(x), float(y)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    exp = root / "experiments" / "exp1"
    header = ["experiment", "trial", "type"]
    for cam in CAMERAS:
        header += [cam, f"{cam}_start", f"{cam}_end"]
    trial = ",".join(f"{cam}_t1.mp4,0," for cam in CAMERAS)
    axis = ",".join(f"Align_{cam}.png,0," for cam in CAMERAS)
    root.mkdir()
    (root / "trials.csv").write_text(
        ",".join(header) + f"\nexp1,trial1,trial,{trial}\nexp1,Align,axis,{axis}\n"
    )
    for cam in CAMERAS:
        path = exp / "videos" / f"{cam}_t1.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(3):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        still = exp / "calibrations" / cam / f"Align_{cam}.png"
        still.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(still), np.zeros((96, 128, 3), dtype=np.uint8))
    entry = save_calibration(exp, build_camera_group(rig()), "charuco", "charuco")
    (root / "project.toml").write_text(toml.dumps({"calibration": {"active": {"exp1": entry.id}}}))
    return make_current(root)


@pytest.fixture
def project(project_root, qapp) -> Project:
    return Project.load(project_root)


@pytest.fixture
def view(project) -> AxisAlignmentView:
    view = AxisAlignmentView(project)
    view.reload_for_experiment()
    return view


def click_all(view, cameras, points):
    """Click each target in every camera, in the order the view asks for them."""
    while view.active_target() is not None:
        target = view.active_target()
        for cam in CAMERAS:
            view.add_pick(cam, *pixel(cameras, cam, points[target]))


class TestPicking:
    def test_starts_on_the_origin(self, view):
        assert view._take_combo.currentText() == "Align"
        assert view.active_target() == "origin"
        assert view._frame_spin.isHidden()   # a 1-frame take

    def test_moves_on_once_every_camera_has_a_click(self, view):
        view.add_pick("cam1", 10.0, 20.0)
        view.add_pick("cam2", 11.0, 21.0)
        assert view.active_target() == "origin"
        view.add_pick("cam3", 12.0, 22.0)
        assert view.active_target() == "x1"

    def test_clicks_are_saved(self, view, project_root):
        view.add_pick("cam1", 10.0, 20.0)
        exp = project_root / "experiments" / "exp1"
        assert read_picks(exp, "Align", 0) == {"origin": {"cam1": (10.0, 20.0)}}
        again = AxisAlignmentView(Project.load(project_root))
        again.reload_for_experiment()
        assert again.picks() == {"origin": {"cam1": (10.0, 20.0)}}

    def test_right_click_removes_the_nearest(self, view):
        view.add_pick("cam1", 10.0, 20.0)
        view.remove_pick_near("cam1", 500.0, 500.0)    # nowhere near
        assert view.picks() == {"origin": {"cam1": (10.0, 20.0)}}
        view.remove_pick_near("cam1", 11.0, 21.0)
        assert view.picks() == {}

    def test_escape_stops_clicking(self, view):
        view.set_active_target(None)
        view.add_pick("cam1", 1.0, 1.0)
        assert view.picks() == {}
        assert not view._grid.get_panel("cam1")._pick_mode

    def test_clear(self, view):
        view.add_pick("cam1", 1.0, 1.0)
        view.clear_picks()
        assert view.picks() == {}
        assert view.active_target() == "origin"

    def test_table_follows_settings(self, view, project_root):
        view._mode.setCurrentIndex(view._mode.findData(2))
        labels = [view._table.item(r, 0).text() for r in range(view._table.rowCount())]
        assert labels == ["Origin", "X near", "X far", "Y near", "Y far", "Z near", "Z far"]
        view._third.setCurrentIndex(view._third.findData("z"))
        assert view._table.rowCount() == 5
        settings = toml.load(project_root / "project.toml")["alignment"]["exp1"]["settings"]
        assert settings["points_per_axis"] == 2 and settings["cross_product"] == "z"

    def test_derived_axis_cannot_stay_primary(self, view):
        view._primary.setCurrentIndex(view._primary.findData("y"))
        view._third.setCurrentIndex(view._third.findData("y"))
        assert view._primary.currentData() == "longest"

    def test_target_labels(self):
        assert target_label("origin", 1) == "Origin"
        assert target_label("y1", 1) == "Y"
        assert target_label("z2", 2) == "Z far"


class TestComputeApply:
    def test_compute_and_apply(self, view, project):
        click_all(view, rig(), ONE_POINT)
        computation = view.compute()
        np.testing.assert_allclose(computation.result.R_align, R_TRUE, atol=1e-5)
        np.testing.assert_allclose(computation.result.origin, O_TRUE, atol=1e-4)
        assert "R_align" in view._results.toPlainText()
        assert view._table.item(0, 2).text() != ""

        applied = []
        view.alignment_applied.connect(lambda: applied.append(True))
        assert view.apply()
        assert applied
        alignment = project.alignment()
        assert alignment.calibration == project.active_calibration_id()
        assert project.alignment_source().take == "Align"
        assert project.alignment_status() == alignment_record.CURRENT
        assert "Alignment in use" in view._status.text()

    def test_compute_names_missing_points(self, view):
        view.add_pick("cam1", 1.0, 1.0)
        assert view.compute() is None
        assert "origin" in view._results.toPlainText()
        assert not view.apply()

    def test_second_apply_reports_the_change(self, view, project):
        click_all(view, rig(), ONE_POINT)
        view.apply()
        view.compute()
        assert "0.00° rotation" in view._results.toPlainText()

    def test_scale_warning(self, view):
        click_all(view, rig(), ONE_POINT)
        view._lengths["x"].setValue(10.0)      # clicked distance is 8 cm
        view.compute()
        assert "(-20.0%)  !" in view._results.toPlainText()


def test_main_window_has_an_align_tab(project, qapp):
    window = MainWindow(project, ProjectState())
    assert [window._tabs.tabText(i) for i in range(window._tabs.count())] == ["Calibrate", "Align", "Label"]
    assert window._tabs.currentIndex() == LABEL_TAB
    assert window._tabs.isTabEnabled(ALIGN_TAB)
    window._tabs.setCurrentIndex(ALIGN_TAB)
    assert window._stack.currentWidget() is window._align_view
    assert window._frame_controls.isHidden()
    window.deleteLater()

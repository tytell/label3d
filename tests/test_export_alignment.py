"""Aligned 3D output (Phase 16.7): export, the CLI and the export dialog."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from label3d import alignment as alignment_record
from label3d.alignment import Alignment, AlignmentSource
from label3d.export import read_alignment_metadata, stale_alignments
from label3d.export_cli import main
from label3d.widgets.export_dialog import ExportDialog
from tests.test_export import KEYS, _export, calibrated  # noqa: F401  (calibrated is a fixture)
from tests.test_project_sync import EXP, project  # noqa: F401  (project is a fixture)

# 90° about z, then a shift: the new x is the calibration's y
R_ALIGN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
ORIGIN = np.array([5.0, -3.0, 90.0])
SOURCE = AlignmentSource(take="Align", type="axis", frame=0)


def _align(project, calibration: str | None = None) -> Alignment:
    alignment = Alignment(
        R_align=R_ALIGN, origin=ORIGIN,
        calibration=calibration or project.active_calibration_id(EXP),
        created=datetime(2026, 11, 4, 12, 0),
    )
    project.save_alignment(alignment, SOURCE, EXP)
    return alignment


def test_rig_rotation_is_proper():
    assert np.linalg.det(R_ALIGN) == pytest.approx(1.0)


class TestAlignedExport:
    @pytest.mark.parametrize("layout", ["wide", "long"])
    def test_3d_is_aligned_and_errors_unchanged(self, calibrated, tmp_path, layout):
        _r, plain = _export(calibrated, tmp_path, layout=layout, format="parquet")
        alignment = _align(calibrated)
        result, aligned = _export(calibrated, tmp_path, layout=layout, format="parquet")
        assert result.aligned == [EXP]

        xyz = ["x.cm", "y.cm", "z.cm"] if layout == "wide" else ["x", "y", "z"]
        rows = plain[xyz].notna().all(axis=1)
        assert rows.any()
        expected = alignment.align(plain.loc[rows, xyz].to_numpy())
        np.testing.assert_allclose(aligned.loc[rows, xyz].to_numpy(), expected, atol=1e-9)
        # Everything else is the same, reprojection errors included
        others = [c for c in plain.columns if c not in xyz]
        pd.testing.assert_frame_equal(aligned[others], plain[others])

    def test_parquet_metadata_holds_the_transform(self, calibrated, tmp_path):
        alignment = _align(calibrated)
        result, _out = _export(calibrated, tmp_path, format="parquet")
        meta = read_alignment_metadata(result.path)
        assert set(meta) == {EXP}
        np.testing.assert_allclose(meta[EXP]["R"], R_ALIGN)
        np.testing.assert_allclose(meta[EXP]["origin"], ORIGIN)
        assert meta[EXP]["calibration"] == alignment.calibration

    def test_no_alignment_no_metadata(self, calibrated, tmp_path):
        result, _out = _export(calibrated, tmp_path, format="parquet")
        assert read_alignment_metadata(result.path) == {}
        assert result.aligned == []


class TestStaleAlignment:
    def test_refused_unless_2d_only(self, calibrated, tmp_path):
        _align(calibrated, calibration="another-calibration")
        assert calibrated.alignment_status(EXP) == alignment_record.STALE
        assert stale_alignments(calibrated) == [EXP]
        with pytest.raises(ValueError, match="2D only"):
            _export(calibrated, tmp_path)

        result, out = _export(calibrated, tmp_path, two_d_only=[EXP])
        assert result.no_3d == [EXP] and result.aligned == []
        assert out[["x.cm", "y.cm", "z.cm", "n_cameras", "calibration"]].isna().all().all()
        assert out[[c for c in out.columns if c.endswith("_x.px")]].notna().all().all()

    def test_current_alignment_is_not_stale(self, calibrated):
        _align(calibrated)
        assert stale_alignments(calibrated) == []

    def test_cli(self, calibrated, tmp_path, capsys):
        _align(calibrated, calibration="another-calibration")
        calibrated.points.save_if_changed()
        out = tmp_path / "cli.parquet"
        assert main([str(calibrated.root), "-o", str(out), "--no-slp"]) == 1
        err = capsys.readouterr().err
        assert EXP in err and "--no-3d" in err
        assert not out.exists()

        assert main([str(calibrated.root), "-o", str(out), "--no-slp", "--no-3d"]) == 0
        df = pd.read_parquet(out)
        assert df["x.cm"].isna().all()
        assert not df.duplicated(KEYS).any()

    def test_dialog(self, calibrated, qapp, monkeypatch):
        _align(calibrated, calibration="another-calibration")
        dialog = ExportDialog(calibrated)
        monkeypatch.setattr(dialog, "_ask_stale_alignment", lambda stale: ExportDialog.TWO_D_ONLY)
        dialog.accept()
        assert dialog.result() == ExportDialog.DialogCode.Accepted
        assert dialog.options().two_d_only == [EXP]

        dialog = ExportDialog(calibrated)
        monkeypatch.setattr(dialog, "_ask_stale_alignment", lambda stale: ExportDialog.RECOMPUTE)
        dialog.accept()
        assert dialog.result() == ExportDialog.DialogCode.Rejected
        assert dialog.recompute_requested == EXP

        dialog = ExportDialog(calibrated)
        monkeypatch.setattr(dialog, "_ask_stale_alignment", lambda stale: ExportDialog.CANCEL)
        dialog.accept()
        assert dialog.recompute_requested is None
        assert dialog.options().two_d_only == []

"""Tests for the Label tab's point legend."""

from __future__ import annotations

from PySide6.QtGui import QColor

from label3d.widgets.point_item import COLOR_GOOD, ColorMode
from label3d.widgets.point_legend import ICON_SIZE, PointLegend, point_icon


def test_error_mode_shows_error_colors(qapp):
    legend = PointLegend()
    entries = legend.visible_entries()
    assert "Error < 5 px" in entries
    assert "Labeled in SLEAP" not in entries
    assert "New since last sync" in entries and "User label" in entries


def test_source_mode_swaps_color_section(qapp):
    legend = PointLegend()
    legend.set_color_mode(ColorMode.SOURCE)
    entries = legend.visible_entries()
    assert {"Labeled in label3d", "Labeled in SLEAP", "Predicted by SLEAP"} <= set(entries)
    assert "Error < 5 px" not in entries
    legend.set_color_mode(ColorMode.ERROR)
    assert "Error < 5 px" in legend.visible_entries()


def test_point_icon_uses_marker_color(qapp):
    center = point_icon("set", COLOR_GOOD).toImage().pixelColor(ICON_SIZE // 2, ICON_SIZE // 2)
    assert center == QColor(COLOR_GOOD)
    ring = point_icon("predicted", COLOR_GOOD).toImage().pixelColor(ICON_SIZE // 2, ICON_SIZE // 2)
    assert ring != QColor(COLOR_GOOD)  # a ring is hollow

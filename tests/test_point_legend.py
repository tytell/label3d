"""Tests for the Label tab's point legend."""

from __future__ import annotations

from PySide6.QtGui import QColor

from label3d.widgets.point_item import COLOR_GOOD, MARK_COLORS, ColorMode
from label3d.widgets.point_legend import ICON_SIZE, PointLegend, point_icon


def test_error_mode_shows_error_colors(qapp):
    legend = PointLegend()
    entries = legend.visible_entries()
    assert "Error < 5 px" in entries
    assert "Labeled in SLEAP" not in entries
    assert "New since last sync" in entries and "User label" in entries


def test_source_mode_swaps_color_section(qapp):
    legend = PointLegend()
    legend.set_coloring(ColorMode.SOURCE, "light_blue")
    entries = legend.visible_entries()
    assert {"Labeled in label3d", "Labeled in SLEAP", "Predicted by SLEAP"} <= set(entries)
    assert "Error < 5 px" not in entries
    legend.set_coloring(ColorMode.ERROR, "light_blue")
    assert "Error < 5 px" in legend.visible_entries()


def test_point_icon_uses_marker_color(qapp):
    center = point_icon("set", COLOR_GOOD).toImage().pixelColor(ICON_SIZE // 2, ICON_SIZE // 2)
    assert center == QColor(COLOR_GOOD)
    ring = point_icon("predicted", COLOR_GOOD).toImage().pixelColor(ICON_SIZE // 2, ICON_SIZE // 2)
    assert ring != QColor(COLOR_GOOD)  # a ring is hollow


def test_score_mode_shows_score_gradient(qapp):
    legend = PointLegend()
    legend.set_coloring(ColorMode.SCORE, "light_blue")
    entries = legend.visible_entries()
    assert {"Score 1.00", "Score 0.50", "Score 0.00", "No score (user label)"} <= set(entries)
    assert "Error < 5 px" not in entries


def test_fixed_color_shows_one_swatch_and_drop_down_follows(qapp):
    legend = PointLegend()
    legend.set_coloring(ColorMode.FIXED, "yellow")
    entries = legend.visible_entries()
    assert "All points" in entries
    assert "Error < 5 px" not in entries
    assert legend._color_combo.currentText() == "Bright yellow"
    icon = legend._icons[ColorMode.FIXED.value][-1].pixmap().toImage()
    assert icon.pixelColor(ICON_SIZE // 2, ICON_SIZE // 2) == MARK_COLORS["yellow"][1]


def test_drop_down_emits_choice(qapp):
    legend = PointLegend()
    chosen = []
    legend.coloring_chosen.connect(lambda mode, key: chosen.append((mode, key)))
    index = legend._color_combo.findText("Black")
    legend._color_combo.activated.emit(index)
    index = legend._color_combo.findText("Source")
    legend._color_combo.activated.emit(index)
    assert chosen == [("fixed", "black"), ("source", "")]

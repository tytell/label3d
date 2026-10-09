"""Refine dialog: adjust a calibration with labelled points (Phase 12).

Opened from the Calibration tab with "Refine with points...". The user picks
what to start from (the charuco takes, or an existing charuco or imported
entry), which points to use and the options, runs the refinement in a
RefineWorker, then saves the result to the registry or discards it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from label3d.calibration_refine import (
    PointSelection,
    RefineOptions,
    count_points,
    gather_points,
    matching_charuco_entry,
    save_refinement,
)
from label3d.calibration_registry import CalibrationEntry, clean_label, save_calibration
from label3d.calibration_solve import CalibrationTake, charuco_metadata
from label3d.points import SOURCE_LABEL3D, SOURCE_SLEAP_PRED, SOURCE_SLEAP_USER
from label3d.project import Project
from label3d.workers.calibration_worker import RefineOutcome, RefineWorker

# Methods an entry can have to be a refinement's starting point.
START_METHODS = ("charuco", "imported")

# (key in RefineResult.errors, row label); the "_before" key is the Stage 1 value.
ERROR_ROWS = [
    ("holdout", "Held-out points, median"),
    ("holdout_p90", "Held-out points, 90th pct"),
    ("points", "Fitted points, median"),
    ("charuco", "Charuco corners, median"),
]


def _entry_text(entry: CalibrationEntry) -> str:
    return f"{entry.label} · {entry.created:%Y-%m-%d %H:%M}"


def _checked(widget: QListWidget) -> list[str]:
    return [
        widget.item(i).text() for i in range(widget.count())
        if widget.item(i).checkState() == Qt.CheckState.Checked
    ]


def _check_list(names: list[str]) -> QListWidget:
    widget = QListWidget()
    widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    widget.setMaximumHeight(110)
    for name in names:
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        widget.addItem(item)
    return widget


class RefineDialog(QDialog):
    """Set up, run and save one refinement of the active experiment's calibration.

    After exec(), `saved_entry` is the refine entry written, or None.
    """

    def __init__(
        self,
        project: Project,
        takes: list[CalibrationTake],
        board_params: dict,
        frame_skip: int,
        n_stage1_repeats: int,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Refine calibration with points — {project.active_experiment}")
        self._project = project
        self._exp_dir = project.root / "experiments" / project.active_experiment
        self._cams = list(project.cameras)
        self._takes = takes
        self._board_params = board_params
        self._frame_skip = frame_skip
        self._n_stage1_repeats = n_stage1_repeats
        self._entries = project.calibrations()
        self._points_df = project.experiment_points_df(project.active_experiment).copy()

        self._worker: RefineWorker | None = None
        self._outcome: RefineOutcome | None = None
        self._run_selection: PointSelection | None = None
        self._run_options: RefineOptions | None = None
        self.saved_entry: CalibrationEntry | None = None

        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(200)
        self._count_timer.timeout.connect(self._update_counts)

        self._build_ui()
        self._update_start_controls()
        self._on_selection_changed()
        self._update_counts()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, 1)
        layout.addLayout(right, 1)

        # Start from: the charuco takes, or an existing entry
        start_group = QGroupBox("Start from")
        start_layout = QVBoxLayout(start_group)
        self._takes_radio = QRadioButton("Calibrate charuco takes")
        self._entry_radio = QRadioButton("Existing calibration")
        self._take_list = _check_list([t.name for t in self._takes])
        self._take_list.itemChanged.connect(self._update_start_controls)
        self._reuse_check = QCheckBox()
        self._reuse_check.setChecked(True)
        self._entry_combo = QComboBox()
        starts = [e for e in self._entries if e.method in START_METHODS]
        for entry in starts:
            self._entry_combo.addItem(_entry_text(entry), entry.id)
        active = self._project.active_calibration_id()
        if active is not None and self._entry_combo.findData(active) >= 0:
            self._entry_combo.setCurrentIndex(self._entry_combo.findData(active))
        self._takes_radio.setEnabled(bool(self._takes))
        self._entry_radio.setEnabled(bool(starts))
        (self._takes_radio if self._takes else self._entry_radio).setChecked(True)
        self._takes_radio.toggled.connect(self._update_start_controls)
        for w in (self._takes_radio, self._take_list, self._reuse_check,
                  self._entry_radio, self._entry_combo):
            start_layout.addWidget(w)
        left.addWidget(start_group)

        # Points
        points_group = QGroupBox("Points")
        points_form = QFormLayout(points_group)
        self._source_checks = {
            SOURCE_LABEL3D: QCheckBox("label3d user labels"),
            SOURCE_SLEAP_USER: QCheckBox("SLEAP user labels"),
            SOURCE_SLEAP_PRED: QCheckBox("SLEAP predictions"),
        }
        self._source_checks[SOURCE_LABEL3D].setChecked(True)
        self._source_checks[SOURCE_SLEAP_USER].setChecked(True)
        sources_row = QVBoxLayout()
        for check in self._source_checks.values():
            check.toggled.connect(self._on_selection_changed)
            sources_row.addWidget(check)
        points_form.addRow("Sources:", sources_row)
        self._score_spin = QDoubleSpinBox()
        self._score_spin.setRange(0.0, 1.0)
        self._score_spin.setSingleStep(0.05)
        self._score_spin.setValue(0.8)
        self._score_spin.valueChanged.connect(self._on_selection_changed)
        points_form.addRow("Prediction score ≥", self._score_spin)
        trials = sorted(self._points_df.index.get_level_values("trial").unique().astype(str))
        self._trial_list = _check_list(trials)
        self._trial_list.itemChanged.connect(self._on_selection_changed)
        points_form.addRow("Trials:", self._trial_list)
        left.addWidget(points_group)

        # Options
        options_group = QGroupBox("Options")
        options_form = QFormLayout(options_group)
        self._extrinsics_check = QCheckBox("Extrinsics only")
        self._extrinsics_check.setChecked(True)
        options_form.addRow(self._extrinsics_check)
        self._holdout_spin = QDoubleSpinBox()
        self._holdout_spin.setRange(0.0, 0.5)
        self._holdout_spin.setSingleStep(0.05)
        self._holdout_spin.setValue(0.2)
        self._holdout_spin.setToolTip("Fraction of frames kept out of the fit to measure the error")
        options_form.addRow("Hold-out fraction:", self._holdout_spin)
        self._repeats_spin = QSpinBox()
        self._repeats_spin.setRange(1, 10)
        self._repeats_spin.setValue(3)
        options_form.addRow("Repeats:", self._repeats_spin)
        self._label_edit = QLineEdit("refine")
        options_form.addRow("Label:", self._label_edit)
        left.addWidget(options_group)

        # Usable points per camera pair
        counts_group = QGroupBox("Usable points")
        counts_layout = QVBoxLayout(counts_group)
        self._counts_label = QLabel()
        self._counts_table = QTableWidget(0, 2)
        self._counts_table.setHorizontalHeaderLabels(["Camera pair", "Points"])
        self._counts_table.verticalHeader().setVisible(False)
        self._counts_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._counts_table.setMaximumHeight(130)
        counts_layout.addWidget(self._counts_label)
        counts_layout.addWidget(self._counts_table)
        right.addWidget(counts_group)

        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._on_run)
        right.addWidget(self._run_btn)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(100)
        right.addWidget(self._log)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self._errors_table = QTableWidget(len(ERROR_ROWS), 2)
        self._errors_table.setHorizontalHeaderLabels(["Before px", "After px"])
        self._errors_table.setVerticalHeaderLabels([label for _, label in ERROR_ROWS])
        self._errors_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._errors_table.setMaximumHeight(150)
        self._drift_table = QTableWidget(0, 3)
        self._drift_table.setHorizontalHeaderLabels(["Camera pair", "Rotation °", "Baseline %"])
        self._drift_table.verticalHeader().setVisible(False)
        self._drift_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._drift_table.setMaximumHeight(130)
        self._scale_label = QLabel()
        results_layout.addWidget(self._errors_table)
        results_layout.addWidget(QLabel("Change from the starting calibration:"))
        results_layout.addWidget(self._drift_table)
        results_layout.addWidget(self._scale_label)
        right.addWidget(results_group)

        buttons = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_active_btn = QPushButton("Save and make active")
        self._close_btn = QPushButton("Discard")
        self._save_btn.clicked.connect(lambda: self._on_save(make_active=False))
        self._save_active_btn.clicked.connect(lambda: self._on_save(make_active=True))
        self._close_btn.clicked.connect(self.reject)
        for btn in (self._save_btn, self._save_active_btn):
            btn.setEnabled(False)
        buttons.addStretch()
        for btn in (self._save_btn, self._save_active_btn, self._close_btn):
            buttons.addWidget(btn)
        right.addLayout(buttons)

    # ── Inputs ────────────────────────────────────────────────────────────

    def _selection(self) -> PointSelection:
        return PointSelection(
            sources={s for s, check in self._source_checks.items() if check.isChecked()},
            score_min=self._score_spin.value(),
            trials=_checked(self._trial_list),
        )

    def _options(self) -> RefineOptions:
        return RefineOptions(
            only_extrinsics=self._extrinsics_check.isChecked(),
            holdout=self._holdout_spin.value(),
            n_repeats=self._repeats_spin.value(),
        )

    def _start_entry(self) -> CalibrationEntry | None:
        """The registry entry to start from, or None to calibrate the takes."""
        if self._takes_radio.isChecked():
            return self._matching_entry() if self._reuse_check.isChecked() else None
        entry_id = self._entry_combo.currentData()
        return next((e for e in self._entries if e.id == entry_id), None)

    def _matching_entry(self) -> CalibrationEntry | None:
        names = _checked(self._take_list)
        if not names:
            return None
        return matching_charuco_entry(self._entries, names, self._takes, self._exp_dir, self._cams)

    def _update_start_controls(self) -> None:
        use_takes = self._takes_radio.isChecked()
        self._take_list.setEnabled(use_takes)
        self._entry_combo.setEnabled(not use_takes)
        match = self._matching_entry() if use_takes else None
        self._reuse_check.setVisible(match is not None)
        if match is not None:
            self._reuse_check.setText(f"Reuse {match.label} (same takes and corners)")

    def _on_selection_changed(self) -> None:
        self._score_spin.setEnabled(self._source_checks[SOURCE_SLEAP_PRED].isChecked())
        self._count_timer.start()

    def _update_counts(self) -> None:
        sel = self._selection()
        counts = count_points(self._points_df, self._cams, sel)
        _, index = gather_points(self._points_df, self._cams, sel)
        n_frames = len({(t, f) for t, f, *_ in index})
        self._counts_label.setText(f"{len(index)} points in {n_frames} frames")
        self._counts_table.setRowCount(len(counts))
        for row, ((a, b), n) in enumerate(counts.items()):
            self._counts_table.setItem(row, 0, QTableWidgetItem(f"{a} – {b}"))
            self._counts_table.setItem(row, 1, QTableWidgetItem(str(n)))

    # ── Running ───────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        start = self._start_entry()
        names = _checked(self._take_list)
        if start is None and not (self._takes_radio.isChecked() and names):
            self._log_msg("Choose charuco takes or an existing calibration to start from.")
            return
        sel = self._selection()
        p2ds, index = gather_points(self._points_df, self._cams, sel)
        if p2ds.shape[1] == 0:
            self._log_msg("No points match the selection.")
            return

        self._run_selection, self._run_options = sel, self._options()
        self._outcome = None
        for btn in (self._save_btn, self._save_active_btn):
            btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._close_btn.setEnabled(False)
        self._log.clear()

        self._worker = RefineWorker(
            self._exp_dir, self._cams, p2ds, index, self._run_options,
            start_entry=start, takes=self._takes, take_names=names,
            board_params=self._board_params, n_stage1_repeats=self._n_stage1_repeats,
            parent=self,
        )
        self._worker.progress.connect(self._log_msg)
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, outcome: RefineOutcome) -> None:
        self._outcome = outcome
        self._run_btn.setEnabled(True)
        self._close_btn.setEnabled(True)
        for btn in (self._save_btn, self._save_active_btn):
            btn.setEnabled(True)
        self._log_msg("Done.")
        self._show_results(outcome)

    def _on_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._close_btn.setEnabled(True)
        self._log_msg(f"Error: {message}")

    def _show_results(self, outcome: RefineOutcome) -> None:
        errors = outcome.result.errors
        for row, (key, _) in enumerate(ERROR_ROWS):
            for col, k in enumerate((f"{key}_before", key)):
                value = errors.get(k)
                text = f"{value:.3f}" if value is not None else "—"
                self._errors_table.setItem(row, col, QTableWidgetItem(text))

        drift = outcome.result.drift
        self._drift_table.setRowCount(len(drift))
        for row, d in enumerate(drift):
            self._drift_table.setItem(row, 0, QTableWidgetItem(f"{d.cam_a} – {d.cam_b}"))
            self._drift_table.setItem(row, 1, QTableWidgetItem(f"{d.rotation_deg:.3f}"))
            self._drift_table.setItem(row, 2, QTableWidgetItem(f"{d.baseline_pct:+.2f}"))

        scale = outcome.result.scale
        text = f"Scale correction: {scale:.4f}"
        if abs(scale - 1.0) > 0.02:
            text += " — large: the points may disagree with the charuco scale"
        spread = outcome.result.spread
        text += (
            f"\nRun-to-run spread: {spread[0]:.3f}° rotation, {spread[1]:.2f}% baseline"
        )
        self._scale_label.setText(text)

    # ── Saving ────────────────────────────────────────────────────────────

    def _stage1_label(self, names: list[str]) -> str:
        if sorted(names) == sorted(t.name for t in self._takes):
            return "charuco-all"
        return clean_label("charuco-" + "-".join(names))

    def _on_save(self, make_active: bool) -> None:
        outcome = self._outcome
        if outcome is None:
            return
        parent = outcome.parent
        try:
            if parent is None:
                names = outcome.stage1_takes
                parent = save_calibration(
                    self._exp_dir, outcome.stage1.cgroup, self._stage1_label(names), "charuco",
                    **charuco_metadata(
                        outcome.stage1, names, self._takes, self._exp_dir, self._cams,
                        self._board_params, self._frame_skip,
                    ),
                )
                self._log_msg(f"Saved the charuco calibration as {parent.id}")
            entry = save_refinement(
                self._exp_dir, outcome.result, parent, self._label_edit.text(),
                self._run_selection, self._run_options, self._points_df,
            )
        except Exception as exc:
            self._log_msg(f"Could not save: {exc}")
            return
        if make_active:
            self._project.set_active_calibration(None, entry.id)
        self.saved_entry = entry
        self.accept()

    def _log_msg(self, text: str) -> None:
        self._log.appendPlainText(text)

    def reject(self) -> None:
        # Bundle adjustment can't be interrupted, so the dialog stays until it ends.
        if self._worker is not None and self._worker.isRunning():
            self._log_msg("The refinement is still running.")
            return
        super().reject()

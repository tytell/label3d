"""Review additional SLEAP videos and select each new experiment's calibrations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from label3d.add_videos import (
    CalibrationVideo, DiscoveryOptions, apply_additions, calibration_candidates,
    candidate_problems, discover_videos, incomplete_trials,
)
from label3d.setup_project import VIDEO_EXTENSIONS
from label3d.workers.add_videos_worker import AddVideosWorker


def _check_item(checked=True):
    item = QTableWidgetItem()
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    return item


def _readonly(text):
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


def _camera_combo(cameras, selected):
    combo = QComboBox()
    combo.addItems(["", *cameras])
    combo.setCurrentText(selected)
    return combo


class CalibrationVideosDialog(QDialog):
    """Explicit selection, camera pairing and take names for one experiment."""

    def __init__(self, project, experiment, regex, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Calibration videos — {experiment}")
        self.resize(900, 420)
        self.project, self.regex = project, regex
        self.videos = []
        self.selection = []
        layout = QVBoxLayout(self)
        label = QLabel(
            f"Choose the calibration recordings for {experiment}. Review their take names "
            "and cameras below, or add calibration later."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        row = QHBoxLayout()
        files = QPushButton("Choose files…")
        folder = QPushButton("Choose folder…")
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        files.clicked.connect(self._files)
        folder.clicked.connect(self._folder)
        row.addWidget(files)
        row.addWidget(folder)
        row.addWidget(self.recursive)
        row.addStretch()
        layout.addLayout(row)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Include", "Take", "Camera", "Video"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.note = QLabel("Calibration videos are copied into this experiment's project folder.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        use = buttons.addButton("Use selected recordings", QDialogButtonBox.ButtonRole.AcceptRole)
        later = buttons.addButton("Add calibration later", QDialogButtonBox.ButtonRole.ActionRole)
        use.clicked.connect(self._use)
        later.clicked.connect(self._later)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose calibration videos", "", "Videos (*.mp4 *.avi *.mov *.mkv *.cine)")
        if files:
            self._populate([Path(f) for f in files])

    def _folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose calibration folder")
        if folder:
            root = Path(folder)
            paths = root.rglob("*") if self.recursive.isChecked() else root.iterdir()
            self._populate([p for p in paths if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])

    def _populate(self, files):
        self.videos = calibration_candidates(self.project, files, self.regex)
        self.table.setRowCount(len(self.videos))
        for row, video in enumerate(self.videos):
            self.table.setItem(row, 0, _check_item(bool(video.take and video.camera)))
            self.table.setItem(row, 1, QTableWidgetItem(video.take))
            self.table.setCellWidget(row, 2, _camera_combo(self.project.cameras, video.camera))
            item = _readonly(video.path.name)
            item.setToolTip(str(video.path))
            self.table.setItem(row, 3, item)
        self.note.setText(f"Found {len(self.videos)} video(s). Select the calibration recordings, then review take names and cameras.")

    def _later(self):
        self.selection = []
        self.accept()

    def _use(self):
        selected = [CalibrationVideo(video.path, self.table.item(row, 1).text().strip(),
                                     self.table.cellWidget(row, 2).currentText())
                    for row, video in enumerate(self.videos)
                    if self.table.item(row, 0).checkState() == Qt.CheckState.Checked]
        if not selected or any(not v.take or not v.camera for v in selected):
            QMessageBox.warning(self, "Calibration videos", "Select recordings and give each one a take name and camera.")
            return
        counts = Counter((v.take, v.camera) for v in selected)
        if any(n > 1 for n in counts.values()):
            QMessageBox.warning(self, "Calibration videos", "Each calibration take can have only one recording per camera.")
            return
        missing = []
        for take in sorted({v.take for v in selected}):
            absent = set(self.project.cameras) - {v.camera for v in selected if v.take == take}
            if absent:
                missing.append(f"{take}: {', '.join(sorted(absent))}")
        if missing and QMessageBox.warning(
            self, "Missing calibration cameras", "These takes are missing cameras:\n" + "\n".join(missing)
            + "\n\nUse these recordings anyway?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.selection = selected
        self.accept()


class AddVideosDialog(QDialog):
    """A scan never writes. Add opens calibration selection, then commits once."""

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add videos from SLEAP")
        self.resize(1120, 650)
        self.project = project
        self.discovery = None
        self.addition_result = None
        self._workers = []
        self._busy = False
        self._filling = False
        options = DiscoveryOptions.from_project(project)
        self._search_dirs = list(options.search_dirs)
        layout = QVBoxLayout(self)
        label = QLabel("Find unregistered videos in this project's SLEAP files. Edit experiment, trial and camera assignments, then select the videos to add.")
        label.setWordWrap(True)
        layout.addWidget(label)
        form = QFormLayout()
        self.strategy = QComboBox()
        self.strategy.addItems(["keyword", "regex", "number", "datestamp", "mtime"])
        self.strategy.setCurrentText(options.match_strategy)
        self.keyword = QLineEdit(options.trial_keyword)
        self.regex = QLineEdit(options.trial_regex)
        self.calib_regex = QLineEdit(options.calib_regex)
        form.addRow("Trial matching:", self.strategy)
        form.addRow("Trial keyword:", self.keyword)
        form.addRow("Trial regex:", self.regex)
        form.addRow("Calibration take regex:", self.calib_regex)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.scan_button = QPushButton("Scan SLEAP files")
        self.scan_button.clicked.connect(self._scan)
        search = QPushButton("Add search folder…")
        search.clicked.connect(self._search_folder)
        locate = QPushButton("Locate selected video…")
        locate.clicked.connect(self._locate)
        self._edit_buttons = [self.scan_button, search, locate]
        for button in self._edit_buttons:
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Include", "Experiment", "Trial", "Camera", "Video", "Status", "SLEAP file"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for column in (0, 1, 2, 3, 5, 6):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemChanged.connect(self._edited)
        layout.addWidget(self.table, stretch=1)
        actions = QHBoxLayout()
        for text, checked in (("Select all", True), ("Select none", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, checked=checked: self._select_all(checked))
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        self.status = QLabel("Ready to scan.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.add_button = self.buttons.addButton("Add selected videos…", QDialogButtonBox.ButtonRole.AcceptRole)
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(self._add)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for widget in (self.strategy, self.keyword, self.regex, self.calib_regex):
            signal = widget.currentTextChanged if isinstance(widget, QComboBox) else widget.textChanged
            signal.connect(self._rules_changed)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._update_status)
        self._initial_scan = True

    def showEvent(self, event):
        super().showEvent(event)
        if self._initial_scan:
            self._initial_scan = False
            QTimer.singleShot(0, lambda: self._scan() if self.isVisible() else None)

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
        else:
            super().closeEvent(event)

    def reject(self):
        if self._busy:
            return
        super().reject()

    def _options(self):
        return DiscoveryOptions(self.strategy.currentText(), self.keyword.text(), self.regex.text(),
                                self.calib_regex.text(), list(self._search_dirs))

    def _rules_changed(self):
        self.add_button.setEnabled(False)
        self.status.setText("Scan again to apply the changed matching rules.")
        self.discovery = None

    def _set_busy(self, busy):
        self._busy = busy
        self.table.setEnabled(not busy)
        for widget in (*self._edit_buttons, self.strategy, self.keyword, self.regex, self.calib_regex):
            widget.setEnabled(not busy)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(not busy)
        self.progress.setVisible(busy)
        self.add_button.setEnabled(False)

    def _report(self, done, total, text):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(text)

    def _scan(self):
        if self._busy:
            return
        options = self._options()
        self._set_busy(True)
        worker = AddVideosWorker(lambda progress: discover_videos(self.project, options, progress), self)
        worker.progress.connect(self._report)
        worker.finished.connect(lambda: self._scan_done(worker))
        self._workers.append(worker)
        worker.start()

    def _scan_done(self, worker):
        self._set_busy(False)
        if worker.error:
            self.status.setText(worker.error)
            return
        self.discovery = worker.result
        groups = defaultdict(list)
        for video in self.discovery.candidates:
            groups[(video.experiment, video.trial)].append(video)
        self._filling = True
        self.table.setRowCount(len(self.discovery.candidates))
        for row, video in enumerate(self.discovery.candidates):
            group = groups[(video.experiment, video.trial)]
            complete = not candidate_problems(group, self.project.cameras) and not incomplete_trials(group, self.project.cameras, self.project.trials_df)
            self.table.setItem(row, 0, _check_item(complete))
            self.table.setItem(row, 1, QTableWidgetItem(video.experiment))
            self.table.setItem(row, 2, QTableWidgetItem(video.trial))
            combo = _camera_combo(self.project.cameras, video.camera)
            combo.currentTextChanged.connect(self._edited)
            self.table.setCellWidget(row, 3, combo)
            item = _readonly(video.path.name if video.path else Path(video.recorded.replace("\\", "/")).name)
            item.setToolTip(f"Recorded: {video.recorded}\nLocated: {video.path or '(not found)'}")
            self.table.setItem(row, 4, item)
            self.table.setItem(row, 5, _readonly(video.issue))
            self.table.setItem(row, 6, _readonly(video.source.name))
        self._filling = False
        self._update_status()

    def _selected(self):
        if self.discovery is None:
            return []
        return [replace(v, experiment=self.table.item(row, 1).text().strip(),
                        trial=self.table.item(row, 2).text().strip(),
                        camera=self.table.cellWidget(row, 3).currentText())
                for row, v in enumerate(self.discovery.candidates)
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked]

    def _edited(self, *_):
        if not self._filling and not self._busy:
            self._timer.start()

    def _update_status(self):
        if self.discovery is None or self._busy:
            return
        selected = self._selected()
        problems = candidate_problems(selected, self.project.cameras)
        trials = {(v.experiment, v.trial) for v in selected}
        missing = incomplete_trials(selected, self.project.cameras, self.project.trials_df)
        self.add_button.setEnabled(bool(selected) and not problems)
        self.status.setText(
            f"Found {len(self.discovery.candidates)} unregistered video(s); "
            f"{len(selected)} selected in {len(trials)} trial(s). "
            + (problems[0] if problems else f"{len(missing)} incomplete trial(s)." if missing else "")
        )
        self._filling = True
        all_values = [replace(v, experiment=self.table.item(row, 1).text().strip(),
                              trial=self.table.item(row, 2).text().strip(),
                              camera=self.table.cellWidget(row, 3).currentText())
                      for row, v in enumerate(self.discovery.candidates)]
        groups = defaultdict(list)
        for value in all_values:
            groups[(value.experiment, value.trial)].append(value)
        for row, v in enumerate(self.discovery.candidates):
            current = all_values[row]
            group = groups[(current.experiment, current.trial)]
            messages = candidate_problems(group, self.project.cameras) or incomplete_trials(group, self.project.cameras, self.project.trials_df)
            self.table.item(row, 5).setText(messages[0] if messages else "Ready")
        self._filling = False

    def _select_all(self, checked):
        if self._busy:
            return
        self._filling = True
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._filling = False
        self._update_status()

    def _search_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose video search folder")
        if folder:
            self._search_dirs.append(Path(folder))
            self._scan()

    def _locate(self):
        row = self.table.currentRow()
        if self.discovery is None or row < 0:
            return
        video = self.discovery.candidates[row]
        filename, _ = QFileDialog.getOpenFileName(self, "Locate this video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.cine)")
        if filename:
            video.path, video.issue = Path(filename).resolve(), ""
            if video.path.parent not in self.discovery.options.search_dirs:
                self.discovery.options.search_dirs.append(video.path.parent)
                self._search_dirs.append(video.path.parent)
            self.table.item(row, 4).setText(video.path.name)
            self.table.item(row, 4).setToolTip(str(video.path))
            self._update_status()

    def _add(self):
        selected = self._selected()
        problems = candidate_problems(selected, self.project.cameras)
        if not selected or problems:
            return
        missing = incomplete_trials(selected, self.project.cameras, self.project.trials_df)
        if missing and QMessageBox.warning(
            self, "Incomplete trials", "These trials have missing cameras:\n" + "\n".join(missing[:20])
            + "\n\nAdd them anyway?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        calibrations = {}
        new = sorted({v.experiment for v in selected} - set(self.project.experiments))
        for experiment in new:
            dialog = CalibrationVideosDialog(self.project, experiment, self.calib_regex.text(), self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            calibrations[experiment] = dialog.selection
        n_trials = len({(v.experiment, v.trial) for v in selected})
        n_calib = sum(len(v) for v in calibrations.values())
        if QMessageBox.question(
            self, "Add videos", f"Add {len(selected)} video(s) in {n_trials} trial(s), "
            f"{len(new)} new experiment(s), and {n_calib} calibration recording(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes,
        ) != QMessageBox.StandardButton.Yes:
            return
        if self.project.points is not None:
            self.project.points.save_if_changed()
        self._set_busy(True)
        worker = AddVideosWorker(lambda progress: apply_additions(
            self.project, self.discovery, selected, calibrations,
            allow_incomplete=bool(missing), progress=progress,
        ), self)
        worker.progress.connect(self._report)
        worker.finished.connect(lambda: self._added(worker))
        self._workers.append(worker)
        worker.start()

    def _added(self, worker):
        self._set_busy(False)
        if worker.error:
            QMessageBox.warning(self, "Could not add videos", worker.error)
            self._update_status()
            return
        self.addition_result = worker.result
        self.accept()

"""Set up Sleap files dialog: assign (experiment, trial, camera) videos to SLP files."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
)

from label3d.adapters.sleap_adapter import SleapAdapter
from label3d.project import Project
from label3d.skeleton import MappingCheck, Skeleton
from label3d.slp_sync import PRESETS, Cell, SlpMapping, SlpMappingEditor

DATA_ROLE = Qt.ItemDataRole.UserRole
# The skeleton tag after a file name in the list, "  [name]"
SKELETON_TAG = re.compile(r"\s+\[[^\]]*\]$")


class SlpSetupDialog(QDialog):
    """Edit the SLP file mapping in memory.  Read the result with mapping() after exec()."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set up Sleap files")
        self.resize(960, 560)
        self._project = project
        self._message = ""

        mapping = project.slp_mapping
        self._editor = SlpMappingEditor(mapping, project.trials_df, project.cameras)
        if not mapping.files and self._existing_slp_files():
            self._adopt_existing()
        self._file_skeletons = project.existing_slp_skeletons(mapping)
        self._check = MappingCheck()

        self._build_ui()
        if self._several_skeletons():
            # One file per camera per experiment never mixes skeletons
            index = self._preset_combo.findData("per_experiment_camera")
            self._preset_combo.setCurrentIndex(index)
        self._refresh()

    def mapping(self) -> SlpMapping:
        return self._editor.mapping()

    def accept(self) -> None:
        """Ask before applying a mapping that leaves some videos without nodes they had."""
        if self._check.warnings:
            reply = QMessageBox.warning(
                self, "Set up Sleap files",
                "\n".join(self._check.warnings)
                + "\n\nPoints of those nodes stay in label3d but are hidden, and "
                "are not written to the SLP files. Apply anyway?",
                QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Apply:
                return
        super().accept()

    # -- Skeletons -------------------------------------------------------------

    def _several_skeletons(self) -> bool:
        skeletons = self._project.skeletons
        if skeletons is None:
            return False
        found: list[Skeleton] = []
        for sk in [skeletons.camera_skeleton(c) for c in self._project.cameras] + list(
            self._file_skeletons.values()
        ):
            if not any(sk.same_topology(f) for f in found):
                found.append(sk)
        return len(found) > 1

    def _skeleton_label(self, path: Path) -> str:
        """The skeleton a file has or will be created with, or "mixed"."""
        if path in self._file_skeletons:
            return self._file_skeletons[path].name or "unnamed"
        skeletons = self._project.skeletons
        cameras = dict.fromkeys(c[2] for c in self._editor.cells_of(path))
        found = [skeletons.camera_skeleton(c) for c in cameras]
        if not found:
            return ""
        if any(not found[0].same_topology(sk) for sk in found[1:]):
            return "mixed"
        return found[0].name or "unnamed"

    # -- UI --------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self._exp_label = QLabel("Experiment:")
        self._exp_combo = QComboBox()
        self._exp_combo.addItems(sorted(self._project.experiments))
        self._exp_combo.setCurrentText(self._project.active_experiment)
        single = len(self._project.experiments) <= 1
        self._exp_label.setVisible(not single)
        self._exp_combo.setVisible(not single)
        self._exp_combo.currentTextChanged.connect(lambda _: self._refresh_grid())
        top.addWidget(self._exp_label)
        top.addWidget(self._exp_combo)
        top.addSpacing(12)

        top.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        for key, (label, _) in PRESETS.items():
            self._preset_combo.addItem(label, key)
        top.addWidget(self._preset_combo)
        self._pattern_edit = QLineEdit(next(iter(PRESETS.values()))[1])
        self._pattern_edit.setToolTip("File name pattern — placeholders: {experiment}, {trial}, {camera}")
        self._pattern_edit.setMinimumWidth(220)
        self._preset_combo.currentIndexChanged.connect(
            lambda _: self._pattern_edit.setText(PRESETS[self._preset_combo.currentData()][1])
        )
        top.addWidget(self._pattern_edit)
        apply_preset = QPushButton("Apply preset")
        apply_preset.setToolTip("Replace all assignments using this pattern")
        apply_preset.clicked.connect(self._on_apply_preset)
        top.addWidget(apply_preset)
        top.addStretch()
        self._adopt_btn = QPushButton("Adopt existing")
        self._adopt_btn.setToolTip("Build assignments from the videos referenced by the files in slp/")
        self._adopt_btn.setEnabled(bool(self._existing_slp_files()))
        self._adopt_btn.clicked.connect(self._on_adopt)
        top.addWidget(self._adopt_btn)
        layout.addLayout(top)

        body = QHBoxLayout()
        self._grid = QTableWidget()
        self._grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._grid.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self._grid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._grid.itemSelectionChanged.connect(self._update_buttons)
        body.addWidget(self._grid, stretch=3)

        side = QVBoxLayout()
        side.addWidget(QLabel("Files (double-click to rename):"))
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._file_list.itemChanged.connect(self._on_file_renamed)
        self._file_list.itemSelectionChanged.connect(self._update_buttons)
        side.addWidget(self._file_list, stretch=1)

        self._split_cam_btn = QPushButton("Split by camera")
        self._split_cam_btn.setToolTip("Split each selected file into one file per camera")
        self._split_cam_btn.clicked.connect(self._on_split_camera)
        self._split_trial_btn = QPushButton("Split by trial")
        self._split_trial_btn.setToolTip(
            "Move the trials selected in the grid out of the selected file into a new file"
        )
        self._split_trial_btn.clicked.connect(self._on_split_trial)
        self._new_file_btn = QPushButton("New file from selection")
        self._new_file_btn.setToolTip("Move the videos selected in the grid into a new file")
        self._new_file_btn.clicked.connect(self._on_new_file)
        self._join_btn = QPushButton("Join")
        self._join_btn.setToolTip("Merge the selected files into the first one")
        self._join_btn.clicked.connect(self._on_join)
        self._assign_btn = QToolButton()
        self._assign_btn.setText("Assign to")
        self._assign_btn.setToolTip("Move the videos selected in the grid into an existing file")
        self._assign_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._assign_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._assign_menu = QMenu(self._assign_btn)
        self._assign_menu.aboutToShow.connect(self._populate_assign_menu)
        self._assign_btn.setMenu(self._assign_menu)
        for button in (
            self._split_cam_btn, self._split_trial_btn, self._new_file_btn,
            self._join_btn, self._assign_btn,
        ):
            side.addWidget(button)
        body.addLayout(side, stretch=2)
        layout.addLayout(body, stretch=1)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        self._apply_btn = self._button_box.button(QDialogButtonBox.StandardButton.Apply)
        self._apply_btn.setToolTip(
            "Back up slp/, read labels from the current files, then write the new files"
        )
        self._apply_btn.clicked.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    # -- Refresh ---------------------------------------------------------------

    def _refresh(self) -> None:
        self._refresh_files()
        self._refresh_grid()
        self._update_status()
        self._update_buttons()

    def _color(self, path: Path) -> QColor:
        index = self._editor.files.index(path)
        return QColor.fromHsv((index * 137) % 360, 70, 245)

    def _swatch(self, path: Path) -> QIcon:
        pixmap = QPixmap(14, 14)
        pixmap.fill(self._color(path))
        return QIcon(pixmap)

    def _refresh_files(self, select: list[Path] | None = None) -> None:
        if select is None:
            select = self._selected_files()
        self._file_list.blockSignals(True)
        self._file_list.clear()
        several = self._several_skeletons()
        for path in self._editor.files:
            label = self._skeleton_label(path) if several else ""
            text = f"{path.name}  [{label}]" if label else path.name
            item = QListWidgetItem(self._swatch(path), text)
            item.setData(DATA_ROLE, path)
            tooltip = f"{path.as_posix()} — {len(self._editor.cells_of(path))} video(s)"
            if label:
                tooltip += f"; skeleton: {label}"
                if label == "mixed":
                    item.setForeground(QColor(176, 0, 32))
            item.setToolTip(tooltip)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._file_list.addItem(item)
            item.setSelected(path in select)
        self._file_list.blockSignals(False)

    def _refresh_grid(self) -> None:
        selected = set(self._selected_cells())
        experiment = self._exp_combo.currentText()
        trials = [t.name for t in self._project.experiments[experiment].trials]
        cameras = self._project.cameras

        self._grid.blockSignals(True)
        self._grid.clear()
        self._grid.setRowCount(len(trials))
        self._grid.setColumnCount(len(cameras))
        self._grid.setVerticalHeaderLabels(trials)
        self._grid.setHorizontalHeaderLabels(cameras)
        for row, trial in enumerate(trials):
            for col, cam in enumerate(cameras):
                cell = (experiment, trial, cam)
                item = QTableWidgetItem()
                item.setForeground(QColor(20, 20, 20))
                if cell not in self._editor.cells_info:
                    item.setText("—")
                    item.setToolTip("No video")
                    item.setFlags(Qt.ItemFlag.NoItemFlags)
                else:
                    item.setData(DATA_ROLE, cell)
                    paths = self._editor.files_of(cell)
                    if not paths:
                        item.setText("unassigned")
                        item.setBackground(QBrush(QColor(150, 150, 150), Qt.BrushStyle.BDiagPattern))
                    elif len(paths) == 1:
                        item.setText(paths[0].name)
                        item.setBackground(self._color(paths[0]))
                    else:
                        item.setText(" + ".join(p.name for p in paths))
                        item.setBackground(QColor(255, 180, 180))
                        item.setToolTip("Assigned to more than one file")
                self._grid.setItem(row, col, item)
                item.setSelected(cell in selected)
        self._grid.blockSignals(False)

    def _update_status(self) -> None:
        self._check = self._project.check_slp_mapping(
            self._editor.mapping(), self._file_skeletons
        )
        errors = (
            self._editor.errors() + list(self._project.skeleton_conflicts)
            + self._check.errors
        )
        self._apply_btn.setEnabled(not errors)
        if errors:
            summary = "; ".join(errors)
        else:
            summary = (
                f"{len(self._editor.files)} file(s); all "
                f"{len(self._editor.cells_info)} videos assigned"
            )
            if self._check.warnings:
                summary += " — " + "; ".join(self._check.warnings)
        text = f"{self._message} — {summary}" if self._message else summary
        self._status.setText(text)
        if errors:
            self._status.setStyleSheet("color: #b00020;")
        elif self._check.warnings:
            self._status.setStyleSheet("color: #a05a00;")
        else:
            self._status.setStyleSheet("")

    def _update_buttons(self) -> None:
        files = self._selected_files()
        cells = self._selected_cells()
        self._split_cam_btn.setEnabled(
            any(len({c[2] for c in self._editor.cells_of(p)}) > 1 for p in files)
        )
        self._split_trial_btn.setEnabled(len(files) == 1 and bool(cells))
        self._new_file_btn.setEnabled(bool(cells))
        self._join_btn.setEnabled(len(files) >= 2)
        self._assign_btn.setEnabled(bool(cells) and bool(self._editor.files))

    # -- Selection -------------------------------------------------------------

    def _selected_files(self) -> list[Path]:
        selected = {item.data(DATA_ROLE) for item in self._file_list.selectedItems()}
        return [p for p in self._editor.files if p in selected]

    def _selected_cells(self) -> list[Cell]:
        cells = [item.data(DATA_ROLE) for item in self._grid.selectedItems()]
        return [tuple(c) for c in cells if c]

    # -- Actions ---------------------------------------------------------------

    def _set_message(self, message: str) -> None:
        self._message = message
        self._update_status()

    def _on_apply_preset(self) -> None:
        try:
            self._editor.apply_preset(self._pattern_edit.text())
        except (KeyError, ValueError, IndexError) as exc:
            self._set_message(f"Invalid file name pattern: {exc}")
            return
        self._message = ""
        self._refresh()

    def _on_adopt(self) -> None:
        self._adopt_existing()
        self._refresh_files(select=[])
        self._refresh()

    def _on_split_camera(self) -> None:
        new_paths = []
        for path in self._selected_files():
            new_paths.extend(self._editor.split_by_camera(path))
        self._message = ""
        self._refresh_files(select=new_paths)
        self._refresh()

    def _on_split_trial(self) -> None:
        (path,) = self._selected_files()
        trials = {(c[0], c[1]) for c in self._selected_cells()}
        new_path = self._editor.split_by_trial(path, trials)
        if new_path is None:
            self._set_message(f"Select some, but not all, of the trials in {path.name}")
            return
        self._message = ""
        self._refresh_files(select=[new_path])
        self._refresh()

    def _on_new_file(self) -> None:
        new_path = self._editor.new_file(self._selected_cells())
        self._message = ""
        self._refresh_files(select=[new_path] if new_path else [])
        self._refresh()

    def _on_join(self) -> None:
        target = self._editor.join(self._selected_files())
        self._message = ""
        self._refresh_files(select=[target] if target else [])
        self._refresh()

    def _populate_assign_menu(self) -> None:
        self._assign_menu.clear()
        for path in self._editor.files:
            action = self._assign_menu.addAction(self._swatch(path), path.name)
            action.triggered.connect(lambda _=False, p=path: self._assign_to(p))

    def _assign_to(self, path: Path) -> None:
        self._editor.move_cells(self._selected_cells(), path)
        self._message = ""
        self._refresh()

    def _on_file_renamed(self, item: QListWidgetItem) -> None:
        old = item.data(DATA_ROLE)
        try:
            name = SKELETON_TAG.sub("", item.text().strip())
            new = self._editor.rename(old, old.parent / name)
        except ValueError as exc:
            self._message = str(exc)
            new = old
        else:
            self._message = ""
        self._refresh_files(select=[new])
        self._refresh()

    # -- Existing files --------------------------------------------------------

    def _existing_slp_files(self) -> list[Path]:
        if not self._project.slp_dir.exists():
            return []
        return sorted(self._project.slp_dir.glob("*.slp"))

    def _adopt_existing(self) -> None:
        files = self._existing_slp_files()
        project = self._project
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            file_map = SleapAdapter().build_file_map(files, project.trials_df, project.cameras)
        except Exception as exc:
            self._message = f"Could not read existing SLP files: {exc}"
            return
        finally:
            QApplication.restoreOverrideCursor()
        cell_map = {cell: path.relative_to(project.root) for cell, path in file_map.items()}
        self._editor = SlpMappingEditor(
            SlpMapping.from_cell_map(cell_map), project.trials_df, project.cameras
        )
        self._message = f"Loaded assignments from {len(files)} existing SLP file(s)"

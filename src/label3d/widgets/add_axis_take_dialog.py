"""Add axis take dialog (Phase 16.4): files per camera, or a frame of a take or trial."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from label3d import axis_takes
from label3d.axis_takes import AxisTakeError
from label3d.frames import shown_frame, stored_frame
from label3d.project import Project
from label3d.setup_project import VIDEO_EXTENSIONS
from label3d.video import IMAGE_EXTENSIONS

NO_FILE = "(none)"
FILES_TAB, FRAME_TAB = 0, 1


class AddAxisTakeDialog(QDialog):
    """Add an axis take to the active experiment.

    The Files tab copies one image or video per camera into the project; the
    Frame tab extracts one frame of a calibration take or trial. Add does it
    (axis_takes.add_from_files / add_from_frame) and closes the dialog; after
    exec(), take() is the new take's name.
    """

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add axis take")
        self.resize(560, 380)
        self._project = project
        self._files: list[Path] = []
        self._added = ""
        self._name_edited = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"An axis take for experiment <b>{project.active_experiment}</b>: an object "
            "with marked x, y, z axes, seen by at least two cameras."
        ))

        self._tabs = QTabWidget()
        self._tabs.addTab(self._files_page(), "Files")
        self._tabs.addTab(self._frame_page(), "Frame of a take or trial")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tabs, stretch=1)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Take name:"))
        self._name = QLineEdit(axis_takes.unique_take_name(project))
        self._name.textEdited.connect(self._on_name_edited)
        name_row.addWidget(self._name, stretch=1)
        layout.addLayout(name_row)

        self._message = QLabel()
        self._message.setWordWrap(True)
        self._message.setStyleSheet("color: #b45309;")
        layout.addWidget(self._message)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._add_button = self._buttons.addButton("Add", QDialogButtonBox.ButtonRole.AcceptRole)
        self._buttons.accepted.connect(self._on_add)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._on_source_changed()
        self._update()

    # -- Pages ---------------------------------------------------------------

    def _files_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        buttons = QHBoxLayout()
        choose_files = QPushButton("Choose files…")
        choose_files.clicked.connect(self._on_choose_files)
        choose_folder = QPushButton("Choose folder…")
        choose_folder.clicked.connect(self._on_choose_folder)
        buttons.addWidget(choose_files)
        buttons.addWidget(choose_folder)
        buttons.addStretch()
        layout.addLayout(buttons)

        grid = QGridLayout()
        self._camera_combos: dict[str, QComboBox] = {}
        for row, cam in enumerate(self._project.cameras):
            grid.addWidget(QLabel(cam), row, 0)
            combo = QComboBox()
            combo.addItem(NO_FILE)
            combo.currentIndexChanged.connect(self._update)
            grid.addWidget(combo, row, 1)
            self._camera_combos[cam] = combo
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self._unmatched = QLabel()
        self._unmatched.setWordWrap(True)
        layout.addWidget(self._unmatched)
        layout.addStretch()
        note = QLabel("The files are copied into the project; the originals stay where they are.")
        note.setWordWrap(True)
        layout.addWidget(note)
        return page

    def _frame_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._source = QComboBox()
        for name, kind in axis_takes.frame_sources(self._project):
            self._source.addItem(f"{name} ({kind})", name)
        self._source.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Take or trial:", self._source)
        self._frame = QSpinBox()
        self._frame.setMinimum(shown_frame(0))
        self._frame.setMaximum(shown_frame(0))
        form.addRow("Frame:", self._frame)
        note = QLabel(
            "Each camera's video is read at its start offset plus this frame, and "
            "saved as a still image in the project."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return page

    # -- Files ---------------------------------------------------------------

    def set_files(self, files: list[Path]) -> None:
        """Match files to cameras and show them (also used by the choose buttons)."""
        match = axis_takes.match_files(files, self._project.cameras, self._project.camera_aliases)
        self._files = sorted(Path(f) for f in files)
        for cam, combo in self._camera_combos.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(NO_FILE)
            for f in self._files:
                combo.addItem(f.name, str(f))
            chosen = match.by_camera.get(cam)
            combo.setCurrentIndex(self._files.index(chosen) + 1 if chosen else 0)
            combo.blockSignals(False)
        self._unmatched.setText(
            "Not matched to a camera: " + ", ".join(f.name for f in match.unmatched)
            if match.unmatched else ""
        )
        if not self._name_edited:
            self._name.setText(axis_takes.unique_take_name(self._project, match.take))
        self._update()

    def chosen_files(self) -> dict[str, Path]:
        return {
            cam: Path(combo.currentData())
            for cam, combo in self._camera_combos.items()
            if combo.currentData()
        }

    def _on_choose_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Axis take files", "", f"Images and videos ({patterns})"
        )
        if files:
            self.set_files([Path(f) for f in files])

    def _on_choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder of axis take files")
        if folder:
            self.set_files(axis_takes.media_files(Path(folder)))

    # -- Frame ---------------------------------------------------------------

    def _on_source_changed(self, *_args) -> None:
        source = self._source.currentData()
        length = axis_takes.source_length(self._project, source) if source else 0
        self._frame.setMaximum(shown_frame(max(length - 1, 0)))
        self._update()

    # -- Common --------------------------------------------------------------

    def _on_tab_changed(self, _index: int) -> None:
        self._update()

    def _on_name_edited(self, _text: str) -> None:
        self._name_edited = True
        self._update()

    def _problem(self) -> str:
        """Why Add can't go ahead yet, or ""."""
        try:
            axis_takes.check_take_name(self._project, self._name.text())
        except AxisTakeError as e:
            return str(e)
        if self._tabs.currentIndex() == FILES_TAB:
            if len(self.chosen_files()) < 2:
                return "Choose a file for at least two cameras."
        elif self._source.currentData() is None:
            return "This experiment has no calibration take or trial."
        return ""

    def _update(self, *_args) -> None:
        problem = self._problem()
        self._add_button.setEnabled(not problem)
        self._message.setText(problem)

    def _on_add(self) -> None:
        take = self._name.text()
        try:
            if self._tabs.currentIndex() == FILES_TAB:
                axis_takes.add_from_files(self._project, take, self.chosen_files())
            else:
                axis_takes.add_from_frame(
                    self._project, take, self._source.currentData(),
                    stored_frame(self._frame.value()),
                )
        except (AxisTakeError, OSError) as e:
            self._message.setText(str(e))
            return
        self._added = take
        self.accept()

    def take(self) -> str:
        """The added take's name, after exec() returned Accepted."""
        return self._added

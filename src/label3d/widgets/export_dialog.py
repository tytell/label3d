"""File > Export points... options (Phase 13.3)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from label3d.export import (
    FORMATS,
    ExportOptions,
    default_output_path,
    uncalibrated_experiments,
    with_format_suffix,
)
from label3d.project import Project


class ExportDialog(QDialog):
    """Experiments, filters, format, layout and output path for the export.

    On Export, experiments without an active calibration are listed in a
    warning; Cancel there returns to this dialog.
    """

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self.setWindowTitle("Export points")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Experiments:"))
        self._experiments = QListWidget()
        for name in project.experiments:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._experiments.addItem(item)
        layout.addWidget(self._experiments)

        self._predictions = QCheckBox("Include predictions")
        self._predictions.setChecked(True)
        self._predictions.setToolTip(
            "Off leaves out SLEAP predictions. Scores are in the file, for filtering later."
        )
        layout.addWidget(self._predictions)

        self._only_3d = QCheckBox("Only points with 3D")
        layout.addWidget(self._only_3d)

        form = QFormLayout()
        self._format = QComboBox()
        self._format.addItem("Parquet", "parquet")
        self._format.addItem("CSV", "csv")
        self._format.currentIndexChanged.connect(self._format_changed)
        form.addRow("Format:", self._format)
        self._layout = QComboBox()
        self._layout.addItem("Wide: one row per point, a block of columns per camera", "wide")
        self._layout.addItem("Long: one row per point and camera, and per 3D point", "long")
        form.addRow("Layout:", self._layout)
        layout.addLayout(form)

        layout.addWidget(QLabel("Output file:"))
        path_row = QHBoxLayout()
        self._path = QLineEdit(str(default_output_path(project, self.format())))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        path_row.addWidget(self._path, stretch=1)
        path_row.addWidget(browse)
        layout.addLayout(path_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(480, self.sizeHint().height())

    def selected_experiments(self) -> list[str]:
        return [
            self._experiments.item(i).text()
            for i in range(self._experiments.count())
            if self._experiments.item(i).checkState() == Qt.CheckState.Checked
        ]

    def format(self) -> str:
        return self._format.currentData()

    def options(self) -> ExportOptions:
        return ExportOptions(
            experiments=self.selected_experiments(),
            include_predictions=self._predictions.isChecked(),
            only_3d=self._only_3d.isChecked(),
            output=with_format_suffix(Path(self._path.text().strip()), self.format()),
            format=self.format(),
            layout=self._layout.currentData(),
        )

    def _format_changed(self) -> None:
        text = self._path.text().strip()
        if text:
            self._path.setText(str(with_format_suffix(Path(text), self.format())))

    def _browse(self) -> None:
        ext = FORMATS[self.format()]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export points", self._path.text(),
            f"{self._format.currentText()} files (*{ext})",
        )
        if path:
            self._path.setText(str(with_format_suffix(Path(path), self.format())))

    def accept(self) -> None:
        names = self.selected_experiments()
        if not names:
            QMessageBox.information(self, "Export points", "Check at least one experiment.")
            return
        if not self._path.text().strip():
            QMessageBox.information(self, "Export points", "Choose an output file.")
            return
        uncalibrated = uncalibrated_experiments(self._project, names)
        if uncalibrated:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("No active calibration")
            box.setText(
                "These experiments have no active calibration, so only their 2D "
                "points will be exported:\n\n" + "\n".join(uncalibrated)
            )
            export = box.addButton("Export anyway", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is not export:
                return
        super().accept()

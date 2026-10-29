"""Offer to upgrade a project made by an older label3d (Phase 20)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from label3d.layout import UPGRADE_BACKUP_DIR, UpgradePlan
from label3d.upgrade import UpgradeResult, earliest_backup, find_originals
from label3d.version import LAYOUT_VERSION, version_string
from label3d.workers.upgrade_worker import UpgradeWorker

KEEP = "Keep the project's copy"


class UpgradeDialog(QDialog):
    """The steps an upgrade takes and, for each .slp file to restore, its original."""

    def __init__(self, plan: UpgradePlan, parent: QWidget | None = None):
        super().__init__(parent)
        self._plan = plan
        self.setWindowTitle(f"Upgrade {plan.root.name}")
        self.setMinimumWidth(640)
        layout = QVBoxLayout(self)

        steps = "".join(f"<li>{text}</li>" for text in plan.describe())
        intro = QLabel(
            f"<b>{plan.root.name}</b> was made by label3d {plan.from_version} or similar. "
            f"To open it, label3d {version_string()} upgrades it to layout {LAYOUT_VERSION}:"
            f"<ul>{steps}</ul>"
            f"Every file that changes is first copied to <tt>{UPGRADE_BACKUP_DIR}/</tt> in the "
            "project. If the upgrade fails, the project is put back as it was."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._combos: dict[Path, QComboBox] = {}
        if plan.restore:
            layout.addWidget(self._restore_section())

        buttons = QDialogButtonBox()
        buttons.addButton("Upgrade", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _restore_section(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        note = QLabel(
            "Syncs in label3d before 0.2.0 dropped the videos and frames outside the project "
            "from these .slp files. Choose the original each one was set up from; the "
            "project's points are written into it, and everything else in it is kept. "
            "The earliest Sync backup is the file as it was before the first Sync."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self._table = QTableWidget(len(self._plan.restore), 2)
        self._table.setHorizontalHeaderLabels([".slp file", "Original"])
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for row, rel in enumerate(self._plan.restore):
            item = QTableWidgetItem(rel.as_posix())
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, item)
            combo = QComboBox()
            backup = earliest_backup(self._plan.root, rel.name)
            if backup is not None:
                folder = backup.parent.relative_to(self._plan.root).as_posix()
                combo.addItem(f"Earliest Sync backup ({folder})", backup)
                combo.setItemData(0, str(backup), Qt.ItemDataRole.ToolTipRole)
            combo.addItem(KEEP, None)
            self._combos[rel] = combo
            self._table.setCellWidget(row, 1, combo)
        layout.addWidget(self._table)

        buttons = QHBoxLayout()
        search = QPushButton("Search folder...")
        search.setToolTip("Find the originals by file name in a folder and its subfolders")
        search.clicked.connect(self._search_folder)
        choose = QPushButton("Choose file...")
        choose.setToolTip("Choose the original for the selected .slp file")
        choose.clicked.connect(self._choose_file)
        buttons.addWidget(search)
        buttons.addWidget(choose)
        buttons.addStretch()
        layout.addLayout(buttons)
        return box

    def _add_choice(self, rel: Path, path: Path, select: bool) -> None:
        combo = self._combos[rel]
        index = combo.findData(path)
        if index < 0:
            combo.insertItem(0, str(path), path)
            combo.setItemData(0, str(path), Qt.ItemDataRole.ToolTipRole)
            index = 0
        if select:
            combo.setCurrentIndex(index)

    def _search_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder holding the original .slp files")
        if not folder:
            return
        found = find_originals([rel.name for rel in self._plan.restore], Path(folder))
        missing = []
        for rel in self._plan.restore:
            matches = found[rel.name]
            for n, path in enumerate(reversed(matches)):
                self._add_choice(rel, path, select=(n == len(matches) - 1))
            if not matches:
                missing.append(rel.name)
        if missing:
            QMessageBox.information(
                self, "Not found", "No file of this name in that folder:\n  " + "\n  ".join(missing)
            )

    def _choose_file(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            row = 0
        rel = self._plan.restore[row]
        path, _ = QFileDialog.getOpenFileName(
            self, f"Original of {rel.name}", "", "SLEAP labels (*.slp)"
        )
        if path:
            self._add_choice(rel, Path(path), select=True)

    def sources(self) -> dict[Path, Path | None]:
        """The chosen original for each file to restore (None keeps the project's copy)."""
        return {rel: combo.currentData() for rel, combo in self._combos.items()}

    def accept(self) -> None:
        kept = [rel.as_posix() for rel, src in self.sources().items() if src is None]
        if kept:
            answer = QMessageBox.warning(
                self, "Keep the project's copies?",
                "These files keep what older Syncs left in them; the videos and frames they "
                "dropped stay lost:\n  " + "\n  ".join(kept),
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Ok:
                return
        super().accept()


def upgrade_project(plan: UpgradePlan, parent: QWidget | None = None) -> bool:
    """Ask, then upgrade the project with a progress dialog. True once it is upgraded."""
    dialog = UpgradeDialog(plan, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False

    progress = QProgressDialog("Upgrading…", None, 0, 0, parent)
    progress.setWindowTitle(f"Upgrade {plan.root.name}")
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumWidth(480)
    progress.setMinimumDuration(0)
    progress.show()

    outcome: dict = {}
    loop = QEventLoop()
    worker = UpgradeWorker(plan.root, dialog.sources())
    worker.progress.connect(progress.setLabelText)
    worker.done.connect(lambda result: outcome.update(result=result))
    worker.error.connect(lambda text: outcome.update(error=text))
    worker.finished.connect(loop.quit)
    worker.start()
    loop.exec()
    progress.close()

    if "error" in outcome:
        QMessageBox.critical(
            parent, "Upgrade failed",
            f"The upgrade stopped; the project is as it was.\n\n{outcome['error']}",
        )
        return False
    result: UpgradeResult = outcome["result"]
    QMessageBox.information(
        parent, "Project upgraded",
        "\n".join(result.messages)
        + f"\n\nThe files as they were are in {result.backup_dir}. "
        "That folder can be deleted once the project has been checked.",
    )
    return True

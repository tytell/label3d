"""Load skeleton dialog: assign a skeleton to all cameras or to chosen ones (Phase 14.2)."""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from label3d.project import AssignmentPreview, Project
from label3d.skeleton import Skeleton


def read_skeleton(path: Path) -> Skeleton:
    """A skeleton from a skeleton.json or the first skeleton of a .slp file."""
    if path.suffix.lower() == ".slp":
        skeleton = Skeleton.from_slp(path)
        if not skeleton.name:
            skeleton.name = path.stem
        return skeleton
    return Skeleton.from_json(path)


class LoadSkeletonDialog(QDialog):
    """Choose a skeleton file and the cameras to assign it to.

    After exec(), read skeleton(), cameras() and remove_points(). Nothing is
    changed by the dialog itself; preview() describes what OK would do.
    """

    def __init__(self, project: Project, path: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load skeleton")
        self.resize(640, 420)
        self._project = project
        self._skeleton: Skeleton | None = None
        self._error = ""
        self._preview: AssignmentPreview | None = None

        layout = QVBoxLayout(self)

        source = QHBoxLayout()
        source.addWidget(QLabel("Skeleton file:"))
        self._path_label = QLabel("none")
        self._path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        source.addWidget(self._path_label, stretch=1)
        choose = QPushButton("Choose…")
        choose.setToolTip("A skeleton.json, or a .slp file whose skeleton is used")
        choose.clicked.connect(self._on_choose)
        source.addWidget(choose)
        layout.addLayout(source)

        self._skeleton_label = QLabel()
        self._skeleton_label.setWordWrap(True)
        layout.addWidget(self._skeleton_label)

        self._all_box = QCheckBox("All cameras")
        self._all_box.setChecked(True)
        self._all_box.toggled.connect(self._on_all_toggled)
        layout.addWidget(self._all_box)

        grid = QGridLayout()
        grid.setContentsMargins(20, 0, 0, 0)
        self._camera_boxes: dict[str, QCheckBox] = {}
        self._camera_notes: dict[str, QLabel] = {}
        for row, cam in enumerate(project.cameras):
            box = QCheckBox(cam)
            box.setChecked(True)
            box.toggled.connect(self._on_camera_toggled)
            note = QLabel()
            note.setWordWrap(True)
            grid.addWidget(box, row, 0)
            grid.addWidget(note, row, 1)
            self._camera_boxes[cam] = box
            self._camera_notes[cam] = note
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self._checks = QLabel()
        self._checks.setWordWrap(True)
        self._checks.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._checks)

        self._outside_label = QLabel()
        self._outside_label.setWordWrap(True)
        self._outside_label.setVisible(False)
        layout.addWidget(self._outside_label)
        self._remove_box = QCheckBox()
        self._remove_box.setVisible(False)
        layout.addWidget(self._remove_box)
        layout.addStretch()

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        if path is not None:
            self.set_path(path)
        self._refresh()

    # -- Results ---------------------------------------------------------------

    def skeleton(self) -> Skeleton | None:
        return self._skeleton

    def cameras(self) -> list[str]:
        return [cam for cam, box in self._camera_boxes.items() if box.isChecked()]

    def remove_points(self) -> bool:
        return self._remove_box.isVisible() and self._remove_box.isChecked()

    def preview(self) -> AssignmentPreview | None:
        return self._preview

    # -- Source ----------------------------------------------------------------

    def _on_choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load skeleton", str(self._project.root),
            "Skeletons (*.json *.slp);;Skeleton files (*.json);;SLEAP files (*.slp)",
        )
        if path:
            self.set_path(Path(path))

    def set_path(self, path: Path) -> None:
        """Read the skeleton at path; an unreadable file leaves no skeleton."""
        self._path_label.setText(str(path))
        try:
            self._skeleton = read_skeleton(path)
            self._error = ""
        except (OSError, ValueError, KeyError) as exc:
            self._skeleton = None
            self._error = f"Could not read {path.name}: {exc}"
        self._refresh()

    # -- Cameras ---------------------------------------------------------------

    def _on_all_toggled(self, checked: bool) -> None:
        for box in self._camera_boxes.values():
            box.blockSignals(True)
            box.setChecked(checked)
            box.blockSignals(False)
        self._refresh()

    def _on_camera_toggled(self, _checked: bool) -> None:
        self._all_box.blockSignals(True)
        self._all_box.setChecked(all(b.isChecked() for b in self._camera_boxes.values()))
        self._all_box.blockSignals(False)
        self._refresh()

    def _camera_note(self, camera: str) -> str:
        """The camera's current skeleton and what its existing .slp files keep."""
        skeletons = self._project.skeletons
        if skeletons is None:
            return "now: no skeleton"
        if camera in skeletons.by_camera:
            parts = [f"now: {skeletons.by_camera[camera].name or 'unnamed'} (assigned)"]
        elif skeletons.inferred(camera) is not None:
            parts = [f"now: {skeletons.inferred(camera).name or 'unnamed'} (from its .slp files)"]
        else:
            parts = ["now: the combined skeleton"]
        files = skeletons.camera_files(camera)
        if files:
            differ = self._skeleton is not None and any(
                not skeletons.by_file[f].same_topology(self._skeleton) for f in files
            )
            if differ:
                parts.append(
                    f"{len(files)} existing .slp file(s) keep their skeleton; "
                    "this applies to new files only"
                )
            else:
                parts.append(f"{len(files)} existing .slp file(s)")
        return " · ".join(parts)

    # -- Checks ----------------------------------------------------------------

    def _refresh(self) -> None:
        sk = self._skeleton
        error = self._error
        if sk is None:
            self._skeleton_label.setText(error or "Choose a skeleton file.")
            self._skeleton_label.setStyleSheet("color: #b00020;" if error else "")
        else:
            name = f" ({sk.name})" if sk.name else ""
            self._skeleton_label.setText(
                f"{len(sk.nodes)} node(s){name}: {', '.join(sk.nodes)}"
            )
            self._skeleton_label.setStyleSheet("")
        for cam, note in self._camera_notes.items():
            note.setText(self._camera_note(cam))

        cameras = self.cameras()
        self._preview = None
        lines: list[str] = []
        ok = sk is not None and bool(cameras)
        if sk is not None and not cameras:
            lines.append(_colored("Choose at least one camera.", "#b00020"))
        if ok:
            self._preview = preview = self._project.preview_assignment(sk, cameras)
            if preview.conflicts:
                ok = False
                lines.append(_colored("The skeletons would conflict:", "#b00020"))
                lines += [_colored(f"• {c}", "#b00020") for c in preview.conflicts]
            for problem in preview.mapping_errors:
                lines.append(_colored(
                    f"• {problem} — change the mapping in Set up Sleap files "
                    "before the next Sync", "#a05a00",
                ))
        self._checks.setText("<br>".join(lines))
        self._ok.setEnabled(ok)

        outside = self._preview.outside if self._preview is not None else {}
        self._outside_label.setVisible(bool(outside))
        self._remove_box.setVisible(bool(outside))
        if outside:
            total = sum(outside.values())
            parts = ", ".join(f"{node} ({n})" for node, n in sorted(outside.items()))
            self._outside_label.setText(
                f"{total} labelled point(s) in the chosen cameras would not be in their "
                f"skeleton: {parts}. Unless removed, they are kept but hidden, and not "
                "triangulated or written to the SLP files."
            )
            self._remove_box.setText(f"Remove these {total} point(s) now")


def _colored(text: str, color: str) -> str:
    return f'<span style="color:{color}">{html.escape(text)}</span>'

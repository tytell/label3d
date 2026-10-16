"""Summary of what a sync or prediction import brought in from SLEAP (Phase 10.5)."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

KIND_NAMES = {
    "sync": "Sync with Sleap",
    "open": "Reading SLP files on open",
    "close": "Sync on close",
    "import": "Import predictions",
    "setup": "Set up Sleap files",
}

COLUMNS = [
    ("experiment", "Experiment"),
    ("trial", "Trial"),
    ("n_new_user", "New SLEAP labels"),
    ("n_new_pred", "New predictions"),
    ("n_updated_pred", "Updated predictions"),
    ("n_conflicts", "Conflicts"),
    ("n_frames", "Frames with new points"),
]


def dropped_text(dropped: dict[tuple[str, str], int]) -> str:
    """Describe SLP points left out because a camera's skeleton lacks the node."""
    total = sum(dropped.values())
    parts = [f"{node} in {cam} ({n})" for (cam, node), n in sorted(dropped.items())]
    return (
        f"{total} point(s) were left out because the camera's skeleton does not "
        f"have the node: {', '.join(parts)}"
    )


class SyncSummaryDialog(QDialog):
    """Table of new points per experiment × trial for one sync.

    rows: the sync_history.csv rows of one sync id.  Rows of the active
    experiment come first.  "Go to first new frame" closes the dialog with
    go_to_first_new() True; it is disabled when the active experiment has no
    new points.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        active_experiment: str,
        unmatched_videos: list[str] | None = None,
        dropped: dict[tuple[str, str], int] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._go_to_first_new = False
        self.setWindowTitle("Sync summary")
        self.resize(720, 360)

        layout = QVBoxLayout(self)

        if rows.empty:
            layout.addWidget(QLabel("No new points were brought in."))
        else:
            first = rows.iloc[0]
            when = str(first["timestamp"]).replace("T", " ")
            kind = KIND_NAMES.get(str(first["kind"]), str(first["kind"]))
            totals = (
                f"{int(rows['n_new_user'].sum())} new SLEAP label(s), "
                f"{int(rows['n_new_pred'].sum())} new and "
                f"{int(rows['n_updated_pred'].sum())} updated prediction(s) "
                f"in {int(rows['n_frames'].sum())} frame(s)"
            )
            layout.addWidget(QLabel(f"<b>{kind}</b> — {when}<br>{totals}"))

        rows = rows.assign(_active=rows["experiment"].astype(str) != active_experiment)
        rows = rows.sort_values(["_active", "experiment", "trial"], kind="stable")

        table = QTableWidget(len(rows), len(COLUMNS))
        table.setHorizontalHeaderLabels([title for _, title in COLUMNS])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for r, (_, row) in enumerate(rows.iterrows()):
            for c, (key, _) in enumerate(COLUMNS):
                value = row[key]
                item = QTableWidgetItem(str(value) if isinstance(value, str) else str(int(value)))
                if key not in ("experiment", "trial"):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(r, c, item)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        layout.addWidget(table)

        if unmatched_videos:
            shown = ", ".join(unmatched_videos[:10])
            more = f" and {len(unmatched_videos) - 10} more" if len(unmatched_videos) > 10 else ""
            note = QLabel(
                f"{len(unmatched_videos)} video(s) in the imported files match no trial "
                f"and were skipped: {shown}{more}"
            )
            note.setWordWrap(True)
            layout.addWidget(note)

        if dropped:
            note = QLabel(dropped_text(dropped))
            note.setWordWrap(True)
            layout.addWidget(note)

        buttons = QDialogButtonBox()
        go_btn = QPushButton("Go to first new frame")
        has_active = bool((rows["experiment"].astype(str) == active_experiment).any())
        go_btn.setEnabled(has_active)
        if not has_active and not rows.empty:
            go_btn.setToolTip("The new points are in other experiments — switch experiment first")
        go_btn.clicked.connect(self._on_go)
        buttons.addButton(go_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def go_to_first_new(self) -> bool:
        return self._go_to_first_new

    def _on_go(self) -> None:
        self._go_to_first_new = True
        self.accept()

"""Application entry point — creates QApplication and launches MainWindow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QProgressDialog

from label3d.project import LoadCancelled, Project
from label3d.state import ProjectState
from label3d.version import version_string
from label3d.widgets.main_window import MainWindow


STARTUP_DIALOG_DELAY_MS = 500


def run(project_dir: str | Path | None = None):
    """Launch the label3d application.

    Args:
        project_dir: Path to the project directory. If None, uses current directory.
    """
    parser = argparse.ArgumentParser(prog="label3d")
    parser.add_argument("project_dir", nargs="?", help="Project directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="version", version=f"label3d {version_string()}")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.debug:
        logging.getLogger("label3d").setLevel(logging.DEBUG)

    app = QApplication.instance() or QApplication(sys.argv)

    if project_dir is None:
        if args.project_dir is not None:
            project_dir = args.project_dir
        elif args.project_dir is None:
            project_dir = QFileDialog.getExistingDirectory(
                None, "Select Project Directory", str(Path.cwd())
            )
            if not project_dir:
                print("No project directory selected. Exiting.", file=sys.stderr)
                sys.exit(1)

    project_dir = Path(project_dir)

    # Checking the videos can be slow on a network share, and very slow the
    # first time (frame counts). The dialog appears only once loading has
    # taken about half a second.
    dialog = QProgressDialog("Starting up…", "Cancel", 0, 0)
    dialog.setWindowTitle(f"label3d — {project_dir.name}")
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(STARTUP_DIALOG_DELAY_MS)
    dialog.setMinimumWidth(480)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)

    def progress(done: int, total: int, text: str) -> bool:
        dialog.setMaximum(total)
        dialog.setLabelText(text)
        dialog.setValue(done)
        app.processEvents()
        return not dialog.wasCanceled()

    try:
        project = Project.load(project_dir, progress=progress)
    except LoadCancelled:
        print("Cancelled while opening the project.", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading project: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        dialog.close()

    state = ProjectState()
    window = MainWindow(project, state)
    window.show()

    sys.exit(app.exec())

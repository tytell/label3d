"""Application entry point — creates QApplication and launches MainWindow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from label3d.project import Project
from label3d.state import ProjectState
from label3d.widgets.main_window import MainWindow


def run(project_dir: str | Path | None = None):
    """Launch the label3d application.

    Args:
        project_dir: Path to the project directory. If None, uses current directory.
    """
    parser = argparse.ArgumentParser(prog="label3d")
    parser.add_argument("project_dir", nargs="?", help="Project directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
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
        project_dir = args.project_dir if args.project_dir is not None else Path.cwd()

    project_dir = Path(project_dir)

    try:
        project = Project.load(project_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading project: {e}", file=sys.stderr)
        sys.exit(1)

    state = ProjectState()
    window = MainWindow(project, state)
    window.show()

    sys.exit(app.exec())

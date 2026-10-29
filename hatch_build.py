"""Hatch build hook: record the git commit in a wheel (Phase 20).

label3d.version.commit() reads git when it runs from a checkout; an installed
wheel has no checkout, so the commit is written to label3d/_build_info.py.
"""

import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        path = Path(self.root) / "src" / "label3d" / "_build_info.py"
        path.write_text(f'COMMIT = "{commit}"\n')
        build_data["force_include"][str(path)] = "label3d/_build_info.py"

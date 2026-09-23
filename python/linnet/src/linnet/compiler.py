"""Running the `linnet` compiler as a subprocess."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class LinnetError(Exception):
    """The compiler rejected the request, or the toolchain is missing."""


def find_compiler() -> str:
    """The `linnet` executable: LINNET_BIN, then PATH."""
    candidate = os.environ.get("LINNET_BIN")
    if candidate and Path(candidate).exists():
        return candidate
    found = shutil.which("linnet")
    if found is None:
        raise LinnetError("cannot find the `linnet` executable; set LINNET_BIN or add it to PATH")
    return found


def run_compiler(*arguments: str, stdin: str | None = None) -> str:
    """Runs one compiler command and returns its standard output."""
    completed = subprocess.run(
        [find_compiler(), *arguments],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise LinnetError(
            f"linnet {' '.join(arguments)} failed:\n{completed.stderr}{completed.stdout}".rstrip()
        )
    return completed.stdout


def std_arguments(std_root: str | Path | None) -> list[str]:
    """`--std <dir>` when a standard library directory is given."""
    return [] if std_root is None else ["--std", str(std_root)]

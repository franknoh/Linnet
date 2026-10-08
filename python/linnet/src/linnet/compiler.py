"""Running the `linnet` compiler as a subprocess."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path


class LinnetError(Exception):
    """The compiler rejected the request, or the toolchain is missing."""


class PlanError(LinnetError):
    """A plan could not be produced, read, or instantiated."""


def find_compiler() -> str:
    """The `linnet` executable: LINNET_BIN, then the one this package's wheel
    installed, then PATH."""
    candidate = os.environ.get("LINNET_BIN")
    if candidate and Path(candidate).exists():
        return candidate
    installed = installed_compiler()
    if installed is not None:
        return str(installed)
    found = shutil.which("linnet")
    if found is None:
        raise LinnetError(
            "cannot find the `linnet` executable; set LINNET_BIN or add it to PATH "
            "(https://linnet.franknoh.dev/guide/installation)"
        )
    return found


def installed_compiler() -> Path | None:
    """The executable a platform wheel of `linnet-lang` installs beside the
    Python scripts, or None for a pure or source install."""
    try:
        files = metadata.distribution("linnet-lang").files or []
    except metadata.PackageNotFoundError:
        return None
    for file in files:
        if file.name in ("linnet", "linnet.exe") and file.parent.name in ("bin", "Scripts"):
            path = Path(str(file.locate())).resolve()
            if path.is_file():
                return path
    return None


def run_compiler(
    *arguments: str, stdin: str | None = None, error: type[LinnetError] | None = None
) -> str:
    """Runs one compiler command and returns its standard output. A failure
    raises `error` (a `LinnetError` by default) with what the compiler said."""
    completed = subprocess.run(
        [find_compiler(), *arguments],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise (error or LinnetError)(
            f"linnet {' '.join(arguments)} failed:\n{completed.stderr}{completed.stdout}".rstrip()
        )
    return completed.stdout


def parse_binding(text: str) -> tuple[str, int | str]:
    """`NAME=VALUE` as a generic's binding: a whole number as an `int`, any
    other value (a dtype) as text."""
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise LinnetError(f"`{text}` is not NAME=VALUE")
    return name, int(value) if value.lstrip("-").isdigit() else value


def bind_arguments(bindings: Mapping[str, object]) -> list[str]:
    """`--bind NAME=VALUE` for each generic binding."""
    return [part for name, value in bindings.items() for part in ("--bind", f"{name}={value}")]


def std_arguments(std_root: str | Path | None) -> list[str]:
    """`--std <dir>` when a standard library directory is given."""
    return [] if std_root is None else ["--std", str(std_root)]

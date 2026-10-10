"""Importing generated Python source as a module of its own."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import re
import sys
from pathlib import Path
from types import ModuleType

from .compiler import PlanError, write_atomically
from .home import home

# Each import a module name of its own: two models whose entries generate
# the same source share its file, never its module's state.
_imports = itertools.count()


def generated_directory() -> Path:
    """Where generated modules are written: `$LINNET_HOME/compiled/modules`."""
    return home() / "compiled" / "modules"


def write_generated(directory: Path, name: str, source: str) -> Path:
    """`source` at `directory/<name>-<digest>.py`: named by its content, so
    a file there is written once and never changes under a reader."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    path = directory / f"{name}-{digest}.py"
    if not path.is_file():
        write_atomically(path, source)
    return path


def import_generated(directory: Path, name: str, source: str) -> ModuleType:
    """Writes `source` into `directory` (`write_generated`) and imports it
    as a new module. Its file stays, so `@triton.jit` and tracebacks can read
    it. The module's `__linnet_path__` is the file."""
    path = write_generated(directory, name, source)
    module_name = "linnet_generated_" + re.sub(r"\W", "_", f"{path.stem}_{next(_imports)}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PlanError(f"cannot load the generated module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    setattr(module, "__linnet_path__", path)  # noqa: B010
    return module


__all__ = ["generated_directory", "import_generated", "write_generated"]

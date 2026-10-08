"""Generated Python source (`linnet torch`, `linnet jax`) imported as a
module of its own."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

from .compiler import PlanError


def import_generated(directory: Path, name: str, source: str) -> ModuleType:
    """Writes `source` to `directory/name.py` and imports it. Its name in
    `sys.modules` comes from the file's path, so models (and functions) whose
    entries share a name never replace each other's modules. The module's
    `__linnet_path__` is the file."""
    path = directory / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    module_name = "linnet_generated_" + re.sub(r"\W", "_", f"{directory.name}_{name}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PlanError(f"cannot load the generated module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    setattr(module, "__linnet_path__", path)  # noqa: B010
    return module


__all__ = ["import_generated"]

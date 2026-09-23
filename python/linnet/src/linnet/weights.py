"""SafeTensors checkpoints and the bindings that map Linnet paths onto them."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from .compiler import LinnetError


def safetensors_files(weights: str | Path) -> list[Path]:
    """The `.safetensors` files a path names: one file, or every file in a directory."""
    path = Path(weights)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    if not files or not all(file.exists() for file in files):
        raise LinnetError(f"no .safetensors files under {weights}")
    return files


def read_arrays(weights: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Loads a checkpoint as NumPy arrays by tensor name.

    `weights` is a mapping (returned as arrays), a `.safetensors` file, or a
    directory of them.
    """
    if isinstance(weights, Mapping):
        return {str(name): np.asarray(value) for name, value in weights.items()}
    from safetensors.numpy import load_file  # type: ignore[import-untyped]

    loaded: dict[str, Any] = {}
    for file in safetensors_files(weights):
        loaded.update(cast(dict[str, Any], load_file(str(file))))
    return loaded


def read_bindings(bindings: str | Path) -> dict[str, str]:
    """Reads a JSON object mapping Linnet parameter paths to checkpoint tensor names."""
    loaded: object = json.loads(Path(bindings).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise LinnetError("bindings must be a JSON object mapping parameter paths to tensor names")
    return {str(path): str(name) for path, name in cast(dict[Any, Any], loaded).items()}


def apply_bindings(loaded: Mapping[str, Any], bindings: str | Path | None) -> dict[str, Any]:
    """Adds every bound Linnet path to a checkpoint mapping, keeping the original names."""
    if bindings is None:
        return dict(loaded)
    mapping = read_bindings(bindings)
    return {
        **loaded,
        **{path: loaded[name] for path, name in mapping.items() if name in loaded},
    }

"""SafeTensors checkpoints and the bindings that map Linnet paths onto them."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class RawTensor:
    """One tensor of a SafeTensors file as bytes: no framework dtype needed."""

    name: str
    dtype: str  # the file's dtype name: BF16, F32, I64, ...
    shape: tuple[int, ...]
    data: bytes


def iter_safetensors(weights: str | Path) -> Iterator[RawTensor]:
    """Reads a checkpoint tensor by tensor from the SafeTensors header.

    Works for every dtype, including `BF16`, which NumPy has no type for.
    """
    for file in safetensors_files(weights):
        with file.open("rb") as handle:
            (size,) = struct.unpack("<Q", handle.read(8))
            header = cast(dict[str, Any], json.loads(handle.read(size).decode("utf-8")))
            base = 8 + size
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                entry = cast(dict[str, Any], info)
                start, end = (int(o) for o in cast(list[Any], entry["data_offsets"]))
                handle.seek(base + start)
                yield RawTensor(
                    name=str(name),
                    dtype=str(entry["dtype"]),
                    shape=tuple(int(d) for d in cast(list[Any], entry["shape"])),
                    data=handle.read(end - start),
                )

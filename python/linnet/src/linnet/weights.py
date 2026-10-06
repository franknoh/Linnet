"""SafeTensors checkpoints and the bindings that map Linnet paths onto them."""

# NumPy's stubs leave `frombuffer` partly unknown on some versions.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from . import ir
from .compiler import LinnetError
from .dtypes import BY_SAFETENSORS, from_safetensors


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
    directory of them. `BF16` tensors come back as `ml_dtypes.bfloat16`
    arrays (installed with JAX).
    """
    if isinstance(weights, Mapping):
        return {str(name): np.asarray(value) for name, value in weights.items()}
    return {tensor.name: numpy_array(tensor) for tensor in iter_safetensors(weights)}


def numpy_array(tensor: RawTensor) -> np.ndarray:
    """A raw tensor as a NumPy array over its bytes, in its own dtype."""
    found = BY_SAFETENSORS.get(tensor.dtype)
    if found is None:
        raise LinnetError(f"`{tensor.name}` has dtype {tensor.dtype}, which Linnet does not read")
    if found.name == "bf16":
        import ml_dtypes  # type: ignore[import-untyped]

        dtype = np.dtype(ml_dtypes.bfloat16)
    else:
        dtype = np.dtype(found.numpy)
    # Over a copy the array owns, so it is writable as any other.
    return np.frombuffer(bytearray(tensor.data), dtype=dtype).reshape(tensor.shape)


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Values as bf16 bit patterns (`uint16`, NumPy having no bf16), rounded
    to nearest, ties to even, as every framework's cast does."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype(np.uint16)


def from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    """bf16 bit patterns (`uint16`) as the f32 values they are."""
    return (np.asarray(bits).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def decode_floats(dtype: str, data: bytes) -> np.ndarray:
    """Floating-point bytes of SafeTensors dtype `dtype` (`F32`, `F16`, `F64`,
    `BF16`) as a flat f32 array."""
    if dtype == "BF16":
        return from_bf16_bits(np.frombuffer(data, dtype=np.uint16))
    found = BY_SAFETENSORS.get(dtype)
    if found is None or not found.is_float:
        raise LinnetError(f"{dtype} is not a floating-point dtype")
    return np.frombuffer(data, dtype=np.dtype(found.numpy)).astype(np.float32)


def encode_floats(values: np.ndarray, dtype: str) -> bytes:
    """Values as bytes of SafeTensors floating-point dtype `dtype`."""
    if dtype == "BF16":
        return to_bf16_bits(values).tobytes()
    found = BY_SAFETENSORS.get(dtype)
    if found is None or not found.is_float:
        raise LinnetError(f"{dtype} is not a floating-point dtype")
    return np.ascontiguousarray(values, dtype=np.float32).astype(np.dtype(found.numpy)).tobytes()


def read_bindings(bindings: str | Path) -> dict[str, str]:
    """Reads a JSON object mapping Linnet parameter paths to checkpoint tensor names."""
    loaded: object = json.loads(Path(bindings).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise LinnetError("bindings must be a JSON object mapping parameter paths to tensor names")
    return {str(path): str(name) for path, name in cast(dict[Any, Any], loaded).items()}


def paths_by_tensor(mapping: Mapping[str, str]) -> dict[str, list[str]]:
    """The Linnet paths bound to each checkpoint tensor, in binding order:
    a tied embedding and output head share one."""
    paths: dict[str, list[str]] = {}
    for path, name in mapping.items():
        paths.setdefault(name, []).append(path)
    return paths


def match_checkpoint(
    program: ir.Program,
    bindings: ir.Bindings,
    tensors: Mapping[str, tuple[tuple[int, ...], str]],
    mapping: Mapping[str, str],
) -> tuple[list[tuple[ir.ManifestEntry, list[tuple[str, str]]]], list[str]]:
    """Each parameter of `program` against a checkpoint's tensors (name ->
    shape and SafeTensors dtype), a path read from `mapping[path]` when
    bound: the tensors found for each manifest entry as (path, tensor
    name), and every problem (a required tensor missing, a shape or dtype
    that differs, a size that does not evaluate)."""
    found: list[tuple[ir.ManifestEntry, list[tuple[str, str]]]] = []
    problems: list[str] = []
    for entry in program.manifest:
        if entry.kind != "param":
            continue
        try:
            shape = ir.evaluate_shape(entry.shape, bindings)
            dtype = ir.evaluate_dtype(entry.dtype, bindings)
            paths = ir.expand_paths(entry, bindings)
        except LinnetError as error:
            problems.append(f"{entry.path}: {error}")
            continue
        located: list[tuple[str, str]] = []
        for path in paths:
            source = mapping.get(path, path)
            if source not in tensors:
                if not entry.optional:
                    problems.append(f"missing tensor `{source}` for `{path}`")
                continue
            found_shape, found_dtype = tensors[source]
            if found_shape != shape:
                problems.append(
                    f"`{source}` has shape {list(found_shape)}, `{path}` needs {list(shape)}"
                )
            elif from_safetensors(found_dtype) != dtype:
                problems.append(f"`{source}` is {found_dtype}, `{path}` needs {dtype}")
            else:
                located.append((path, source))
        found.append((entry, located))
    return found, problems


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


@dataclass(frozen=True, slots=True)
class TensorLocation:
    """Where a tensor lives in a SafeTensors file, from its header."""

    file: Path
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offsets in the file
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    def read(self) -> bytes:
        with self.file.open("rb") as handle:
            handle.seek(self.start)
            return handle.read(self.nbytes)


def read_header(file: str | Path) -> dict[str, Any]:
    """The JSON header of a SafeTensors file on disk."""
    return _header(Path(file))[0]


def _header(file: Path) -> tuple[dict[str, Any], int]:
    """A file's header and where its tensor data starts."""
    with file.open("rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        return cast(dict[str, Any], json.loads(handle.read(size).decode("utf-8"))), 8 + size


def header_tensors(header: Mapping[str, Any]) -> dict[str, tuple[tuple[int, ...], str]]:
    """Each tensor of a SafeTensors header as its shape and dtype name."""
    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        entry = cast(dict[str, Any], info)
        shape = tuple(int(d) for d in cast(list[Any], entry["shape"]))
        tensors[str(name)] = (shape, str(entry["dtype"]))
    return tensors


def safetensors_index(weights: str | Path) -> dict[str, TensorLocation]:
    """Every tensor of a checkpoint by name, from the file headers alone."""
    index: dict[str, TensorLocation] = {}
    for file in safetensors_files(weights):
        header, base = _header(file)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            entry = cast(dict[str, Any], info)
            start, end = (int(o) for o in cast(list[Any], entry["data_offsets"]))
            index[str(name)] = TensorLocation(
                file=file,
                dtype=str(entry["dtype"]),
                shape=tuple(int(d) for d in cast(list[Any], entry["shape"])),
                start=base + start,
                end=base + end,
            )
    return index


def iter_safetensors(weights: str | Path) -> Iterator[RawTensor]:
    """Reads a checkpoint tensor by tensor from the SafeTensors header.

    Works for every dtype, including `BF16`, which NumPy has no type for.
    """
    for name, location in safetensors_index(weights).items():
        yield RawTensor(name, location.dtype, location.shape, location.read())


@dataclass(frozen=True, slots=True)
class LazyBytes:
    """A tensor's bytes, produced only when written: `nbytes` long, from
    `read()`. Saving a model holds one tensor's copy at a time."""

    nbytes: int
    read: Callable[[], bytes]


def write_safetensors(
    path: str | Path,
    tensors: Iterable[tuple[str, str, tuple[int, ...], TensorLocation | LazyBytes | bytes]],
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Writes a SafeTensors file from raw tensors, streaming those given as locations.

    `tensors` yields `(name, dtype, shape, data)` with `data` bytes, a
    `TensorLocation` to copy from, or `LazyBytes` read as it is written; sizes
    come from the location or the `nbytes` given, so a checkpoint larger than
    memory copies without loading it whole.
    """
    entries = list(tensors)
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for name, dtype, shape, data in entries:
        size = data.nbytes if isinstance(data, TensorLocation | LazyBytes) else len(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)  # the header is padded to 8 bytes
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for name, _, _, data in entries:
            if isinstance(data, TensorLocation):
                with data.file.open("rb") as source:
                    source.seek(data.start)
                    remaining = data.nbytes
                    while remaining:
                        chunk = source.read(min(remaining, 64 << 20))
                        out.write(chunk)
                        remaining -= len(chunk)
            elif isinstance(data, LazyBytes):
                chunk = data.read()
                if len(chunk) != data.nbytes:
                    raise LinnetError(f"`{name}` produced {len(chunk)} bytes, not {data.nbytes}")
                out.write(chunk)
            else:
                out.write(data)
    return target

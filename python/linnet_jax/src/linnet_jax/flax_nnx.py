"""A Linnet model as a Flax NNX module.

`to_nnx` mirrors the block hierarchy of a loaded entry as nested
`nnx.Module`s: every `param` is an `nnx.Param`, every `buffer` an
`nnx.Variable`, a `sub` member a child module, and a sub array an `nnx.List`,
so the parameter paths of the model (`layers.0.attn.q`) are the paths of the
module's state. Calling the module runs the compiled entry with the arrays
the module currently holds, so state produced by `nnx.split`, checkpoints,
or sharding flows into the call; the entry itself has no VJP.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp

from .compiler import LinnetError
from .load import LinnetFunction, load


def _import_nnx() -> Any:
    try:
        from flax import nnx  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise LinnetError("Flax is not installed; `pip install flax`") from error
    return nnx


def to_nnx(function: LinnetFunction) -> Any:
    """The loaded entry as an `nnx.Module` owning its parameters."""
    nnx = _import_nnx()
    plan = function.plan
    blocks = cast(dict[str, Any], plan["blocks"])
    root_name = str(cast(dict[str, Any], plan["root"])["name"])
    weights = function.weights

    class LinnetBlock(nnx.Module):  # type: ignore[misc]
        """One block of the hierarchy; its attributes are the block's members."""

        def __init__(self, block: str, prefix: str) -> None:
            for member in cast(list[dict[str, Any]], blocks[block]["members"]):
                name = str(member["name"])
                path = f"{prefix}{name}"
                kind = cast(dict[str, Any], member["type"])
                if member["kind"] == "sub":
                    setattr(self, name, _child(kind, path))
                elif path in weights:
                    variable = nnx.Param if member["kind"] == "param" else nnx.Variable
                    setattr(self, name, variable(jnp.asarray(weights[path])))
                else:
                    setattr(self, name, None)  # an optional parameter that is absent

    def _child(kind: dict[str, Any], path: str) -> Any:
        if kind["kind"] == "array":
            element = cast(dict[str, Any], kind["element"])
            return nnx.List(
                [_child(element, f"{path}.{i}") for i in range(_length(kind, path, function))]
            )
        if kind["kind"] != "block":
            raise LinnetError(f"member `{path}` has an unexpected type {kind['kind']}")
        return LinnetBlock(str(kind["name"]), f"{path}.")

    class LinnetModule(LinnetBlock):
        """The root block; calling it runs the entry on the module's arrays."""

        def __init__(self) -> None:
            super().__init__(root_name, "")

        def __call__(self, *inputs: Any) -> Any:
            return function.apply(_collect(self, ""), *inputs)

    LinnetModule.__name__ = LinnetModule.__qualname__ = root_name
    return LinnetModule()


def _length(kind: dict[str, Any], path: str, function: LinnetFunction) -> int:
    """The element count of a sub array: from its type when the length is a
    literal or a bound generic, otherwise from the weights it has."""
    length = kind["length"]
    if isinstance(length, int):
        return length
    if isinstance(length, dict) and "name" in length:
        bound = function.generics.get(str(cast(dict[str, Any], length)["name"]))
        if isinstance(bound, int):
            return bound
    pattern = re.compile(re.escape(path) + r"\.(\d+)(\.|$)")
    indices = [int(m.group(1)) for m in map(pattern.match, function.weights) if m]
    if not indices:
        raise LinnetError(f"cannot tell how many elements `{path}` has")
    return max(indices) + 1


def _collect(module: Any, prefix: str) -> dict[str, Any]:
    """Every variable under `module` by parameter path."""
    nnx = _import_nnx()
    out: dict[str, Any] = {}
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        path = f"{prefix}{name}"
        if isinstance(value, nnx.Variable):
            out[path] = value[...]
        elif isinstance(value, nnx.List):
            for i, element in enumerate(cast(list[Any], value)):
                out.update(_collect(element, f"{path}.{i}."))
        elif isinstance(value, nnx.Module):
            out.update(_collect(value, f"{path}."))
    return out


def load_nnx(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path | Mapping[str, Any],
    root: str | None = None,
    entry: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
) -> Any:
    """`load` followed by `to_nnx`."""
    return to_nnx(
        load(
            source,
            generics=generics,
            weights=weights,
            root=root,
            entry=entry,
            bindings=bindings,
            std_root=std_root,
        )
    )

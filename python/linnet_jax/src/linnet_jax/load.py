"""Materializing a Linnet entry as a JAX function.

`load` reads the model's plan to learn the entry's signature and parameter
paths, then, for every input shape it is called with, asks the compiler for
the StableHLO of the entry with those dimensions bound and wraps the module
as a `jax.export.Exported`. Calling the result runs under XLA and composes
with `jax.jit`; weights travel as ordinary array arguments, so nothing
model-specific exists on the Python side. Inference only: the wrapped module
has no VJP.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from .compiler import LinnetError, run_compiler

_DTYPES: dict[str, Any] = {
    "i1": np.bool_,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "ui8": np.uint8,
    "ui16": np.uint16,
    "ui32": np.uint32,
    "ui64": np.uint64,
    "f16": np.float16,
    "bf16": jnp.bfloat16,
    "f32": np.float32,
    "f64": np.float64,
}

_LINNET_DTYPES: dict[str, Any] = {
    "bool": np.bool_,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    "f16": np.float16,
    "bf16": jnp.bfloat16,
    "f32": np.float32,
    "f64": np.float64,
}


def _std_arguments(std_root: str | Path | None) -> list[str]:
    return [] if std_root is None else ["--std", str(std_root)]


def _read_weights(weights: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(weights, Mapping):
        return {str(k): np.asarray(v) for k, v in weights.items()}
    from safetensors.numpy import load_file  # type: ignore[import-untyped]

    path = Path(weights)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    if not files:
        raise LinnetError(f"no .safetensors files under {weights}")
    loaded: dict[str, Any] = {}
    for file in files:
        loaded.update(cast(dict[str, Any], load_file(str(file))))
    return loaded


class LinnetFunction:
    """A Linnet entry as a callable over JAX arrays."""

    def __init__(
        self,
        source: Path,
        plan: dict[str, Any],
        generics: Mapping[str, int | str],
        weights: dict[str, Any],
        std_root: str | Path | None,
        root: str,
        entry: str,
    ) -> None:
        self._source = source
        self._generics = dict(generics)
        self._weights = weights
        self._std_root = std_root
        self.root = root
        self.entry = entry
        functions = cast(list[dict[str, Any]], plan["functions"])
        matching = [f for f in functions if f["name"].endswith(f"::{root}.{entry}")]
        if not matching:
            raise LinnetError(f"block `{root}` has no entry `{entry}`")
        self._signature = matching[0]
        manifest = cast(list[dict[str, Any]], plan["manifest"])
        self.parameter_paths: list[str] = [entry["path"] for entry in manifest]
        self.optional_paths: set[str] = {e["path"] for e in manifest if e["optional"]}
        self._cache: dict[tuple[Any, ...], Callable[..., Any]] = {}
        self._check_weights(manifest)

    def _check_weights(self, manifest: list[dict[str, Any]]) -> None:
        # Manifest paths spell array elements `[*]`; weights spell them `.0`.
        def matches(pattern: str, path: str) -> bool:
            return re.fullmatch(re.escape(pattern).replace(r"\[\*\]", r"\.\d+"), path) is not None

        for entry in manifest:
            if entry["optional"]:
                continue
            if not any(matches(entry["path"], path) for path in self._weights):
                raise LinnetError(f"missing weights for `{entry['path']}`")
        present = [p for p in self.optional_paths if any(matches(p, w) for w in self._weights)]
        if present and len(present) != len(self.optional_paths):
            raise LinnetError(
                "optional parameters must all be present or all absent; present: "
                + ", ".join(sorted(present))
            )
        self.optionals_present = bool(present)

    # ---- entry generics from input shapes

    def _bindings_for(self, inputs: Sequence[Any]) -> dict[str, int | str]:
        arguments = cast(list[dict[str, Any]], self._signature["body"]["args"])[1:]
        if len(arguments) != len(inputs):
            raise LinnetError(
                f"entry `{self.entry}` takes {len(arguments)} inputs, got {len(inputs)}"
            )
        bindings: dict[str, int | str] = dict(self._generics)
        for argument, value in zip(arguments, inputs, strict=True):
            declared = cast(dict[str, Any], argument["type"])
            if declared.get("kind") != "tensor":
                continue
            shape = cast(list[Any], declared["shape"])
            actual = tuple(int(d) for d in np.shape(value))
            if len(shape) != len(actual):
                raise LinnetError(
                    f"input `{argument['name']}` has rank {len(actual)}, expected {len(shape)}"
                )
            for unit, size in zip(shape, actual, strict=True):
                if isinstance(unit, dict) and "sym" in unit:
                    name = str(cast(dict[str, Any], unit)["name"])
                    previous = bindings.get(name)
                    if previous is not None and previous != size:
                        raise LinnetError(f"dimension `{name}` is both {previous} and {size}")
                    bindings[name] = size
                elif isinstance(unit, int) and unit != size:
                    raise LinnetError(
                        f"input `{argument['name']}` has extent {size} where {unit} is declared"
                    )
        return bindings

    def _compile(self, bindings: Mapping[str, int | str]) -> Callable[..., Any]:
        arguments = ["stablehlo", "--root", self.root, "--entry", self.entry]
        arguments += ["--optionals", "present" if self.optionals_present else "absent"]
        for name, value in bindings.items():
            arguments += ["--bind", f"{name}={value}"]
        text = run_compiler(*arguments, *_std_arguments(self._std_root), str(self._source))
        paths = re.findall(r'linnet\.path = "([^"]+)"', text)
        missing = [path for path in paths if path not in self._weights]
        if missing:
            raise LinnetError("missing weights: " + ", ".join(missing))
        exported = _wrap_module(text)
        parameters = [jnp.asarray(self._weights[path]) for path in paths]

        def call(*inputs: Any) -> Any:
            return exported.call(*inputs, *parameters)

        return call

    def __call__(self, *inputs: Any) -> Any:
        bindings = self._bindings_for(inputs)
        key = tuple(sorted(bindings.items()))
        if key not in self._cache:
            self._cache[key] = self._compile(bindings)
        return self._cache[key](*inputs)


def _wrap_module(text: str) -> Any:
    """A `jax.export.Exported` around a StableHLO module whose function is
    `@main`, so JAX can call it like one of its own exports."""
    from jax import export
    from jax._src.interpreters import mlir as jax_mlir  # pyright: ignore[reportPrivateUsage]
    from jax._src.lib.mlir import ir  # pyright: ignore[reportPrivateUsage]
    from jax.tree_util import tree_flatten

    mlir: Any = jax_mlir
    with mlir.make_ir_context():
        module = cast(Any, ir.Module).parse(text)
        function = module.body.operations[0]
        block = function.regions[0].blocks[0]
        in_types = [cast(Any, ir.RankedTensorType)(a.type) for a in block.arguments]
        signature = cast(Any, ir.FunctionType)(function.attributes["function_type"].value)
        out_types = [cast(Any, ir.RankedTensorType)(r) for r in signature.results]
        serialized = mlir.module_to_bytecode(module)

    from jax import core as jax_core

    def aval(tensor_type: Any) -> Any:
        element = str(tensor_type.element_type)
        if element not in _DTYPES:
            raise LinnetError(f"unsupported element type {element}")
        return jax_core.ShapedArray(tuple(tensor_type.shape), _DTYPES[element])

    in_avals = tuple(aval(t) for t in in_types)
    out_avals = tuple(aval(t) for t in out_types)
    in_tree = tree_flatten((tuple(0 for _ in in_avals), {}))[1]
    out_tree = tree_flatten(0 if len(out_avals) == 1 else tuple(0 for _ in out_avals))[1]
    exported_type: Any = export.Exported
    return exported_type(
        fun_name="main",
        in_tree=in_tree,
        in_avals=in_avals,
        out_tree=out_tree,
        out_avals=out_avals,
        _in_named_shardings=(None,) * len(in_avals),
        _out_named_shardings=(None,) * len(out_avals),
        in_shardings_hlo=(None,) * len(in_avals),
        out_shardings_hlo=(None,) * len(out_avals),
        nr_devices=1,
        platforms=(jax.default_backend(),),
        ordered_effects=(),
        unordered_effects=(),
        disabled_safety_checks=(),
        mlir_module_serialized=serialized,
        calling_convention_version=cast(Any, export).maximum_supported_calling_convention_version,
        module_kept_var_idx=tuple(range(len(in_avals))),
        uses_global_constants=False,
        _get_vjp=None,
    )


def load(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path | Mapping[str, Any],
    root: str | None = None,
    entry: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
) -> LinnetFunction:
    """Materializes the entry of a root block as a JAX callable.

    `generics` binds the root block's generic parameters; the entry's own are
    bound from the input shapes at each call. `weights` is a mapping from
    parameter path to array, a `.safetensors` file, or a directory of them;
    `bindings` is an optional JSON file mapping parameter paths to tensor
    names in the weights.
    """
    source_path = Path(source)
    plan_arguments = ["plan", "--no-optimize"]
    if root is not None:
        plan_arguments += ["--root", root]
    plan = cast(
        dict[str, Any],
        json.loads(run_compiler(*plan_arguments, *_std_arguments(std_root), str(source_path))),
    )
    root_name = str(cast(dict[str, Any], plan["root"])["name"])
    entries = [
        f["name"].rsplit(".", 1)[1]
        for f in cast(list[dict[str, Any]], plan["functions"])
        if f["kind"] == "entry" and f["block"] == root_name
    ]
    entry_name = entry
    if entry_name is None:
        if len(entries) != 1:
            raise LinnetError(f"block `{root_name}` has {len(entries)} entries; name one")
        entry_name = entries[0]
    loaded = _read_weights(weights)
    if bindings is not None:
        mapping = cast(dict[str, str], json.loads(Path(bindings).read_text(encoding="utf-8")))
        loaded = {
            **loaded,
            **{path: loaded[name] for path, name in mapping.items() if name in loaded},
        }
    return LinnetFunction(source_path, plan, generics, loaded, std_root, root_name, entry_name)


__all__ = ["LinnetFunction", "load"]

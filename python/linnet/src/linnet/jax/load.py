"""Materializing a Linnet entry as a JAX function.

`load` reads the model's plan to learn the entry's signature and parameter
paths, then, for every input shape it is called with, asks the compiler for
the StableHLO of the entry with those dimensions bound and wraps the module
as a `jax.export.Exported`. Calling the result runs under XLA and composes
with `jax.jit`; weights travel as ordinary array arguments, so nothing
model-specific exists on the Python side. Inference only: the wrapped module
has no VJP.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..compiler import LinnetError, run_compiler, std_arguments
from ..weights import apply_bindings, read_arrays

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
        numerics: str = "equivalent",
    ) -> None:
        self._source = source
        self.numerics = numerics
        self.plan = plan
        self.generics = dict(generics)
        self._generics = self.generics
        self.weights = weights
        self._weights = self.weights
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
        self._cache: dict[tuple[Any, ...], CompiledEntry] = {}
        self._check_weights(manifest)

    def _check_weights(self, manifest: list[dict[str, Any]]) -> None:
        # Manifest paths spell array elements `[*]`; weights spell them `.0`.
        def matches(pattern: str, path: str) -> bool:
            return re.fullmatch(re.escape(pattern).replace(r"\[\*\]", r"\.\d+"), path) is not None

        for entry in manifest:
            if entry["optional"] or entry["kind"] == "state":  # state is never a weight
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

    def _compile(self, bindings: Mapping[str, int | str]) -> CompiledEntry:
        arguments = ["stablehlo", "--root", self.root, "--entry", self.entry]
        arguments += ["--numerics", self.numerics]
        arguments += ["--optionals", "present" if self.optionals_present else "absent"]
        for name, value in bindings.items():
            arguments += ["--bind", f"{name}={value}"]
        text = run_compiler(*arguments, *std_arguments(self._std_root), str(self._source))
        paths = re.findall(r'linnet\.path = "([^"]+)"', text)
        missing = [path for path in paths if path not in self._weights]
        if missing:
            raise LinnetError("missing weights: " + ", ".join(missing))
        # State threaded by the exporter: inputs read before the call and
        # results assigned by it, both by path.
        state_inputs = re.findall(r'linnet\.state = "([^"]+)"', text)
        listed = re.search(r"linnet\.states = \[([^\]]*)\]", text)
        state_outputs = re.findall(r'"([^"]+)"', listed.group(1)) if listed else []
        exported = _wrap_module(text)
        # The compiled program under `jit`, so a call is one dispatch; the
        # loaded weights go to the device once rather than per call.
        call = jax.jit(exported.call)
        arrays = [_device_array(self._weights[path]) for path in paths]
        state_avals = dict(
            zip(
                state_inputs,
                exported.in_avals[len(exported.in_avals) - len(state_inputs) :],
                strict=True,
            )
        )
        return CompiledEntry(
            paths, state_inputs, state_outputs, state_avals, exported, call, arrays
        )

    def apply(self, parameters: Mapping[str, Any], *inputs: Any, state: Any = None) -> Any:
        """Runs the entry with `parameters` (path -> array) in place of the
        loaded weights, so a framework module can own the arrays. An entry
        that touches `state` members takes their values before the call in
        `state` (path -> array; a missing one starts at zeros) and returns
        `(result, new_state)`; other entries return the result alone."""
        bindings = self._bindings_for(inputs)
        key = tuple(sorted(bindings.items()))
        if key not in self._cache:
            self._cache[key] = self._compile(bindings)
        compiled = self._cache[key]
        if parameters is self._weights:
            arrays = list(compiled.arrays)
        else:
            missing = [path for path in compiled.parameters if path not in parameters]
            if missing:
                raise LinnetError("missing parameters: " + ", ".join(missing))
            arrays = [_device_array(parameters[path]) for path in compiled.parameters]
        given: Mapping[str, Any] = state or {}
        for path in compiled.state_inputs:
            aval = compiled.state_avals[path]
            arrays.append(
                _device_array(given[path]) if path in given else jnp.zeros(aval.shape, aval.dtype)
            )
        outputs = compiled.call(*inputs, *arrays)
        if not compiled.state_inputs and not compiled.state_outputs:
            return outputs
        outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]
        count = len(compiled.state_outputs)
        results, new_states = outputs[: len(outputs) - count], outputs[len(outputs) - count :]
        result = results[0] if len(results) == 1 else tuple(results)
        new_state = dict(given)
        new_state.update(zip(compiled.state_outputs, new_states, strict=True))
        return result, new_state

    def __call__(self, *inputs: Any, state: Any = None) -> Any:
        return self.apply(self._weights, *inputs, state=state)


def _platform() -> str:
    """The platform name `Exported` checks calls against: `cuda`/`rocm`
    rather than the `gpu` backend name."""
    backend = jax.default_backend()
    if backend != "gpu":
        return backend
    kind = jax.devices()[0].device_kind.lower()
    return "rocm" if "amd" in kind or kind.startswith("mi") else "cuda"


@dataclasses.dataclass
class CompiledEntry:
    parameters: list[str]  # parameter paths, in argument order after the inputs
    state_inputs: list[str]  # state paths read before the call, after the parameters
    state_outputs: list[str]  # state paths assigned, as results after the entry's own
    state_avals: dict[str, Any]
    exported: Any
    call: Any  # `exported.call` under `jax.jit`
    arrays: list[Any]  # the loaded weights as device arrays, in `parameters` order
    source_path: Path | None = None  # generated JAX source, when the entry runs as code


def _device_array(value: Any) -> Any:
    """`value` as a JAX array; arrays already on a device pass through."""
    return value if isinstance(value, jax.Array) else jnp.asarray(value)


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
    fields = {
        "fun_name": "main",
        "in_tree": in_tree,
        "in_avals": in_avals,
        "out_tree": out_tree,
        "out_avals": out_avals,
        "_has_named_shardings": False,  # older jax only
        "_in_named_shardings": (None,) * len(in_avals),
        "_out_named_shardings": (None,) * len(out_avals),
        "in_shardings_hlo": (None,) * len(in_avals),
        "out_shardings_hlo": (None,) * len(out_avals),
        "nr_devices": 1,
        "platforms": (_platform(),),
        "ordered_effects": (),
        "unordered_effects": (),
        "disabled_safety_checks": (),
        "mlir_module_serialized": serialized,
        "calling_convention_version": cast(
            Any, export
        ).maximum_supported_calling_convention_version,
        "module_kept_var_idx": tuple(range(len(in_avals))),
        "uses_global_constants": False,
        "_get_vjp": None,
    }
    # The dataclass gained and lost private fields across jax releases.
    names = {f.name for f in dataclasses.fields(exported_type)}
    return exported_type(**{k: v for k, v in fields.items() if k in names})


def load(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path | Mapping[str, Any],
    root: str | None = None,
    entry: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
    numerics: str = "fast",
) -> LinnetFunction:
    """Materializes the entry of a root block as a JAX callable.

    `generics` binds the root block's generic parameters; the entry's own are
    bound from the input shapes at each call. `weights` is a mapping from
    parameter path to array, a `.safetensors` file, or a directory of them;
    `bindings` is an optional JSON file mapping parameter paths to tensor
    names in the weights.

    `numerics` is `"fast"` (the default: layer normalization and attention
    accumulate in the input dtype, as framework reference implementations
    do), `"equivalent"` (the f32 accumulation the canonical bodies specify),
    or `"exact"` (every library operation as its canonical decomposition).
    """
    if numerics not in ("exact", "equivalent", "fast"):
        raise LinnetError('numerics must be "exact", "equivalent", or "fast"')
    source_path = Path(source)
    plan_arguments = ["plan", "--no-optimize"]
    if root is not None:
        plan_arguments += ["--root", root]
    plan = cast(
        dict[str, Any],
        json.loads(run_compiler(*plan_arguments, *std_arguments(std_root), str(source_path))),
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
    loaded = apply_bindings(read_arrays(weights), bindings)
    return LinnetFunction(
        source_path, plan, generics, loaded, std_root, root_name, entry_name, numerics
    )


__all__ = ["LinnetFunction", "load"]

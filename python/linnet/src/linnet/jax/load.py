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
import functools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jax.sharding import SingleDeviceSharding

from .. import ir
from ..compiler import (
    LinnetError,
    bind_arguments,
    check_numerics,
    lora_arguments,
    run_compiler,
    std_arguments,
)
from ..plan import compile_plan
from ..results import Result
from ..weights import apply_bindings, read_arrays
from .dtypes import MLIR_TYPES

if TYPE_CHECKING:
    from types import ModuleType
    from typing import TypeAlias

    from jax._src.lib.mlir import ir as mlir_ir
    from jax.core import ShapedArray
    from jax.sharding import Sharding
    from jax.typing import ArrayLike, DTypeLike

    # A `state` value by path, as an entry that touches state takes and returns it.
    State: TypeAlias = dict[str, jax.Array]
    # The cache key of one compilation: its generic bindings, sorted.
    _Key: TypeAlias = tuple[tuple[str, int | str], ...]


class LinnetFunction:
    """A Linnet entry as a callable over JAX arrays."""

    def __init__(
        self,
        source: Path,
        program: ir.Program,
        generics: Mapping[str, int | str],
        weights: dict[str, ArrayLike],
        std_root: str | Path | None,
        root: str,
        entry: str,
        numerics: str = "equivalent",
        cast_dtype: bool = False,
    ) -> None:
        self._source = source
        self.numerics = numerics
        self.cast_dtype = cast_dtype
        self.program = program
        self.generics = dict(generics)
        self._root = ir.bind_generics(program.root.generics, self.generics)
        self.weights = weights
        self._weights = self.weights
        self._std_root = std_root
        self.root = root
        self.entry = entry
        # How a weight goes to the devices when an entry first compiles:
        # `placement(path, host_array)`, or onto the default device.
        self.placement: Callable[[str, ArrayLike], jax.Array] | None = None
        matching = [f for f in program.functions.values() if f.name.endswith(f"::{root}.{entry}")]
        if not matching:
            raise LinnetError(f"block `{root}` has no entry `{entry}`")
        self._signature = matching[0]
        manifest = program.manifest
        self.parameter_paths: list[str] = [entry.path for entry in manifest]
        self.optional_paths: set[str] = {e.path for e in manifest if e.optional}
        self._cache: dict[_Key, CompiledEntry] = {}
        # Set by `LinnetModel`, which runs several entries over one copy of
        # the weights and keeps their state on the device: the weights are
        # written back once placed (and cast), so the next entry finds them
        # there, and the state an entry replaces is donated to it.
        self.share_weights = False
        self.donate_state = False
        # Set by `LinnetModel` (inference only): weight-only work runs once, in
        # `prepare`, its results kept here by key and shared across entries.
        self.prepare_weights = False
        self.prepared: dict[str, jax.Array] = {}
        self._check_weights(manifest)

    def _export(self, target: str, bindings: Mapping[str, int | str]) -> str:
        """`linnet <target>` for this entry, with each optional parameter
        compiled in exactly when the weights have it. The first export takes
        every optional and reads back which paths the weights lack; after
        that the set is known and reused."""
        arguments = [target, "--root", self.root, "--entry", self.entry]
        arguments += ["--numerics", self.numerics, "--optionals", "present"]
        if target == "jax" and self.prepare_weights:
            arguments.append("--prepare")
        if target == "jax":
            for unit in getattr(self, "sharded", ()):
                arguments += ["--fully-shard", unit]
            for unit in getattr(self, "remat", ()):
                arguments += ["--remat", unit]
        if target == "jax":
            arguments += lora_arguments(getattr(self, "lora", None))
        arguments += bind_arguments(bindings)
        arguments += ["--absent-file", "-", *std_arguments(self._std_root), str(self._source)]

        def run(absent: list[str]) -> str:
            return run_compiler(*arguments, stdin="".join(f"{path}\n" for path in absent))

        if self.absent is None:
            if target != "stablehlo":
                self._export("stablehlo", bindings)
            else:
                probe = run([])
                paths = re.findall(r'linnet\.path = "([^"]+)"', probe)
                optional = [
                    path
                    for path in paths
                    if any(self._matches(pattern, path) for pattern in self.optional_paths)
                ]
                self.absent = sorted(path for path in optional if path not in self._weights)
                if not self.absent:
                    return probe
        assert self.absent is not None
        return run(self.absent)

    @staticmethod
    def _matches(pattern: str, path: str) -> bool:
        """Manifest paths spell array elements `[*]`; weights spell them `.0`."""
        return re.fullmatch(re.escape(pattern).replace(r"\[\*\]", r"\.\d+"), path) is not None

    def _check_weights(self, manifest: Sequence[ir.ManifestEntry]) -> None:
        for entry in manifest:
            if entry.optional or entry.kind == "state":  # state is never a weight
                continue
            if not any(self._matches(entry.path, path) for path in self._weights):
                raise LinnetError(f"missing weights for `{entry.path}`")
        # Which optional parameters are absent is read off the first export
        # (see `_export`): checkpoints mix them, a bias on some projections
        # and not others, so it is decided per parameter.
        self.absent: list[str] | None = None

    # ---- entry generics from input shapes

    def _bindings_for(self, inputs: Sequence[ArrayLike]) -> dict[str, int | str]:
        arguments = self._signature.params
        if len(arguments) != len(inputs):
            raise LinnetError(
                f"entry `{self.entry}` takes {len(arguments)} inputs, got {len(inputs)}"
            )
        env = self._root.copy()
        for argument, value in zip(arguments, inputs, strict=True):
            if isinstance(argument.type, ir.TensorType):
                ir.bind_input(env, argument, [int(d) for d in np.shape(value)], None)
        return {**self.generics, **ir.bind_names(env, self._signature.generics)}

    def _compile(self, bindings: Mapping[str, int | str]) -> CompiledEntry:
        text = self._export("stablehlo", bindings)
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
        donated = self._donated(len(exported.in_avals), state_inputs, state_outputs)
        options = compiler_options(text)
        call = jax.jit(exported.call, donate_argnums=donated, compiler_options=options)
        first = len(exported.in_avals) - len(state_inputs) - len(paths)
        declared = exported.in_avals[first : first + len(paths)]
        arrays: list[jax.Array] = []
        # Paths bound to one checkpoint tensor (a tied embedding and output
        # head) share one device array.
        uploaded: dict[int, jax.Array] = {}
        for path, aval in zip(paths, declared, strict=True):
            host = self._weights[path]
            array = uploaded.get(id(host))
            if array is None:
                array = self.placement(path, host) if self.placement else _device_array(host)
                if (
                    self.cast_dtype
                    and array.dtype != aval.dtype
                    and jnp.issubdtype(array.dtype, jnp.floating)
                    and jnp.issubdtype(aval.dtype, jnp.floating)
                ):
                    # The export declares each parameter's dtype; a floating
                    # array of another width is converted to it on the device,
                    # one at a time, so the uncast copy of only one parameter
                    # is ever there (all of them at once is a bf16 model's
                    # weights twice over).
                    array = array.astype(aval.dtype)
                uploaded[id(host)] = array
            arrays.append(array)
        if self.share_weights:
            self._weights.update(zip(paths, arrays, strict=True))
        state_avals = dict(
            zip(
                state_inputs,
                exported.in_avals[len(exported.in_avals) - len(state_inputs) :],
                strict=True,
            )
        )
        return CompiledEntry(
            paths,
            state_inputs,
            state_outputs,
            state_avals,
            exported,
            call,
            arrays,
            dtypes=[aval.dtype for aval in declared],
            options=options,
        )

    def _donated(
        self, arguments: int, state_inputs: Sequence[str], state_outputs: Sequence[str]
    ) -> tuple[int, ...]:
        """The argument positions of the states this entry replaces, when
        `donate_state` is set: XLA then writes the new value into the old
        one's memory (a cache updated in place) instead of beside it."""
        if not self.donate_state:
            return ()
        first = arguments - len(state_inputs)
        return tuple(first + i for i, path in enumerate(state_inputs) if path in state_outputs)

    def parameters_for(self, *inputs: ArrayLike) -> dict[str, jax.Array]:
        """The weights the entry takes for inputs of these shapes, compiling
        it for them first: path -> device array, in the dtype the entry
        computes in. Paths bound to one checkpoint tensor share one array."""
        bindings = self._bindings_for(inputs)
        key = tuple(sorted(bindings.items()))
        if key not in self._cache:
            self._cache[key] = self._compile(bindings)
        compiled = self._cache[key]
        return dict(zip(compiled.parameters, compiled.arrays, strict=True))

    def apply(
        self,
        parameters: Mapping[str, ArrayLike],
        *inputs: ArrayLike,
        state: Mapping[str, jax.Array] | None = None,
    ) -> Result:  # pyright: ignore[reportExplicitAny]
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
            if self.cast_dtype:
                # Mixed precision: master parameters (f32, say) cast to the
                # dtype the export declares on every call, so `jax.grad`
                # returns gradients in the masters' own dtype. A weight the
                # code gathers itself (`--fully-shard`) is cast there, after
                # its gradient is reduced in the master's dtype.
                gathered: set[str] = set(getattr(compiled.module, "GATHERED", None) or [])
                arrays = [
                    array.astype(dtype)
                    if path not in gathered
                    and array.dtype != dtype
                    and jnp.issubdtype(array.dtype, jnp.floating)
                    and jnp.issubdtype(dtype, jnp.floating)
                    else array
                    for path, array, dtype in zip(
                        compiled.parameters, arrays, compiled.dtypes, strict=True
                    )
                ]
        given: Mapping[str, jax.Array] = state or {}
        missing = [p for p in compiled.state_inputs if p not in given]
        zeros: State = {}
        if missing:
            # Zeros placed where the weights are: a call's new state comes
            # back committed to that device, and jit compiles again for
            # committed arrays where the first call had uncommitted ones.
            # Weights split over a mesh leave the placement to the model
            # (`shard`).
            shardings = {
                a.sharding
                for a in arrays
                if isinstance(a, jax.Array)  # pyright: ignore[reportUnnecessaryIsInstance]
                and not isinstance(a, Tracer)
            }
            placement = None
            if len(shardings) == 1 and isinstance(next(iter(shardings)), SingleDeviceSharding):
                placement = next(iter(shardings))
            shapes = tuple(
                (tuple(compiled.state_avals[p].shape), compiled.state_avals[p].dtype)
                for p in missing
            )
            zeros = dict(zip(missing, _zeros(shapes, placement)(), strict=True))
        for path in compiled.state_inputs:
            arrays.append(_device_array(given[path]) if path in given else zeros[path])
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

    def __call__(self, *inputs: ArrayLike, state: Mapping[str, jax.Array] | None = None) -> Result:  # pyright: ignore[reportExplicitAny]
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
    state_avals: dict[str, ShapedArray]
    exported: jax.export.Exported
    # `exported.call` under `jax.jit`.
    call: Callable[..., Result]  # pyright: ignore[reportExplicitAny]
    arrays: list[jax.Array]  # the loaded weights as device arrays, in `parameters` order
    source_path: Path | None = None  # generated JAX source, when the entry runs as code
    dtypes: list[np.dtype[np.generic]] = dataclasses.field(  # declared, each
        default_factory=lambda: list[np.dtype[np.generic]]()
    )
    module: ModuleType | None = None  # the generated module, when the entry runs as code
    prepared: list[jax.Array] = dataclasses.field(  # `prepare`'s
        default_factory=lambda: list[jax.Array]()
    )
    options: dict[str, str | bool] | None = None  # XLA's, by `compiler_options`


# XLA options for a program bound by arithmetic (a prompt's pass): cuBLAS
# products, which beat XLA's Triton ones over many rows, and kernels launched
# one by one rather than from command buffers, whose update holds the host for
# milliseconds before the device starts; here each kernel runs long enough to
# hide its launch. A program bound by memory (a decoding step) keeps XLA's
# defaults, which suit it.
_COMPUTE_BOUND_OPTIONS: dict[str, str | bool] = {
    "xla_gpu_enable_triton_gemm": False,
    "xla_gpu_enable_command_buffer": "",
}
# Products' FLOPs per byte of the program's arguments (its weights, mostly)
# from which it counts as bound by arithmetic: about where an H100's bf16
# products stop waiting on memory.
_COMPUTE_BOUND = 256
_DOT = re.compile(
    r'"stablehlo\.dot_general"\(.*?lhs_contracting_dimensions = \[([\d, ]*)\].*?'
    r"-> tensor<([^>]*)>"
)
_DOT_OPERAND = re.compile(r": \(tensor<([^>]*)>")
_ARGUMENT = re.compile(r"%[\w.]+: tensor<([^>]*)>")


def compiler_options(text: str) -> dict[str, str | bool] | None:
    """XLA's options for the StableHLO module `text`: those for a program
    bound by arithmetic when its `intensity` reaches `_COMPUTE_BOUND` on a
    GPU, otherwise None (the defaults)."""
    if jax.default_backend() != "gpu" or intensity(text) < _COMPUTE_BOUND:
        return None
    return dict(_COMPUTE_BOUND_OPTIONS)


def intensity(text: str) -> float:
    """The FLOPs of the StableHLO module `text`'s products per byte of its
    `main`'s arguments."""
    flops = 0
    arguments = 0
    for line in text.splitlines():
        if "func.func" in line and "@main" in line:
            arguments = sum(math.prod(_dims(t)) * _width(t) for t in _ARGUMENT.findall(line))
        found = _DOT.search(line)
        operand = _DOT_OPERAND.search(line)
        if found is None or operand is None:
            continue
        lhs = _dims(operand.group(1))
        contracted = [int(axis) for axis in found.group(1).replace(" ", "").split(",") if axis]
        flops += 2 * math.prod(_dims(found.group(2))) * math.prod(lhs[a] for a in contracted)
    return flops / arguments if arguments else 0.0


def _dims(tensor: str) -> list[int]:
    """`1x512x2048xbf16` -> [1, 512, 2048]."""
    return [int(d) for d in tensor.split("x")[:-1]]


def _width(tensor: str) -> int:
    """Bytes of one element of `1x512x2048xbf16`."""
    bits = re.search(r"\d+", tensor.split("x")[-1])
    return max(1, int(bits.group()) // 8) if bits else 1


@functools.cache
def _zeros(
    shapes: tuple[tuple[tuple[int, ...], DTypeLike], ...], placement: Sharding | None
) -> Callable[[], tuple[jax.Array, ...]]:
    """One compiled function making every missing state member's zeros:
    one dispatch rather than one per cache."""
    return jax.jit(
        lambda: tuple(jnp.zeros(shape, dtype) for shape, dtype in shapes),
        out_shardings=None if placement is None else tuple(placement for _ in shapes),
    )


def _device_array(value: ArrayLike) -> jax.Array:
    """`value` as a JAX array; arrays already on a device pass through."""
    return value if isinstance(value, jax.Array) else jnp.asarray(value)


class _ExportedFields(Protocol):
    """`jax.export.Exported`'s constructor, given its fields by name."""

    def __call__(self, **fields: object) -> jax.export.Exported: ...


def _wrap_module(text: str) -> jax.export.Exported:
    """A `jax.export.Exported` around a StableHLO module whose function is
    `@main`, so JAX can call it like one of its own exports."""
    from jax import export
    from jax._src.interpreters import mlir as jax_mlir  # pyright: ignore[reportPrivateUsage]
    from jax._src.lib.mlir import ir  # pyright: ignore[reportPrivateUsage]
    from jax.tree_util import tree_flatten

    with jax_mlir.make_ir_context():
        module = ir.Module.parse(text)
        function = module.body.operations[0]
        block = function.regions[0].blocks[0]
        in_types = [ir.RankedTensorType(a.type) for a in block.arguments]
        # The bindings hand back the attribute as the `TypeAttr` it is.
        function_type = cast("mlir_ir.TypeAttr", function.attributes["function_type"])
        signature = ir.FunctionType(function_type.value)
        out_types = [ir.RankedTensorType(r) for r in signature.results]
        serialized = jax_mlir.module_to_bytecode(module)

    from jax import core as jax_core

    def aval(tensor_type: mlir_ir.RankedTensorType) -> ShapedArray:
        element = str(tensor_type.element_type)
        if element not in MLIR_TYPES:
            raise LinnetError(f"unsupported element type {element}")
        return jax_core.ShapedArray(tuple(tensor_type.shape), MLIR_TYPES[element])

    in_avals = tuple(aval(t) for t in in_types)
    out_avals = tuple(aval(t) for t in out_types)
    in_tree = tree_flatten((tuple(0 for _ in in_avals), {}))[1]
    out_tree = tree_flatten(0 if len(out_avals) == 1 else tuple(0 for _ in out_avals))[1]
    fields: dict[str, object] = {
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
        "calling_convention_version": export.maximum_supported_calling_convention_version,
        "module_kept_var_idx": tuple(range(len(in_avals))),
        "uses_global_constants": False,
        "_get_vjp": None,
    }
    # The dataclass gained and lost private fields across jax releases, so
    # it is built from whichever of these it has.
    names = {f.name for f in dataclasses.fields(export.Exported)}
    exported_type = cast(_ExportedFields, export.Exported)
    return exported_type(**{k: v for k, v in fields.items() if k in names})


def load(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path | Mapping[str, ArrayLike],
    root: str | None = None,
    entry: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
    numerics: str = "fast",
    cast_dtype: bool = False,
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

    `cast_dtype=True` converts floating-point weights to the dtype the export
    declares for them, so a checkpoint published in f32 runs in bf16 (or the
    reverse) as the generics ask; an integer where a float is declared is
    still an error.
    """
    check_numerics(numerics)
    source_path = Path(source)
    program = compile_plan(source_path, root=root, std_root=std_root, optimize=False)
    return function_of(
        program, source_path, generics, weights, bindings, std_root, entry, numerics, cast_dtype
    )


def function_of(
    program: ir.Program,
    source: Path,
    generics: Mapping[str, int | str],
    weights: str | Path | Mapping[str, ArrayLike],
    bindings: str | Path | None,
    std_root: str | Path | None,
    entry: str | None,
    numerics: str,
    cast_dtype: bool,
) -> LinnetFunction:
    """`load` for a program already compiled from `source`."""
    root_name = program.root.name
    entries = [f.short_name for f in program.entries()]
    entry_name = ir.choose_entry(root_name, entries, entry)
    loaded = apply_bindings(read_arrays(weights), bindings)
    return LinnetFunction(
        source, program, generics, loaded, std_root, root_name, entry_name, numerics, cast_dtype
    )


__all__ = ["LinnetFunction", "load"]

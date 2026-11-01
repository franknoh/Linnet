"""Every entry of a Linnet block on MLX, over one copy of the weights.

`load_model` is to MLX what `linnet.jax.load_model` is to XLA: each entry is
generated as `mlx.core` source (`linnet mlx`) for the shapes it is called
with and compiled by `mx.compile`, every entry sharing the weights, which are
read onto the device once. The block's `state` (a decoder's KV caches) stays
there between calls, each call's new state the next call's input.

    from linnet.mlx import load_model

    model = load_model("model.linnet", generics={..., "T": "bf16"},
                       weights="model.safetensors", cast_dtype=True)
    logits = model.run_entry("prefill", [tokens, np.int32(0)])

MLX runs on Apple silicon's GPU (Metal), on NVIDIA GPUs (`mlx[cuda13]`) and
on the CPU (`mlx[cpu]` on Linux).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from .. import ir
from ..compiler import LinnetError, bind_arguments, check_numerics, run_compiler, std_arguments
from ..dtypes import BY_SAFETENSORS, DTYPES
from ..generated import generated_directory, import_generated
from ..plan import compile_plan
from ..weights import check_problems, read_bindings, safetensors_index, tensor_problem

if TYPE_CHECKING:
    from types import ModuleType

    import mlx.core as mx
    from numpy.typing import ArrayLike

    from ..weights import TensorLocation

# An entry compiled for one binding of its generics: its name and generics.
_Key = tuple[str, tuple[tuple[str, str], ...]]


class _Options(TypedDict):
    """`load_model`'s arguments that each entry's export takes."""

    generics: dict[str, int | str]
    root: str | None
    std_root: str | Path | None
    numerics: str
    cast_dtype: bool
    compile: bool


@dataclass
class _Entry:
    """An entry compiled for one shape binding."""

    module: ModuleType
    call: Callable[..., tuple[mx.array, ...]]
    parameters: list[mx.array]
    inputs: list[mx.Dtype | None]  # each input's dtype, None for a non-tensor


class MlxModel:
    """The entries of a block (`entries`, by name) over one copy of its
    weights on MLX's default device. `run_entry` runs one; `state` holds the
    block's state members by path, as MLX arrays; `reset_state()` clears
    it."""

    def __init__(
        self,
        source: Path,
        program: ir.Program,
        options: _Options,
        weights: str | Path | Mapping[str, ArrayLike],
        bindings: str | Path | None,
    ) -> None:
        self._source = source
        self.program = program
        self._options = options
        self._signatures = {f.short_name: f for f in program.entries()}
        self.entries = list(self._signatures)
        self.generics = dict(options["generics"])
        self._root = ir.bind_root(program, self.generics)
        self._mapping = read_bindings(bindings) if bindings is not None else {}
        # A checkpoint on disk is read a tensor at a time, when an entry
        # first needs it; arrays given are used as they are.
        self._index: dict[str, TensorLocation] = {}
        self._given: dict[str, np.ndarray] = {}
        if isinstance(weights, Mapping):
            self._given = {str(name): np.asarray(value) for name, value in weights.items()}
        else:
            self._index = safetensors_index(weights)
        self._optional = [
            re.compile(re.escape(entry.path).replace(r"\[\*\]", r"\.\d+"))
            for entry in program.manifest
            if entry.optional
        ]
        self._weights: dict[tuple[str, str], mx.array] = {}  # (tensor, dtype) -> array
        self._compiled: dict[_Key, _Entry] = {}
        self._work = generated_directory()
        self.state: dict[str, mx.array] = {}

    # ---- entries

    def run_entry(
        self, name: str, inputs: Sequence[ArrayLike | mx.array]
    ) -> mx.array | tuple[mx.array, ...]:
        """Runs entry `name` on `inputs` (MLX or NumPy arrays, or numbers):
        its result, or a tuple of several, as evaluated MLX arrays."""
        import mlx.core as mx

        signature = self._signatures.get(name)
        if signature is None:
            raise LinnetError(f"the block has no entry `{name}`")
        arrays = [v if isinstance(v, mx.array) else mx.array(np.asarray(v)) for v in inputs]
        env = self._bind(signature, arrays)
        bindings = ir.bind_names(env, signature.generics)
        key: _Key = (name, tuple(sorted(bindings.items())))
        entry = self._compiled.get(key)
        if entry is None:
            entry = self._compiled[key] = self._compile(name, signature, bindings, env)
        arrays = [
            a if wanted is None or a.dtype == wanted else a.astype(wanted)
            for a, wanted in zip(arrays, entry.inputs, strict=True)
        ]
        module = entry.module
        states = [
            self._state(path, shape, dtype)
            for path, (shape, dtype) in zip(module.STATES, module.STATE_TYPES, strict=True)
        ]
        outputs = entry.call(*arrays, *entry.parameters, *states)
        results = outputs[: module.RESULTS]
        for path, value in zip(module.NEXT_STATES, outputs[module.RESULTS :], strict=True):
            self.state[path] = value
        # Evaluated now: a state left lazy would carry every call's graph.
        mx.eval(*outputs)  # pyright: ignore[reportUnknownMemberType]
        return results[0] if len(results) == 1 else tuple(results)

    def reset_state(self) -> None:
        self.state = {}

    def _state(self, path: str, shape: Sequence[int], dtype: str) -> mx.array:
        """State `path`, zeros until an entry writes it."""
        import mlx.core as mx

        found = self.state.get(path)
        if found is None:
            found = self.state[path] = mx.zeros(tuple(shape), dtype=mlx_dtype(dtype))
        return found

    def _bind(self, signature: ir.Function, inputs: Sequence[mx.array]) -> ir.Bindings:
        """The root's bindings with the entry's own generics bound, from
        the shapes of its inputs."""
        given = {
            g.name: self.generics[g.name] for g in signature.generics if g.name in self.generics
        }
        return ir.bind_entry(
            self._root,
            signature,
            inputs,
            lambda param, value: ir.shape_only(param, [int(d) for d in value.shape]),
            given,
        )

    def _compile(
        self, name: str, signature: ir.Function, bindings: Mapping[str, str], env: ir.Bindings
    ) -> _Entry:
        import mlx.core as mx

        options = self._options
        arguments = ["mlx", "--entry", name, "--numerics", options["numerics"]]
        if options["root"] is not None:
            arguments += ["--root", options["root"]]
        arguments += ["--optionals", "present", "--absent-file", "-"]
        arguments += bind_arguments({**self.generics, **bindings})
        arguments += [*std_arguments(options["std_root"]), str(self._source)]
        # Every optional parameter compiled in first; those the weights lack
        # are compiled out again, by path.
        text = run_compiler(*arguments, stdin="")
        module = import_generated(self._work, f"mlx_{name}", text)
        absent = [
            path
            for path in module.PARAMETERS
            if not self._has(path) and any(p.fullmatch(path) for p in self._optional)
        ]
        if absent:
            text = run_compiler(*arguments, stdin="".join(f"{path}\n" for path in absent))
            module = import_generated(self._work, f"mlx_{name}", text)
        problems: list[str] = []
        parameters = [
            self._parameter(path, tuple(shape), dtype, problems)
            for path, (shape, dtype) in zip(module.PARAMETERS, module.PARAMETER_TYPES, strict=True)
        ]
        check_problems(problems)
        main = cast("Callable[..., tuple[mx.array, ...]]", module.main)
        # A runtime loop reads its condition back, which tracing cannot.
        call = main if module.DYNAMIC or not options["compile"] else mx.compile(main)
        inputs: list[mx.Dtype | None] = []
        for param in signature.params:
            if isinstance(param.type, ir.TensorType | ir.ScalarType):
                inputs.append(mlx_dtype(ir.evaluate_dtype(param.type.dtype, env)))
            else:
                inputs.append(None)
        return _Entry(module, call, [p for p in parameters if p is not None], inputs)

    # ---- weights

    def _has(self, path: str) -> bool:
        tensor = self._mapping.get(path, path)
        return tensor in self._index or tensor in self._given

    def _parameter(
        self, path: str, shape: tuple[int, ...], dtype: str, problems: list[str]
    ) -> mx.array | None:
        """Parameter `path` (`shape`, Linnet `dtype`) on the device, read
        once per tensor and dtype: a tensor serving several parameters (an
        output head tied to the embedding) is one array."""
        import mlx.core as mx

        tensor = self._mapping.get(path, path)
        found = self._weights.get((tensor, dtype))
        if found is not None:
            return found
        stored = DTYPES[dtype].storage_shape(shape)
        if tensor in self._index:
            location = self._index[tensor]
            code, data = location.dtype, location.read()
            problem = tensor_problem(
                tensor, path, location.shape, code, stored, dtype, self._options["cast_dtype"]
            )
            if problem is not None:
                problems.append(problem)
                return None
            array = _from_bytes(data, code, location.shape)
        elif tensor in self._given:
            host = self._given[tensor]
            if tuple(host.shape) != tuple(stored):
                problems.append(f"`{tensor}` has shape {list(host.shape)}, `{path}` needs {stored}")
                return None
            array = mx.array(host)
        else:
            problems.append(f"missing tensor `{tensor}` for `{path}`")
            return None
        wanted = mlx_dtype(dtype)
        if array.dtype != wanted:
            array = array.astype(wanted)
        mx.eval(array)  # pyright: ignore[reportUnknownMemberType]
        self._weights[(tensor, dtype)] = array
        return array


def mlx_dtype(name: str) -> mx.Dtype:
    """The MLX dtype a Linnet dtype is held in: its bytes (`uint8`) for a
    storage format MLX has no type for."""
    import mlx.core as mx

    found = DTYPES.get(name)
    if found is None:
        raise LinnetError(f"`{name}` is not a Linnet dtype")
    if found.storage:
        return mx.uint8
    return mx.bool_ if found.name == "bool" else cast("mx.Dtype", getattr(mx, found.numpy))


def _from_bytes(data: bytes, code: str, shape: Sequence[int]) -> mx.array:
    """A SafeTensors tensor's bytes as an MLX array: bf16 by its bits, FP8
    and the other storage formats as bytes."""
    import mlx.core as mx

    found = BY_SAFETENSORS.get(code)
    if found is None:
        raise LinnetError(f"a tensor has dtype {code}, which Linnet does not read")
    if found.name == "bf16":
        bits = np.frombuffer(data, dtype=np.uint16).reshape(tuple(shape))
        return mx.array(bits).view(mx.bfloat16)
    host = np.frombuffer(data, dtype=np.uint8 if found.storage else np.dtype(found.numpy))
    return mx.array(host.reshape(tuple(shape)))


def load_model(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path | Mapping[str, ArrayLike],
    root: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
    numerics: str = "fast",
    cast_dtype: bool = False,
    compile: bool = True,
) -> MlxModel:
    """Every entry of the root block on MLX over one copy of the weights,
    with its state kept on the device. `cast_dtype=True` converts the
    checkpoint to the dtype the generics ask for (`T`); `compile=False` runs
    the generated code without `mx.compile`."""
    try:
        import mlx.core  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError:
        raise LinnetError(
            "MLX is not installed: pip install mlx on Apple silicon, mlx[cuda13] or mlx[cpu] "
            "on Linux"
        ) from None
    check_numerics(numerics)
    program = compile_plan(source, root=root, std_root=std_root, optimize=False)
    options: _Options = {
        "generics": dict(generics),
        "root": root,
        "std_root": std_root,
        "numerics": numerics,
        "cast_dtype": cast_dtype,
        "compile": compile,
    }
    return MlxModel(Path(source), program, options, weights, bindings)


__all__ = ["MlxModel", "load_model", "mlx_dtype"]

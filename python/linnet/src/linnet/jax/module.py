"""Every entry of a Linnet block in JAX, over one copy of the weights.

`load_model` is what decoding and serving need from the JAX backend: a
prompt entry and a step entry that read the same parameters, and the block's
`state` (its KV caches) kept on the device between calls. Each state an entry
replaces is donated to it, so XLA writes the new cache into the old one's
memory rather than beside it.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..compiler import LinnetError, check_numerics
from ..plan import compile_plan
from .load import LinnetFunction, function_of
from .source import SourceFunction


class LinnetModel:
    """A root block's entries as JAX callables sharing weights and state.

    `run_entry(name, inputs)` runs one entry; the entry's own generics are
    bound from the input shapes, as `load` does. `state` holds the block's
    state members by path (zeros until an entry writes them), and
    `reset_state()` clears it.
    """

    def __init__(self, first: LinnetFunction, generated: bool) -> None:
        self._first = first
        self._generated = generated
        self._functions: dict[str, LinnetFunction] = {}
        self.program = first.program
        self.generics = first.generics
        self.weights = first.weights
        self.entries = [f.short_name for f in first.program.entries()]
        self.state: dict[str, Any] = {}
        # Weight-only values (`prepare`), shared by every entry by key.
        self._prepared: dict[str, Any] = {}
        self.mesh: Any = None  # set by `shard`
        self._split_shardings: dict[int, Any] = {}  # id -> a state sharding known split right

    def _function(self, name: str) -> LinnetFunction:
        if name not in self._functions:
            if name not in self.entries:
                raise LinnetError(f"block `{self._first.root}` has no entry `{name}`")
            first = self._first
            kind = SourceFunction if self._generated else LinnetFunction
            function = kind(
                first._source,  # pyright: ignore[reportPrivateUsage]
                first.program,
                first.generics,
                self.weights,
                first._std_root,  # pyright: ignore[reportPrivateUsage]
                first.root,
                name,
                first.numerics,
                first.cast_dtype,
            )
            function.share_weights = True
            function.donate_state = True
            if self._generated:
                function.prepare_weights = True
                function.prepared = self._prepared
            # Which optional parameters the weights lack is the same for every
            # entry; the first export found it.
            known = [f.absent for f in self._functions.values() if f.absent is not None]
            if known:
                function.absent = known[0]
            self._functions[name] = function
        return self._functions[name]

    def run_entry(self, name: str, inputs: Sequence[Any]) -> Any:
        function = self._function(name)
        outcome = function.apply(function.weights, *inputs, state=self.state)
        if not isinstance(outcome, tuple) or len(outcome) != 2 or not isinstance(outcome[1], dict):
            return outcome
        result, state = outcome
        if self.mesh is not None:
            state = {path: self._shard_state(value) for path, value in state.items()}
        self.state = state
        return result

    def _shard_state(self, value: Any) -> Any:
        """A state array split by heads over the mesh, as the key and value
        projections that fill a KV cache are; others stay as XLA left them."""
        import jax
        from jax.sharding import NamedSharding, PartitionSpec

        from ..parallel import state_axis

        axis = state_axis(value.shape, self.mesh.devices.size)
        if axis is None:
            return value
        # XLA hands the same sharding back call after call, spelled
        # `P(None, 'model')` for `P(None, 'model', None, None)`: equivalent,
        # not equal. Each one seen is checked once.
        given = value.sharding
        if id(given) in self._split_shardings:
            return value
        spec = [None] * value.ndim
        spec[axis] = self.mesh.axis_names[0]
        wanted = NamedSharding(self.mesh, PartitionSpec(*spec))
        if given.is_equivalent_to(wanted, value.ndim):
            self._split_shardings[id(given)] = given
            return value
        return jax.device_put(value, wanted)

    def copy_weights(self, parameters: Mapping[str, Any]) -> None:
        """Replaces the weights with `parameters` (path -> array): a policy
        being trained into the model a serving engine samples from, adapters
        merged in first (`linnet.jax.train.merge_lora`). Each is cast to the
        dtype its entries compute in and placed as the weight it replaces;
        compiled entries stay, and weight-only values (`prepare`) are
        computed again."""
        import jax
        import jax.numpy as jnp

        # Every function that may hold the old arrays: the one the model was
        # made from too. One left holding them keeps a whole copy alive.
        functions = list({id(f): f for f in (self._first, *self._functions.values())}.values())
        replaced: dict[str, Any] = {}
        for function in functions:
            for compiled in function._cache.values():  # pyright: ignore[reportPrivateUsage]
                for i, path in enumerate(compiled.parameters):
                    if path not in parameters:
                        continue
                    if path not in replaced:
                        old = compiled.arrays[i]
                        new = jnp.asarray(parameters[path]).astype(compiled.dtypes[i])
                        # Placed already (the policy's own arrays, on the
                        # same device): taken as they are, not copied.
                        if isinstance(old, jax.Array) and not new.sharding.is_equivalent_to(
                            old.sharding, new.ndim
                        ):
                            new = jax.device_put(new, old.sharding)
                        replaced[path] = new
                    compiled.arrays[i] = replaced[path]
        for path, value in parameters.items():
            if path in self.weights:
                self.weights[path] = replaced.get(path, value)
        for function in functions:
            # `SourceFunction.parameters`: the weights it loaded, as arrays.
            loaded = getattr(function, "parameters", None)
            if isinstance(loaded, dict):
                for path in loaded.keys() & replaced.keys():
                    loaded[path] = replaced[path]
        self._prepared.clear()
        for function in self._functions.values():
            if not isinstance(function, SourceFunction):
                continue
            for compiled in function._cache.values():  # pyright: ignore[reportPrivateUsage]
                if compiled.module is not None:
                    compiled.prepared = function._prepared_values(  # pyright: ignore[reportPrivateUsage]
                        compiled.module, compiled.arrays
                    )

    def reset_state(self) -> None:
        self.state = {}

    def shard(self, mesh: Any, rules: Mapping[str, int | None] | None = None) -> None:
        """Splits the weights over `mesh` (a one-axis `Mesh`, or a device
        count) as `rules` say; entries then run partitioned across it."""
        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        from ..parallel import split_axis

        if isinstance(mesh, int):
            mesh = Mesh(np.array(jax.devices()[:mesh]), ("model",))
        axis_name = mesh.axis_names[0]
        devices = mesh.devices.size
        self.mesh = mesh
        manifest = self._first.parameter_paths
        for path, value in list(self.weights.items()):
            # Checkpoint names the bindings map from stay on the host; only
            # the model's own paths go to the devices.
            if not any(LinnetFunction._matches(pattern, path) for pattern in manifest):  # pyright: ignore[reportPrivateUsage]
                continue
            shape = tuple(np.shape(value))
            axis = split_axis(path, shape, devices, rules)
            spec = [None] * len(shape)
            if axis is not None:
                spec[axis] = axis_name
            self.weights[path] = jax.device_put(value, NamedSharding(mesh, PartitionSpec(*spec)))
        self.state = {}


def load_model(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path | Mapping[str, Any],
    root: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
    numerics: str = "fast",
    cast_dtype: bool = False,
    generated: bool = True,
    mesh: Any = None,
    rules: Mapping[str, int | None] | None = None,
) -> LinnetModel:
    """Every entry of the root block over one copy of the weights, with its
    state kept on the device. `generated=True` runs the entries as generated
    JAX source (`linnet jax`), whose cache writes are scatters and slice
    updates; `False` runs the StableHLO export. The other arguments are
    `load`'s.

    `mesh` runs the model tensor-parallel: a `jax.sharding.Mesh` with one
    axis, or a number of devices to make one from. Each weight is split over
    it along the axis `rules` give (`linnet.parallel.DEFAULT_RULES` unless
    given) and XLA partitions every entry, adding the collectives the split
    needs; a KV cache is split by heads."""
    # Any entry will do for the first function, which checks the weights against the
    # plan; the model builds a function for each entry it is asked to run.
    check_numerics(numerics)
    program = compile_plan(source, root=root, std_root=std_root, optimize=False)
    name = program.root.name
    entries = [f.short_name for f in program.entries()]
    if not entries:
        raise LinnetError(f"block `{name}` has no entries")
    first = function_of(
        program,
        Path(source),
        generics,
        weights,
        bindings,
        std_root,
        entries[0],
        numerics,
        cast_dtype,
    )
    model = LinnetModel(first, generated)
    if mesh is not None:
        model.shard(mesh, rules)
    return model

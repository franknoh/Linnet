"""Every entry of a Linnet block in JAX, over one copy of the weights.

`load_model` is what decoding and serving need from the JAX backend: a
prompt entry and a step entry that read the same parameters, and the block's
`state` (its KV caches) kept on the device between calls. Each state an entry
replaces is donated to it, so XLA writes the new cache into the old one's
memory rather than beside it.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..compiler import LinnetError, run_compiler, std_arguments
from .load import LinnetFunction, load
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
        self.plan = first.plan
        self.generics = first.generics
        self.weights = first.weights
        root = first.root
        self.entries = [
            f["name"].rsplit(".", 1)[1]
            for f in first.plan["functions"]
            if f["kind"] == "entry" and f["block"] == root
        ]
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
                first.plan,
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
    # Any entry will do for `load`, which checks the weights against the
    # plan; the model builds a function for each entry it is asked to run.
    plan = json.loads(
        run_compiler(
            "plan",
            "--no-optimize",
            *(["--root", root] if root is not None else []),
            *std_arguments(std_root),
            str(source),
        )
    )
    name = str(plan["root"]["name"])
    entries = [
        str(f["name"]).rsplit(".", 1)[1]
        for f in plan["functions"]
        if f["kind"] == "entry" and f["block"] == name
    ]
    if not entries:
        raise LinnetError(f"block `{name}` has no entries")
    first = load(
        source,
        generics=generics,
        weights=weights,
        root=root,
        entry=entries[0],
        bindings=bindings,
        std_root=std_root,
        numerics=numerics,
        cast_dtype=cast_dtype,
    )
    model = LinnetModel(first, generated)
    if mesh is not None:
        model.shard(mesh, rules)
    return model

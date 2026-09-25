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
        self.state = state
        return result

    def reset_state(self) -> None:
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
) -> LinnetModel:
    """Every entry of the root block over one copy of the weights, with its
    state kept on the device. `generated=True` runs the entries as generated
    JAX source (`linnet jax`), whose cache writes are scatters and slice
    updates; `False` runs the StableHLO export. The other arguments are
    `load`'s."""
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
    return LinnetModel(first, generated)

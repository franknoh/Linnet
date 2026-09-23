"""A Linnet entry as generated `jax.numpy` code.

`load_source` is `load` with the entry compiled by `linnet jax` instead of
`linnet stablehlo`: for each input shape the compiler writes a Python module
of straight-line JAX, which is executed as the entry. Because that code is
ordinary JAX, `jax.grad` differentiates it — this is how a Linnet model is
trained in JAX — and `jax.jit`/`jax.vmap` compose with it as usual.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax

from .compiler import LinnetError, run_compiler
from .load import CompiledEntry, LinnetFunction, load, std_arguments


class SourceFunction(LinnetFunction):
    """`LinnetFunction` whose compiled entries are generated JAX source.

    `apply(parameters, *inputs, state=None)` runs the entry on `parameters`
    (path -> array), so `jax.grad(lambda p: loss(f.apply(p, x)))(f.parameters)`
    trains it; `parameters` holds the loaded weights as device arrays.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.parameters: dict[str, Any] = {}
        self._work = Path(tempfile.mkdtemp(prefix="linnet-jax-"))

    def _compile(self, bindings: Mapping[str, int | str]) -> CompiledEntry:
        # The StableHLO export supplies the argument order and state avals;
        # the JAX source supplies the function that runs.
        compiled = super()._compile(bindings)
        arguments = ["jax", "--root", self.root, "--entry", self.entry]
        arguments += ["--numerics", self.numerics]
        arguments += ["--optionals", "present" if self.optionals_present else "absent"]
        for name, value in bindings.items():
            arguments += ["--bind", f"{name}={value}"]
        text = run_compiler(*arguments, *std_arguments(self._std_root), str(self._source))
        path = self._work / f"{self.entry}_{len(self._cache)}.py"
        path.write_text(text, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"linnet_jax_generated_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise LinnetError(f"cannot load the generated module at {path}")
        module: Any = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if (
            list(module.PARAMETERS) != compiled.parameters
            or list(module.STATES) != compiled.state_inputs
        ):
            raise LinnetError(
                "internal: the JAX source and StableHLO exports disagree on arguments"
            )
        jitted = jax.jit(module.main)

        def call(*arguments: Any) -> Any:
            # A lone result is returned bare, as the StableHLO path does.
            outputs = jitted(*arguments)
            return outputs[0] if len(outputs) == 1 else outputs

        compiled.call = call
        compiled.source_path = path
        for name, array in zip(compiled.parameters, compiled.arrays, strict=True):
            self.parameters.setdefault(name, array)
        return compiled

    def generated_source(self) -> str:
        """The JAX source of the most recently compiled entry."""
        for compiled in reversed(list(self._cache.values())):
            if compiled.source_path is not None:
                return compiled.source_path.read_text(encoding="utf-8")
        raise LinnetError("no entry has been compiled yet")


def load_source(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path | Mapping[str, Any],
    root: str | None = None,
    entry: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
    numerics: str = "equivalent",
) -> SourceFunction:
    """`load`, with entries running as generated JAX source (differentiable)."""
    function = load(
        source,
        generics=generics,
        weights=weights,
        root=root,
        entry=entry,
        bindings=bindings,
        std_root=std_root,
        numerics=numerics,
    )
    return SourceFunction(
        function._source,  # pyright: ignore[reportPrivateUsage]
        function.plan,
        function.generics,
        function.weights,
        function._std_root,  # pyright: ignore[reportPrivateUsage]
        function.root,
        function.entry,
        function.numerics,
    )

"""A Linnet entry as generated `jax.numpy` code.

`load_source` is `load` with the entry compiled by `linnet jax` instead of
`linnet stablehlo`: for each input shape the compiler writes a Python module
of straight-line JAX, which is executed as the entry. Because that code is
ordinary JAX, `jax.grad` differentiates it — this is how a Linnet model is
trained in JAX — and `jax.jit`/`jax.vmap` compose with it as usual.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax

from .. import lora
from ..compiler import LinnetError
from ..generated import import_generated
from .load import CompiledEntry, LinnetFunction, load

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np
    from jax.typing import ArrayLike

    from .. import ir


class SourceFunction(LinnetFunction):
    """`LinnetFunction` whose compiled entries are generated JAX source.

    `apply(parameters, *inputs, state=None)` runs the entry on `parameters`
    (path -> array), so `jax.grad(lambda p: loss(f.apply(p, x)))(f.parameters)`
    trains it; `parameters` holds the loaded weights as device arrays.
    """

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
        super().__init__(
            source, program, generics, weights, std_root, root, entry, numerics, cast_dtype
        )
        self.parameters: dict[str, jax.Array] = {}
        self._work = Path(tempfile.mkdtemp(prefix="linnet-jax-"))
        # Low-rank adapters (`add_lora`): patterns, rank, alpha; and the seed
        # their first values are drawn from.
        self.lora: tuple[tuple[str, ...], int, float] | None = None
        self._lora_seed = 0
        # Blocks whose weights the code gathers from their parts where they
        # run, and blocks the backward pass computes again (`blocks`).
        self.sharded: tuple[str, ...] = ()
        self.remat: tuple[str, ...] = ()

    def blocks(self, *, sharded: Sequence[str] = (), remat: Sequence[str] = ()) -> None:
        """Compiles the entry again with each block in `sharded` gathering its
        weights from their parts where it first uses them (under
        `shard_map`: `linnet.jax.fsdp`), and each block in `remat` computed
        again in the backward pass instead of kept (`jax.checkpoint`). Blocks
        are paths: `layers.3`, `lm_head` (`linnet.jax.fsdp.units`)."""
        sharded, remat = tuple(sharded), tuple(remat)
        if (sharded, remat) == (self.sharded, self.remat):
            return
        self.sharded, self.remat = sharded, remat
        self._cache.clear()
        self.parameters = {}

    def add_lora(
        self, patterns: str | Sequence[str], *, rank: int = 16, alpha: float = 32.0, seed: int = 0
    ) -> None:
        """Gives every linear weight whose path matches a glob pattern
        (`"layers.*.attention.*_proj.weight"`) a low-rank adapter (LoRA): the
        layer computes `x @ W.T + (x @ A.T) @ B.T * alpha / rank`, with `A`
        ([rank, in], random) and `B` ([out, rank], zero) the parameters
        `<block>.lora_a` and `<block>.lora_b`. Entries compile again with
        them; `parameters_for` then returns them with the weights, and
        `apply` takes them."""
        if self.lora is not None:
            raise LinnetError("the model already has adapters")
        if rank <= 0:
            raise LinnetError("the adapter rank must be positive")
        chosen = (patterns,) if isinstance(patterns, str) else tuple(patterns)
        self.lora = (chosen, rank, float(alpha))
        self._lora_seed = seed
        self._cache.clear()
        self.parameters = {}

    def _compile(self, bindings: Mapping[str, int | str]) -> CompiledEntry:
        # The StableHLO export supplies the argument order and state avals;
        # the JAX source supplies the function that runs.
        compiled = super()._compile(bindings)
        text = self._export("jax", bindings)
        module = import_generated(self._work, f"{self.entry}_{len(self._cache)}", text)
        path: Path = module.__linnet_path__
        # Adapters come after the parameters both exports share.
        shared = list(module.PARAMETERS)[: len(compiled.parameters)]
        if shared != compiled.parameters or list(module.STATES) != compiled.state_inputs:
            raise LinnetError(
                "internal: the JAX source and StableHLO exports disagree on arguments"
            )
        adapters = list(module.PARAMETERS)[len(compiled.parameters) :]
        if adapters:
            values, dtypes = self._adapters(adapters, compiled)
            compiled.parameters = list(module.PARAMETERS)
            compiled.arrays = [*compiled.arrays, *values]
            compiled.dtypes = [*compiled.dtypes, *dtypes]
        # The StableHLO export's state layout is the source's, past the
        # adapters, so the same states are donated.
        donated = self._donated(
            len(compiled.exported.in_avals) + len(adapters),
            compiled.state_inputs,
            compiled.state_outputs,
        )
        jitted = jax.jit(module.main, donate_argnums=donated)
        compiled.module = module
        compiled.prepared = self._prepared_values(module, compiled.arrays)

        def call(*arguments: ArrayLike) -> jax.Array | tuple[jax.Array, ...]:
            # A lone result is returned bare, as the StableHLO path does. The
            # prepared values are read here: new weights replace them.
            outputs = jitted(*arguments, *compiled.prepared)
            return outputs[0] if len(outputs) == 1 else outputs

        compiled.call = call
        compiled.source_path = path
        for name, array in zip(compiled.parameters, compiled.arrays, strict=True):
            self.parameters.setdefault(name, array)
        return compiled

    def _adapters(
        self, paths: list[str], compiled: CompiledEntry
    ) -> tuple[list[jax.Array], list[np.dtype[np.generic]]]:
        """First values of the adapters at `paths`: `lora_a` uniform in
        +-1/sqrt(in), `lora_b` zero, in their weight's dtype."""
        import numpy as np

        assert self.lora is not None
        rank = self.lora[1]
        weights = dict(zip(compiled.parameters, compiled.arrays, strict=True))
        draws = np.random.default_rng(self._lora_seed)
        values: list[jax.Array] = []
        dtypes: list[np.dtype[np.generic]] = []
        for path in paths:
            block, adapter = path.rsplit(".", 1)
            weight = weights[block + ".weight"]
            out_features, in_features = weight.shape
            if adapter == lora.DOWN:
                host = lora.initial_down(rank, in_features, draws)
            else:
                host = np.zeros((out_features, rank), dtype=np.float32)
            value = jax.numpy.asarray(host, dtype=weight.dtype)
            if self.placement is not None:
                value = self.placement(path, value)
            values.append(value)
            dtypes.append(weight.dtype)
        return values, dtypes

    def _prepared_values(self, module: ModuleType, arrays: Sequence[jax.Array]) -> list[jax.Array]:
        """The entry's weight-only values (`prepare`), computed once and kept
        by key in `self.prepared`, which `LinnetModel` shares across entries."""
        keys = list(getattr(module, "PREPARED", []))
        if not keys:
            return []
        missing = [i for i, key in enumerate(keys) if key not in self.prepared]
        if missing:
            # Only the values not already kept: XLA drops the rest of
            # `prepare`, so a second copy of what another entry prepared
            # (a mixture's dequantized experts) is never made.
            inputs = [arrays[int(name[1:])] for name in module.PREPARE_INPUTS]

            def only_missing(*values: jax.Array) -> tuple[jax.Array, ...]:
                prepared = module.prepare(*values)
                return tuple(prepared[i] for i in missing)

            values = jax.jit(only_missing)(*inputs)
            for i, value in zip(missing, values, strict=True):
                self.prepared[keys[i]] = value
        return [self.prepared[key] for key in keys]

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
    weights: str | Path | Mapping[str, ArrayLike],
    root: str | None = None,
    entry: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
    numerics: str = "fast",
    cast_dtype: bool = False,
) -> SourceFunction:
    """`load`, with entries running as generated JAX source (differentiable).

    With `cast_dtype=True` and the generics asking for a 16-bit `T`, this is
    mixed precision: `apply` takes master parameters in f32, casts them to
    `T` on every call, and `jax.grad` returns f32 gradients for them."""
    function = load(
        source,
        generics=generics,
        weights=weights,
        root=root,
        entry=entry,
        bindings=bindings,
        std_root=std_root,
        numerics=numerics,
        cast_dtype=cast_dtype,
    )
    return SourceFunction(
        function._source,  # pyright: ignore[reportPrivateUsage]
        function.program,
        function.generics,
        function.weights,
        function._std_root,  # pyright: ignore[reportPrivateUsage]
        function.root,
        function.entry,
        function.numerics,
        function.cast_dtype,
    )

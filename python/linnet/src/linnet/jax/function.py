"""Module-level entries as JAX functions.

An `entry` declared at module level, outside any block, has no parameters
or state: it is a function of its inputs alone -- a loss, a preprocessing
step, a reward. `load_function` compiles one to `jax.numpy` code with
`linnet jax`, once for each binding of its generics, and returns it as a
function of arrays under `jax.jit`. The generated code is ordinary JAX, so
`jax.grad`, `jax.jit`, and `jax.vmap` compose with it: inside a transform
the function sees abstract arrays, whose shapes and dtypes bind its
generics the same way.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from .. import ir
from ..compiler import LinnetError, bind_arguments, check_numerics, run_compiler, std_arguments
from ..dtypes import BY_NUMPY
from ..generated import import_generated
from ..plan import compile_plan
from .dtypes import NUMPY_TYPES

if TYPE_CHECKING:
    from typing import TypeAlias

    from numpy.typing import NDArray

    from .load import Result

    # An input: an array, or a Python number for a scalar input.
    Input: TypeAlias = jax.Array | NDArray[np.generic] | bool | int | float
    # The compiled function: the generated `main` under `jax.jit`, all results in a sequence.
    _Compiled: TypeAlias = Callable[..., Sequence[jax.Array]]


class Function:
    """A module-level entry as a function of JAX arrays (see `load_function`)."""

    def __init__(
        self,
        program: ir.Program,
        name: str | None,
        *,
        source: Path,
        std_root: str | Path | None,
        numerics: str,
    ) -> None:
        self.program = program
        self.function = program.module_entry(name)
        self.name: str = self.function.name.rsplit("::", 1)[1]
        self._source = source
        self._std_root = std_root
        self._numerics = numerics
        self._compiled: dict[tuple[tuple[str, str], ...], tuple[_Compiled, Path]] = {}
        self._work: Path | None = None

    @property
    def inputs(self) -> list[str]:
        """The names of the function's inputs, in order."""
        return [argument.name for argument in self.function.params]

    def __call__(self, *inputs: Input, **generics: int | str) -> Result:
        return self.run(list(inputs), generics)

    def run(
        self, inputs: Sequence[Input], generics: Mapping[str, int | str] | None = None
    ) -> Result:
        """Calls the function on arrays (or Python numbers for scalar inputs);
        `generics` binds by name what the inputs do not determine. A single
        result is returned as it is, several as a tuple."""
        params = self.function.params
        if len(params) != len(inputs):
            raise LinnetError(f"`{self.name}` takes {len(params)} inputs, got {len(inputs)}")
        env = ir.Bindings()
        ir.bind_named(env, self.function.generics, generics or {})
        for param, value in zip(params, inputs, strict=True):
            if isinstance(value, bool | int | float):
                if not isinstance(param.type, ir.ScalarType):
                    raise LinnetError(f"input `{param.name}` is a tensor; pass an array")
                continue
            found = BY_NUMPY.get(jnp.dtype(value.dtype).name)
            dtype = str(value.dtype) if found is None else found.name
            ir.bind_input(env, param, [int(size) for size in np.shape(value)], dtype)
        ir.require_bound(env, self.function)
        bindings = ir.bind_names(env, self.function.generics)
        values = [
            self._number(param, value, env) if isinstance(value, bool | int | float) else value
            for param, value in zip(params, inputs, strict=True)
        ]
        key = tuple(sorted(bindings.items()))
        if key not in self._compiled:
            self._compiled[key] = self._compile(bindings)
        outputs = self._compiled[key][0](*values)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def generated_source(self) -> str:
        """The JAX source of the most recent compilation, for reading."""
        if not self._compiled:
            raise LinnetError(f"`{self.name}` has not been compiled yet")
        return next(reversed(self._compiled.values()))[1].read_text(encoding="utf-8")

    def _number(self, param: ir.Value, value: bool | float, env: ir.Bindings) -> jax.Array:
        declared = param.type
        assert isinstance(declared, ir.ScalarType)
        return jnp.asarray(value, dtype=NUMPY_TYPES[env.dtype(declared.dtype)])

    # ---- one compilation per binding

    def _compile(self, bindings: Mapping[str, str]) -> tuple[_Compiled, Path]:
        arguments = ["jax", "--entry", self.name, "--numerics", self._numerics]
        arguments += bind_arguments(bindings)
        text = run_compiler(*arguments, *std_arguments(self._std_root), str(self._source))
        if self._work is None:
            self._work = Path(tempfile.mkdtemp(prefix="linnet-jax-function-"))
        module = import_generated(self._work, f"{self.name}_{len(self._compiled)}", text)
        path: Path = module.__linnet_path__
        if module.PARAMETERS or module.STATES:
            raise LinnetError(f"internal: `{self.name}` compiled with parameters or state")
        return jax.jit(module.main), path


def load_function(
    source: str | Path,
    name: str | None = None,
    *,
    std_root: str | Path | None = None,
    numerics: str = "fast",
) -> Function:
    """Compiles the module-level entry `name` of a source file (the only one
    when `name` is omitted) and returns it as a JAX function: generated
    `jax.numpy` code under `jax.jit`, differentiable with `jax.grad`.

    `numerics` is `"fast"` (the default), `"equivalent"`, or `"exact"`, as
    in `load`."""
    check_numerics(numerics)
    program = compile_plan(source, std_root=std_root, optimize=False, functions=True)
    # Linnet's `i64` needs 64-bit integers. The generated code turns them on
    # when it first loads; on now, the caller's `int64` inputs stay 64-bit.
    jax.config.update("jax_enable_x64", True)
    return Function(program, name, source=Path(source), std_root=std_root, numerics=numerics)


__all__ = ["Function", "load_function"]

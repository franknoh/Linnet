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

import importlib.util
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..compiler import LinnetError, bind_arguments, run_compiler, std_arguments
from ..dtypes import BY_NUMPY, CLASSES
from ..plan import Plan, bind_shape_names, compile_plan
from .dtypes import NUMPY_TYPES


class Function:
    """A module-level entry as a function of JAX arrays (see `load_function`)."""

    def __init__(
        self,
        plan: Plan,
        name: str | None,
        *,
        source: Path,
        std_root: str | Path | None,
        numerics: str,
    ) -> None:
        self.plan = plan
        self.function = plan.module_entry(name)
        self.name: str = self.function["name"].rsplit("::", 1)[1]
        self._source = source
        self._std_root = std_root
        self._numerics = numerics
        self._compiled: dict[tuple[tuple[str, str], ...], tuple[Callable[..., Any], Path]] = {}
        self._work: Path | None = None

    @property
    def inputs(self) -> list[str]:
        """The names of the function's inputs, in order."""
        return [str(argument["name"]) for argument in self.function["body"]["args"]]

    def __call__(self, *inputs: Any, **generics: int | str) -> Any:
        return self.run(list(inputs), generics)

    def run(self, inputs: Sequence[Any], generics: Mapping[str, int | str] | None = None) -> Any:
        """Calls the function on arrays (or Python numbers for scalar inputs);
        `generics` binds by name what the inputs do not determine. A single
        result is returned as it is, several as a tuple."""
        params: list[dict[str, Any]] = self.function["body"]["args"]
        if len(params) != len(inputs):
            raise LinnetError(f"`{self.name}` takes {len(params)} inputs, got {len(inputs)}")
        bindings = self._bindings(params, inputs, generics or {})
        values = [
            self._number(param, value, bindings) if isinstance(value, bool | int | float) else value
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

    # ---- generics from the inputs

    def _bindings(
        self,
        params: list[dict[str, Any]],
        inputs: Sequence[Any],
        given: Mapping[str, int | str],
    ) -> dict[str, str]:
        declared = {str(generic["name"]): generic for generic in self.function["generics"]}
        for name in given:
            if name not in declared:
                raise LinnetError(f"`{self.name}` has no generic parameter `{name}`")
        bindings = {name: str(value) for name, value in given.items()}
        for param, value in zip(params, inputs, strict=True):
            kind = param["type"]["kind"]
            name = param["name"]
            if isinstance(value, bool | int | float):
                if kind != "scalar":
                    raise LinnetError(f"input `{name}` is a tensor; pass an array")
                continue
            if kind == "scalar":
                if np.ndim(value) != 0:
                    raise LinnetError(f"input `{name}` must be a scalar")
                self._bind_dtype(param["type"]["dtype"], value, name, bindings)
                continue
            if kind != "tensor":
                raise LinnetError(f"input `{name}` has a type that cannot be passed from JAX")
            self._bind_dtype(param["type"]["dtype"], value, name, bindings)
            shape = [int(size) for size in np.shape(value)]
            bind_shape_names(param["type"]["shape"], shape, name, bindings)
        for generic in declared.values():
            if generic["name"] not in bindings:
                raise LinnetError(
                    f"cannot determine `{generic['name']}` of `{self.name}` from its inputs; "
                    f"give it by name, `{self.name}(..., {generic['name']}=...)`"
                )
            if generic["kind"] == "dtype":
                admitted = CLASSES[generic.get("class", "any")]
                if bindings[generic["name"]] not in admitted:
                    raise LinnetError(
                        f"`{generic['name']}` of `{self.name}` is {generic['class']}, "
                        f"not {bindings[generic['name']]}"
                    )
        # Sizes bound from shapes arrive as integers; `--bind` takes text.
        return {name: str(value) for name, value in bindings.items()}

    def _bind_dtype(
        self, spec: str | dict[str, Any], value: Any, name: str, bindings: dict[str, str]
    ) -> None:
        found = BY_NUMPY.get(jnp.dtype(value.dtype).name)
        actual = None if found is None else found.name
        if actual is None:
            raise LinnetError(f"input `{name}` has dtype {value.dtype}, which Linnet lacks")
        wanted = spec if isinstance(spec, str) else bindings.setdefault(str(spec["name"]), actual)
        if actual != wanted:
            raise LinnetError(f"input `{name}` has dtype {actual}, expected {wanted}")

    def _number(self, param: dict[str, Any], value: Any, bindings: Mapping[str, str]) -> Any:
        spec = param["type"]["dtype"]
        dtype = spec if isinstance(spec, str) else bindings.get(str(spec["name"]))
        if dtype is None:
            raise LinnetError(
                f"the dtype of `{param['name']}` is not bound; give it by name or pass an array"
            )
        return jnp.asarray(value, dtype=NUMPY_TYPES[dtype])

    # ---- one compilation per binding

    def _compile(self, bindings: Mapping[str, str]) -> tuple[Callable[..., Any], Path]:
        arguments = ["jax", "--entry", self.name, "--numerics", self._numerics]
        arguments += bind_arguments(bindings)
        text = run_compiler(*arguments, *std_arguments(self._std_root), str(self._source))
        if self._work is None:
            self._work = Path(tempfile.mkdtemp(prefix="linnet-jax-function-"))
        path = self._work / f"{self.name}_{len(self._compiled)}.py"
        path.write_text(text, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"linnet_jax_function_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise LinnetError(f"cannot load the generated module at {path}")
        module: Any = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
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
    if numerics not in ("exact", "equivalent", "fast"):
        raise LinnetError('numerics must be "exact", "equivalent", or "fast"')
    plan = compile_plan(source, std_root=std_root, optimize=False, functions=True)
    # Linnet's `i64` needs 64-bit integers. The generated code turns them on
    # when it first loads; on now, the caller's `int64` inputs stay 64-bit.
    jax.config.update("jax_enable_x64", True)
    return Function(plan, name, source=Path(source), std_root=std_root, numerics=numerics)


__all__ = ["Function", "load_function"]

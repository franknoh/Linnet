"""Module-level entries as PyTorch functions.

An `entry` declared at module level, outside any block, has no parameters
or state: it is a function of its inputs alone -- a loss, a preprocessing
step, a reward. `load_function` compiles one and returns it as a callable
over tensors. Its generic parameters are bound from the inputs at each call
(dimensions from shapes, dtype generics from dtypes), so one function serves
every batch size and dtype; one the inputs do not determine (an output
length) is given by name, as in `positions(offset, N=8)`. Every operation is
ordinary PyTorch arithmetic, interpreted or generated, so autograd
differentiates it: `cross_entropy(model(x), labels).backward()` reaches the
model's parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .. import ir
from ..compiler import bind_arguments, run_compiler, std_arguments
from ..generated import generated_directory, import_generated
from ..plan import PlanError, compile_plan
from ..results import Result
from .dtypes import torch_dtype
from .interp import Interpreter
from .module import bind_input

if TYPE_CHECKING:
    from typing import TypeAlias

    # What a function returns: one tensor, or a tuple of them when it has several results.
    # The cache key of one compilation: its generic bindings, sorted, the device, the backend.
    _Key: TypeAlias = tuple[tuple[tuple[str, str], ...], torch.device, str | None]


class Function:
    """A module-level entry as a callable over tensors (see `load_function`)."""

    def __init__(
        self,
        program: ir.Program,
        name: str | None,
        *,
        source: Path,
        std_root: str | Path | None,
        numerics: str,
        compile: bool | str | None,
    ) -> None:
        self.program = program
        self.function = program.module_entry(name)
        self.name: str = self.function.name.rsplit("::", 1)[1]
        self._source = source
        self._std_root = std_root
        self._numerics = numerics
        self._compile = compile
        self._interpreters: dict[torch.device, Interpreter] = {}
        self._generated: dict[_Key, _Generated] = {}

    @property
    def inputs(self) -> list[str]:
        """The names of the function's inputs, in order."""
        return [argument.name for argument in self.function.params]

    def __call__(self, *inputs: torch.Tensor | float, **generics: int | str) -> Result:  # pyright: ignore[reportExplicitAny]
        return self.run(list(inputs), generics)

    def run(
        self,
        inputs: Sequence[torch.Tensor | float],
        generics: Mapping[str, int | str] | None = None,
        compile: bool | str | None = None,
    ) -> Result:  # pyright: ignore[reportExplicitAny]
        """Calls the function. `inputs` are tensors, or Python numbers for
        scalar inputs; `generics` binds by name what the inputs do not
        determine; `compile` overrides `load_function`'s for this call. A
        single result is returned as it is, several as a tuple."""
        params = self.function.params
        if len(params) != len(inputs):
            raise PlanError(f"`{self.name}` takes {len(params)} inputs, got {len(inputs)}")
        device = next(
            (value.device for value in inputs if isinstance(value, torch.Tensor)),
            torch.device("cpu"),
        )
        env = ir.Bindings()
        ir.bind_named(env, self.function.generics, generics or {})
        # Tensors first: they bind the dtype generics a number's type may name.
        for param, value in zip(params, inputs, strict=True):
            if isinstance(value, torch.Tensor):
                bind_input(env, param, value)
        ir.require_bound(env, self.function)
        values = [
            value if isinstance(value, torch.Tensor) else self._number(param, value, env, device)
            for param, value in zip(params, inputs, strict=True)
        ]
        mode = self._compile if compile is None else compile
        if mode is None:
            mode = device.type == "cuda"
        if not mode:
            # An entry's results are tensors (a scalar as a 0-d one).
            return self._interpreter(device).call(self.function, env, values)
        return self._call_generated(env, values, device, mode if isinstance(mode, str) else None)

    def generated_source(self) -> str:
        """The PyTorch source of the most recent compilation, for reading."""
        if not self._generated:
            raise PlanError(f"`{self.name}` has not been compiled yet")
        return next(reversed(self._generated.values())).path.read_text(encoding="utf-8")

    # ---- inputs and generics

    def _number(
        self, param: ir.Value, value: object, env: ir.Bindings, device: torch.device
    ) -> torch.Tensor:
        declared = param.type
        if not isinstance(declared, ir.ScalarType):
            raise PlanError(f"input `{param.name}` is a tensor; pass a torch.Tensor")
        if not isinstance(value, bool | int | float):
            raise PlanError(f"input `{param.name}` is a scalar; pass a number or a 0-d tensor")
        return torch.tensor(value, dtype=torch_dtype(env, declared.dtype), device=device)

    # ---- interpreted

    def _interpreter(self, device: torch.device) -> Interpreter:
        if device not in self._interpreters:
            self._interpreters[device] = Interpreter(self.program, device)
        return self._interpreters[device]

    # ---- generated

    def _call_generated(
        self,
        env: ir.Bindings,
        values: list[torch.Tensor],
        device: torch.device,
        backend: str | None,
    ) -> Result:  # pyright: ignore[reportExplicitAny]
        bindings = ir.bind_names(env, self.function.generics)
        key = (tuple(sorted(bindings.items())), device, backend)
        generated = self._generated.get(key)
        if generated is None:
            generated = self._generated[key] = self._compile_entry(bindings, device, backend)
        outputs = list(generated.main(*values, *generated.constants))
        if backend in ("reduce-overhead", "cudagraphs"):
            # A CUDA graph's outputs are overwritten by its next replay.
            outputs = [value.clone() for value in outputs]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _compile_entry(
        self, bindings: Mapping[str, str], device: torch.device, backend: str | None
    ) -> _Generated:
        text = run_compiler(
            "torch",
            "--entry",
            self.name,
            "--numerics",
            self._numerics,
            *std_arguments(self._std_root),
            *bind_arguments(bindings),
            str(self._source),
            error=PlanError,
        )
        module = import_generated(generated_directory(), self.name, text)
        path: Path = module.__linnet_path__
        if module.PARAMETERS or module.STATES:
            raise PlanError(f"internal: `{self.name}` compiled with parameters or state")
        main: Callable[..., Iterable[torch.Tensor]] = module.main
        if backend in ("reduce-overhead", "cudagraphs"):
            main = torch.compile(main, mode="reduce-overhead")
        elif backend is not None:
            main = torch.compile(main, backend=backend)
        # Input-independent values (index grids, tables), computed once.
        with torch.no_grad():
            constants = list(module.constants(device))
        return _Generated(path, main, constants)


@dataclass
class _Generated:
    """The function compiled for one binding of its generics, on one device."""

    path: Path
    main: Callable[..., Iterable[torch.Tensor]]
    constants: list[torch.Tensor]


def load_function(
    source: str | Path,
    name: str | None = None,
    *,
    std_root: str | Path | None = None,
    numerics: str = "fast",
    compile: bool | str | None = None,
    optimize: bool = True,
) -> Function:
    """Compiles the module-level entry `name` of a source file (the only one
    when `name` is omitted) and returns it as a PyTorch function.

    `numerics` selects the kernels as in `load`. `compile=True` runs the
    function as PyTorch source generated by `linnet torch` for each binding of
    its generics; a backend name such as `"inductor"` also passes that source
    through `torch.compile`, and `"reduce-overhead"` replays it as CUDA
    graphs. Unset, it is `True` for inputs on a CUDA device and `False` (the
    interpreter) elsewhere. Either way the function is differentiable.
    """
    program = compile_plan(
        source, std_root=std_root, optimize=optimize, numerics=numerics, functions=True
    )
    return Function(
        program,
        name,
        source=Path(source),
        std_root=std_root,
        numerics=numerics,
        compile=compile,
    )


__all__ = ["Function", "load_function"]

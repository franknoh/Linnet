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

import importlib.util
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..compiler import find_compiler
from ..plan import Env, Plan, PlanError, compile_plan
from .dtypes import torch_dtype
from .interp import Interpreter
from .module import bind_generics, bind_input

# The dtypes each class of dtype generic admits (`T: Float`).
_FLOATS = {"f16", "bf16", "f32", "f64"}
_INTEGERS = {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}
_CLASSES: dict[str, set[str]] = {
    "float": _FLOATS,
    "integer": _INTEGERS,
    "numeric": _FLOATS | _INTEGERS,
    "any": _FLOATS | _INTEGERS | {"bool"},
}


class Function:
    """A module-level entry as a callable over tensors (see `load_function`)."""

    def __init__(
        self,
        plan: Plan,
        name: str | None,
        *,
        source: Path,
        std_root: str | Path | None,
        numerics: str,
        compile: bool | str | None,
    ) -> None:
        self.plan = plan
        self.function = plan.module_entry(name)
        self.name: str = self.function["name"].rsplit("::", 1)[1]
        self._source = source
        self._std_root = std_root
        self._numerics = numerics
        self._compile = compile
        self._interpreters: dict[torch.device, Interpreter] = {}
        self._generated: dict[tuple[Any, ...], _Generated] = {}
        self._work: Path | None = None

    @property
    def inputs(self) -> list[str]:
        """The names of the function's inputs, in order."""
        return [str(argument["name"]) for argument in self.function["body"]["args"]]

    def __call__(self, *inputs: Any, **generics: int | str) -> Any:
        return self.run(list(inputs), generics)

    def run(
        self,
        inputs: Sequence[Any],
        generics: Mapping[str, int | str] | None = None,
        compile: bool | str | None = None,
    ) -> Any:
        """Calls the function. `inputs` are tensors, or Python numbers for
        scalar inputs; `generics` binds by name what the inputs do not
        determine; `compile` overrides `load_function`'s for this call. A
        single result is returned as it is, several as a tuple."""
        params: list[dict[str, Any]] = self.function["body"]["args"]
        if len(params) != len(inputs):
            raise PlanError(f"`{self.name}` takes {len(params)} inputs, got {len(inputs)}")
        device = next(
            (value.device for value in inputs if isinstance(value, torch.Tensor)),
            torch.device("cpu"),
        )
        env = Env()
        bind_generics(env, self.function["generics"], generics or {})
        # Tensors first: they bind the dtype generics a number's type may name.
        for param, value in zip(params, inputs, strict=True):
            if isinstance(value, torch.Tensor):
                bind_input(env, param, value)
        values = [
            value if isinstance(value, torch.Tensor) else self._number(param, value, env, device)
            for param, value in zip(params, inputs, strict=True)
        ]
        self._check_generics(env)
        mode = self._compile if compile is None else compile
        if mode is None:
            mode = device.type == "cuda"
        if not mode:
            return self._interpreter(device).call(self.function, env, values)
        return self._call_generated(env, values, device, mode if isinstance(mode, str) else None)

    def generated_source(self) -> str:
        """The PyTorch source of the most recent compilation, for reading."""
        if not self._generated:
            raise PlanError(f"`{self.name}` has not been compiled yet")
        return next(reversed(self._generated.values())).path.read_text(encoding="utf-8")

    # ---- inputs and generics

    def _number(
        self, param: dict[str, Any], value: Any, env: Env, device: torch.device
    ) -> torch.Tensor:
        declared = param["type"]
        if declared["kind"] != "scalar":
            raise PlanError(f"input `{param['name']}` is a tensor; pass a torch.Tensor")
        if not isinstance(value, bool | int | float):
            raise PlanError(f"input `{param['name']}` is a scalar; pass a number or a 0-d tensor")
        return torch.tensor(value, dtype=torch_dtype(env, declared["dtype"]), device=device)

    def _check_generics(self, env: Env) -> None:
        for generic in self.function["generics"]:
            name = generic["name"]
            if generic["kind"] == "dim":
                bound = int(generic["sym"]) in env.dims
            elif generic["kind"] == "shape":
                bound = int(generic["sym"]) in env.packs
            else:
                dtype = env.dtypes.get(int(generic["var"]))
                bound = dtype is not None
                if dtype is not None and dtype not in _CLASSES[generic.get("class", "any")]:
                    raise PlanError(f"`{name}` of `{self.name}` is {generic['class']}, not {dtype}")
            if not bound:
                raise PlanError(
                    f"cannot determine `{name}` of `{self.name}` from its inputs; "
                    f"give it by name, `{self.name}(..., {name}=...)`"
                )
        for constraint in self.function["constraints"]:
            if not env.relation_holds(constraint):
                raise PlanError(f"the inputs break the `where` clause of `{self.name}`")

    # ---- interpreted

    def _interpreter(self, device: torch.device) -> Interpreter:
        if device not in self._interpreters:
            self._interpreters[device] = Interpreter(self.plan, device)
        return self._interpreters[device]

    # ---- generated

    def _call_generated(
        self, env: Env, values: list[torch.Tensor], device: torch.device, backend: str | None
    ) -> Any:
        bindings: dict[str, str] = {}
        for generic in self.function["generics"]:
            if generic["kind"] == "dim":
                bindings[generic["name"]] = str(env.dims[int(generic["sym"])])
            elif generic["kind"] == "dtype":
                bindings[generic["name"]] = env.dtypes[int(generic["var"])]
            else:
                raise PlanError(
                    f"`{self.name}` has a shape pack, which generated code cannot take yet; "
                    "run it interpreted (compile=False)"
                )
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
        command = [find_compiler(), "torch", "--entry", self.name, "--numerics", self._numerics]
        if self._std_root is not None:
            command += ["--std", str(self._std_root)]
        for name, value in bindings.items():
            command += ["--bind", f"{name}={value}"]
        completed = subprocess.run(
            [*command, str(self._source)], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise PlanError(completed.stderr.strip() or "`linnet torch` failed")
        if self._work is None:
            self._work = Path(tempfile.mkdtemp(prefix="linnet-function-"))
        path = self._work / f"{self.name}_{len(self._generated)}.py"
        path.write_text(completed.stdout, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"linnet_function_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise PlanError(f"cannot load the generated module at {path}")
        module: Any = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if module.PARAMETERS or module.STATES:
            raise PlanError(f"internal: `{self.name}` compiled with parameters or state")
        main: Callable[..., Any] = module.main
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
    main: Callable[..., Any]
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
    if numerics not in ("exact", "equivalent", "fast"):
        raise PlanError('numerics must be "exact", "equivalent", or "fast"')
    plan = compile_plan(
        source, std_root=std_root, optimize=optimize, numerics=numerics, functions=True
    )
    return Function(
        plan,
        name,
        source=Path(source),
        std_root=std_root,
        numerics=numerics,
        compile=compile,
    )


__all__ = ["Function", "load_function"]

"""A root block whose entries run as generated PyTorch code.

`linnet torch` prints an entry, for one binding of every generic, as a
Python module of straight-line PyTorch: no interpreter in the loop, library
operations dispatched to native kernels, and a function `torch.compile` can
trace whole. `CompiledLinnetModule` keeps the same parameter and state
hierarchy as `LinnetModule` and compiles each entry the first time it sees
an input shape; weights, `state_dict()`, `reset_state()`, and `bind_weights`
work unchanged.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..compiler import find_compiler
from ..plan import Env, Plan, PlanError
from .module import BlockModule, LinnetModule, bind_generics, bind_input, owner_of


class CompiledLinnetModule(LinnetModule):
    """`LinnetModule` whose entries execute generated PyTorch source."""

    def __init__(
        self,
        plan: Plan,
        generics: Mapping[str, int | str],
        device: torch.device,
        *,
        source: Path,
        std_root: str | Path | None,
        numerics: str,
        backend: str | None,
    ) -> None:
        super().__init__(plan, generics, device)
        self._source = Path(source)
        self._std_root = std_root
        self._numerics = numerics
        # A `torch.compile` backend, or a mode ("reduce-overhead": CUDA graphs,
        # which remove the per-kernel launch cost that dominates decoding),
        # or None for eager.
        self._backend = backend
        self._cuda_graphs = backend in ("reduce-overhead", "cudagraphs")
        self._generic_arguments = dict(generics)
        self._compiled: dict[tuple[Any, ...], _Generated] = {}
        self._work = Path(tempfile.mkdtemp(prefix="linnet-torch-"))

    def run_entry(
        self,
        name: str,
        inputs: list[torch.Tensor],
        generics: Mapping[str, int | str] | None = None,
    ) -> Any:
        function = self.entries[name]
        params = function["body"]["args"][1:]
        if len(params) != len(inputs):
            raise PlanError(f"entry `{name}` takes {len(params)} inputs, got {len(inputs)}")
        bindings = self._bindings(function, inputs, generics or {})
        key = (name, tuple(sorted(bindings.items())), self._optionals_present())
        if key not in self._compiled:
            self._compiled[key] = self._compile(name, bindings)
        generated = self._compiled[key]
        parameters = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        arguments: list[Any] = list(inputs)
        arguments += [parameters[f"root.{path}"] for path in generated.parameters]
        arguments += [buffers[f"root.{path}"] for path in generated.states]
        arguments += generated.constants
        outputs = generated.main(*arguments)
        if self._cuda_graphs:
            # Graph outputs are overwritten by the next replay; keep copies.
            outputs = [value.clone() for value in outputs]
        results = list(outputs[: generated.results])
        for path, value in zip(generated.next_states, outputs[generated.results :], strict=True):
            owner, leaf = owner_of(self, path)
            setattr(owner, leaf, value.detach())
        return results[0] if len(results) == 1 else tuple(results)

    # ---- one compilation per entry and shape

    def _bindings(
        self,
        function: dict[str, Any],
        inputs: list[torch.Tensor],
        given: Mapping[str, int | str],
    ) -> dict[str, str]:
        """Every generic the export needs: the root's, then the entry's from
        `given` and the input shapes, by name."""
        bindings = {name: str(value) for name, value in self._generic_arguments.items()}
        env = Env(dict(self.root.env.dims), dict(self.root.env.packs), dict(self.root.env.dtypes))
        bind_generics(env, function["generics"], given)
        for param, value in zip(function["body"]["args"][1:], inputs, strict=True):
            bind_input(env, param, value)
        for generic in function["generics"]:
            if generic["kind"] == "dim":
                symbol = int(generic["sym"])
                if symbol not in env.dims:
                    raise PlanError(f"cannot determine `{generic['name']}` from the inputs")
                bindings[generic["name"]] = str(env.dims[symbol])
            elif generic["kind"] == "dtype":
                symbol = int(generic["var"])
                if symbol in env.dtypes:
                    bindings[generic["name"]] = env.dtypes[symbol]
            else:
                raise PlanError("shape-pack generics of entries cannot be compiled per call yet")
        return bindings

    def _optionals_present(self) -> bool:
        for module in self.modules():
            if isinstance(module, BlockModule) and module.optional_params:
                return not module.absent_params
        return False

    def _compile(self, entry: str, bindings: dict[str, str]) -> _Generated:
        command = [find_compiler(), "torch", "--root", self.plan.root["name"], "--entry", entry]
        command += ["--numerics", self._numerics]
        command += ["--optionals", "present" if self._optionals_present() else "absent"]
        if self._std_root is not None:
            command += ["--std", str(self._std_root)]
        for name, value in bindings.items():
            command += ["--bind", f"{name}={value}"]
        completed = subprocess.run(
            [*command, str(self._source)], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise PlanError(completed.stderr.strip() or "`linnet torch` failed")
        path = self._work / f"{entry}_{len(self._compiled)}.py"
        path.write_text(completed.stdout, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"linnet_generated_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise PlanError(f"cannot load the generated module at {path}")
        module: Any = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        main: Callable[..., Any] = module.main
        if self._cuda_graphs:
            main = torch.compile(main, mode="reduce-overhead")
        elif self._backend is not None:
            main = torch.compile(main, backend=self._backend)
        # Input-independent values (rotary tables, masks) are computed once
        # here and passed to every call.
        constants: list[torch.Tensor] = []
        if hasattr(module, "constants"):
            with torch.no_grad():
                constants = list(module.constants(self.interpreter.device))
        return _Generated(
            path,
            main,
            list(module.PARAMETERS),
            list(module.STATES),
            list(module.NEXT_STATES),
            int(module.RESULTS),
            constants,
        )

    def generated_source(self, entry: str | None = None) -> str:
        """The PyTorch source of the most recently compiled entry, for reading."""
        for key in reversed(list(self._compiled)):
            if entry is None or key[0] == entry:
                return self._compiled[key].path.read_text(encoding="utf-8")
        raise PlanError("no entry has been compiled yet")


@dataclass
class _Generated:
    """One entry compiled for one shape: the module `linnet torch` wrote."""

    path: Path
    main: Callable[..., Any]
    parameters: list[str]  # paths, in argument order after the inputs
    states: list[str]  # paths, after the parameters
    next_states: list[str]  # paths of the results after the entry's own
    results: int
    constants: list[torch.Tensor]  # `constants(device)`, after the states

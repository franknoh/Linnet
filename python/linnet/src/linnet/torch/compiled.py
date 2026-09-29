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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch

from ..compiler import find_compiler
from ..plan import Env, Plan, PlanError
from .module import BlockModule, LinnetModule, bind_generics, bind_input, owner_of
from .placement import Placement


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
        placement: Placement | None = None,
    ) -> None:
        super().__init__(plan, generics, device)
        # Set by `linnet.torch.load` once the weights are bound and placed.
        self.placement = placement
        self._source = Path(source)
        self._std_root = std_root
        self._numerics = numerics
        # A `torch.compile` backend, or a mode ("reduce-overhead": CUDA graphs,
        # which remove the per-kernel launch cost that dominates decoding),
        # or None for eager.
        self._backend = backend
        self._generic_arguments = dict(generics)
        self._compiled: dict[tuple[Any, ...], _Generated] = {}
        # Per call signature (entry, input shapes and dtypes, generics, backend):
        # the compiled entry and where its parameters and states live, so a
        # decoding step does not re-derive them. `bind_weights` clears it.
        self._fast: dict[tuple[Any, ...], _Prepared] = {}
        # Weight-only work the generated entries share (`prepare`), by key:
        # computed once per bound weights, whichever entry asks first.
        self._prepared: dict[str, Any] = {}  # a tensor, or a tuple of them
        # The memory every hand-captured CUDA graph allocates from.
        self._graph_pool: Any = None
        self._work = Path(tempfile.mkdtemp(prefix="linnet-torch-"))

    def _run_entry(
        self,
        name: str,
        inputs: list[torch.Tensor],
        generics: Mapping[str, int | str] | None = None,
        compile: bool | str | None = None,
    ) -> Any:
        """Runs entry `name`. `compile` overrides the module's own setting for
        this entry (`True`: the generated source as it is; a backend or mode
        name: through `torch.compile`), so a server can replay its decoding
        step as CUDA graphs and run prompts of many lengths without them."""
        backend = (
            self._backend if compile is None else compile if isinstance(compile, str) else None
        )
        cuda_graphs = backend in ("reduce-overhead", "cudagraphs")
        signature = (
            name,
            tuple((tuple(value.shape), value.dtype) for value in inputs),
            tuple(sorted((generics or {}).items())),
            backend,
        )
        prepared = self._fast.get(signature)
        if prepared is None:
            prepared = self._prepare(name, inputs, generics or {}, backend)
            self._fast[signature] = prepared
        if prepared.generated.captured:
            return self._replay(prepared, inputs)
        return self._call(prepared, inputs, cuda_graphs)

    def _call(self, prepared: _Prepared, inputs: list[torch.Tensor], cuda_graphs: bool) -> Any:
        generated = prepared.generated
        if self.placement is not None:
            # Inputs enter where the first unit runs.
            inputs = [value.to(self.placement.devices[0]) for value in inputs]
        arguments: list[Any] = list(inputs)
        arguments += [getattr(owner, leaf) for owner, leaf in prepared.parameters]
        arguments += [getattr(owner, leaf) for owner, leaf in prepared.states]
        arguments += generated.constants
        arguments += prepared.prepared
        if generated.placed:
            assert self.placement is not None
            arguments.append(self.placement.devices)
        states = arguments[len(inputs) + len(generated.parameters) :][: len(generated.states)]
        by_path = dict(zip(generated.states, states, strict=True))
        if cuda_graphs:
            # A state written in place is an input the graph mutates, which
            # CUDA graphs allow only at an address that never changes.
            for path in generated.in_place:
                _mark_static(by_path[path])
            # Prepared values and constants never change between calls either;
            # unmarked, a replay would copy each of them in first.
            for value in (*generated.constants, *prepared.prepared):
                _mark_static(value)
        outputs = list(generated.main(*arguments))
        if cuda_graphs:
            # Graph outputs are overwritten by the next replay; keep copies
            # of everything but a state written in place, which is the
            # state's own tensor.
            outputs = [
                value
                if i >= generated.results
                and _same(value, by_path.get(generated.next_states[i - generated.results]))
                else value.clone()
                for i, value in enumerate(outputs)
            ]
        results = list(outputs[: generated.results])
        if self.placement is not None:
            # Results come back where they were computed; hand them over on
            # the first device, where the inputs went in.
            results = [value.to(self.placement.devices[0]) for value in results]
        for (owner, leaf), path, value in zip(
            prepared.next_states, generated.next_states, outputs[generated.results :], strict=True
        ):
            if _same(value, by_path.get(path)):
                continue  # written in place: the buffer already holds it
            setattr(owner, leaf, value.detach())
        return results[0] if len(results) == 1 else tuple(results)

    def _replay(self, prepared: _Prepared, inputs: list[torch.Tensor]) -> Any:
        """A call of an entry replayed as one CUDA graph captured by hand: the
        first call runs as it is (compiling what it needs), the second
        captures the step and replays it, and every later one copies its
        inputs into the graph's and replays. No guard or argument is checked
        per call but that the states and weights the graph read are still
        the module's."""
        if not all(value.is_cuda for value in inputs):
            return self._call(prepared, inputs, cuda_graphs=False)
        graph = prepared.graph
        if graph is not None and not graph.current():
            graph = prepared.graph = None  # a state or weight was replaced
        if prepared.static is None:
            # The first call runs as it is, on the buffers the graph will
            # read, so what it compiles is what the capture then traces.
            prepared.static = [value.contiguous().clone() for value in inputs]
            return self._call(prepared, prepared.static, cuda_graphs=False)
        for static, value in zip(prepared.static, inputs, strict=True):
            static.copy_(value)
        if graph is None:
            graph = prepared.graph = self._capture(prepared, prepared.static)
        graph.graph.replay()
        results = [value.clone() for value in graph.results]
        return results[0] if len(results) == 1 else tuple(results)

    def _capture(self, prepared: _Prepared, static: list[torch.Tensor]) -> _Graph:
        generated = prepared.generated
        parameters = [getattr(owner, leaf) for owner, leaf in prepared.parameters]
        states = [getattr(owner, leaf) for owner, leaf in prepared.states]
        arguments = [*static, *parameters, *states, *generated.constants, *prepared.prepared]
        by_path = dict(zip(generated.states, states, strict=True))
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool):
            outputs = list(generated.main(*arguments))
            if self.tensor_parallel is not None:
                from .parallel import whole

                # Gathered inside the graph: a replay hands back whole
                # tensors, with no collective or DTensor dispatch after it.
                outputs[: generated.results] = [whole(v) for v in outputs[: generated.results]]
            # A state the step returns rather than writes in place goes into
            # its own buffer, inside the graph, so it stays where it is.
            for (owner, leaf), path, value in zip(
                prepared.next_states,
                generated.next_states,
                outputs[generated.results :],
                strict=True,
            ):
                state = by_path.get(path)
                if state is None:
                    setattr(owner, leaf, value.detach())
                elif not _same(value, state):
                    state.copy_(value)
        watched = [
            (table, leaf, table[leaf])
            for table, leaf in (
                _table(owner, leaf) for owner, leaf in (*prepared.parameters, *prepared.states)
            )
        ]
        return _Graph(graph, outputs[: generated.results], watched)

    def _prepare(
        self,
        name: str,
        inputs: list[torch.Tensor],
        generics: Mapping[str, int | str],
        backend: str | None,
    ) -> _Prepared:
        function = self.entries[name]
        params = function["body"]["args"][1:]
        if len(params) != len(inputs):
            raise PlanError(f"entry `{name}` takes {len(params)} inputs, got {len(inputs)}")
        bindings = self._bindings(function, inputs, generics)
        key = (name, tuple(sorted(bindings.items())), self._absent_optionals(), backend)
        if key not in self._compiled:
            self._compiled[key] = self._compile(name, bindings, backend)
        generated = self._compiled[key]
        return _Prepared(
            generated,
            [owner_of(self, path) for path in generated.parameters],
            [owner_of(self, path) for path in generated.states],
            [owner_of(self, path) for path in generated.next_states],
            self._prepared_values(generated),
        )

    def _prepared_values(self, generated: _Generated) -> list[Any]:
        """The entry's weight-only values: shared with every entry that
        computes the same thing, and computed here when none has yet."""
        if not generated.prepared_keys:
            return []
        if not all(key in self._prepared for key in generated.prepared_keys):
            self._lay_out(generated.fused)
            inputs: list[Any] = []
            for name in generated.prepare_inputs:
                if name.startswith("p"):
                    owner, leaf = owner_of(self, generated.all_parameters[int(name[1:])])
                    inputs.append(getattr(owner, leaf))
                else:
                    inputs.append(generated.constants[generated.constant_names.index(name)])
            assert generated.prepare is not None
            with torch.no_grad():
                values = generated.prepare(*inputs, self.interpreter.device)
            for key, value in zip(generated.prepared_keys, values, strict=True):
                self._prepared.setdefault(key, value)
        return [self._prepared[key] for key in generated.prepared_keys]

    def _lay_out(self, groups: list[list[str]]) -> None:
        """Puts each group of parameters `prepare` joins (`FUSED`: a layer's
        query, key, and value weights, say) one after another in one buffer,
        the parameters slices of it, so that the joined weight is a view and
        the fused product costs no memory. Copied once; a group already so
        laid out stays as it is."""
        for paths in groups:
            owners = [owner_of(self, path) for path in paths]
            tensors: list[torch.Tensor] = [getattr(owner, leaf) for owner, leaf in owners]
            first = tensors[0]
            if _adjacent(tensors) or any(
                t.dtype != first.dtype or t.device != first.device or t.shape[1:] != first.shape[1:]
                for t in tensors
            ):
                continue
            rows = sum(t.shape[0] for t in tensors)
            buffer = torch.empty((rows, *first.shape[1:]), dtype=first.dtype, device=first.device)
            offset = 0
            with torch.no_grad():
                for (owner, leaf), tensor in zip(owners, tensors, strict=True):
                    part = buffer[offset : offset + tensor.shape[0]]
                    part.copy_(tensor)
                    if isinstance(tensor, torch.nn.Parameter):
                        part = torch.nn.Parameter(part, requires_grad=tensor.requires_grad)
                    setattr(owner, leaf, part)
                    offset += tensor.shape[0]

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

    def _absent_optionals(self) -> tuple[str, ...]:
        """Every optional parameter the bound weights leave out, by path.

        Checkpoints mix them: a ResNet's convolutions have no bias and its
        classifier has one, Qwen2.5 biases its query/key/value projections and
        nothing else. The compiled source has to know each one, as the
        interpreter does -- deciding from the first block with an optional
        dropped the classifier's bias from every compiled ResNet."""
        absent: list[str] = []
        named = cast("Iterable[tuple[str, torch.nn.Module]]", self.named_modules())
        for name, module in named:
            if isinstance(module, BlockModule):
                path = name.removeprefix("root").removeprefix(".")
                prefix = f"{path}." if path else ""
                absent += [prefix + leaf for leaf in sorted(module.absent_params)]
        return tuple(absent)

    def _compile(self, entry: str, bindings: dict[str, str], backend: str | None) -> _Generated:
        command = [find_compiler(), "torch", "--root", self.plan.root["name"], "--entry", entry]
        command += ["--numerics", self._numerics]
        command += ["--optionals", "present"]
        absent = self._absent_optionals()
        if absent:
            listing = self._work / "absent.txt"
            listing.write_text("\n".join(absent) + "\n", encoding="utf-8")
            command += ["--absent-file", str(listing)]
        if self._std_root is not None:
            command += ["--std", str(self._std_root)]
        for name, value in bindings.items():
            command += ["--bind", f"{name}={value}"]
        if self.placement is not None and not self.placement.trivial:
            command += self.placement.flags()
        elif not any(parameter.requires_grad for parameter in self.parameters()):
            # Weight-only work once at load; a model being trained keeps it
            # in the graph, where gradients flow through it.
            command.append("--prepare")
            if self.tensor_parallel is not None:
                # Joining split weights would gather them onto every process.
                command.append("--no-fuse")
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
        if self.tensor_parallel is not None:
            from .parallel import SplitFunctional

            module.F = SplitFunctional()
        main: Callable[..., Any] = module.main
        # CUDA graphs are captured by hand around the whole step, unless the
        # model is spread over devices by placement or trains: replaying one
        # graph costs a copy per input, where `torch.compile`'s own checks
        # every argument of hundreds on every call. Split over a mesh, every
        # process captures the same step, collectives included, and replays
        # it in step with the others.
        captured = (
            backend in ("reduce-overhead", "cudagraphs")
            and self.interpreter.device.type == "cuda"
            and (self.placement is None or self.placement.trivial)
            and not any(parameter.requires_grad for parameter in self.parameters())
        )
        if captured:
            if backend == "reduce-overhead":
                main = torch.compile(main)
        elif backend in ("reduce-overhead", "cudagraphs"):
            main = torch.compile(main, mode="reduce-overhead")
        elif backend is not None:
            main = torch.compile(main, backend=backend)
        # Input-independent values (rotary tables, masks) are computed once
        # here and passed to every call.
        placed = hasattr(module, "SLOTS")
        constants: list[torch.Tensor] = []
        if hasattr(module, "constants"):
            with torch.no_grad():
                if placed:
                    assert self.placement is not None
                    constants = list(module.constants(self.placement.devices))
                else:
                    constants = list(module.constants(self.interpreter.device))
        return _Generated(
            path,
            main,
            list(module.PARAMETERS),
            list(module.STATES),
            list(module.NEXT_STATES),
            int(module.RESULTS),
            constants,
            placed,
            list(getattr(module, "IN_PLACE", [])),
            list(getattr(module, "PREPARED", [])),
            list(getattr(module, "PREPARE_INPUTS", [])),
            getattr(module, "prepare", None),
            list(module.PARAMETERS),
            list(getattr(module, "CONSTANTS", [])),
            captured,
            [list(group) for group in getattr(module, "FUSED", [])],
        )

    def generated_source(self, entry: str | None = None) -> str:
        """The PyTorch source of the most recently compiled entry, for reading."""
        for key in reversed(list(self._compiled)):
            if entry is None or key[0] == entry:
                return self._compiled[key].path.read_text(encoding="utf-8")
        raise PlanError("no entry has been compiled yet")


@dataclass
class _Prepared:
    """A compiled entry and where its arguments live, for one call signature."""

    generated: _Generated
    parameters: list[tuple[Any, str]]  # (module, attribute) of each parameter
    states: list[tuple[Any, str]]
    next_states: list[tuple[Any, str]]
    prepared: list[torch.Tensor]  # the entry's weight-only values, after the constants
    static: list[torch.Tensor] | None = None  # inputs a hand-captured graph reads
    graph: _Graph | None = None  # the step captured by hand, after one call


@dataclass
class _Graph:
    """One step captured as a CUDA graph: the results it writes, and the
    weights and states it was captured over."""

    graph: Any  # torch.cuda.CUDAGraph
    results: list[torch.Tensor]
    # (a module's parameter or buffer table, name, tensor): checked on every
    # replay, which a lookup in the table itself keeps to a few microseconds
    # for hundreds of weights, where `getattr` on the module takes a hundred.
    watched: list[tuple[dict[str, Any], str, torch.Tensor]]

    def current(self) -> bool:
        return all(table.get(leaf) is tensor for table, leaf, tensor in self.watched)


def _table(owner: Any, leaf: str) -> tuple[dict[str, Any], str]:
    """Where a module keeps `leaf`: its parameters, or its buffers (states)."""
    parameters: dict[str, Any] = owner._parameters
    return (parameters if leaf in parameters else owner._buffers), leaf


def _mark_static(tensor: Any) -> None:
    # A prepared value can be a tuple (packed weights and their scales).
    if isinstance(tensor, tuple):
        for part in tensor:  # pyright: ignore[reportUnknownVariableType]
            _mark_static(part)
        return
    if not isinstance(tensor, torch.Tensor):
        return
    if not getattr(tensor, "_linnet_static", False):
        torch._dynamo.mark_static_address(tensor)  # pyright: ignore[reportPrivateUsage]
        tensor._linnet_static = True  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


def _adjacent(tensors: list[torch.Tensor]) -> bool:
    """Whether `tensors` lie one after another in one buffer, as `_adjacent`
    in generated source joins them without a copy."""
    first = tensors[0]
    storage = first.untyped_storage().data_ptr()
    offset = first.storage_offset()
    for tensor in tensors:
        if (
            not tensor.is_contiguous()
            or tensor.dtype != first.dtype
            or tensor.untyped_storage().data_ptr() != storage
            or tensor.storage_offset() != offset
        ):
            return False
        offset += tensor.numel()
    return True


def _same(value: torch.Tensor, state: torch.Tensor | None) -> bool:
    """Whether `value` is `state` itself, as a state written in place comes back."""
    return (
        state is not None
        and value.data_ptr() == state.data_ptr()
        and value.shape == state.shape
        and value.dtype == state.dtype
    )


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
    placed: bool = False  # `main` also takes the device of every slot
    in_place: list[str] = field(default_factory=list[str])  # states `main` writes into
    prepared_keys: list[str] = field(default_factory=list[str])  # `PREPARED`
    prepare_inputs: list[str] = field(default_factory=list[str])  # `PREPARE_INPUTS`
    prepare: Callable[..., Any] | None = None
    all_parameters: list[str] = field(default_factory=list[str])  # `PARAMETERS`
    constant_names: list[str] = field(default_factory=list[str])  # `CONSTANTS`
    captured: bool = False  # replayed as a CUDA graph captured by hand
    fused: list[list[str]] = field(default_factory=list[list[str]])  # `FUSED`

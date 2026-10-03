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

import fnmatch
import importlib.util
import math
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
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
        # The process group of a sharded model (`load(tensor_parallel=...)` on
        # a card with `Shards`), which each generated module's `_all_reduce`
        # sums over.
        self.shard_group: Any = None
        self._generic_arguments = dict(generics)
        self._compiled: dict[tuple[Any, ...], _Generated] = {}
        # Per call signature (entry, input shapes and dtypes, generics,
        # backend, whether the model trains): the compiled entry and where its
        # parameters and states live, so a decoding step does not re-derive
        # them. `bind_weights` clears it.
        self._fast: dict[tuple[Any, ...], _Prepared] = {}
        # The parameters, listed once: whether any requires gradients is
        # asked on every call. Cleared when a parameter object is replaced.
        self._parameter_list: list[torch.nn.Parameter] | None = None
        # Low-rank adapters (`add_lora`): the patterns, rank and alpha the
        # generated source adds them for, and the weights adapted.
        self.lora: tuple[tuple[str, ...], int, float] | None = None
        self.lora_paths: list[str] = []
        # Units whose parameters are split across processes (`fully_shard`).
        self.fully_sharded: tuple[str, ...] = ()
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
        trains = self.trains()
        signature = (
            name,
            tuple((tuple(value.shape), value.dtype) for value in inputs),
            tuple(sorted((generics or {}).items())),
            backend,
            trains,
        )
        prepared = self._fast.get(signature)
        if prepared is None:
            prepared = self._prepare(name, inputs, generics or {}, backend, trains)
            self._fast[signature] = prepared
        if prepared.generated.captured:
            return self._replay(prepared, inputs)
        return self._call(prepared, inputs, cuda_graphs)

    def trains(self) -> bool:
        """Whether any parameter requires gradients. A model being trained
        runs without weight-only work done ahead (`prepare`), without joined
        weights, and without hand-captured CUDA graphs: each would hold values
        computed from the weights before an update, or replace the parameter
        objects an optimizer holds."""
        if self._parameter_list is None:
            self._parameter_list = list(self.parameters())
        return any(parameter.requires_grad for parameter in self._parameter_list)

    def add_lora(
        self,
        patterns: str | Sequence[str],
        *,
        rank: int = 16,
        alpha: float = 32.0,
        seed: int = 0,
    ) -> list[str]:
        """Adds a low-rank adapter (LoRA) to every linear weight whose path
        matches a glob pattern (`"layers.*.attention.*_proj.weight"`): the
        layer computes `x @ W.T + (x @ A.T) @ B.T * alpha / rank`, `A`
        ([rank, in], random) and `B` ([out, rank], zero) parameters of the
        weight's block named `lora_a` and `lora_b`. The output is unchanged
        until `B` trains. Afterwards only the adapters train (`set_trainable`
        changes that). Returns the adapted weights' paths."""
        if self.fully_sharded:
            raise PlanError("add adapters before `fully_shard`")
        if self.lora is not None:
            raise PlanError("the model already has adapters; merge them first")
        if rank <= 0:
            raise PlanError("the adapter rank must be positive")
        chosen = [patterns] if isinstance(patterns, str) else list(patterns)
        generator = torch.Generator().manual_seed(seed)
        adapted: list[str] = []
        # Listed first: registering adapters changes the blocks' parameters.
        for path, weight in list(self.root.named_parameters(remove_duplicate=False)):
            owner, leaf = owner_of(self, path)
            if (
                leaf != "weight"
                or weight.dim() != 2
                or not weight.is_floating_point()
                or leaf in owner.absent_params
                or not any(fnmatch.fnmatchcase(path, pattern) for pattern in chosen)
            ):
                continue
            out_features, in_features = weight.shape
            bound = 1.0 / math.sqrt(in_features)
            down = (torch.rand(rank, in_features, generator=generator) * 2 - 1) * bound
            owner.register_parameter(
                "lora_a",
                torch.nn.Parameter(down.to(dtype=weight.dtype, device=weight.device)),
            )
            zero = torch.zeros(out_features, rank, dtype=weight.dtype, device=weight.device)
            owner.register_parameter("lora_b", torch.nn.Parameter(zero))
            adapted.append(path)
        if not adapted:
            raise PlanError("no linear weight matches " + ", ".join(chosen))
        self.lora = (tuple(chosen), rank, alpha)
        self.lora_paths = adapted
        self._recompile()
        self.set_trainable(["*.lora_a", "*.lora_b"])
        return adapted

    def merge_lora(self) -> list[str]:
        """Adds each adapter's product into its weight, `W += B @ A * alpha /
        rank` (in f32), and removes the adapters: the model computes the same
        as before with plain weights, ready to serve or export. Returns the
        weights changed."""
        if self.lora is None:
            return []
        if self.fully_sharded:
            raise PlanError(
                "a sharded model's weights are split across processes: save the adapters and "
                "merge them into a model on one process"
            )
        _, rank, alpha = self.lora
        with torch.no_grad():
            for path in self.lora_paths:
                owner, _ = owner_of(self, path)
                weight = owner.get_parameter("weight")
                delta = (
                    owner.get_parameter("lora_b").float() @ owner.get_parameter("lora_a").float()
                )
                weight.copy_((weight.float() + delta * (alpha / rank)).to(weight.dtype))
                del owner._parameters["lora_a"]
                del owner._parameters["lora_b"]
        merged = self.lora_paths
        self.lora = None
        self.lora_paths = []
        self._recompile()
        return merged

    def copy_weights(self, source: LinnetModule) -> None:
        """Copies `source`'s weights in (see `LinnetModule.copy_weights`).
        Compiled entries and captured CUDA graphs stay; the weight-only
        values computed ahead are computed again into their own tensors."""
        super().copy_weights(source)
        self._refresh_prepared()

    def _refresh_prepared(self) -> None:
        """Computes every prepared value again from the current weights, into
        the tensor that already holds it."""
        done: set[str] = set()
        for generated in self._compiled.values():
            keys = generated.prepared_keys
            if not keys or all(key in done for key in keys) or generated.prepare is None:
                continue
            inputs: list[Any] = []
            for name in generated.prepare_inputs:
                if name.startswith("p"):
                    owner, leaf = owner_of(self, generated.all_parameters[int(name[1:])])
                    inputs.append(getattr(owner, leaf))
                else:
                    inputs.append(generated.constants[generated.constant_names.index(name)])
            with torch.no_grad():
                values = generated.prepare(*inputs, self.interpreter.device)
                for key, value in zip(keys, values, strict=True):
                    if key in done:
                        continue
                    done.add(key)
                    if key in self._prepared:
                        _copy_into(self._prepared[key], value)
                    else:
                        self._prepared[key] = value

    def _recompile(self) -> None:
        """Drops every compiled entry and prepared value: the parameters
        changed."""
        self._compiled.clear()
        self._fast.clear()
        self._prepared.clear()
        self.forget_parameters()

    def forget_parameters(self) -> None:
        """Called when parameter objects are replaced (bound weights tied, a
        group joined into one buffer)."""
        self._parameter_list = None

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
        if self.fully_sharded and generated.eager:
            from .fsdp import regathered

            with regathered():
                outputs = list(generated.main(*arguments))
        else:
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
                # Written in place: the buffer already holds it. A write from
                # values that require gradients leaves the buffer in this
                # call's graph; the next call starts from its value alone.
                if value.requires_grad:
                    value.detach_()
                continue
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
            from torch._dynamo.exc import TorchRuntimeError  # pyright: ignore[reportPrivateUsage]

            try:
                graph = prepared.graph = self._capture(prepared, prepared.static)
            except TorchRuntimeError:
                # The step had to compile again -- a guard the first call
                # set no longer held -- which nothing can under a capture.
                # This call runs as it is, compiling, and the next captures.
                return self._call(prepared, prepared.static, cuda_graphs=False)
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
        _warm_blas(self.interpreter.device)
        graph = torch.cuda.CUDAGraph()
        # Thread-local: a server's other threads (Triton's, say) may call
        # into CUDA while this one captures, which in the default global mode
        # invalidates the capture.
        with torch.cuda.graph(graph, pool=self._graph_pool, capture_error_mode="thread_local"):
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
        trains: bool,
    ) -> _Prepared:
        function = self.entries[name]
        params = function["body"]["args"][1:]
        if len(params) != len(inputs):
            raise PlanError(f"entry `{name}` takes {len(params)} inputs, got {len(inputs)}")
        bindings = self._bindings(function, inputs, generics)
        key = (name, tuple(sorted(bindings.items())), self._absent_optionals(), backend, trains)
        if key not in self._compiled:
            self._compiled[key] = self._compile(name, bindings, backend, trains)
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
            self.forget_parameters()

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
                symbol = int(generic["sym"])
                if symbol not in env.packs:
                    raise PlanError(f"cannot determine `{generic['name']}` from the inputs")
                # A shape pack's dimensions, as `linnet torch --bind S=2,3` takes them.
                bindings[generic["name"]] = ",".join(map(str, env.packs[symbol]))
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
                absent += [prefix + sub for sub in sorted(module.absent_subs)]
        return tuple(absent)

    def _compile(
        self, entry: str, bindings: dict[str, str], backend: str | None, trains: bool
    ) -> _Generated:
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
        if self.lora is not None:
            patterns, rank, alpha = self.lora
            for pattern in patterns:
                command += ["--lora", pattern]
            command += ["--lora-rank", str(rank), "--lora-alpha", repr(float(alpha))]
        for unit in self.fully_sharded:
            command += ["--fully-shard", unit]
        if self.placement is not None and not self.placement.trivial:
            command += self.placement.flags()
        elif not trains and self.lora is None and not self.fully_sharded:
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
        if self.shard_group is not None and hasattr(module, "_GROUP"):
            module._GROUP = self.shard_group
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
            and not trains
            and not self.fully_sharded
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
            backend is None,
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


_blas_threads = threading.local()


def _warm_blas(device: torch.device) -> None:
    """Makes this thread's cuBLAS and cuBLASLt handles, outside any capture:
    they are made per thread on first use, which a capture cannot do, and a
    server may capture a step on a worker thread after warming up on
    another."""
    if getattr(_blas_threads, "warm", False):
        return
    small = torch.ones(16, 16, dtype=torch.bfloat16, device=device)
    torch.addmm(small[0], small, small)
    torch.matmul(small.float(), small.float())
    _blas_threads.warm = True


def _copy_into(kept: Any, value: Any) -> None:
    """Writes `value` into the tensors `kept` holds (a tensor, or a tuple of
    them); a tensor that is already `value`'s memory stays as it is."""
    if isinstance(kept, tuple):
        for old, new in zip(cast(tuple[Any, ...], kept), cast(tuple[Any, ...], value), strict=True):
            _copy_into(old, new)
        return
    old_tensor = cast(torch.Tensor, kept)
    new_tensor = cast(torch.Tensor, value)
    same = (
        old_tensor.data_ptr() == new_tensor.data_ptr()
        and old_tensor.shape == new_tensor.shape
        and old_tensor.stride() == new_tensor.stride()
    )
    if not same:
        old_tensor.copy_(new_tensor)


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
    eager: bool = True  # `main` runs without `torch.compile`

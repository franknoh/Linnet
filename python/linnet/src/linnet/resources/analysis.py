"""Memory analysis of a model under an execution configuration.

`MemoryModel` traces the entry once, with the batch size, the sequence
length and the cache length kept as free symbols when the configuration
asks for them, and `analyze` evaluates it for given values: one sweep over
the steps, not a new trace. That is what makes the fit search in
`linnet.resources.planner` cheap.

    from linnet.resources import ExecutionConfig, MemoryModel

    model = MemoryModel("llama-3.1-8b-instruct", ExecutionConfig(batch=16, context=8192))
    print(model.analyze().expected_peak)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .. import dtypes, ir, nest
from ..parallel import layout_stages, split_axis, state_axis
from ..plan import holds
from ..weights import paths_by_tensor, read_bindings
from . import expr as ex
from .backends import BackendResourceModel, Channels, Estimate, backend_model
from .config import ExecutionConfig
from .graph import Category, Confidence, TensorGraph, lifetimes, peak, peak_expr, plan_buffers
from .kvcache import kv_state_paths
from .performance import DeviceSpec, Throughput, step_time
from .pipeline import StagePart, split_graph
from .result import Allocation, DeviceMemory, MemoryAnalysisResult, MemoryComponent
from .storage import tensor_bytes
from .trace import (
    Lowering,
    SymEnv,
    TraceError,
    TraceOptions,
    constant,
    entry_env,
    root_env,
    trace,
)
from .training import TrainingTimeline, timeline

ROLES = ("batch", "context", "cache")
_FORMULA_LIMIT = 4000

# Compiled programs by source, root, standard library, numerics and the
# source's modification time: a planner builds a model per candidate, and
# compiling is the slow part.
_programs: dict[tuple[str, str | None, str | None, str, int], ir.Program] = {}


def _program(
    source: Path, root: str | None, std_root: str | Path | None, numerics: str
) -> ir.Program:
    key = (
        str(source.resolve()),
        root,
        None if std_root is None else str(std_root),
        numerics,
        source.stat().st_mtime_ns,
    )
    if key not in _programs:
        _programs[key] = ir.load_program(source, root=root, std_root=std_root, numerics=numerics)
    return _programs[key]


def entry_roles(
    model: str | Path | nest.Card,
    config: ExecutionConfig,
    *,
    root: str | None = None,
    std_root: str | Path | None = None,
) -> set[str]:
    """The roles (`batch`, `context`, `cache`) the configuration's entry
    has generics for; for an entry with none, those of the model's root
    (a decoding step sized by the root's `Batch`)."""
    card, source = nest.model_source(model)
    program = _program(source, root or (card.root if card else None), std_root, config.numerics)
    function = program.entry(config.entry, prefer="forward")
    roles = {
        "batch": config.batch_generics,
        "context": config.context_generics,
        "cache": config.cache_generics,
    }
    for generics in (function.generics, program.root.generics):
        names = {g.name for g in generics}
        found = {role for role, bound in roles.items() if names & set(bound)}
        if found:
            return found
    return set()


def _split_derived(
    graph: TensorGraph, paths: Collection[str], devices: int
) -> tuple[TensorGraph, set[int]]:
    """`graph` as DTensor runs it: a value computed from a split weight stays
    split, one part per device, so it holds a `devices`-th of its bytes.
    Returns the graph and the values so split."""
    split = {o.id for o in graph.objects if o.persistent and o.path in paths}
    derived: set[int] = set()
    for step in graph.steps:
        if any(graph.objects[i].storage in split | derived for i in step.inputs):
            derived |= {graph.objects[i].storage for i in step.outputs}
    objects = tuple(
        replace(o, nbytes=ex.floordiv(o.nbytes, ex.const(devices)))
        if o.id in derived and o.owns_storage and not o.persistent
        else o
        for o in graph.objects
    )
    return replace(graph, objects=objects), derived


def _units(program: ir.Program, env: SymEnv) -> list[str]:
    """The root's blocks a pipeline places, in declaration order: each
    sub-block, and each element of a block array."""
    found: list[str] = []
    for member in program.blocks[program.root.name].members:
        if member.kind != "sub":
            continue
        kind = member.type.inner if isinstance(member.type, ir.OptionalType) else member.type
        if isinstance(kind, ir.ArrayType):
            length = constant(env.dim(kind.length), f"the length of `{member.name}`")
            found += [f"{member.name}.{i}" for i in range(length)]
        else:
            found.append(member.name)
    return found


def _ranges(units: Sequence[str]) -> str:
    """`embedding, layers.0-15` for a run of blocks."""
    out: list[str] = []
    run: list[tuple[str, int]] = []

    def flush() -> None:
        if run:
            name, first = run[0]
            last = run[-1][1]
            out.append(f"{name}.{first}" if first == last else f"{name}.{first}-{last}")
            run.clear()

    for unit in units:
        name, _, index = unit.rpartition(".")
        if name and index.isdigit():
            if run and (run[-1][0] != name or run[-1][1] + 1 != int(index)):
                flush()
            run.append((name, int(index)))
        else:
            flush()
            out.append(unit)
    flush()
    return ", ".join(out)


def _arrays(program: ir.Program) -> tuple[str, ...]:
    """The block arrays of the hierarchy, outermost first: `layers`."""
    found: list[str] = []
    for entry in program.manifest:
        if "[*]" in entry.path:
            found.append(entry.path.split("[*]", 1)[0])
    return tuple(dict.fromkeys(found))


class MemoryModel:
    """One entry of a model, traced once, analyzed for any value of the
    roles left free."""

    def __init__(
        self,
        model: str | Path | nest.Card,
        config: ExecutionConfig,
        *,
        root: str | None = None,
        std_root: str | Path | None = None,
        bindings: bool = True,
        free: tuple[str, ...] = (),
        backend: BackendResourceModel | None = None,
    ) -> None:
        """`model` is a `.linnet` file or anything `linnet.nest.load` takes:
        a Nest name, a Hugging Face Hub repo, a model directory, or a
        `nest.Card`. A card's generics are defaults the configuration's
        `bindings` override; with `bindings`, its weight bindings decide
        which parameters are tied and which optional ones are present."""
        self.config = config
        processes = config.tensor_parallel
        self.backend = backend or backend_model(
            config.backend,
            context_bytes=config.context_bytes,
            compiled=config.compiled,
            processes=max(
                processes,
                config.pipeline_parallel,
                config.training.shards if config.training else 1,
            ),
            # A group each of tensor and pipeline parallelism and sharding.
            communicators=(processes > 1)
            + (config.pipeline_parallel > 1)
            + bool(config.training and config.training.shards > 1),
        )
        numerics = config.numerics
        card, self.source_path = nest.model_source(model)
        self.name = card.name if card is not None else self.source_path.stem
        self.generics: Mapping[str, int | str] = dict(card.generics) if card is not None else {}
        source_root = root if root is not None or card is None else card.root
        program = _program(self.source_path, source_root, std_root, numerics)
        self.card = card
        self.program = program
        self.root = source_root
        # Paths bound to one checkpoint tensor share it; bound paths are
        # the optional parameters present.
        self.tied: dict[str, str] = {}
        self.present: frozenset[str] = frozenset()
        mapping = card.bindings_path if bindings and card is not None else None
        if mapping is not None and mapping.exists():
            bound = read_bindings(mapping)
            for paths in paths_by_tensor(bound).values():
                self.tied.update((path, paths[0]) for path in paths[1:])
            self.present = frozenset(bound)
        function = program.entry(config.entry, prefer="forward")
        self.entry = function.short_name
        self.free = tuple(
            r
            for r in ROLES
            if r in free or getattr(config, r if r != "cache" else "cache") is not None
        )
        if "context" in self.free and config.cache is None:
            self.free = tuple(dict.fromkeys([*self.free, "cache"]))
        self.roles = {
            "batch": config.batch_generics,
            "context": config.context_generics,
            "cache": config.cache_generics,
        }
        root_values: dict[str, int | str | ex.Expr] = dict(self.generics)
        root_values.update(config.bindings)
        for generic in program.root.generics:
            role = self._role(generic.name)
            if generic.kind == "dim" and role is not None:
                root_values[generic.name] = ex.sym(role)
            if (
                generic.kind == "dtype"
                and config.dtype is not None
                and generic.dtype_class
                in (
                    "float",
                    None,
                )
            ):
                root_values[generic.name] = config.dtype
        inputs: dict[str, int | ex.Expr] = {}
        for generic in function.generics:
            role = self._role(generic.name)
            if role is not None:
                inputs[generic.name] = ex.sym(role)
            elif isinstance(config.bindings.get(generic.name), int):
                inputs[generic.name] = int(config.bindings[generic.name])
        # Tensor parallelism: a model with a `Shards` generic is each
        # process's own program, its collectives real; any other is split
        # as DTensors split it, weights and caches only.
        self.splitting = "none"
        self._split_paths: set[str] = set()  # the weights and caches DTensor splits
        self._derived: set[int] = set()  # values computed from them
        if processes > 1:
            if any(g.name == "Shards" for g in program.root.generics):
                root_values["Shards"] = processes
                self.splitting = "shards"
            elif config.training is not None:
                raise TraceError(
                    "training under DTensor tensor parallelism is not analyzed; a model with a "
                    "`Shards` generic splits itself"
                )
            elif config.pipeline_parallel > 1:
                raise TraceError(
                    "a pipeline splits its stages over processes only for a model that says how "
                    "it splits (a `Shards` generic)"
                )
            else:
                self.splitting = "dtensor"
        if config.pipeline_parallel > 1 and config.microbatches < config.pipeline_parallel:
            raise TraceError(
                f"a pipeline of {config.pipeline_parallel} stages needs at least as many "
                f"micro-batches, not {config.microbatches}"
            )
        present = set(self.present) | set(config.optionals)
        options = TraceOptions(
            present=lambda path: path in present or any(p.startswith(f"{path}.") for p in present),
            tied=self.tied,
            kv_states=kv_state_paths(program),
            parts=self._parts if self.splitting == "dtensor" else lambda path, kind, shape: 1,
        )
        lowering = Lowering(identities=frozenset()) if processes > 1 else Lowering()
        self.graph: TensorGraph = trace(
            program, function.short_name, root_values, inputs, lowering=lowering, options=options
        )
        if self.splitting == "dtensor":
            self.graph, self._derived = _split_derived(self.graph, self._split_paths, processes)
        self.spans = lifetimes(self.graph)
        self.arrays = _arrays(program)
        base = root_env(program, root_values)
        bound = entry_env(function, base, inputs)
        self.constraints = [
            (c.relation, base.dim(c.lhs), base.dim(c.rhs)) for c in program.root.constraints
        ] + [(c.relation, bound.dim(c.lhs), bound.dim(c.rhs)) for c in function.constraints]
        # A pipeline: the stages `linnet.torch.pipeline` runs, and the role
        # its micro-batches cut (the first input's first axis).
        self.stages: list[StagePart] = []
        self.microbatch_role: str | None = None
        if config.pipeline_parallel > 1:
            kept = (Category.STATE, Category.KV_CACHE)
            touched = {
                self.graph.objects[self.graph.objects[i].storage].category
                for step in self.graph.steps
                for i in (*step.inputs, *step.outputs)
            }
            if touched & set(kept):
                raise TraceError("a pipeline runs entries that keep no state between calls")
            units = _units(program, base)
            sizes = self._unit_bytes(units)
            try:
                assigned = layout_stages(units, sizes, config.pipeline_parallel, config.stages)
            except ValueError as error:
                raise TraceError(str(error)) from None
            self.stages = split_graph(self.graph, units, assigned)
            first = function.params[0].type if function.params else None
            axis = first.shape[0] if isinstance(first, ir.TensorType) and first.shape else None
            if isinstance(axis, ir.DimSymbol):
                self.microbatch_role = self._role(axis.name)
            if config.microbatches > 1 and self.microbatch_role is None:
                raise TraceError(
                    "micro-batches cut the first input's first axis, which is not a free batch "
                    "or sequence length here"
                )

    def _unit_bytes(self, units: Sequence[str]) -> list[int]:
        """Each unit's parameter bytes, a tied parameter in every unit that
        holds it, as `linnet.torch.pipeline` balances them."""
        sizes = [0] * len(units)
        ones = {role: 1 for role in self.free}
        for obj in self.graph.objects:
            if obj.category != Category.PARAMETER or obj.path is None:
                continue
            for i, unit in enumerate(units):
                if obj.path.startswith(unit + "."):
                    whole = tensor_bytes(obj.dtype, ex.product(obj.shape))
                    sizes[i] += ex.evaluate(whole, ones)
        return sizes

    def _parts(self, path: str, kind: str, shape: tuple[ex.Expr, ...]) -> int:
        """How many parts DTensor splits the tensor at `path` into: the
        rules of `linnet.parallel`, on sizes the configuration fixes."""
        devices = self.config.tensor_parallel
        sizes = [d.value if isinstance(d, ex.Const) else -1 for d in shape]
        axis = split_axis(path, sizes, devices) if kind == "param" else state_axis(sizes, devices)
        if axis is None:
            return 1
        self._split_paths.add(path)
        return devices

    def _role(self, name: str) -> str | None:
        for role in self.free:
            if name in self.roles[role]:
                return role
        return None

    # ---- evaluation

    def env(
        self, batch: int | None = None, context: int | None = None, cache: int | None = None
    ) -> dict[str, int]:
        """Values for the free roles: those given, else the configuration's;
        the cache length follows the context unless set."""
        values = {
            "batch": batch if batch is not None else self.config.batch,
            "context": context if context is not None else self.config.context,
            "cache": cache if cache is not None else self.config.cache,
        }
        if values["cache"] is None:
            values["cache"] = values["context"]
        env: dict[str, int] = {}
        for role in self.free:
            value = values[role]
            if value is None:
                raise TraceError(f"give the {role} a value")
            env[role] = value
        return env

    def satisfied(self, env: Mapping[str, int]) -> bool:
        """Whether the model's own `where` clauses hold at these values."""
        return all(
            holds(relation, ex.evaluate(lhs, env), ex.evaluate(rhs, env))
            for relation, lhs, rhs in self.constraints
        )

    def analyze(
        self, batch: int | None = None, context: int | None = None, cache: int | None = None
    ) -> MemoryAnalysisResult:
        env = self.env(batch, context, cache)
        if self.stages:
            return self._pipeline(env)
        if self.config.training is not None:
            return self._training(env)
        return self._inference(env)

    # ---- pipelines

    def _micro(self, env: Mapping[str, int]) -> dict[str, int]:
        """`env` for one micro-batch: the role the first input's first axis
        has, cut into the configuration's micro-batches."""
        count = self.config.microbatches
        micro = dict(env)
        role = self.microbatch_role
        if count > 1 and role is None:
            raise TraceError(
                "micro-batches cut the first input's first axis, which is not a free batch "
                "or sequence length here"
            )
        if count > 1 and role is not None:
            if micro[role] % count:
                raise TraceError(f"the {role} ({micro[role]}) does not divide into {count} parts")
            micro[role] //= count
        return micro

    def throughput(
        self,
        device: DeviceSpec,
        batch: int | None = None,
        context: int | None = None,
        cache: int | None = None,
        replicas: int = 1,
    ) -> Throughput:
        """The predicted time of one step on `device` (see
        `linnet.resources.performance`), over `replicas` copies of this
        layout each on its own data."""
        env = self.env(batch, context, cache)
        # Tokens are the positions of token ids (the first input, an
        # integer tensor); any other input counts its samples.
        first = self.graph.objects[self.graph.inputs[0]] if self.graph.inputs else None
        tokens = 1
        if first is not None and first.shape:
            ids = not dtypes.dtype(first.dtype).is_float
            tokens = ex.evaluate(ex.product(first.shape) if ids else first.shape[0], env)
        training = self.config.training

        def repeated(graph: TensorGraph, at: Mapping[str, int]) -> float:
            """The share of the forward pass checkpointing runs again."""
            if training is None or training.checkpoint.kind == "none":
                return 0.0
            steps = timeline(graph, at, training, self.backend, self.arrays, self.tied)
            return steps.recomputed_flops / steps.forward_flops if steps.forward_flops else 0.0

        if self.stages:
            micro = self._micro(env)
            whole = self.graph.objects
            parts = [
                (
                    part.graph,
                    micro,
                    sum(ex.evaluate(whole[i].nbytes, micro) for i in part.receives),
                    repeated(part.graph, micro),
                )
                for part in self.stages
            ]
        else:
            parts = [(self.graph, env, 0, repeated(self.graph, env))]
        gradients = sum(
            ex.evaluate(o.nbytes, env)
            for o in self.graph.objects
            if o.category == Category.PARAMETER and o.owns_storage
        )
        return step_time(
            parts,
            device,
            tokens=tokens,
            training=training is not None,
            processes=self.config.tensor_parallel,
            microbatches=self.config.microbatches,
            replicas=replicas,
            gradient_bytes=gradients if training is not None else 0,
            compiled=self.config.compiled,
            optimizer_states=training.optimizer.states if training is not None else 0,
            shards=training.shards if training is not None else 1,
        )

    def _pipeline(self, env: Mapping[str, int]) -> MemoryAnalysisResult:
        """Each stage's device: its part of one micro-batch's step, plus what
        the schedule keeps of the others and the buffers they arrive in."""
        config = self.config
        count = config.microbatches
        micro = self._micro(env)
        stages = len(self.stages)
        whole = self.graph.objects

        def size(ids: Sequence[int], floats: bool = False) -> int:
            return sum(
                ex.evaluate(whole[i].nbytes, micro)
                for i in ids
                if not floats or dtypes.dtype(whole[i].dtype).is_float
            )

        results: list[MemoryAnalysisResult] = []
        for part in self.stages:
            received, sent = size(part.receives), size(part.sends)
            extras: list[MemoryComponent] = []
            if config.training is None:
                ends: dict[int, Channels] = {0: "send", stages - 1: "receive"}
                base = self._inference(micro, part.graph, ends.get(part.stage, "both"))
                if count > 1:
                    extras.append(
                        MemoryComponent(
                            "Micro-batches held",
                            Category.ACTIVATION,
                            (count - 1) * (received + sent),
                            Confidence.MODELED,
                            note="the other micro-batches' received and sent values, kept to "
                            "the end of the step",
                        )
                    )
                if count > 1 and part.stage == stages - 1:
                    # At the end of the step one micro-batch's activations
                    # are gone, and every micro-batch's results lie beside
                    # their join: the stage's peak is the larger moment.
                    transient = base.total(Category.ACTIVATION) + sum(
                        c.nbytes or 0 for c in base.components if c.name == "Backend workspace"
                    )
                    during = transient + (count - 1) * (received + sent)
                    end = count * received + 2 * count * sent
                    if end > during:
                        extras.append(
                            MemoryComponent(
                                "Joined results",
                                Category.ACTIVATION,
                                end - during,
                                Confidence.MODELED,
                                note="at the step's end, every micro-batch's results joined "
                                "into one beside them, beyond the forward pass's peak",
                            )
                        )
            else:
                in_flight = count if config.schedule == "gpipe" else min(stages - part.stage, count)

                def flight(
                    steps: TrainingTimeline, part: StagePart = part, in_flight: int = in_flight
                ) -> list[MemoryComponent]:
                    kept = (Category.ACTIVATION, Category.INPUT)
                    held = steps.live_at(len(part.graph.steps) - 1, kept)
                    found: list[MemoryComponent] = []
                    if in_flight > 1:
                        found.append(
                            MemoryComponent(
                                "Micro-batches in flight",
                                Category.ACTIVATION,
                                (in_flight - 1) * held,
                                Confidence.MODELED,
                                note=f"{in_flight} of {count} forward passes kept for their "
                                f"backward ({config.schedule})",
                            )
                        )
                    buffers = (count - in_flight) * size(part.receives)
                    if part.stage < stages - 1:
                        buffers += count * size(part.sends, floats=True)
                    if part.stage > 0:
                        buffers += size(part.receives, floats=True)
                    if buffers:
                        found.append(
                            MemoryComponent(
                                "Pipeline buffers",
                                Category.COMMUNICATION,
                                buffers,
                                Confidence.MODELED,
                                note="every micro-batch's receive buffers, the gradients that "
                                "arrive, and the one sent back",
                            )
                        )
                    return found

                accumulated = config.schedule == "1f1b"
                results.append(self._training(micro, part.graph, flight, accumulated, "both"))
                continue
            added = sum(c.nbytes or 0 for c in extras)
            results.append(
                replace(
                    base,
                    components=(*base.components, *extras),
                    expected_peak=base.expected_peak + added,
                )
            )
        devices = tuple(
            DeviceMemory(
                f"stage {part.stage}",
                _ranges(part.units),
                result.graph_peak,
                result.expected_peak,
                result.total(Category.RUNTIME),
            )
            for part, result in zip(self.stages, results, strict=True)
        )
        stage = max(range(stages), key=lambda k: results[k].expected_peak)
        heaviest = results[stage]
        return replace(
            heaviest,
            devices=devices,
            peak_at=f"stage {stage}, {heaviest.peak_at}",
            assumptions=(
                *heaviest.assumptions,
                f"a pipeline of {stages} stages, {count} micro-batches under {config.schedule}; "
                "each stage holds its own blocks' weights",
            ),
        )

    # ---- inference

    def _persistent(
        self, env: Mapping[str, int], graph: TensorGraph
    ) -> tuple[list[MemoryComponent], int, int]:
        """Parameters, buffers, state and caches: components, graph bytes,
        and bytes as the backend lays the caches out."""
        components: list[MemoryComponent] = []
        graph_total = 0
        laid_out = 0
        groups = [
            (Category.PARAMETER, "Weights"),
            (Category.BUFFER, "Persistent buffers"),
            (Category.STATE, "State"),
        ]
        for category, name in groups:
            objects = graph.by_category(category)
            if not objects and category != Category.PARAMETER:
                continue
            total = ex.total(o.nbytes for o in objects)
            size = ex.evaluate(total, env)
            graph_total += size
            laid_out += size
            components.append(
                MemoryComponent(name, category, size, Confidence.EXACT, _formula(total))
            )
        caches = graph.by_category(Category.KV_CACHE)
        if caches:
            total = ex.total(o.nbytes for o in caches)
            declared = ex.evaluate(total, env)
            laid = [self.config.kv_layout.nbytes(c, env, self.config.kv_dtype) for c in caches]
            size = sum(c.nbytes for c in laid)
            confidence = (
                Confidence.EXACT
                if all(c.confidence == Confidence.EXACT for c in laid)
                else Confidence.MODELED
            )
            graph_total += declared
            laid_out += size
            note = f"{len(caches)} tensors, {self.config.kv_layout.name}"
            components.append(
                MemoryComponent(
                    "KV cache", Category.KV_CACHE, size, confidence, _formula(total), note
                )
            )
        return components, graph_total, laid_out

    def _runtime(
        self, training: bool, channels: Channels = "none"
    ) -> tuple[list[MemoryComponent], int, list[str]]:
        components: list[MemoryComponent] = []
        total = 0
        unknown: list[str] = []
        if channels == "none" and self._split_paths:
            channels = "exchange"
        for item in self.backend.runtime(training, channels):
            estimate = item.estimate
            if estimate.nbytes is None:
                unknown.append(f"{item.name}: {estimate.note}" if estimate.note else item.name)
                continue
            total += estimate.nbytes
            components.append(
                MemoryComponent(
                    item.name,
                    item.category,
                    estimate.nbytes,
                    estimate.confidence,
                    note=estimate.note,
                )
            )
        return components, total, unknown

    def _workspaces(
        self, env: Mapping[str, int], graph: TensorGraph
    ) -> tuple[dict[int, int], dict[int, Estimate], list[str]]:
        sizes: dict[int, int] = {}
        estimates: dict[int, Estimate] = {}
        missing: Counter[str] = Counter()
        for step in graph.steps:
            estimate = self.backend.workspace(step, graph, env)
            if estimate.nbytes is None:
                missing[step.implementation or step.kind] += 1
                continue
            if estimate.nbytes:
                sizes[step.index] = estimate.nbytes
                estimates[step.index] = estimate
        unknown = [f"workspace of `{name}` ({count} calls)" for name, count in missing.items()]
        return sizes, estimates, unknown

    def _inference(
        self, env: Mapping[str, int], graph: TensorGraph | None = None, channels: Channels = "none"
    ) -> MemoryAnalysisResult:
        graph = graph or self.graph
        spans = self.spans if graph is self.graph else lifetimes(graph)
        components, graph_persistent, laid_out = self._persistent(env, graph)
        transient = peak(graph, env, spans)
        work, estimates, unknown = self._workspaces(env, graph)
        gathered = [i for i in graph.outputs if graph.objects[i].storage in self._derived]
        if gathered and graph is self.graph and graph.steps:
            # DTensor hands each split result back whole: a gathered copy
            # beside the part, when the entry returns.
            last = len(graph.steps) - 1
            whole = self.config.tensor_parallel * sum(
                ex.evaluate(graph.objects[i].nbytes, env) for i in gathered
            )
            work[last] = work.get(last, 0) + whole
            estimates[last] = Estimate(
                work[last], Confidence.ESTIMATED, "results gathered whole from the processes"
            )
        with_work = peak(graph, env, spans, work)
        activations = ex.evaluate(ex.total(graph.objects[i].nbytes for i in with_work.live), env)
        at_work = work.get(with_work.step, 0)
        formula = peak_expr(graph, spans)
        whole = self.splitting == "dtensor"
        components.append(
            MemoryComponent(
                "Peak activations",
                Category.ACTIVATION,
                transient.nbytes,
                Confidence.ESTIMATED if whole else Confidence.EXACT,
                _formula(formula),
                "inputs and outputs included"
                + ("; a split weight's results counted split" if whole else ""),
            )
        )
        runtime, runtime_total, runtime_unknown = self._runtime(False, channels)
        if at_work:
            confidence = estimates[with_work.step].confidence
            components.append(
                MemoryComponent(
                    "Backend workspace",
                    Category.WORKSPACE,
                    at_work,
                    confidence,
                    note=estimates[with_work.step].note,
                )
            )
        components.extend(runtime)
        graph_peak = graph_persistent + transient.nbytes
        expected = laid_out + activations + at_work + runtime_total
        step = graph.steps[transient.step] if graph.steps else None
        where = (
            f"step {transient.step}: {step.label} in `{step.scope or 'the root'}`"
            if step
            else "start"
        )
        plan = plan_buffers(graph, env, spans, self.backend.alignment())
        warnings = self._warnings(env)
        return MemoryAnalysisResult(
            configuration=self._describe(),
            components=tuple(components),
            graph_peak=graph_peak,
            expected_peak=max(expected, graph_peak),
            peak_at=where,
            symbols=dict(env),
            unknown=tuple([*unknown, *runtime_unknown]),
            warnings=tuple(warnings),
            assumptions=self._assumptions(),
            formulas={"activations": _formula(formula) or "too long to show"},
            allocation=Allocation(plan.naive, plan.planned, plan.lower_bound),
        )

    # ---- training

    def _training(
        self,
        env: Mapping[str, int],
        graph: TensorGraph | None = None,
        flight: Callable[[TrainingTimeline], list[MemoryComponent]] | None = None,
        accumulated: bool = True,
        channels: Channels = "none",
    ) -> MemoryAnalysisResult:
        """A training step. With `flight` (a pipeline stage), the peak is one
        micro-batch's forward or backward with what `flight` adds for the
        others in flight and their buffers: with every gradient already
        accumulated (`accumulated`, 1F1B's steady state), or with the
        gradients one micro-batch has at that point (GPipe, whose micro-
        batches are all in flight before the first backward). The optimizer
        step, when it is larger, is the peak instead."""
        config = self.config.training
        assert config is not None
        graph = graph or self.graph
        # A pipeline stage keeps its sharded weights whole through the step.
        steps = timeline(
            graph, env, config, self.backend, self.arrays, self.tied, held=flight is not None
        )
        extras: list[MemoryComponent] = []
        if flight is None:
            at, total, parts = steps.peak()
        else:
            optimizer = steps.length - 1
            gradients = sum(i.nbytes for i in steps.intervals if i.category == Category.GRADIENT)
            if accumulated:
                at, total, parts = steps.peak(skip=(Category.GRADIENT,), before=optimizer)
                parts[Category.GRADIENT] = gradients
                total += gradients
            else:
                at, total, parts = steps.peak(before=optimizer)
            extras = flight(steps)
            total += sum(c.nbytes or 0 for c in extras)
            stepping = sum(steps.persistent.values()) + gradients
            if stepping > total:
                at, total, parts, extras = optimizer, stepping, dict(steps.persistent), []
                parts[Category.GRADIENT] = gradients
        components: list[MemoryComponent] = []
        names = [
            (Category.PARAMETER, "Weights"),
            (Category.MASTER, "Master weights"),
            (Category.GRADIENT, "Gradients"),
            (Category.OPTIMIZER, f"Optimizer states ({config.optimizer.name})"),
            (Category.BUFFER, "Persistent buffers"),
            (Category.STATE, "State"),
            (Category.KV_CACHE, "KV cache"),
            (Category.INPUT, "Inputs"),
            (Category.ACTIVATION, "Saved activations"),
            (Category.TEMPORARY, "Backward temporaries"),
            (Category.WORKSPACE, "Backend workspace"),
            (Category.COMMUNICATION, "Gathered parameters"),
        ]
        graph_categories = {
            Category.PARAMETER,
            Category.MASTER,
            Category.GRADIENT,
            Category.OPTIMIZER,
            Category.BUFFER,
            Category.STATE,
            Category.KV_CACHE,
            Category.INPUT,
            Category.ACTIVATION,
            Category.TEMPORARY,
        }
        graph_peak = 0
        for category, name in names:
            size = parts.get(category, 0)
            if not size and category not in (Category.PARAMETER, Category.GRADIENT):
                continue
            confidence = steps.persistent_confidence.get(category, Confidence.EXACT)
            if category in (
                Category.ACTIVATION,
                Category.TEMPORARY,
                Category.WORKSPACE,
                Category.COMMUNICATION,
            ):
                confidence = Confidence.MODELED
            components.append(MemoryComponent(name, category, size, confidence))
            if category in graph_categories:
                graph_peak += size
        runtime, runtime_total, runtime_unknown = self._runtime(True, channels)
        components.extend(extras)
        components.extend(runtime)
        phase = (
            "the optimizer step"
            if at == steps.length - 1
            else ("the forward pass" if at < len(graph.steps) else "the backward pass")
        )
        recompute = (
            steps.recomputed_flops / (3 * steps.forward_flops)
            if config.checkpoint.kind != "none" and steps.forward_flops
            else None
        )
        return MemoryAnalysisResult(
            configuration=self._describe(),
            components=tuple(components),
            graph_peak=graph_peak,
            expected_peak=total + runtime_total,
            peak_at=f"{phase} (timeline step {at} of {steps.length})",
            symbols=dict(env),
            unknown=tuple([*steps.unknown, *runtime_unknown]),
            warnings=tuple(self._warnings(env)),
            assumptions=(
                *self._assumptions(),
                "autograd keeps what each operation's backward reads, by PyTorch's rules for "
                "native kernels and by the operation's arithmetic otherwise",
                "a backward pass costs about twice its forward pass",
            ),
            recompute=recompute,
        )

    # ---- reporting

    def _describe(self) -> dict[str, object]:
        described = self.config.describe()
        described["model"] = self.name
        described["entry"] = self.entry
        return described

    def _assumptions(self) -> tuple[str, ...]:
        found = [
            f"lowering `{self.backend.name}`: library calls run what the plan selected for "
            f"`{self.config.numerics}` numerics",
            "reshape is a view of contiguous storage and a copy otherwise; permute, "
            "broadcast and slice are views",
            "scalars occupy no memory",
        ]
        processes = self.config.tensor_parallel
        if self.splitting == "shards":
            found.append(
                f"tensor parallelism: one of {processes} processes, its `Shards` part of each "
                "weight, collectives allocating their results"
            )
        elif self.splitting == "dtensor":
            found.append(
                f"tensor parallelism: one of {processes} processes; weights and caches split "
                "as `linnet.parallel` rules split them, activations counted whole"
            )
        return tuple(found)

    def _warnings(self, env: Mapping[str, int]) -> list[str]:
        warnings: list[str] = []
        if not self.satisfied(env):
            warnings.append("these values break one of the model's `where` clauses")
        if any(s.kind == "if" for s in self.graph.steps):
            warnings.append("both branches of a runtime `if` were traced")
        return warnings


def _formula(expr: ex.Expr) -> str | None:
    text = ex.format_expr(expr)
    return text if len(text) <= _FORMULA_LIMIT else None

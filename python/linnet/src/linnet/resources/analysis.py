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

import fnmatch
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

from .. import ir, nest
from ..weights import read_bindings
from . import expr as ex
from .backends import BackendResourceModel, Estimate, backend_model
from .config import ExecutionConfig
from .graph import Category, Confidence, TensorGraph, lifetimes, peak, peak_expr, plan_buffers
from .kvcache import kv_state_paths
from .result import Allocation, MemoryAnalysisResult, MemoryComponent
from .trace import Lowering, TraceError, TraceOptions, entry_env, root_env, trace
from .training import timeline

ROLES = ("batch", "context", "cache")
_FORMULA_LIMIT = 4000

# Compiled programs by source, root, standard library, numerics and the
# source's modification time: a planner builds a model per candidate, and
# compiling is the slow part.
_programs: dict[tuple[str, str | None, str | None, str, int], ir.Program] = {}


def _compiled(
    source: Path,
    root: str | None,
    std_root: str | Path | None,
    numerics: str,
    compile: Callable[[], ir.Program],
) -> ir.Program:
    key = (
        str(source.resolve()),
        root,
        None if std_root is None else str(std_root),
        numerics,
        source.stat().st_mtime_ns,
    )
    if key not in _programs:
        _programs[key] = compile()
    return _programs[key]


def _arrays(program: ir.Program) -> tuple[str, ...]:
    """The block arrays of the hierarchy, outermost first: `layers`."""
    found: list[str] = []
    for entry in program.manifest:
        if "[*]" in entry.path:
            found.append(entry.path.split("[*]", 1)[0])
    return tuple(dict.fromkeys(found))


def _choose_entry(program: ir.Program, name: str | None) -> ir.Function:
    if name is not None:
        return program.entry(name)
    entries = program.entries()
    if len(entries) == 1:
        return entries[0]
    for entry in entries:
        if entry.short_name == "forward":
            return entry
    names = ", ".join(e.short_name for e in entries)
    raise TraceError(f"the model has several entries ({names}); name one")


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
        if config.tensor_parallel > 1 or config.pipeline_parallel > 1:
            raise TraceError(
                "tensor and pipeline parallelism are not analyzed yet; "
                "data parallelism with sharded training state is"
            )
        self.config = config
        self.backend = backend or backend_model(
            config.backend, context_bytes=config.context_bytes, compiled=config.compiled
        )
        numerics = config.numerics
        file = None if isinstance(model, nest.Card) else Path(model)
        card: nest.Card | None = None
        if isinstance(model, nest.Card):
            card = model
        elif file is None or not (file.suffix == ".linnet" and file.is_file()):
            card = nest.resolve(model)
        if card is None:
            assert file is not None
            self.source_path = file
            self.name = self.source_path.stem
            self.generics: Mapping[str, int | str] = {}
            source_root = root
            program = _compiled(
                self.source_path,
                root,
                std_root,
                numerics,
                lambda: ir.load_program(
                    self.source_path, root=root, std_root=std_root, numerics=numerics
                ),
            )
        else:
            loaded = card
            self.source_path = card.source_path
            self.name = card.name
            self.generics = dict(card.generics)
            source_root = root or card.root
            program = _compiled(
                self.source_path,
                source_root,
                std_root,
                numerics,
                lambda: (
                    loaded.program(std_root, numerics=numerics)
                    if root is None
                    else ir.load_program(
                        loaded.source_path, root=root, std_root=std_root, numerics=numerics
                    )
                ),
            )
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
            first: dict[str, str] = {}
            for path, tensor in bound.items():
                if tensor in first:
                    self.tied[path] = first[tensor]
                else:
                    first[tensor] = path
            self.present = frozenset(bound)
        function = _choose_entry(program, config.entry)
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
        present = set(self.present) | set(config.optionals)
        options = TraceOptions(
            present=lambda path: path in present or any(p.startswith(f"{path}.") for p in present),
            tied=self.tied,
            kv_states=kv_state_paths(program),
        )
        self.graph: TensorGraph = trace(
            program, function.short_name, root_values, inputs, lowering=Lowering(), options=options
        )
        self.spans = lifetimes(self.graph)
        self.arrays = _arrays(program)
        base = root_env(program, root_values)
        bound = entry_env(function, base, inputs)
        self.constraints = [
            (c.relation, base.dim(c.lhs), base.dim(c.rhs)) for c in program.root.constraints
        ] + [(c.relation, bound.dim(c.lhs), bound.dim(c.rhs)) for c in function.constraints]

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
        for relation, lhs, rhs in self.constraints:
            a, b = ex.evaluate(lhs, env), ex.evaluate(rhs, env)
            holds = {
                "==": a == b,
                "!=": a != b,
                "<": a < b,
                "<=": a <= b,
                ">": a > b,
                ">=": a >= b,
            }[relation]
            if not holds:
                return False
        return True

    def analyze(
        self, batch: int | None = None, context: int | None = None, cache: int | None = None
    ) -> MemoryAnalysisResult:
        env = self.env(batch, context, cache)
        if self.config.training is not None:
            return self._training(env)
        return self._inference(env)

    # ---- inference

    def _persistent(self, env: Mapping[str, int]) -> tuple[list[MemoryComponent], int, int]:
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
            objects = self.graph.by_category(category)
            if not objects and category != Category.PARAMETER:
                continue
            total = ex.total(o.nbytes for o in objects)
            size = ex.evaluate(total, env)
            graph_total += size
            laid_out += size
            components.append(
                MemoryComponent(name, category, size, Confidence.EXACT, _formula(total))
            )
        caches = self.graph.by_category(Category.KV_CACHE)
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

    def _runtime(self, training: bool) -> tuple[list[MemoryComponent], int, list[str]]:
        components: list[MemoryComponent] = []
        total = 0
        unknown: list[str] = []
        for item in self.backend.runtime(training):
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
        self, env: Mapping[str, int]
    ) -> tuple[dict[int, int], dict[int, Estimate], list[str]]:
        sizes: dict[int, int] = {}
        estimates: dict[int, Estimate] = {}
        missing: Counter[str] = Counter()
        for step in self.graph.steps:
            estimate = self.backend.workspace(step, self.graph, env)
            if estimate.nbytes is None:
                missing[step.implementation or step.kind] += 1
                continue
            if estimate.nbytes:
                sizes[step.index] = estimate.nbytes
                estimates[step.index] = estimate
        unknown = [f"workspace of `{name}` ({count} calls)" for name, count in missing.items()]
        return sizes, estimates, unknown

    def _inference(self, env: Mapping[str, int]) -> MemoryAnalysisResult:
        components, graph_persistent, laid_out = self._persistent(env)
        transient = peak(self.graph, env, self.spans)
        work, estimates, unknown = self._workspaces(env)
        with_work = peak(self.graph, env, self.spans, work)
        activations = ex.evaluate(
            ex.total(self.graph.objects[i].nbytes for i in with_work.live), env
        )
        at_work = work.get(with_work.step, 0)
        formula = peak_expr(self.graph, self.spans)
        components.append(
            MemoryComponent(
                "Peak activations",
                Category.ACTIVATION,
                transient.nbytes,
                Confidence.EXACT,
                _formula(formula),
                "inputs and outputs included",
            )
        )
        runtime, runtime_total, runtime_unknown = self._runtime(False)
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
        step = self.graph.steps[transient.step] if self.graph.steps else None
        where = (
            f"step {transient.step}: {step.label} in `{step.scope or 'the root'}`"
            if step
            else "start"
        )
        plan = plan_buffers(self.graph, env, self.spans, self.backend.alignment())
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

    def _training(self, env: Mapping[str, int]) -> MemoryAnalysisResult:
        config = self.config.training
        assert config is not None
        steps = timeline(self.graph, env, config, self.backend, self.arrays)
        at, total, parts = steps.peak()
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
        runtime, runtime_total, runtime_unknown = self._runtime(True)
        components.extend(runtime)
        phase = (
            "the optimizer step"
            if at == steps.length - 1
            else ("the forward pass" if at < len(self.graph.steps) else "the backward pass")
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
        return (
            f"lowering `{self.backend.name}`: library calls run what the plan selected for "
            f"`{self.config.numerics}` numerics",
            "reshape is a view of contiguous storage and a copy otherwise; permute, "
            "broadcast and slice are views",
            "scalars occupy no memory",
        )

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


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)

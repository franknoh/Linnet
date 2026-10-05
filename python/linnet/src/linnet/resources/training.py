"""Training memory: a training step as a timeline of the forward pass, the
backward pass and the optimizer step, with every block of memory an
interval on it.

The forward steps come from the traced entry (a loss). Autograd keeps some
of each step's inputs and outputs for its backward (`saved`); a kept tensor
lives until that backward runs, which is what makes training activations
larger than inference's. A gradient of an activation lives from the
backward of its last reader to the backward of its producer; a parameter's
gradient from its first backward to the optimizer step. Activation
checkpointing (`CheckpointPolicy`) runs each checkpointed region's forward
without keeping anything, then runs it again just before its backward, so
only one region's saved tensors are live at a time.

Optimizer memory comes from an `OptimizerModel` (SGD keeps nothing, SGD
with momentum one tensor per parameter, Adam and AdamW two), master weights
and gradient dtypes from the `TrainingConfig`, and sharding (FSDP) divides
parameters, gradients, master weights and optimizer states across devices
and adds the layer being gathered.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from . import expr as ex
from .backends import BackendResourceModel, Estimate
from .graph import Category, Confidence, Step, TensorGraph, sweep, weakest
from .storage import is_float, storage


@dataclass(frozen=True, slots=True)
class OptimizerModel:
    """The state an optimizer keeps for each trainable parameter: `states`
    tensors of its size, in `state_dtype` (None: the dtype of the tensors it
    updates, the master weights when there are any). `fused` steps update
    in place; others take transient memory nothing here models."""

    name: str
    states: int
    state_dtype: str | None = None
    fused: bool = True


OPTIMIZERS: dict[str, OptimizerModel] = {
    "sgd": OptimizerModel("sgd", 0),
    "sgd-momentum": OptimizerModel("sgd-momentum", 1),
    "adam": OptimizerModel("adam", 2),
    "adamw": OptimizerModel("adamw", 2),
}


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Which parts of the forward pass are recomputed in backward.

    `none` keeps everything; `blocks` checkpoints each element of every
    block array (each decoder layer, say); `regions` checkpoints each
    instance of the block paths `patterns` matches, such as `layers.*` or
    `layers.*.mlp` (fnmatch over block paths)."""

    kind: Literal["none", "blocks", "regions"] = "none"
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """How a model is trained, as far as memory goes.

    `gradient_dtype` defaults to the dtype the optimizer updates: the master
    weights' when `master_dtype` is set (f32 masters of bf16 weights), the
    parameters' otherwise. `trainable` are fnmatch patterns over parameter
    paths. `shards` divides parameters, gradients, master weights and
    optimizer states across that many devices (FSDP)."""

    optimizer: OptimizerModel = field(default_factory=lambda: OPTIMIZERS["adamw"])
    gradient_dtype: str | None = None
    master_dtype: str | None = None
    trainable: tuple[str, ...] = ("*",)
    checkpoint: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    shards: int = 1


# Operations whose backward needs nothing but the incoming gradient.
_KEEPS_NOTHING = frozenset(
    {"add", "sub", "neg", "cast", "reshape", "permute", "broadcast", "slice", "concat"}
)
# Operations whose backward needs their result.
_KEEPS_OUTPUT = frozenset({"exp", "sqrt", "rsqrt", "tanh"})
# Operations that have no gradient.
_NOT_DIFFERENTIABLE = frozenset(
    {
        "compare",
        "and",
        "or",
        "bitand",
        "bitor",
        "bitxor",
        "shl",
        "shr",
        "not",
        "fill",
        "iota",
    }
)


@dataclass(frozen=True, slots=True)
class Interval:
    start: int
    end: int
    nbytes: int
    category: Category
    confidence: Confidence = Confidence.EXACT


@dataclass(frozen=True, slots=True)
class TrainingTimeline:
    """Every transient block of a training step on one timeline, the
    persistent memory, and what checkpointing costs in recomputation."""

    intervals: tuple[Interval, ...]
    length: int
    persistent: Mapping[Category, int]
    persistent_confidence: Mapping[Category, Confidence]
    forward_flops: int
    recomputed_flops: int
    unknown: tuple[str, ...]

    def peak(self) -> tuple[int, int, dict[Category, int]]:
        """The step of the timeline with the most memory, its bytes, and
        their categories (persistent memory included)."""
        delta = [0] * (self.length + 1)
        for item in self.intervals:
            delta[item.start] += item.nbytes
            delta[item.end + 1] -= item.nbytes
        live = sweep(delta[: self.length])
        at = max(range(self.length), key=live.__getitem__)
        parts: dict[Category, int] = dict(self.persistent)
        for item in self.intervals:
            if item.start <= at <= item.end:
                parts[item.category] = parts.get(item.category, 0) + item.nbytes
        return at, live[at] + sum(self.persistent.values()), parts


def _trainable(path: str | None, patterns: Sequence[str]) -> bool:
    return path is not None and any(fnmatch.fnmatchcase(path, p) for p in patterns)


def _regions(
    steps: Sequence[Step], policy: CheckpointPolicy, arrays: Sequence[str]
) -> list[tuple[int, int]]:
    """The checkpointed regions as inclusive step ranges."""
    if policy.kind == "none":
        return []
    patterns = list(policy.patterns) if policy.kind == "regions" else [f"{a}.*" for a in arrays]

    def key(scope: str) -> str | None:
        parts = scope.split(".") if scope else []
        for k in range(1, len(parts) + 1):
            prefix = ".".join(parts[:k])
            if any(fnmatch.fnmatchcase(prefix, p) for p in patterns):
                return prefix
        return None

    regions: list[tuple[int, int]] = []
    current: str | None = None
    start = 0
    for step in steps:
        k = key(step.scope)
        if k != current:
            if current is not None:
                regions.append((start, step.index - 1))
            current, start = k, step.index
    if current is not None and steps:
        regions.append((start, steps[-1].index))
    return regions


def timeline(
    graph: TensorGraph,
    env: Mapping[str, int],
    config: TrainingConfig,
    backend: BackendResourceModel,
    arrays: Sequence[str] = (),
) -> TrainingTimeline:
    """The training step of `graph` (an entry returning a loss) as intervals.

    `arrays` are the block arrays of the hierarchy (`layers`), which the
    `blocks` checkpoint policy and sharding treat as units."""
    objects = graph.objects
    steps = graph.steps
    n = len(steps)
    unknown: list[str] = []

    def nbytes(id: int) -> int:
        return ex.evaluate(objects[id].nbytes, env)

    # ---- which tensors carry gradients
    grad: dict[int, bool] = {}
    for obj in objects:
        owner = objects[obj.storage]
        grad[obj.id] = (
            owner.category == Category.PARAMETER and _trainable(owner.path, config.trainable)
        ) and is_float(obj.dtype)
    differentiable: list[bool] = []
    for step in steps:
        flows = step.kind not in _NOT_DIFFERENTIABLE and any(
            grad.get(i, False) for i in step.inputs
        )
        differentiable.append(flows)
        for out in step.outputs:
            grad[out] = flows and is_float(objects[out].dtype)

    # ---- the schedule: forward, then backward, recomputing each region first
    regions = _regions(steps, config.checkpoint, arrays)
    region_of: dict[int, int] = {}
    for r, (a, b) in enumerate(regions):
        for k in range(a, b + 1):
            region_of[k] = r
    t_back = [0] * n
    t_redo = [0] * n
    time = n
    k = n - 1
    while k >= 0:
        r = region_of.get(k)
        if r is not None:
            a, b = regions[r]
            for j in range(a, b + 1):
                t_redo[j] = time
                time += 1
            for j in range(b, a - 1, -1):
                t_back[j] = time
                time += 1
            k = a - 1
        else:
            t_back[k] = time
            time += 1
            k -= 1
    step_time = time  # the optimizer step
    length = step_time + 1

    # ---- what each step keeps for its backward
    saved_by: dict[int, list[int]] = {}
    intervals: list[Interval] = []
    for step in steps:
        if not differentiable[step.index]:
            continue
        kept = _saved(step, graph, env, backend, unknown)
        for id in kept.objects:
            owner = objects[id].storage
            if not objects[owner].persistent:
                saved_by.setdefault(owner, []).append(step.index)
        if kept.extra.nbytes:
            start = t_redo[step.index] if step.index in region_of else step.index
            intervals.append(
                Interval(
                    start,
                    t_back[step.index],
                    kept.extra.nbytes,
                    Category.ACTIVATION,
                    kept.extra.confidence,
                )
            )
        if kept.extra.nbytes is None:
            unknown.append(f"saved tensors of `{step.implementation}`")

    # ---- activations
    readers: dict[int, list[int]] = {}
    for step in steps:
        for id in step.inputs:
            readers.setdefault(objects[id].storage, []).append(step.index)
    outputs = {objects[i].storage for i in graph.outputs}
    inputs = {objects[i].storage for i in graph.inputs}
    for obj in objects:
        if not obj.owns_storage or obj.persistent:
            continue
        size = nbytes(obj.id)
        if obj.id in inputs:
            intervals.append(Interval(0, length - 1, size, Category.INPUT))
            continue
        if obj.producer is None:
            continue
        p = obj.producer
        uses = readers.get(obj.id, [])
        keepers = saved_by.get(obj.id, [])
        if obj.id in outputs:
            intervals.append(Interval(p, length - 1, size, Category.ACTIVATION))
            continue
        if p in region_of:
            r = region_of[p]
            inside = [u for u in uses if region_of.get(u) == r]
            # The first, gradient-free pass through the region; a later
            # checkpointed region that reads it recomputes from it.
            end = max(
                [
                    p,
                    *uses,
                    *(t_back[s] for s in keepers if region_of.get(s) != r),
                    *(t_back[u] for u in uses if u in region_of and region_of[u] != r),
                ]
            )
            intervals.append(Interval(p, end, size, Category.ACTIVATION))
            # The recomputation just before the region's backward.
            redo_end = max(
                [
                    t_redo[p],
                    *(t_redo[u] for u in inside),
                    *(t_back[s] for s in keepers if region_of.get(s) == r),
                ]
            )
            if inside or keepers:
                intervals.append(Interval(t_redo[p], redo_end, size, Category.ACTIVATION))
            continue
        end = max([p, *uses, *(t_back[s] for s in keepers)])
        # A checkpointed region recomputes from its inputs.
        end = max([end, *(t_back[u] for u in uses if u in region_of)])
        intervals.append(Interval(p, end, size, Category.ACTIVATION))

    # ---- workspaces, in the forward pass, the recomputation and backward
    for step in steps:
        work = backend.workspace(step, graph, env)
        if work.nbytes is None:
            if step.implementation:
                unknown.append(f"workspace of `{step.implementation}`")
            continue
        if not work.nbytes:
            continue
        times = [step.index]
        if step.index in region_of:
            times.append(t_redo[step.index])
        if differentiable[step.index]:
            times.append(t_back[step.index])
        for t in times:
            intervals.append(Interval(t, t, work.nbytes, Category.WORKSPACE, work.confidence))

    # ---- gradients of activations, and of parameters
    for obj in objects:
        if not obj.owns_storage or not grad.get(obj.id, False):
            continue
        consumers = [u for u in readers.get(obj.id, []) if differentiable[u]]
        if not consumers:
            continue
        start = min(t_back[u] for u in consumers)
        if obj.persistent:
            continue
        if obj.producer is None:
            continue
        intervals.append(Interval(start, t_back[obj.producer], nbytes(obj.id), Category.TEMPORARY))

    persistent, confidence, parameter_grads = _persistent(
        graph, env, config, readers, differentiable, t_back
    )
    for start, size in parameter_grads:
        intervals.append(Interval(start, step_time, size, Category.GRADIENT))
    if not config.optimizer.fused:
        unknown.append(f"the {config.optimizer.name} step's temporaries (not fused)")
    if config.shards > 1:
        intervals.extend(_gathered(graph, env, config, arrays, t_back))

    forward = sum(ex.evaluate(s.flops, env) for s in steps)
    redone = sum(ex.evaluate(steps[k].flops, env) for k in region_of)
    return TrainingTimeline(
        intervals=tuple(intervals),
        length=length,
        persistent=persistent,
        persistent_confidence=confidence,
        forward_flops=forward,
        recomputed_flops=redone,
        unknown=tuple(dict.fromkeys(unknown)),
    )


def _saved(
    step: Step,
    graph: TensorGraph,
    env: Mapping[str, int],
    backend: BackendResourceModel,
    unknown: list[str],
) -> _Kept:
    if step.implementation is not None:
        kept = backend.saved(step, graph, env, True)
        if kept is not None:
            return _Kept(kept.objects, kept.extra)
        unknown.append(
            f"what `{step.implementation}` keeps for backward (all inputs and outputs assumed)"
        )
        return _Kept((*step.inputs, *step.outputs), Estimate(0, Confidence.ESTIMATED))
    if step.kind in _KEEPS_NOTHING:
        return _Kept(())
    if step.kind in _KEEPS_OUTPUT:
        return _Kept(step.outputs)
    if step.kind == "reduce":
        return _Kept((*step.inputs, *step.outputs))
    return _Kept(step.inputs)


@dataclass(frozen=True, slots=True)
class _Kept:
    objects: tuple[int, ...]
    extra: Estimate = field(default_factory=lambda: Estimate(0, Confidence.EXACT))


def _persistent(
    graph: TensorGraph,
    env: Mapping[str, int],
    config: TrainingConfig,
    readers: Mapping[int, list[int]],
    differentiable: Sequence[bool],
    t_back: Sequence[int],
) -> tuple[dict[Category, int], dict[Category, Confidence], list[tuple[int, int]]]:
    """Parameters, buffers and state; master weights and optimizer states;
    and each trainable parameter's gradient as (first backward, bytes)."""
    shards = max(config.shards, 1)
    totals: dict[Category, int] = {}
    confidence: dict[Category, Confidence] = {}
    grads: list[tuple[int, int]] = []

    def put(category: Category, size: int, how: Confidence = Confidence.EXACT) -> None:
        totals[category] = totals.get(category, 0) + size
        confidence[category] = weakest([confidence.get(category, Confidence.EXACT), how])

    for obj in graph.objects:
        if not obj.persistent or not obj.owns_storage:
            continue
        size = ex.evaluate(obj.nbytes, env)
        if obj.category != Category.PARAMETER:
            put(obj.category, size)
            continue
        put(Category.PARAMETER, -(-size // shards))
        if not _trainable(obj.path, config.trainable) or not is_float(obj.dtype):
            continue
        elements = ex.evaluate(ex.product(obj.shape), env)
        share = -(-elements // shards)
        updated = config.master_dtype or obj.dtype
        if config.master_dtype is not None and config.master_dtype != obj.dtype:
            put(Category.MASTER, share * storage(config.master_dtype).element_bytes)
        state_dtype = config.optimizer.state_dtype or updated
        if config.optimizer.states:
            put(
                Category.OPTIMIZER,
                share * config.optimizer.states * storage(state_dtype).element_bytes,
            )
        gradient_dtype = config.gradient_dtype or updated
        consumers = [u for u in readers.get(obj.id, []) if differentiable[u]]
        if consumers:
            grads.append(
                (min(t_back[u] for u in consumers), share * storage(gradient_dtype).element_bytes)
            )
    return totals, confidence, grads


def _gathered(
    graph: TensorGraph,
    env: Mapping[str, int],
    config: TrainingConfig,
    arrays: Sequence[str],
    t_back: Sequence[int],
) -> list[Interval]:
    """Sharded parameters gathered whole where they run: each array
    element's while its steps run forward and backward, the rest throughout
    the step."""
    shards = max(config.shards, 1)
    units: dict[str, int] = {}
    loose = 0
    for obj in graph.objects:
        if not obj.persistent or not obj.owns_storage or obj.category != Category.PARAMETER:
            continue
        size = ex.evaluate(obj.nbytes, env)
        unit = _unit(obj.path or "", arrays)
        if unit is None:
            loose += size - -(-size // shards)
        else:
            units[unit] = units.get(unit, 0) + size - -(-size // shards)
    out: list[Interval] = []
    length = max(t_back, default=0) + 2
    if loose:
        out.append(Interval(0, length - 1, loose, Category.COMMUNICATION, Confidence.MODELED))
    spans: dict[str, list[int]] = {}
    for step in graph.steps:
        unit = _unit(step.scope, arrays)
        if unit is not None:
            spans.setdefault(unit, []).append(step.index)
    for unit, indices in spans.items():
        size = units.get(unit, 0)
        if not size:
            continue
        out.append(
            Interval(min(indices), max(indices), size, Category.COMMUNICATION, Confidence.MODELED)
        )
        back = [t_back[i] for i in indices]
        out.append(Interval(min(back), max(back), size, Category.COMMUNICATION, Confidence.MODELED))
    return out


def _unit(path: str, arrays: Sequence[str]) -> str | None:
    for array in arrays:
        if path.startswith(f"{array}."):
            index = path[len(array) + 1 :].split(".", 1)[0]
            return f"{array}.{index}"
    return None

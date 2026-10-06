"""How long a step takes on a device, operation by operation.

Each step of the graph is costed at the device's measured rates
(`python -m linnet.resources.calibrate`):

- matrix products at the rate of their weight's size, or reading their
  operands at memory bandwidth, whichever is longer;
- fused attention at its kernel's rate (causal, or under a mask), and one
  query under a mask as the generated code runs it: two products around a
  softmax, bound by memory;
- collectives at the link's bandwidth plus a latency each;
- everything else that makes a tensor reads its inputs and writes its
  outputs at the bandwidth an elementwise kernel reaches, lower when an
  operand is broadcast;
- views take no device time.

The calls are the generated code's: what it computes once per shape is left
out, an inference entry's products that read the same input run as one
(their weights joined), and a causal mask runs as `is_causal`.

The generated code runs eagerly: the host issues one call after another,
`dispatch` seconds each, and the device runs each kernel once it is issued
and the previous one is done. A step with small kernels waits on the host;
a step with large ones keeps the host ahead. Training adds the backward pass
(products twice, attention at its backward rate, other kernels reading the
gradient and writing one) and one optimizer pass a step. A pipeline adds its
bubble, `(stages - 1) / (micro-batches + stages - 1)` of the step.

What is not modeled: kernels below their measured rate at unusual shapes,
the host's own variance, overlap of communication with computation, and
`torch.compile`'s fusion (compiled models drop the host time only).
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import expr as ex
from .graph import Category, TensorGraph

MIB = 1 << 20


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """A device's rates, as `python -m linnet.resources.calibrate` measures
    them.

    - `flops`: matrix products' FLOP/s by compute dtype; `products`: the
      same by the weight's size (`(n, FLOP/s)` for 8192 rows times an `n`
      by `n` weight).
    - `bandwidth`: a large copy's bytes read plus written per second;
      `elementwise` and `broadcast`: the same for an operation on two
      same-shape tensors and for one with a broadcast operand.
    - `attention`: fused attention's FLOP/s by kind, head width and, past
      bf16, dtype (`causal/64`, `causal/64/f32`): `causal` counts the half of the scores its kernel
      computes, `masked` all of them, and `_backward` rates the same count
      over the backward pass's time. `decode`: the bytes of cache one query
      per row reads per second under a mask, as the generated code runs it.
    - `dispatch`: the host's seconds per eager call; `kernel`: a small
      kernel's device seconds, issued eagerly.
    - `link`: an all-reduce's bus bandwidth; `latency`: a small one's
      seconds; `reduce`: one token's as the generated code calls it with the
      next call reading it, host included.
    - `memory`: what the driver reports; `source`: where the figures come
      from."""

    name: str
    memory: int
    flops: Mapping[str, float]
    bandwidth: float
    link: float
    latency: float
    attention: Mapping[str, float] = field(default_factory=dict[str, float])
    decode: float = 0.0
    elementwise: float = 0.0
    broadcast: float = 0.0
    products: Mapping[str, tuple[tuple[float, float], ...]] = field(
        default_factory=dict[str, tuple[tuple[float, float], ...]]
    )
    dispatch: float = 0.0
    kernel: float = 0.0
    reduce: float = 0.0
    source: str = "data sheet"

    def peak(self, dtype: str) -> float:
        return self.flops.get(dtype, self.flops["f32"])

    def product(self, dtype: str, size: int) -> float:
        """A matrix product's FLOP/s whose weight's size is `size`: between
        the measured sizes on a log scale, the peak past them."""
        peak = self.peak(dtype)
        measured = self.products.get("bf16" if dtype == "f16" else dtype, ())
        if not measured:
            return peak
        points = [*sorted(measured), (2 * max(n for n, _ in measured), peak)]
        if size <= points[0][0]:
            return points[0][1] * size / points[0][0]
        for (low, slow), (high, fast) in itertools.pairwise(points):
            if size <= high:
                share = math.log(size / low) / math.log(high / low)
                return slow + share * (fast - slow)
        return peak

    def attention_rate(self, kind: str, width: int, dtype: str = "bf16") -> float:
        """Fused attention's rate in `dtype` at the measured head width
        nearest `width`."""
        suffix = "" if dtype in ("bf16", "f16") else f"/{dtype}"
        widths = [
            int(parts[1])
            for parts in (key.split("/") for key in self.attention)
            if len(parts) > 1 and parts[0] == kind and "/".join(["", *parts[2:]]) == suffix
        ]
        if widths:
            nearest = min(widths, key=lambda w: abs(math.log(w / max(width, 1))))
            return self.attention[f"{kind}/{nearest}{suffix}"]
        return self.attention.get(kind) or self.peak(dtype)

    @classmethod
    def load(cls, path: str | Path) -> DeviceSpec:
        """A profile `python -m linnet.resources.calibrate` wrote."""
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(data["name"]),
            memory=int(data["memory"]),
            flops={k: float(v) for k, v in data["flops"].items()},
            bandwidth=float(data["bandwidth"]),
            link=float(data.get("link") or 0.0) or _H100.link,
            latency=float(data.get("latency") or 0.0) or _H100.latency,
            attention={k: float(v) for k, v in data.get("attention", {}).items()},
            decode=float(data.get("decode", 0.0)),
            elementwise=float(data.get("elementwise", 0.0)),
            broadcast=float(data.get("broadcast", 0.0)),
            products={
                dtype: tuple((float(n), float(r)) for n, r in points)
                for dtype, points in data.get("products", {}).items()
            },
            dispatch=float(data.get("dispatch", 0.0)),
            kernel=float(data.get("kernel", 0.0)),
            reduce=float(data.get("reduce") or 0.0) or _H100.reduce,
            source=str(data.get("source", "measured")),
        )


# Measured on an H100 SXM with PyTorch 2.14.1 and CUDA 13 by
# `python -m linnet.resources.calibrate`; link, latency and reduce between
# two of them under torchrun (NVLink, NCCL 2.30).
_H100 = DeviceSpec(
    "h100-80gb",
    85017493504,
    {"bf16": 804.8e12, "f16": 804.8e12, "f32": 50.85e12},
    3.015e12,
    302.5e9,
    18.4e-6,
    attention={
        "causal/64": 377.7e12,
        "causal_backward/64": 129.6e12,
        "masked/64": 237.3e12,
        "masked_backward/64": 85.2e12,
        "causal/128": 550.2e12,
        "causal_backward/128": 169.5e12,
        "masked/128": 320.6e12,
        "masked_backward/128": 114.7e12,
        "causal/64/f32": 25.6e12,
        "causal_backward/64/f32": 10.75e12,
        "masked/64/f32": 27.8e12,
        "masked_backward/64/f32": 12.9e12,
        "causal/128/f32": 38.9e12,
        "causal_backward/128/f32": 12.1e12,
        "masked/128/f32": 39.0e12,
        "masked_backward/128/f32": 13.8e12,
    },
    decode=1.947e12,
    elementwise=3.008e12,
    broadcast=1.562e12,
    products={
        "bf16": (
            (128, 26.1e12),
            (256, 103.6e12),
            (512, 361.2e12),
            (1024, 552.1e12),
            (2048, 700.5e12),
            (4096, 784.3e12),
        ),
        "f32": (
            (256, 30.6e12),
            (512, 45.3e12),
            (1024, 49.4e12),
            (2048, 50.8e12),
            (4096, 50.6e12),
        ),
    },
    dispatch=6.32e-6,
    kernel=2.11e-6,
    reduce=19.8e-6,
    source="measured: H100 SXM, PyTorch 2.14.1+cu130",
)


def _from_sheet(
    name: str, memory: int, bf16: float, f32: float, bandwidth: float, link: float
) -> DeviceSpec:
    """A device known by its data sheet: the H100's measured rates scaled by
    the ratio of the two data sheets, and the H100 host's dispatch."""
    compute = bf16 / 989.4e12
    memory_ratio = bandwidth / 3.35e12
    return DeviceSpec(
        name,
        memory,
        {
            "bf16": _H100.flops["bf16"] * compute,
            "f16": _H100.flops["f16"] * compute,
            "f32": _H100.flops["f32"] * f32 / 66.9e12,
        },
        _H100.bandwidth * memory_ratio,
        _H100.link * link / 450e9,
        _H100.latency,
        attention={
            k: v * (f32 / 66.9e12 if k.endswith("/f32") else compute)
            for k, v in _H100.attention.items()
        },
        decode=_H100.decode * memory_ratio,
        elementwise=_H100.elementwise * memory_ratio,
        broadcast=_H100.broadcast * memory_ratio,
        products={
            dtype: tuple(
                (n, r * (compute if dtype == "bf16" else f32 / 66.9e12)) for n, r in points
            )
            for dtype, points in _H100.products.items()
        },
        dispatch=_H100.dispatch,
        kernel=_H100.kernel,
        reduce=_H100.reduce,
        source="data sheet, at the H100's measured share of its own",
    )


DEVICES: dict[str, DeviceSpec] = {
    d.name: d
    for d in (
        _H100,
        _from_sheet("h200", 143771 * MIB, 989.4e12, 66.9e12, 4.8e12, 450e9),
        _from_sheet("a100-80gb", 81920 * MIB, 312e12, 19.5e12, 2.039e12, 300e9),
        _from_sheet("a100-40gb", 40960 * MIB, 312e12, 19.5e12, 1.555e12, 300e9),
        _from_sheet("l4", 23034 * MIB, 121e12, 30.3e12, 300e9, 32e9),
    )
}


def device(name: str) -> DeviceSpec:
    """A device by its name in `DEVICES`, or a calibrated profile's path."""
    if name in DEVICES:
        return DEVICES[name]
    if Path(name).is_file():
        return DeviceSpec.load(name)
    known = ", ".join(sorted(DEVICES))
    raise ValueError(f"unknown device `{name}`: one of {known}, or a calibrated profile's path")


@dataclass(frozen=True, slots=True)
class Throughput:
    """A step's predicted time and what sets it.

    `tokens` are the positions one step processes over every replica;
    `compute`, `memory` and `communication` are the slowest device's
    kernel seconds limited by each, `host` the seconds it waits on the
    host to issue them, and `bubble` the share of a pipelined step its
    stages sit idle."""

    tokens: int
    seconds: float
    compute: float
    memory: float
    communication: float
    bubble: float
    limit: Literal["compute", "memory", "communication", "host"]
    host: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": self.tokens,
            "seconds": self.seconds,
            "tokens_per_second": self.tokens_per_second,
            "compute_seconds": self.compute,
            "memory_seconds": self.memory,
            "communication_seconds": self.communication,
            "host_seconds": self.host,
            "bubble": self.bubble,
            "limit": self.limit,
        }


_PRODUCTS = ("torch.nn.functional.linear", "torch.matmul")
_WRITES = ("torch.Tensor.index_copy", "torch.Tensor.index_put", "torch.Tensor.index_put(tokens)")
_COLLECTIVES = ("torch.distributed.all_reduce", "torch.distributed.all_gather")
_LOSSES = ("linnet.linear_cross_entropy", "linnet.linear_token_log_probs")
# The chunked loss's passes over its f32 logits, bytes per logit: making
# them (bf16, then f32) and their log-sum-exp; with gradients also the
# softmax, its scaling and its cast back.
_LOSS_BYTES = 16
_LOSS_GRAD_BYTES = 46


@dataclass(slots=True)
class _Op:
    """Device work the host issues in `calls` calls (and `host` seconds
    more): `seconds` on the device, and which of compute, memory or
    communication sets it."""

    calls: int
    seconds: float
    by: Literal["compute", "memory", "communication"]
    host: float = 0.0


@dataclass(slots=True)
class _Clock:
    """The host issuing calls and the device running them, in order."""

    dispatch: float
    host: float = 0.0
    device: float = 0.0
    spent: dict[str, float] = field(
        default_factory=lambda: {"compute": 0.0, "memory": 0.0, "communication": 0.0}
    )

    def run(self, op: _Op) -> None:
        self.host += op.calls * self.dispatch + op.host
        self.device = max(self.device, self.host) + op.seconds
        self.spent[op.by] += op.seconds


def _bytes(graph: TensorGraph, id: int, env: Mapping[str, int]) -> int:
    return ex.evaluate(graph.objects[id].nbytes, env)


def _numel(graph: TensorGraph, id: int, env: Mapping[str, int]) -> int:
    return math.prod(ex.evaluate(d, env) for d in graph.objects[id].shape)


def _dims(graph: TensorGraph, id: int, env: Mapping[str, int]) -> list[int]:
    return [ex.evaluate(d, env) for d in graph.objects[id].shape]


def _causal(graph: TensorGraph, mask: int) -> bool:
    """A square mask `causal_mask` made, which the generated code passes as
    `is_causal`."""
    owner = graph.objects[graph.objects[mask].storage]
    made = graph.steps[owner.producer] if owner.producer is not None else None
    shape = graph.objects[mask].shape
    return made is not None and made.label == "causal_mask" and shape[-1] == shape[-2]


def _hoisted(graph: TensorGraph) -> set[int]:
    """The steps the generated code runs once per shape (`constants`), not
    per call: those reading neither an input nor a persistent object."""
    varying = set(graph.inputs) | {o.id for o in graph.objects if o.persistent}
    hoisted: set[int] = set()
    for step in graph.steps:
        if any(i in varying or graph.objects[i].storage in varying for i in step.inputs):
            varying.update(step.outputs)
        else:
            hoisted.add(step.index)
    return hoisted


def _fused(graph: TensorGraph) -> dict[int, list[int]]:
    """Products the generated inference code runs as one: those reading the
    same input, each by a weight of its own, whose weights `prepare` joins
    (the query, key and value projections, say). Keyed by the first."""
    groups: dict[int, list[int]] = {}
    for step in graph.steps:
        if step.implementation != "torch.nn.functional.linear" or len(step.inputs) != 2:
            continue
        weight = graph.objects[step.inputs[1]]
        if weight.category == Category.PARAMETER and weight.persistent:
            groups.setdefault(step.inputs[0], []).append(step.index)
    return {members[0]: members for members in groups.values() if len(members) > 1}


def _costs(
    graph: TensorGraph,
    env: Mapping[str, int],
    device: DeviceSpec,
    processes: int,
    training: bool,
) -> tuple[list[_Op], list[_Op]]:
    """The device work of each call the generated code makes for a step,
    forward, and in training backward."""
    forward: list[_Op] = []
    backward: list[_Op] = []
    hoisted = _hoisted(graph)
    # Split models keep their weights apart (`--no-fuse`).
    fused = {} if training or processes > 1 else _fused(graph)
    joined = {i for members in fused.values() for i in members[1:]}

    stream = device.elementwise or device.bandwidth
    spread = device.broadcast or stream

    def memory(nbytes: float, calls: int = 1, rate: float = stream) -> _Op:
        return _Op(calls, max(nbytes / rate, calls * device.kernel), "memory")

    def bounded(flops: float, rate: float, nbytes: float, calls: int = 1) -> _Op:
        compute, moved = flops / rate, nbytes / device.bandwidth
        floor = calls * device.kernel
        if compute >= moved:
            return _Op(calls, max(compute, floor), "compute")
        return _Op(calls, max(moved, floor), "memory")

    for step in graph.steps:
        if step.index in hoisted or step.index in joined:
            continue
        implementation = (step.implementation or "").split("(")[0]
        full = step.implementation or ""
        owned = [o for o in step.outputs if graph.objects[o].owns_storage]
        read = sum(_bytes(graph, i, env) for i in dict.fromkeys(step.inputs))
        written = sum(_bytes(graph, o, env) for o in owned)
        flops = float(ex.evaluate(step.flops, env))
        operands = len(step.inputs)
        if implementation in _PRODUCTS and operands >= 2:
            members = fused.get(step.index, [step.index])
            if len(members) > 1:
                steps = [graph.steps[i] for i in members]
                flops = sum(float(ex.evaluate(s.flops, env)) for s in steps)
                read = _bytes(graph, step.inputs[0], env) + sum(
                    _bytes(graph, s.inputs[1], env) for s in steps
                )
                written = sum(_bytes(graph, o, env) for s in steps for o in s.outputs)
            weight = _dims(graph, step.inputs[1], env)
            outputs = sum(_dims(graph, graph.steps[i].inputs[1], env)[-2] for i in members)
            width = weight[-1] if weight else 1
            size = math.isqrt(max(outputs * width, 1)) if len(weight) >= 2 else 1
            dtype = graph.objects[step.inputs[0]].dtype
            # A joined product is one call, then a slice for each part.
            calls = 1 + len(members) if len(members) > 1 else 1
            op = bounded(flops, device.product(dtype, size), read + written, calls)
            forward.append(op)
            backward.append(_Op(2, 2 * op.seconds, op.by))
        elif implementation == "torch.nn.functional.scaled_dot_product_attention":
            inputs = step.inputs
            query = _dims(graph, inputs[0], env)
            masked = operands > 3 and graph.objects[inputs[3]].dtype == "bool"
            causal = masked and _causal(graph, inputs[3])
            cache = sum(_bytes(graph, i, env) for i in inputs[1:3])
            if masked and not causal and query[-2] == 1:
                # One query under a mask: two products around an f32
                # softmax, twelve calls.
                seconds = cache / (device.decode or device.bandwidth)
                op = _Op(12, max(seconds, 8 * device.kernel), "memory")
                forward.append(op)
                backward.append(_Op(op.calls, 2 * op.seconds, "memory"))
                continue
            kind = "masked" if masked and not causal else "causal"
            work = flops / 2 if causal else flops
            # Exact numerics run it in f32.
            dtype = graph.objects[inputs[0]].dtype if "(input dtype)" in full else "f32"
            rate = device.attention_rate(kind, query[-1], dtype)
            backward_rate = device.attention_rate(f"{kind}_backward", query[-1], dtype)
            moved = (read + written) / device.bandwidth
            compute = work / rate
            forward.append(
                _Op(
                    1,
                    max(compute, moved, device.kernel),
                    "compute" if compute >= moved else "memory",
                )
            )
            backward.append(_Op(1, max(work / backward_rate, 2 * moved, device.kernel), "compute"))
        elif implementation in _COLLECTIVES:
            if processes <= 1:
                continue  # the generated helper returns its argument
            size = _bytes(graph, step.outputs[0], env)
            reduce = implementation == "torch.distributed.all_reduce"
            share = (processes - 1) / processes
            carried = (2 if reduce else 1) * share
            seconds = device.latency + carried * size / device.link
            if reduce and device.reduce:
                # The generated code's sum costs the host more than a call:
                # `reduce` is measured with the call that reads it.
                host = max(device.reduce - 2 * device.dispatch, 0.0)
                op = _Op(1, seconds, "communication", host)
            else:
                op = _Op(1, seconds, "communication")
            forward.append(op)
            backward.append(op)
        elif implementation in _LOSSES:
            hidden, weight = _dims(graph, step.inputs[0], env), _dims(graph, step.inputs[1], env)
            logits = math.prod(hidden[:-1]) * weight[0]
            rate = device.product(
                graph.objects[step.inputs[0]].dtype, math.isqrt(weight[0] * weight[1])
            )
            gradients = implementation == "linnet.linear_cross_entropy" and training
            products = 3 * flops if gradients else flops
            per_logit = _LOSS_GRAD_BYTES if gradients else _LOSS_BYTES
            forward.append(bounded(products, rate, logits * per_logit + read))
            if training and not gradients:
                # Log-probabilities keep their log-sum-exp and recompute the
                # logits for the gradients.
                backward.append(bounded(3 * flops, rate, logits * _LOSS_GRAD_BYTES + read))
            elif training:
                backward.append(memory(2 * read))
        elif implementation == "torch.nn.functional.embedding":
            forward.append(memory(2 * written))
            # A dense gradient the table's size, rows added into it.
            backward.append(memory(_bytes(graph, step.inputs[1], env) + 3 * written, 2))
        elif implementation in _WRITES:
            forward.append(memory(2 * _bytes(graph, step.inputs[1], env)))
        elif implementation in ("torch.rms_norm", "torch.nn.functional.layer_norm"):
            # The fast normalizations are two calls: the normalization, then
            # times the (broadcast) weight.
            normalized = memory(2 * written)
            scaled = memory(2 * written, rate=spread)
            op = _Op(2, normalized.seconds + scaled.seconds, "memory")
            forward.append(op)
            backward.append(_Op(op.calls, 2 * op.seconds, "memory"))
        elif owned:
            largest = max(_numel(graph, o, env) for o in owned)
            broadcast = any(0 < _numel(graph, i, env) < largest for i in step.inputs)
            op = memory(read + written, rate=spread if broadcast else stream)
            forward.append(op)
            backward.append(_Op(op.calls, 2 * op.seconds, "memory"))
        else:
            forward.append(_Op(1, 0.0, "memory"))  # a view
    return forward, backward


@dataclass(frozen=True, slots=True)
class _Part:
    seconds: float  # one micro-batch through the stage, forward and backward
    host: float
    spent: Mapping[str, float]
    optimizer: float  # once a step


def _part(
    graph: TensorGraph,
    env: Mapping[str, int],
    device: DeviceSpec,
    *,
    training: bool,
    processes: int,
    compiled: bool,
    received: int = 0,
    recomputed: float = 0.0,
    optimizer_states: int = 2,
) -> _Part:
    """One device's share of one (micro-)step. `recomputed` is the share of
    the forward pass checkpointing runs again."""
    forward, backward = _costs(graph, env, device, processes, training)
    clock = _Clock(0.0 if compiled else device.dispatch)
    if received:
        clock.run(_Op(1, device.latency + received / device.link, "communication"))
    for op in forward:
        clock.run(op)
    optimizer = 0.0
    if training:
        if recomputed:
            for op in forward:
                clock.run(_Op(op.calls, op.seconds * recomputed, op.by))
        for op in reversed(backward):
            clock.run(op)
        weights = sum(
            ex.evaluate(o.nbytes, env)
            for o in graph.objects
            if o.category == Category.PARAMETER and o.owns_storage
        )
        # The update reads each weight and its gradient, writes the weight,
        # and reads and writes each state.
        optimizer = (3 + 2 * optimizer_states) * weights / device.bandwidth
    busy = sum(clock.spent.values())
    return _Part(clock.device, max(clock.device - busy, 0.0), dict(clock.spent), optimizer)


def step_time(
    parts: Sequence[tuple[TensorGraph, Mapping[str, int], int, float]],
    device: DeviceSpec,
    *,
    tokens: int,
    training: bool,
    processes: int = 1,
    microbatches: int = 1,
    replicas: int = 1,
    gradient_bytes: int = 0,
    compiled: bool = False,
    optimizer_states: int = 2,
) -> Throughput:
    """The time of a step whose devices run `parts`: one per pipeline stage
    (its graph, the environment of one micro-batch, the bytes it receives
    and the share of its forward pass checkpointing repeats), or one for
    the whole step. `replicas` copies of the layout each take their own
    `tokens`; in training they then sum `gradient_bytes` over the link."""
    stages = [
        _part(
            graph,
            env,
            device,
            training=training,
            processes=processes,
            compiled=compiled,
            received=received,
            recomputed=recomputed,
            optimizer_states=optimizer_states,
        )
        for graph, env, received, recomputed in parts
    ]
    slowest = max(stages, key=lambda p: p.seconds)
    count = microbatches if len(stages) > 1 else 1
    slots = count + len(stages) - 1
    seconds = slots * slowest.seconds + slowest.optimizer
    bubble = (len(stages) - 1) / slots
    totals = {k: v * slots for k, v in slowest.spent.items()}
    totals["memory"] += slowest.optimizer
    totals["host"] = slowest.host * slots
    if training and replicas > 1 and gradient_bytes:
        # Summing gradients overlaps the backward pass; the longer one sets the step.
        summed = 2 * (replicas - 1) / replicas * gradient_bytes / device.link
        if summed > seconds:
            totals["communication"] += summed - seconds
            seconds = summed
    limit: Literal["compute", "memory", "communication", "host"] = max(
        ("compute", "memory", "communication", "host"), key=lambda k: totals[k]
    )  # type: ignore[assignment]
    return Throughput(
        tokens * replicas,
        seconds,
        totals["compute"],
        totals["memory"],
        totals["communication"],
        bubble,
        limit,
        totals["host"],
    )


__all__ = ["DEVICES", "DeviceSpec", "Throughput", "device", "step_time"]

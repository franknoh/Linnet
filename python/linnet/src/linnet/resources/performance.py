"""How fast a configuration can run on a device, as a roofline bound.

A step takes at least as long as its arithmetic at the device's peak rate,
and at least as long as reading what it must read at the device's memory
bandwidth: every weight once, every cache once, and in training the
gradients and optimizer states as the optimizer updates them. Collectives
add their transfer over the link and a latency each; a pipeline adds its
bubble, `(stages - 1) / (micro-batches + stages - 1)` of the step.

These are bounds. Kernels reach a fraction of peak, and the fraction
differs between prefill, decoding and training; nothing here is fitted to a
measurement. The device figures are data-sheet values (dense peak rates).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .. import dtypes
from . import expr as ex
from .graph import Category, TensorGraph

MIB = 1 << 20


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """A device's memory (as the driver reports it), dense peak rates by
    compute dtype (FLOP/s), memory bandwidth and link bandwidth each way to
    another device of the node (bytes/s), and a collective's latency (s)."""

    name: str
    memory: int
    flops: Mapping[str, float]
    bandwidth: float
    link: float
    latency: float

    def peak(self, dtype: str) -> float:
        return self.flops.get(dtype, self.flops["f32"])


DEVICES: dict[str, DeviceSpec] = {
    d.name: d
    for d in (
        DeviceSpec(
            "h100-80gb",
            81559 * MIB,
            {"bf16": 989.4e12, "f16": 989.4e12, "f32": 66.9e12},
            3.35e12,
            450e9,
            10e-6,
        ),
        DeviceSpec(
            "h200",
            143771 * MIB,
            {"bf16": 989.4e12, "f16": 989.4e12, "f32": 66.9e12},
            4.8e12,
            450e9,
            10e-6,
        ),
        DeviceSpec(
            "a100-80gb",
            81920 * MIB,
            {"bf16": 312e12, "f16": 312e12, "f32": 19.5e12},
            2.039e12,
            300e9,
            10e-6,
        ),
        DeviceSpec(
            "a100-40gb",
            40960 * MIB,
            {"bf16": 312e12, "f16": 312e12, "f32": 19.5e12},
            1.555e12,
            300e9,
            10e-6,
        ),
        DeviceSpec(
            "l4",
            23034 * MIB,
            {"bf16": 121e12, "f16": 121e12, "f32": 30.3e12},
            300e9,
            32e9,
            20e-6,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class Throughput:
    """A step's time bound and what sets it.

    `tokens` are the positions one step processes over every replica;
    `compute`, `memory` and `communication` are the slowest device's
    seconds for each (compute and memory overlap, communication does not),
    and `bubble` the share of a pipelined step its stages sit idle."""

    tokens: int
    seconds: float
    compute: float
    memory: float
    communication: float
    bubble: float
    limit: Literal["compute", "memory", "communication"]

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
            "bubble": self.bubble,
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class _Part:
    compute: float
    memory: float
    communication: float
    optimizer: float  # once per step, after every micro-batch


def _part(
    graph: TensorGraph,
    env: Mapping[str, int],
    device: DeviceSpec,
    *,
    training: bool,
    processes: int,
    received: int = 0,
    recomputed: int = 0,
) -> _Part:
    """One device's share of one (micro-)step; `recomputed` is the forward
    arithmetic checkpointing runs again."""
    flops = sum(ex.evaluate(step.flops, env) for step in graph.steps)
    weights = 0
    caches = 0
    by_dtype: dict[str, int] = {}
    for obj in graph.objects:
        if not obj.persistent or not obj.owns_storage:
            continue
        size = ex.evaluate(obj.nbytes, env)
        if obj.category == Category.PARAMETER:
            weights += size
            by_dtype[obj.dtype] = by_dtype.get(obj.dtype, 0) + size
        elif obj.category in (Category.KV_CACHE, Category.STATE, Category.BUFFER):
            caches += size
    compute_dtype = max(by_dtype, key=by_dtype.__getitem__, default="bf16")
    if not dtypes.dtype(compute_dtype).is_float:
        compute_dtype = "bf16"
    reads = weights + caches
    optimizer = 0.0
    if training:
        # Forward, then backward's two products per weight, each reading the
        # weights; once a step, the optimizer reads weights and gradients
        # and reads and writes two states.
        flops = 3 * flops + recomputed
        reads = 3 * weights + caches
        optimizer = 6 * weights / device.bandwidth
    communication = received / device.link if received else 0.0
    if processes > 1:
        share = (processes - 1) / processes
        for step in graph.steps:
            if step.implementation == "torch.distributed.all_reduce":
                size = ex.evaluate(graph.objects[step.outputs[0]].nbytes, env)
                communication += 2 * share * size / device.link + device.latency
            elif step.implementation == "torch.distributed.all_gather":
                size = ex.evaluate(graph.objects[step.outputs[0]].nbytes, env)
                communication += share * size / device.link + device.latency
    return _Part(
        flops / device.peak(compute_dtype), reads / device.bandwidth, communication, optimizer
    )


def step_bound(
    parts: Sequence[tuple[TensorGraph, Mapping[str, int], int, int]],
    device: DeviceSpec,
    *,
    tokens: int,
    training: bool,
    processes: int = 1,
    microbatches: int = 1,
    replicas: int = 1,
    gradient_bytes: int = 0,
) -> Throughput:
    """The bound for a step whose devices run `parts`: one per pipeline
    stage (its graph, the environment of one micro-batch, the bytes it
    receives and the arithmetic checkpointing repeats), or one for the
    whole step. `replicas` copies of the layout
    each take their own `tokens`; in training they then sum `gradient_bytes`
    over the link."""
    stages = [
        _part(
            graph,
            env,
            device,
            training=training,
            processes=processes,
            received=received,
            recomputed=recomputed,
        )
        for graph, env, received, recomputed in parts
    ]
    slowest = max(stages, key=lambda p: max(p.compute, p.memory) + p.communication)
    per_micro = max(slowest.compute, slowest.memory) + slowest.communication
    count = microbatches if len(stages) > 1 else 1
    seconds = (count + len(stages) - 1) * per_micro + slowest.optimizer
    bubble = (len(stages) - 1) / (count + len(stages) - 1)
    communication = slowest.communication * (count + len(stages) - 1)
    if training and replicas > 1 and gradient_bytes:
        # Summing gradients overlaps the backward pass; the longer one sets the step.
        summed = 2 * (replicas - 1) / replicas * gradient_bytes / device.link
        if summed > seconds:
            communication += summed - seconds
            seconds = summed
    totals = {
        "compute": slowest.compute * (count + len(stages) - 1),
        "memory": slowest.memory * (count + len(stages) - 1) + slowest.optimizer,
        "communication": communication,
    }
    limit: Literal["compute", "memory", "communication"] = max(
        ("compute", "memory", "communication"), key=lambda k: totals[k]
    )  # type: ignore[assignment]
    return Throughput(
        tokens * replicas,
        seconds,
        totals["compute"],
        totals["memory"],
        totals["communication"],
        bubble,
        limit,
    )


__all__ = ["DEVICES", "DeviceSpec", "Throughput", "step_bound"]

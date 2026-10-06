"""Backend resource models: what a lowered operation allocates beyond the
tensors the graph shows, and what the runtime holds besides.

The graph knows a native call's inputs and outputs; only the backend knows
the workspace a kernel takes, the tensors its autograd keeps, and the
memory the process holds before any model loads. A model answers those
questions with a confidence (`linnet.resources.graph.Confidence`): a
number it can derive from the implementation is `backend-modeled`, a
typical value for something the runtime decides is `estimated`, and an
implementation it does not know is `unknown`, which is reported as such
and never counted as zero.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from .. import dtypes
from ..parallel import ONE_SHOT_BYTES
from . import expr as ex
from .graph import Category, Confidence, MemoryObject, Step, TensorGraph

MIB = 1 << 20

# When a step runs: in inference, in a training forward pass (gradients
# wanted), or in its backward.
Phase = Literal["inference", "forward", "backward"]


@dataclass(frozen=True, slots=True)
class Estimate:
    """Bytes with how they were obtained; `nbytes` is None when unknown."""

    nbytes: int | None
    confidence: Confidence
    note: str = ""


ZERO = Estimate(0, Confidence.MODELED)


@dataclass(frozen=True, slots=True)
class Saved:
    """What autograd keeps from a step for its backward: some of the step's
    own inputs and outputs (object ids), and extra tensors the kernel makes
    for itself (a softmax's log-sum-exp)."""

    objects: tuple[int, ...]
    extra: Estimate = ZERO


# Point-to-point channels a process opens, and what NCCL 2.30 allocates for
# them outside the allocator on two H100s: a pipeline's first stage sends,
# its last receives, a training step goes both ways, and DTensor's
# all-to-all exchanges.
Channels = Literal["none", "send", "receive", "both", "exchange"]
_CHANNELS = {"send": 421, "receive": 677, "both": 745, "exchange": 440}


@dataclass(frozen=True, slots=True)
class RuntimeItem:
    """Memory the backend holds whatever the model: a library handle's
    workspace, the CUDA context."""

    name: str
    category: Category
    estimate: Estimate


class BackendResourceModel(Protocol):
    """What a backend allocates for lowered operations."""

    name: str

    def workspace(
        self, step: Step, graph: TensorGraph, env: Mapping[str, int], phase: Phase = "inference"
    ) -> Estimate:
        """Transient bytes a step takes while it runs, beyond its outputs."""
        ...

    def saved(
        self, step: Step, graph: TensorGraph, env: Mapping[str, int], grads: Mapping[int, bool]
    ) -> Saved | None:
        """What autograd keeps from a native step for backward, or None to
        apply the generic rule; `grads` says which objects carry gradients."""
        ...

    def runtime(self, training: bool, channels: Channels = "none") -> tuple[RuntimeItem, ...]:
        """Memory held regardless of the model. `channels` are the
        point-to-point channels the process opens: a pipeline stage's sends
        and receives, or DTensor's all-to-all exchanges."""
        ...

    def alignment(self) -> int:
        """The allocation granularity of transient storage."""
        ...


def _numel(obj: MemoryObject, env: Mapping[str, int]) -> int:
    return ex.evaluate(ex.product(obj.shape), env)


def _bytes(obj: MemoryObject, env: Mapping[str, int]) -> int:
    return ex.evaluate(obj.nbytes, env) if obj.owns_storage else _numel(obj, env) * _element(obj)


def _element(obj: MemoryObject) -> int:
    return dtypes.dtype(obj.dtype).element_bytes


def _base(implementation: str) -> str:
    """`torch.softmax(input dtype)` is `torch.softmax` computed in the input
    dtype: the same storage."""
    return implementation.removesuffix("(input dtype)")


class GenericBackend:
    """A backend nothing is known about: every native operation's workspace
    is unknown, and so is the runtime's own memory."""

    name = "generic"

    def workspace(
        self, step: Step, graph: TensorGraph, env: Mapping[str, int], phase: Phase = "inference"
    ) -> Estimate:
        if step.implementation is None:
            return ZERO
        return Estimate(None, Confidence.UNKNOWN, f"`{step.implementation}`")

    def saved(
        self, step: Step, graph: TensorGraph, env: Mapping[str, int], grads: Mapping[int, bool]
    ) -> Saved | None:
        return None

    def runtime(self, training: bool, channels: Channels = "none") -> tuple[RuntimeItem, ...]:
        return (
            RuntimeItem(
                "Runtime overhead",
                Category.RUNTIME,
                Estimate(None, Confidence.UNKNOWN, "no runtime model for this backend"),
            ),
        )

    def alignment(self) -> int:
        return 1


# The f32 logits of one block of rows in `linnet.torch.loss`.
LOSS_BLOCK_BYTES = 1 << 30


def _cublas_workspace() -> Estimate:
    """PyTorch's cuBLAS workspace per handle: CUBLAS_WORKSPACE_CONFIG when
    set (`:SIZE_KIB:COUNT` pairs), else its default."""
    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config:
        total = 0
        parts = [p for p in config.split(":") if p]
        for size, count in zip(parts[::2], parts[1::2], strict=False):
            total += int(size) * 1024 * int(count)
        return Estimate(total, Confidence.MODELED, "CUBLAS_WORKSPACE_CONFIG")
    return Estimate(
        32 * MIB,
        Confidence.ESTIMATED,
        "PyTorch's default on Hopper; about 8 MiB on earlier GPUs",
    )


class CudaTorchBackend:
    """The generated PyTorch code on a CUDA device.

    Workspaces and saved tensors come from what each native implementation
    is known to allocate: PyTorch's kernels, and `linnet.torch`'s own (the
    chunked loss). The CUDA context is a typical value; the caching
    allocator's fragmentation and any compilation's autotuning are unknown.
    `context_bytes` overrides the context estimate. `processes` is the size
    of the group a process belongs to: with more than one, collectives
    allocate and the group has communication buffers."""

    name = "cuda"

    def __init__(
        self,
        context_bytes: int | None = None,
        compiled: bool = False,
        processes: int = 1,
    ) -> None:
        self.context_bytes = context_bytes
        self.compiled = compiled
        self.processes = processes

    # ---- workspace

    def workspace(
        self, step: Step, graph: TensorGraph, env: Mapping[str, int], phase: Phase = "inference"
    ) -> Estimate:
        if step.implementation is None:
            return ZERO
        implementation = _base(step.implementation)
        objects = [graph.objects[i] for i in step.inputs]
        if implementation.startswith("torch.nn.functional.scaled_dot_product_attention"):
            query = objects[0]
            rows = _numel(query, env) // ex.evaluate(query.shape[-1], env)
            bias = _mask_bias(objects, env)
            return Estimate(
                rows * 4 + bias,
                Confidence.ESTIMATED,
                "a fused kernel's f32 log-sum-exp and the mask as an additive bias; PyTorch's "
                "math fallback, which it picks when no fused kernel fits, materializes the "
                "scores instead",
            )
        if implementation in ("linnet.linear_cross_entropy", "linnet.linear_token_log_probs"):
            return _loss_workspace(implementation, objects, env, phase)
        if implementation == "torch.distributed.all_gather" and self.processes > 1:
            gathered = graph.objects[step.outputs[0]]
            return Estimate(
                _bytes(gathered, env),
                Confidence.MODELED,
                "the processes' parts stacked before they are laid side by side",
            )
        if implementation in _NO_WORKSPACE:
            return ZERO
        return Estimate(None, Confidence.UNKNOWN, f"`{implementation}`")

    # ---- autograd

    def saved(
        self, step: Step, graph: TensorGraph, env: Mapping[str, int], grads: Mapping[int, bool]
    ) -> Saved | None:
        if step.implementation is None:
            return None
        implementation = _base(step.implementation)
        inputs, outputs = step.inputs, step.outputs

        def needs(index: int) -> bool:
            return index < len(inputs) and grads.get(inputs[index], False)

        if implementation in ("torch.nn.functional.linear", "torch.matmul"):
            # Each operand for the other's gradient.
            return Saved(tuple(inputs[i] for i in (0, 1) if needs(1 - i)))
        if implementation in ("torch.distributed.all_reduce", "torch.distributed.all_gather"):
            return Saved(())
        if implementation.startswith("torch.nn.functional.scaled_dot_product_attention"):
            objects = [graph.objects[i] for i in inputs]
            rows = _numel(objects[0], env) // ex.evaluate(objects[0].shape[-1], env)
            return Saved(
                (*inputs[:3], *outputs),
                Estimate(
                    rows * 4 + _mask_bias(objects, env),
                    Confidence.ESTIMATED,
                    "log-sum-exp, and the mask as an additive bias",
                ),
            )
        if implementation in ("torch.rms_norm", "torch.nn.functional.layer_norm"):
            # Generated as the normalization, then times the weight (plus the
            # bias): the normalization keeps its input and f32 statistics,
            # the product the normalized values for the weight's gradient.
            source = graph.objects[inputs[0]]
            rows = _numel(source, env) // ex.evaluate(source.shape[-1], env)
            statistics = rows * 4 * (1 if implementation == "torch.rms_norm" else 2)
            normalized = _bytes(source, env) if needs(1) else 0
            return Saved(
                inputs[:2],
                Estimate(
                    statistics + normalized,
                    Confidence.MODELED,
                    "f32 statistics, and the normalized values the weight's gradient reads",
                ),
            )
        if implementation in ("torch.softmax", "torch.sigmoid", "torch.relu", "torch.tanh"):
            return Saved(outputs)
        if implementation in (
            "torch.nn.functional.silu",
            "torch.nn.functional.gelu",
            "torch.nn.functional.gelu(tanh)",
        ):
            return Saved(inputs[:1])
        if implementation in ("torch.nn.functional.embedding", "torch.index_select"):
            return Saved(inputs[:1])
        if implementation == "torch.Tensor.mean":
            return Saved(())
        if implementation == "linnet.linear_cross_entropy":
            hidden, weight = graph.objects[inputs[0]], graph.objects[inputs[1]]
            gradients = _bytes(hidden, env) + _numel(weight, env) * 4
            return Saved(
                (), Estimate(gradients, Confidence.MODELED, "gradients computed with the loss")
            )
        if implementation == "linnet.linear_token_log_probs":
            hidden = graph.objects[inputs[0]]
            rows = _numel(hidden, env) // ex.evaluate(hidden.shape[-1], env)
            return Saved(inputs[:3], Estimate(rows * 4, Confidence.MODELED, "log-sum-exp"))
        if implementation in (
            "torch.tril",
            "torch.Tensor.index_copy",
            "torch.Tensor.index_put",
            "torch.Tensor.index_put(tokens)",
        ):
            return Saved(())
        return None

    # ---- the process

    def runtime(self, training: bool, channels: Channels = "none") -> tuple[RuntimeItem, ...]:
        # Measured outside the caching allocator on an H100 with PyTorch
        # 2.14, CUDA 13 and NCCL 2.30 (`python -m linnet.resources.probe`
        # measures them on any machine): they vary with the GPU, the driver
        # and the libraries' versions.
        context = (
            Estimate(self.context_bytes, Confidence.MODELED, "given")
            if self.context_bytes is not None
            else Estimate(621 * MIB, Confidence.ESTIMATED, "measured with PyTorch 2.14 on an H100")
        )
        items = [
            RuntimeItem("CUDA context", Category.RUNTIME, context),
            RuntimeItem(
                "CUDA libraries",
                Category.RUNTIME,
                Estimate(
                    (146 if training else 74) * MIB,
                    Confidence.ESTIMATED,
                    "cuBLAS and attention kernels, and in training their backward kernels, "
                    "loaded when first called",
                ),
            ),
            RuntimeItem("cuBLAS workspace", Category.WORKSPACE, _cublas_workspace()),
            RuntimeItem(
                "Allocator fragmentation",
                Category.RUNTIME,
                Estimate(
                    None,
                    Confidence.UNKNOWN,
                    "the caching allocator's unused reserve depends on the allocation order",
                ),
            ),
        ]
        if self.processes > 1:
            items += [
                RuntimeItem(
                    "One-shot all-reduce buffers",
                    Category.RUNTIME,
                    Estimate(
                        2 * ONE_SHOT_BYTES,
                        Confidence.ESTIMATED,
                        "the symmetric memory small sums go through (`linnet.torch.collectives`)",
                    ),
                ),
                RuntimeItem(
                    "NCCL communicator",
                    Category.RUNTIME,
                    Estimate(
                        842 * MIB,
                        Confidence.ESTIMATED,
                        "measured for two H100s over NVLink; grows with the GPUs and channels",
                    ),
                ),
            ]
        if channels != "none":
            items.append(
                RuntimeItem(
                    "NCCL send and receive",
                    Category.RUNTIME,
                    Estimate(
                        _CHANNELS[channels] * MIB,
                        Confidence.ESTIMATED,
                        {
                            "send": "a pipeline stage's channel to the next",
                            "receive": "a pipeline stage's channel from the one before",
                            "both": "a pipeline stage's channels both ways",
                            "exchange": "the channels DTensor's all-to-all redistributions open",
                        }[channels]
                        + "; measured on H100s, beside the context, libraries and communicator",
                    ),
                )
            )
        if self.compiled:
            items.append(
                RuntimeItem(
                    "Compilation",
                    Category.RUNTIME,
                    Estimate(None, Confidence.UNKNOWN, "torch.compile autotuning and caches"),
                )
            )
        return tuple(items)

    def alignment(self) -> int:
        # The caching allocator rounds each block up to 512 bytes.
        return 512


def _mask_bias(objects: list[MemoryObject], env: Mapping[str, int]) -> int:
    """A boolean mask (the fourth input) as the additive bias the fused
    kernels take: its elements in the query's dtype, not broadcast."""
    if len(objects) < 4 or objects[3].dtype != "bool":
        return 0
    return _numel(objects[3], env) * _element(objects[0])


def _loss_workspace(
    implementation: str, objects: list[MemoryObject], env: Mapping[str, int], phase: Phase
) -> Estimate:
    """`linnet.torch.loss`, a block of rows at a time. Python keeps the last
    block's tensors until their names are reassigned, so two blocks overlap:
    with gradients, the last block's f32 probabilities and input-dtype
    gradient stay while the next block's f32 logits and `logsumexp`'s f32
    temporary exist; and each block's weight-gradient product is made in the
    input dtype and in f32 before it is added."""
    hidden, weight = objects[0], objects[1]
    vocab = ex.evaluate(weight.shape[0], env)
    width = ex.evaluate(weight.shape[1], env)
    rows = _numel(hidden, env) // ex.evaluate(hidden.shape[-1], env)
    block = min(rows, max(1, LOSS_BLOCK_BYTES // (4 * vocab)))
    second = min(block, rows - block)  # the next block, when there is one
    e = _element(hidden)
    update = vocab * width * (e + 4)
    if implementation == "linnet.linear_cross_entropy":
        if phase == "backward":
            return Estimate(0, Confidence.MODELED, "the gradients were computed with the loss")
        if phase == "forward":
            sizes = [
                block * vocab * (e + 4),
                block * vocab * 8,
                block * vocab * (4 + e) + update,
                block * vocab * (4 + e) + second * vocab * 8,
            ]
            return Estimate(max(sizes), Confidence.MODELED, "blocks of logits and their gradient")
    elif phase == "backward":
        sizes = [
            block * vocab * (4 + e) + second * vocab * (4 + e),
            block * vocab * (4 + e) + update,
        ]
        return Estimate(
            max(sizes) + vocab * width * 4 + rows * width * e,
            Confidence.MODELED,
            "blocks of logits again, and the f32 weight gradient",
        )
    sizes = [
        block * vocab * (e + 4),
        block * vocab * 8,
        block * vocab * 4 + second * vocab * (e + 4),
    ]
    return Estimate(
        max(sizes), Confidence.MODELED, "blocks of logits, in the input dtype and in f32"
    )


_NO_WORKSPACE = frozenset(
    {
        "torch.nn.functional.linear",
        "torch.matmul",
        "torch.nn.functional.embedding",
        "torch.index_select",
        "torch.nn.functional.batch_norm",
        "torch.Tensor.mean",
        "torch.nn.functional.gelu(tanh)",
        "torch.Tensor.index_put",
        "torch.Tensor.index_put(tokens)",
        "torch.rms_norm",
        "torch.nn.functional.layer_norm",
        "torch.softmax",
        "torch.sigmoid",
        "torch.relu",
        "torch.tanh",
        "torch.nn.functional.silu",
        "torch.nn.functional.gelu",
        "torch.tril",
        "torch.Tensor.index_copy",
        # A collective's result is its own output; `all_gather` across
        # processes is the exception, above.
        "torch.distributed.all_reduce",
        "torch.distributed.all_gather",
    }
)


def backend_model(name: str, **options: object) -> BackendResourceModel:
    """The resource model of a backend by name: `cuda` or `generic`."""
    if name == "cuda":
        context = options.get("context_bytes")
        processes = options.get("processes", 1)
        return CudaTorchBackend(
            context_bytes=int(context) if isinstance(context, int) else None,
            compiled=bool(options.get("compiled", False)),
            processes=processes if isinstance(processes, int) else 1,
        )
    if name == "generic":
        return GenericBackend()
    raise ValueError(f"no resource model for backend `{name}` (cuda, generic)")

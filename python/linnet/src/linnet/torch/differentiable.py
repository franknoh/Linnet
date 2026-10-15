"""A plan's backend steps made part of autograd (docs/backends.md).

A backend's executable makes its outputs outside autograd, from weights it
reads detached from the context. With gradients on and a weight or an input
that wants one, each such step runs as a `torch.autograd.Function` instead:
its backward is the executable's own (`backward(inputs, grad_outputs,
context)`, giving the inputs' gradients and the weights' by path) where it
has one, else the region again on the generated source, differentiated
there. The gradients reach the model's own parameters.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import torch

from ..backends import Context, Executable, ExecutableMeta, Region

if TYPE_CHECKING:
    from .compiled import CompiledLinnetModule


class _Backward(Protocol):
    def __call__(
        self,
        inputs: Sequence[torch.Tensor],
        grad_outputs: Sequence[torch.Tensor | None],
        context: Context[torch.Tensor],
    ) -> tuple[Sequence[torch.Tensor | None], dict[str, torch.Tensor]]: ...


class Differentiable:
    """`inner`, the executable of `region`, run through autograd when a
    gradient is wanted; `recompute` makes the generated source's executable
    for `region`, which a backward without one of `inner`'s own runs."""

    def __init__(
        self,
        inner: Executable[torch.Tensor],
        region: Region,
        module: CompiledLinnetModule,
        recompute: Callable[[], Executable[torch.Tensor]],
    ) -> None:
        self.inner = inner
        self.region = region
        self.module = module
        self._recompute = recompute
        self._fallback: Executable[torch.Tensor] | None = None
        self._owners: list[tuple[torch.nn.Module, str]] | None = None

    @property
    def meta(self) -> ExecutableMeta:
        return self.inner.meta

    def prepare(self, context: Context[torch.Tensor]) -> None:
        self.inner.prepare(context)

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        if not torch.is_grad_enabled():
            return self.inner.execute(inputs, outputs, context)
        weights = self.weights()
        if not any(t.requires_grad for t in (*inputs, *weights)):
            return self.inner.execute(inputs, outputs, context)
        made = _Region.apply(self, context, len(inputs), *inputs, *weights)
        return list(cast("tuple[torch.Tensor, ...]", made))

    def weights(self) -> list[torch.Tensor]:
        """The model's own parameters the region reads, live."""
        if self._owners is None:
            from .module import owner_of

            self._owners = [owner_of(self.module, path) for path in self.region.parameters]
        return [cast("torch.Tensor", getattr(owner, leaf)) for owner, leaf in self._owners]

    def backward(
        self,
        inputs: Sequence[torch.Tensor],
        grad_outputs: Sequence[torch.Tensor | None],
        context: Context[torch.Tensor],
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        """The gradients of the region's inputs and of its weights (in
        `region.parameters` order)."""
        own = cast("_Backward | None", getattr(self.inner, "backward", None))
        if own is not None:
            by_input, by_path = own(inputs, grad_outputs, context)
            return list(by_input), [by_path.get(p) for p in self.region.parameters]
        # The region again, on the generated source, differentiated there.
        if self._fallback is None:
            self._fallback = self._recompute()
        weights = self.weights()
        wanting = [t.detach().requires_grad_(t.is_floating_point()) for t in inputs]
        with torch.enable_grad():
            made = self._fallback.execute(wanting, [], context)
            pairs = [(o, g) for o, g in zip(made, grad_outputs, strict=True) if g is not None]
            differentiable = [t for t in (*wanting, *weights) if t.requires_grad]
            grads = torch.autograd.grad(
                [o for o, _ in pairs],
                differentiable,
                [g for _, g in pairs],
                allow_unused=True,
            )
        by_tensor = dict(zip(map(id, differentiable), grads, strict=True))
        return (
            [by_tensor.get(id(t)) for t in wanting],
            [by_tensor.get(id(t)) for t in weights],
        )


class _Region(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        step: Differentiable,
        context: Context[torch.Tensor],
        count: int,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        inputs = list(tensors[:count])
        made = step.inner.execute(inputs, [], context)
        ctx.save_for_backward(*inputs)
        ctx.step = step  # pyright: ignore[reportAttributeAccessIssue]
        ctx.context = context  # pyright: ignore[reportAttributeAccessIssue]
        return tuple(made)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        step = cast("Differentiable", ctx.step)  # pyright: ignore[reportAttributeAccessIssue]
        context = cast("Context[torch.Tensor]", ctx.context)  # pyright: ignore[reportAttributeAccessIssue]
        inputs = cast("tuple[torch.Tensor, ...]", ctx.saved_tensors)  # pyright: ignore[reportAttributeAccessIssue]
        by_input, by_weight = step.backward(list(inputs), list(grad_outputs), context)
        return (None, None, None, *by_input, *by_weight)


class TrainingStep:
    """`inner` run as a training step runs it, to be timed: its floating
    inputs wanting gradients, then backward from its outputs."""

    def __init__(self, inner: Executable[torch.Tensor]) -> None:
        self.inner = inner

    @property
    def meta(self) -> ExecutableMeta:
        # Timed between events: a backward is not recorded in a CUDA graph.
        return dataclasses.replace(self.inner.meta, capturable=False)

    def prepare(self, context: Context[torch.Tensor]) -> None:
        self.inner.prepare(context)

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        with torch.enable_grad():
            wanting = [t.detach().requires_grad_(t.is_floating_point()) for t in inputs]
            made = self.inner.execute(wanting, outputs, context)
            ends = [o for o in made if o.requires_grad]
            if ends:
                torch.autograd.backward(ends, [torch.ones_like(o) for o in ends])
        return made


__all__ = ["Differentiable", "TrainingStep"]

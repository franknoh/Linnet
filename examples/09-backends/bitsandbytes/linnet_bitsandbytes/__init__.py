"""bitsandbytes' 4-bit product as a Linnet backend, a plugin from a package
of its own (docs/backends.md): a `std.quant::linear_nf4` call (`Nf4Linear`)
as `bitsandbytes.matmul_4bit`, which multiplies one row straight from the
4-bit codes and longer inputs over the weight it widens.

bitsandbytes holds a weight's codes flattened, the first of each pair in
the high four bits, where Linnet's `nf4` puts it in the low four: `prepare`
keeps the codes once more in bitsandbytes' order, half a byte a weight. The
scales, one per block of a row, are the same numbers in the same order.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import Protocol, cast

import torch

from linnet.backends import (
    Capabilities,
    Context,
    Cost,
    ExecutableMeta,
    Graph,
    Options,
    Region,
    Target,
    TensorSpec,
)
from linnet.compiler import PlanError
from linnet.torch.dtypes import TORCH_DTYPES

_CALLEE = "std.quant::linear_nf4"
_DTYPES = ("bf16", "f16", "f32")


class _Product(Protocol):
    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        quant_state: object,
        out: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


def _bitsandbytes() -> ModuleType | None:
    try:
        return importlib.import_module("bitsandbytes")
    except ImportError:
        return None


class BitsAndBytes:
    """Takes the NF4 `linear` calls it can run."""

    contract = 1  # the version of `linnet.backends` it was written for

    @property
    def name(self) -> str:
        return "bitsandbytes"

    @property
    def version(self) -> str:
        module = _bitsandbytes()
        return str(getattr(module, "__version__", "missing"))

    def probe(self, target: Target) -> Capabilities | None:
        if target.kind != "cuda" or _bitsandbytes() is None:
            return None
        return Capabilities({"bitsandbytes": self.version}, frozenset(_DTYPES))

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        persistent = set(graph.parameters.values())
        for call in graph.semantic_calls(_CALLEE):
            if len(call.arguments) != 4:
                continue
            x, q, absmax, bias = call.arguments
            region = Region.of(graph, call.start, call.stop, call.index)
            if (
                x.dtype in _DTYPES
                and q.value in persistent
                and absmax.value in persistent
                and (bias.value is None or bias.value in persistent)
                and region.inputs == (x.value,)
            ):
                yield region

    def compile(self, region: Region, target: Target, options: Options) -> _Nf4Product:
        module = _bitsandbytes()
        if region.call is None or module is None:
            raise PlanError("bitsandbytes takes a `linear_nf4` call, and must be installed")
        graph = region.graph
        call = graph.calls[region.call]
        x, q, absmax, bias = call.arguments
        path = {value: path for path, value in graph.parameters.items()}
        out_features, in_features = q.shape
        return _Nf4Product(
            region,
            cast("_Product", module.matmul_4bit),
            path[cast("int", q.value)],
            path[cast("int", absmax.value)],
            path[bias.value] if bias.value is not None else None,
            (out_features, in_features),
            in_features // absmax.shape[1],
            TORCH_DTYPES[x.dtype],
        )

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


class _Nf4Product:
    def __init__(
        self,
        region: Region,
        product: _Product,
        codes: str,
        absmax: str,
        bias: str | None,
        shape: tuple[int, int],
        block: int,
        dtype: torch.dtype,
    ) -> None:
        values = region.graph.values
        self._product = product
        self._codes_path = codes
        self._absmax_path = absmax
        self._bias = bias
        self._shape = shape
        self._block = block
        self._dtype = dtype  # the input's, which the product widens to
        self._codes: torch.Tensor | None = None  # in bitsandbytes' order
        self._state: object | None = None
        self._meta = ExecutableMeta(
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.inputs),
            tuple(TensorSpec(values[v].shape, values[v].dtype) for v in region.outputs),
            holds=shape[0] * shape[1] // 2,
        )

    @property
    def meta(self) -> ExecutableMeta:
        return self._meta

    def prepare(self, context: Context[torch.Tensor]) -> None:
        functional = importlib.import_module("bitsandbytes.functional")
        packed = context.parameters[self._codes_path]
        # Each byte's two codes swapped, flattened: bitsandbytes' layout.
        codes = ((packed << 4) | (packed >> 4)).reshape(-1, 1)
        absmax = context.parameters[self._absmax_path].reshape(-1).contiguous()
        if self._codes is not None and self._state is not None:
            # Again, after the weights changed: into the same tensors, which
            # a captured graph reads.
            self._codes.copy_(codes)
            self._state.absmax.copy_(absmax)  # pyright: ignore[reportAttributeAccessIssue]
            return
        self._codes = codes.contiguous()
        self._state = functional.QuantState(
            absmax=absmax,
            shape=torch.Size(self._shape),
            code=functional.get_4bit_type("nf4", device=packed.device),
            blocksize=self._block,
            quant_type="nf4",
            dtype=self._dtype,
        )

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        if self._codes is None or self._state is None:
            self.prepare(context)
        assert self._codes is not None
        (x,) = inputs
        bias = context.parameters[self._bias] if self._bias is not None else None
        return [self._product(x, self._codes.t(), quant_state=self._state, bias=bias)]


__all__ = ["BitsAndBytes"]

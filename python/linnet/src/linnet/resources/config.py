"""An execution configuration: what to analyze a model under.

The batch size, the sequence length (`context`) and the number of cache
positions (`cache`) are roles rather than generics: each binds every
generic named in its list (`B` and `Batch` for the batch, by default), so
one number sizes an entry's inputs and the caches the model allocates for
them alike. Parallelism degrees and the sharding policy are part of the
configuration so that a planner can vary them; the analysis supports data
parallelism with sharded training state (FSDP) and refuses the others
rather than guessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from .kvcache import ContiguousLayout, KVLayout
from .training import TrainingConfig

Numerics = Literal["exact", "equivalent", "fast"]


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """How a model runs.

    `dtype` binds the root block's float dtype generics (the compute dtype,
    which is also the dtype parameters are stored in: a model that stores
    weights otherwise declares them so in its source). `kv_dtype` and
    `kv_layout` describe a backend that stores caches differently from
    their declaration. `bindings` gives any other generic a value."""

    backend: str = "cuda"
    entry: str | None = None
    numerics: Numerics = "fast"
    dtype: str | None = None
    kv_dtype: str | None = None
    kv_layout: KVLayout = field(default_factory=ContiguousLayout)
    batch: int | None = None
    context: int | None = None
    cache: int | None = None
    bindings: Mapping[str, int | str] = field(default_factory=lambda: MappingProxyType({}))
    optionals: frozenset[str] = frozenset()
    training: TrainingConfig | None = None
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    sharding: Literal["none", "fsdp"] = "none"
    batch_generics: tuple[str, ...] = ("B", "Batch")
    context_generics: tuple[str, ...] = ("S", "P", "Seq", "SeqLen")
    cache_generics: tuple[str, ...] = ("MaxSeq",)
    context_bytes: int | None = None
    compiled: bool = False

    def describe(self) -> dict[str, Any]:
        """The configuration as plain values, for reports."""
        out: dict[str, Any] = {
            "backend": self.backend,
            "entry": self.entry,
            "numerics": self.numerics,
            "dtype": self.dtype,
            "batch": self.batch,
            "context": self.context,
            "cache": self.cache if self.cache is not None else self.context,
            "kv_dtype": self.kv_dtype,
            "kv_layout": self.kv_layout.name,
            "bindings": dict(self.bindings),
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "data_parallel": self.data_parallel,
            "sharding": self.sharding,
        }
        if self.training is not None:
            t = self.training
            out["training"] = {
                "optimizer": t.optimizer.name,
                "gradient_dtype": t.gradient_dtype,
                "master_dtype": t.master_dtype,
                "trainable": list(t.trainable),
                "checkpoint": t.checkpoint.kind
                if t.checkpoint.kind != "regions"
                else list(t.checkpoint.patterns),
                "shards": t.shards,
            }
        return out

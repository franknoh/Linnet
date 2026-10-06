"""Static resource analysis: the memory a model needs, before it runs.

The layers, each its own module:

- `trace`: a compiled entry (`linnet.ir`) as memory objects and steps, with
  the backend's lowering decisions (`Lowering`) applied;
- `graph`: the resource analysis IR, liveness, peaks and buffer planning;
- `expr` and `storage`: symbolic byte counts and dtype sizes;
- `backends`: what lowered operations and the runtime allocate, with how
  sure each number is (`BackendResourceModel`);
- `kvcache`, `training`: caches, and a training step's timeline with
  optimizer states and checkpointing;
- `analysis`: `MemoryModel`, which puts them together for a configuration;
- `planner`: the largest batch, context or cache that fits, and
  feasibility checks over candidate configurations;
- `validate`: predictions against a CUDA device's measurements.

`linnet memory` and `linnet fit` are the command-line front ends.
"""

from .analysis import MemoryModel
from .backends import BackendResourceModel, CudaTorchBackend, Estimate, GenericBackend
from .config import ExecutionConfig
from .graph import Category, Confidence, MemoryObject, Step, TensorGraph
from .kvcache import ContiguousLayout, KVLayout, PagedLayout
from .planner import ExecutionPlanner, FitResult, ResourceConstraint, fit
from .result import MemoryAnalysisResult, MemoryComponent
from .trace import Lowering, trace
from .training import OPTIMIZERS, CheckpointPolicy, OptimizerModel, TrainingConfig

__all__ = [
    "OPTIMIZERS",
    "BackendResourceModel",
    "Category",
    "CheckpointPolicy",
    "Confidence",
    "ContiguousLayout",
    "CudaTorchBackend",
    "Estimate",
    "ExecutionConfig",
    "ExecutionPlanner",
    "FitResult",
    "GenericBackend",
    "KVLayout",
    "Lowering",
    "MemoryAnalysisResult",
    "MemoryComponent",
    "MemoryModel",
    "MemoryObject",
    "OptimizerModel",
    "PagedLayout",
    "ResourceConstraint",
    "Step",
    "TensorGraph",
    "TrainingConfig",
    "fit",
    "trace",
]

"""Predictions against a CUDA device's measurements.

    python -m linnet.resources.validate llama-3.1-8b-instruct --batch 4 --seq-len 2048
    python -m linnet.resources.validate model.linnet --bind H=1024 --entry loss --training

The model loads in PyTorch as generated source (`compile=True`), with zero
weights unless `--weights` names a checkpoint. The entry runs once to warm
up and once measured; for `--training`, a whole step (forward, backward and
the optimizer) runs twice, so the measured step already has its optimizer
states. Reported:

- the graph-visible prediction (`graph_peak`) against the caching
  allocator's peak (`torch.cuda.max_memory_allocated`);
- the allocator prediction (graph, workspaces and library workspaces)
  against the same peak;
- the process prediction (`expected_peak`, with the CUDA context) against
  the device memory in use at the end (`torch.cuda.mem_get_info`).

The errors are what this tool is for. They are reported, never folded back
into the formulas: a constant tuned until one benchmark matches would only
hide where the model is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from .. import ir
from ..compiler import LinnetError
from . import expr as ex
from .analysis import MemoryModel
from .cli import add_arguments, config_from_args
from .graph import Category
from .result import MemoryAnalysisResult, format_bytes
from .training import CheckpointPolicy


@dataclass(frozen=True, slots=True)
class Comparison:
    """A prediction and a measurement; `predicted` is None for what the
    analysis lists as unknown, which is measured to show its size."""

    name: str
    predicted: int | None
    measured: int

    @property
    def error(self) -> int | None:
        return None if self.predicted is None else self.predicted - self.measured

    @property
    def relative(self) -> float | None:
        error = self.error
        return None if error is None or not self.measured else error / self.measured


def root_values(model: MemoryModel, env: Mapping[str, int]) -> dict[str, int | str]:
    """The root block's generics as the analysis bound them, with the roles
    at `env`."""
    program = model.program
    values: dict[str, int | str] = {}
    given: dict[str, int | str] = {**model.generics, **model.config.bindings}
    for generic in program.root.generics:
        role = next((r for r in model.free if generic.name in model.roles[r]), None)
        if generic.kind == "dim" and role is not None:
            values[generic.name] = env[role]
        elif generic.kind == "dtype" and model.config.dtype is not None:
            values[generic.name] = model.config.dtype
        elif generic.name in given:
            values[generic.name] = given[generic.name]
    return values


def _inputs(model: MemoryModel, env: Mapping[str, int], device: Any) -> list[Any]:
    import torch

    program = model.program
    function = program.entry(model.entry)
    from ..torch.dtypes import TORCH_DTYPES
    from .trace import entry_env, root_env

    root = root_env(program, {**root_values(model, env)})
    inputs: dict[str, int | ex.Expr] = {}
    for generic in function.generics:
        role = next((r for r in model.free if generic.name in model.roles[r]), None)
        if role is not None:
            inputs[generic.name] = env[role]
        elif isinstance(model.config.bindings.get(generic.name), int):
            inputs[generic.name] = int(model.config.bindings[generic.name])
    bound = entry_env(function, root, inputs)
    values: list[Any] = []
    for param in function.params:
        if isinstance(param.type, ir.TensorType):
            shape = [ex.evaluate(d, {}) for d in bound.shape(param.type.shape)]
            dtype = TORCH_DTYPES[bound.dtype(param.type.dtype)]
            values.append(
                torch.randn(shape, dtype=dtype, device=device)
                if dtype.is_floating_point
                else torch.zeros(shape, dtype=dtype, device=device)
            )
        elif isinstance(param.type, ir.ScalarType):
            values.append(
                torch.tensor(0, dtype=TORCH_DTYPES[bound.dtype(param.type.dtype)], device=device)
            )
        else:
            raise LinnetError(f"cannot make an input of `{param.name}`'s type")
    return values


def measure(
    model: MemoryModel, weights: str | None = None
) -> tuple[MemoryAnalysisResult, list[Comparison]]:
    """Runs the configured entry on CUDA and compares it with the prediction."""
    import torch

    from .. import torch as linnet_torch

    if not torch.cuda.is_available():
        raise LinnetError("validation needs a CUDA device")
    env = model.env()
    prediction = model.analyze()
    config = model.config
    training = config.training
    if training is not None and (
        training.master_dtype or training.checkpoint.kind != "none" or training.shards > 1
    ):
        raise LinnetError(
            "validation runs plain training steps: no master weights, checkpointing or sharding"
        )
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.init()
    # Every process of a split model takes the same inputs.
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    generics = root_values(model, env)
    inputs = _inputs(model, env, device)
    if config.pipeline_parallel > 1:
        pipe = linnet_torch.pipeline(
            model.source_path,
            generics=generics,
            root=model.program.root.name,
            weights=weights,
            strict=weights is not None,
            entry=model.entry,
            microbatches=config.microbatches,
            schedule=config.schedule,
            stages=config.stages,
            numerics=config.numerics,
            trainable=training is not None,
            device=device,
        )
        parameters = pipe.parameters()

        def step() -> None:
            if training is None:
                pipe.run(*inputs)
            else:
                pipe.step(*inputs)

    else:
        mesh: Any = None
        if config.tensor_parallel > 1:
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        module = linnet_torch.load(
            model.source_path,
            generics=generics,
            root=model.program.root.name,
            weights=weights,
            device=device,
            numerics=config.numerics,
            compile=True,
            trainable=training is not None,
            strict=weights is not None,
            tensor_parallel=mesh,
        )
        entry = getattr(module, model.entry)
        parameters = [p for p in module.parameters() if p.requires_grad]

        def step() -> None:
            if training is None:
                with torch.no_grad():
                    entry(*inputs)
            else:
                entry(*inputs).backward()

    optimizer: Any = None
    if training is not None:
        name = training.optimizer.name
        if name == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=1e-6)
        elif name == "sgd-momentum":
            optimizer = torch.optim.SGD(parameters, lr=1e-6, momentum=0.9)
        elif name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=1e-6, fused=True)
        else:
            optimizer = torch.optim.AdamW(parameters, lr=1e-6, fused=True)
    # The second step is measured: by then the optimizer has its states.
    for repeat in range(2):
        if repeat == 1:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        step()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    if prediction.devices:
        # This process's stage: its own peaks, with the runtime the rest has.
        own = prediction.devices[rank]
        prediction = replace(prediction, graph_peak=own.graph_peak, expected_peak=own.expected_peak)
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    free, total = torch.cuda.mem_get_info()
    used = int(total - free)
    outside = used - int(torch.cuda.memory_reserved())
    runtime = sum(c.nbytes or 0 for c in prediction.components if c.category == Category.RUNTIME)
    context = sum(c.nbytes or 0 for c in prediction.components if c.name == "CUDA context")
    comparisons = [
        Comparison("graph-visible vs allocator peak", prediction.graph_peak, allocated),
        Comparison(
            "graph + workspaces vs allocator peak", prediction.expected_peak - runtime, allocated
        ),
        Comparison("allocator reserve beyond its peak", None, reserved - allocated),
        Comparison("CUDA context vs outside the allocator", context, outside),
        Comparison("whole process vs device in use", prediction.expected_peak, used),
    ]
    return prediction, comparisons


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m linnet.resources.validate")
    add_arguments(parser)
    parser.add_argument("--weights", help="a checkpoint to load (default: zero weights)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    import os

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        # Under torchrun: one process per GPU, each measuring its own.
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl")
    try:
        policy = args.checkpoint[0] if args.checkpoint else CheckpointPolicy()
        # Zero weights load without bindings: no tied tensors, no optional
        # parameters. Predict that module, not the card's.
        model = MemoryModel(
            args.model,
            config_from_args(args, policy),
            root=args.root,
            std_root=args.std,
            bindings=args.weights is not None,
        )
        prediction, comparisons = measure(model, args.weights)
    except LinnetError as error:
        print(f"linnet: {error}", file=sys.stderr)
        return 1
    report: dict[str, Any] = {
        "prediction": prediction.to_dict(),
        "comparisons": [
            {
                "name": c.name,
                "predicted": c.predicted,
                "measured": c.measured,
                "error": c.error,
                "relative": c.relative,
            }
            for c in comparisons
        ],
    }
    lines: list[str] = []
    for c in comparisons:
        relative = "" if c.relative is None else f"  error {c.relative * 100:+.1f}%"
        lines.append(
            f"{c.name:<40} predicted {format_bytes(c.predicted):>12}  measured "
            f"{format_bytes(c.measured):>12}{relative}"
        )
    reports: list[Any] = [report]
    texts: list[Any] = [lines]
    if distributed:
        import torch.distributed as dist

        reports = [report] * dist.get_world_size()
        texts = [lines] * dist.get_world_size()
        dist.all_gather_object(reports, report)
        dist.all_gather_object(texts, lines)
        first = dist.get_rank() == 0
        dist.destroy_process_group()
        if not first:
            return 0
    if args.json:
        json.dump(reports[0] if len(reports) == 1 else reports, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    for rank, text in enumerate(texts):
        if len(texts) > 1:
            print(f"rank {rank}")
        print("\n".join(cast(list[str], text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pipeline parallelism for `linnet.torch`: a model's blocks split into
stages, one process each, micro-batches flowing through them.

    import torch, torch.distributed as dist
    from linnet.torch import pipeline

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    pipe = pipeline("src/lib.linnet", generics={...}, weights="weights/",
                    entry="loss_packed", microbatches=8)
    optimizer = torch.optim.AdamW(pipe.parameters(), lr=1e-5)
    loss = pipe.step(tokens, positions, segments, targets, weights)
    optimizer.step()
    optimizer.zero_grad()

Every process of the group calls the same thing with the same inputs; rank
`k` runs stage `k`. The stages are contiguous runs of the root's blocks
(each sub-block, each element of a block array), balanced by parameter
bytes unless `stages` names where each one starts. A process holds only its
own blocks' weights: the model is built on the `meta` device and only the
stage's blocks are given memory and read from the checkpoint.

The entry is the generated PyTorch source (`linnet torch`) for one
micro-batch, split into one function per stage by its data flow
(`linnet.torch.stages`): only values computed from weights cross between
stages; masks and other values of the inputs are computed again where they
are read. `torch.distributed.pipelining` runs the stages under GPipe or 1F1B.

`step` trains: the inputs' first axis is cut into `microbatches` equal
parts, and the last stage returns the sum of the entry's result over them,
which is also what the gradients are of. A packed entry (`loss_packed`)
takes one row of packed sequences: pack each micro-batch's part on its own,
so that no sequence spans two. `run` is the forward pass alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ..parallel import assign_stages, pipeline_stages, stage_of_path, stage_starts
from ..plan import PlanError, compile_plan
from ..weights import paths_by_tensor
from .compiled import CompiledLinnetModule
from .module import bind_weights, owner_of
from .placement import units_of
from .stages import Split, Stage, split

Schedule = Literal["1f1b", "gpipe"]


class Pipeline:
    """One process's stage of a model run as a pipeline."""

    def __init__(
        self,
        module: CompiledLinnetModule,
        entry: str,
        generics: Mapping[str, int | str],
        *,
        units: Sequence[str],
        assigned: Sequence[int],
        microbatches: int,
        schedule: Schedule,
        group: Any,
        device: torch.device,
        ties: Sequence[tuple[list[str], Any]],
        compile: str | None,
    ) -> None:
        import torch.distributed as dist

        self.module = module
        self.entry = entry
        self.generics = dict(generics)
        self.units = tuple(units)
        self.assigned = tuple(assigned)
        self.starts = tuple(stage_starts(units, assigned))
        self.microbatches = microbatches
        self.schedule = schedule
        self.group = group
        self.device = device
        self.stage = dist.get_rank(group)
        self.stages = dist.get_world_size(group)
        self.compile = compile
        # Parameters tied to one checkpoint tensor across stages (an
        # embedding and an output head), with the group of the stages
        # holding them: their gradients are summed after every step.
        self.ties = list(ties)
        self._built: dict[tuple[Any, ...], _Built] = {}

    @property
    def is_first(self) -> bool:
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        return self.stage == self.stages - 1

    def stage_of(self, path: str) -> int:
        """The stage holding the parameter or state at `path`."""
        stage = stage_of_path(self.units, self.assigned, path)
        if stage is None:
            raise PlanError(f"`{path}` is in none of the model's blocks")
        return stage

    def parameters(self) -> list[nn.Parameter]:
        """This stage's parameters, for its optimizer."""
        return [p for _, p in self.named_parameters()]

    def named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        seen: set[int] = set()
        found: list[tuple[str, nn.Parameter]] = []
        for name, parameter in self.module.root.named_parameters(remove_duplicate=False):
            if self.stage_of(name) != self.stage or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            found.append((name, parameter))
        return found

    def step(self, *inputs: torch.Tensor) -> torch.Tensor | None:
        """One training step's forward and backward over the micro-batches.
        Returns the sum of the entry's results on the last stage, None on
        the others; gradients are left on each stage's parameters."""
        built = self._build(inputs)
        if built.split.results != 1:
            raise PlanError("a training step needs an entry with one result: the loss")
        losses: list[torch.Tensor] | None = [] if self.is_last else None
        built.schedule.step(
            kwarg_mbs=self._microbatches(built, inputs),
            target_mbs=[None] * self.microbatches if self.is_last else None,
            losses=losses,
            return_outputs=False,
        )
        self._sum_tied_gradients()
        if losses is None:
            return None
        return torch.stack([loss.detach().float() for loss in losses]).sum()

    def run(self, *inputs: torch.Tensor) -> Any:
        """The entry's forward pass over the micro-batches, without
        gradients. On the last stage, each result joined along its first
        axis (a scalar result: one value per micro-batch); None elsewhere."""
        built = self._build(inputs)
        # The last stage hands each scalar result over as one value, so that
        # the schedule joins the micro-batches' results along the first axis.
        built.module.joining = True
        try:
            with torch.no_grad():
                merged: Any = built.schedule.eval(
                    kwarg_mbs=self._microbatches(built, inputs),
                    target_mbs=[None] * self.microbatches if self.is_last else None,
                )
        finally:
            built.module.joining = False
        return merged if self.is_last else None

    # ---- one build per input signature

    def _microbatches(
        self, built: _Built, inputs: Sequence[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Each micro-batch's inputs this stage reads, by argument name."""
        reads = set(built.stage.inputs)
        parts: list[dict[str, torch.Tensor]] = [{} for _ in range(self.microbatches)]
        for name, value in zip(built.split.inputs, inputs, strict=True):
            if name not in reads:
                continue
            for index, chunk in enumerate(value.to(self.device).chunk(self.microbatches)):
                parts[index][name] = chunk
        return parts

    def _build(self, inputs: Sequence[torch.Tensor]) -> _Built:
        key = tuple((tuple(value.shape), value.dtype) for value in inputs)
        built = self._built.get(key)
        if built is not None:
            return built
        for value in inputs:
            if value.dim() == 0 or value.shape[0] % self.microbatches != 0:
                raise PlanError(
                    f"every input's first axis must divide into {self.microbatches} "
                    f"micro-batches; one has shape {tuple(value.shape)}"
                )
        one = [value[: value.shape[0] // self.microbatches] for value in inputs]
        bindings = self.module.bindings_for(self.entry, one, self.generics)
        source = self.module.source_for(self.entry, bindings, trains=True, prepare=False)
        pieces = split(source, self.stage_of, self.stages)
        generated = self.module.import_source(
            pieces.source, f"{self.entry}_stages_{len(self._built)}"
        )
        stage = pieces.stages[self.stage]
        with torch.no_grad():
            computed = generated.constants(self.device) if pieces.constants else ()
            constants = dict(zip(pieces.constants, computed, strict=True))
        function: Callable[..., Any] = getattr(generated, stage.name)
        if self.compile is not None:
            function = torch.compile(function, backend=self.compile)
        blocks = [
            self.module.root.get_submodule(unit)
            for unit, stage_index in zip(self.units, self.assigned, strict=True)
            if stage_index == self.stage
        ]
        wrapped = _StageModule(
            function,
            [owner_of(self.module, pieces.parameters[i]) for i in stage.parameters],
            [constants[name] for name in stage.constants],
            blocks,
            self.device,
            last=self.is_last,
        )
        from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

        stage_runner = PipelineStage(
            wrapped, self.stage, self.stages, self.device, group=self.group
        )
        kind = Schedule1F1B if self.schedule == "1f1b" else ScheduleGPipe
        runner = kind(
            stage_runner,
            n_microbatches=self.microbatches,
            loss_fn=_result,
            scale_grads=False,
        )
        built = _Built(pieces, stage, runner, wrapped)
        self._built[key] = built
        return built

    def _sum_tied_gradients(self) -> None:
        import torch.distributed as dist

        for paths, group in self.ties:
            mine = [p for p in paths if self.stage_of(p) == self.stage]
            if not mine:
                continue
            owner, leaf = owner_of(self.module, mine[0])
            parameter: torch.Tensor = getattr(owner, leaf)
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, group=group)


@dataclass
class _Built:
    split: Split
    stage: Stage
    schedule: Any
    module: _StageModule


class _StageModule(nn.Module):
    """A stage function as the module `PipelineStage` runs: the values that
    cross into it positionally, the entry's inputs by keyword."""

    def __init__(
        self,
        function: Callable[..., Any],
        parameters: list[tuple[nn.Module, str]],
        constants: list[torch.Tensor],
        blocks: list[nn.Module],
        device: torch.device,
        *,
        last: bool,
    ) -> None:
        super().__init__()
        # Registered so that `parameters()` is the stage's own.
        self.blocks = nn.ModuleList(blocks)
        self.function = function
        self.locations = parameters
        self.constants = constants
        self.device = device
        self.last = last
        self.joining = False

    def forward(self, *received: torch.Tensor, **inputs: torch.Tensor) -> Any:
        weights = [getattr(owner, leaf) for owner, leaf in self.locations]
        out = self.function(*received, *weights, *self.constants, _device=self.device, **inputs)
        if self.last and self.joining:
            out = tuple(value.reshape(1) if value.dim() == 0 else value for value in out)
        return out[0] if self.last and len(out) == 1 else out


def _result(output: torch.Tensor, target: Any) -> torch.Tensor:
    """The loss of a micro-batch: the entry's own result."""
    return output


def _materialize(
    module: CompiledLinnetModule, keep: Callable[[str], bool], device: torch.device
) -> None:
    """Gives the parameters and buffers `keep` accepts zeros on `device`;
    the rest stay on `meta`, holding no memory."""
    for name, parameter in list(module.root.named_parameters(remove_duplicate=False)):
        if keep(name):
            owner, leaf = owner_of(module, name)
            zeros = torch.zeros(parameter.shape, dtype=parameter.dtype, device=device)
            setattr(owner, leaf, nn.Parameter(zeros, requires_grad=False))
    for name, buffer in list(module.root.named_buffers(remove_duplicate=False)):
        if keep(name):
            owner, leaf = owner_of(module, name)
            setattr(owner, leaf, torch.zeros(buffer.shape, dtype=buffer.dtype, device=device))
    module.forget_parameters()


def pipeline(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    entry: str,
    microbatches: int,
    root: str | None = None,
    std_root: str | Path | None = None,
    weights: str | Path | None = None,
    bindings: str | Path | None = None,
    strict: bool = True,
    numerics: str = "fast",
    trainable: bool | str | Sequence[str] = True,
    stages: Sequence[str] | None = None,
    schedule: Schedule = "1f1b",
    group: Any = None,
    device: str | torch.device | None = None,
    cast_dtype: bool = False,
    compile: str | None = None,
) -> Pipeline:
    """Loads this process's stage of `source`'s root block for a pipeline
    over `group` (the default process group), with `microbatches` per step.

    `stages` names the block each stage after the first starts at
    (`["layers.11", "layers.22"]`); by default the blocks are balanced by
    parameter bytes (`linnet.parallel.pipeline_stages`). `schedule` is
    `"1f1b"` (each stage keeps at most as many micro-batches in flight as
    there are stages after it) or `"gpipe"` (every micro-batch's forward,
    then every backward). `device` defaults to the current CUDA device, or
    the CPU without one. `compile` passes each stage function through
    `torch.compile` with that backend. The rest is as `linnet.torch.load`."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise PlanError("a pipeline runs in a torch.distributed job: call init_process_group first")
    if microbatches < 1:
        raise PlanError("a pipeline needs at least one micro-batch")
    if schedule not in ("1f1b", "gpipe"):
        raise PlanError('schedule must be "1f1b" or "gpipe"')
    group = group if group is not None else dist.group.WORLD
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    device = torch.device(device)
    plan = compile_plan(source, root=root, std_root=std_root, numerics=numerics)
    module = CompiledLinnetModule(
        plan,
        generics,
        torch.device("meta"),
        source=Path(source),
        std_root=std_root,
        numerics=numerics,
        backend=None,
    )
    found = units_of(module)
    names = [unit.path for unit in found]
    try:
        assigned = (
            assign_stages(names, stages)
            if stages is not None
            else pipeline_stages([unit.bytes for unit in found], size)
        )
    except ValueError as error:
        raise PlanError(str(error)) from None
    if max(assigned, default=0) + 1 != size:
        raise PlanError(f"{max(assigned, default=0) + 1} stages for a group of {size} processes")
    for name, _ in module.root.named_parameters(remove_duplicate=False):
        if stage_of_path(names, assigned, name) is None:
            raise PlanError(f"`{name}` belongs to no block; a pipeline splits the root's blocks")

    def mine(path: str) -> bool:
        return stage_of_path(names, assigned, path) == rank

    _materialize(module, mine, device)
    bound: dict[str, str] = {}
    if weights is not None:
        bound = bind_weights(
            module, weights, bindings, strict=strict, cast_dtype=cast_dtype, only=mine
        )
    # Parameters read from one checkpoint tensor: one parameter within a
    # stage, and summed gradients across stages.
    by_tensor = paths_by_tensor(bound)
    ties: list[tuple[list[str], Any]] = []
    for tensor in sorted(by_tensor):
        paths = by_tensor[tensor]
        if len(paths) < 2:
            continue
        local = [p for p in paths if mine(p)]
        if len(local) > 1:
            first_owner, first_leaf = owner_of(module, local[0])
            kept = getattr(first_owner, first_leaf)
            for path in local[1:]:
                owner, leaf = owner_of(module, path)
                setattr(owner, leaf, kept)
        holders = sorted({stage_of_path(names, assigned, p) or 0 for p in paths})
        if len(holders) > 1:
            # Every process makes every group, in the same order.
            ranks = [dist.get_global_rank(group, s) for s in holders]
            ties.append((paths, dist.new_group(ranks)))
    module.forget_parameters()
    if trainable:
        module.set_trainable(trainable)
        # Parameters of other stages hold no memory; never train them here.
        for name, parameter in module.root.named_parameters(remove_duplicate=False):
            if not mine(name):
                parameter.requires_grad_(False)
    return Pipeline(
        module,
        entry,
        generics,
        units=names,
        assigned=assigned,
        microbatches=microbatches,
        schedule=schedule,
        group=group,
        device=device,
        ties=ties,
        compile=compile,
    )


__all__ = ["Pipeline", "pipeline"]

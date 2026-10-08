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
from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from ..parallel import layout_stages, stage_of_path, stage_starts
from ..plan import PlanError, compile_plan
from ..results import Result
from ..weights import paths_by_tensor
from .compiled import CompiledLinnetModule
from .fsdp import Held, sharded
from .module import bind_weights, owner_of
from .placement import units_of
from .stages import Split, Stage, split

if TYPE_CHECKING:
    from typing import TypeAlias

    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh

    # What the entry returns on the last stage: one tensor, or a tuple of them
    # when it has several results.
    # A stage function of the generated source: its results as a tuple.
    StageFunction: TypeAlias = Callable[..., tuple[torch.Tensor, ...]]

Schedule = Literal["1f1b", "gpipe"]


class _Runner(Protocol):
    """A `torch.distributed.pipelining` schedule, as a `Pipeline` drives it."""

    def step(
        self,
        *,
        kwarg_mbs: list[dict[str, torch.Tensor]],
        target_mbs: list[torch.Tensor] | None,
        losses: list[torch.Tensor] | None,
        return_outputs: bool,
    ) -> object: ...

    def eval(
        self,
        *,
        kwarg_mbs: list[dict[str, torch.Tensor]],
        target_mbs: list[torch.Tensor] | None,
    ) -> Result | None: ...


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
        group: ProcessGroup,
        device: torch.device,
        ties: Sequence[tuple[list[str], ProcessGroup | None]],
        compile: str | None,
        data_parallel: ProcessGroup | None = None,
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
        # The processes training this stage on batches of their own, when
        # its weights are sharded across them.
        self.data_parallel = data_parallel
        self._built: dict[tuple[tuple[tuple[int, ...], torch.dtype], ...], _Built] = {}

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
        # The gradients summed with other processes' below hold this step's
        # alone then; what they held before is added back after.
        aside: list[tuple[nn.Parameter, torch.Tensor]] = []
        for parameter in self._summed():
            if parameter.grad is not None:
                aside.append((parameter, parameter.grad))
                parameter.grad = None
        held = self._hold(built, trains=True)
        try:
            built.schedule.step(
                kwarg_mbs=self._microbatches(built, inputs),
                target_mbs=self._targets(),
                losses=losses,
                return_outputs=False,
            )
            if held is not None:
                held.finish()
        finally:
            built.module.held = None
        self._sum_tied_gradients()
        self._sum_whole_gradients()
        for parameter, grad in aside:
            if parameter.grad is not None:
                grad.add_(parameter.grad)
            parameter.grad = grad
        if losses is None:
            return None
        return torch.stack([loss.detach().float() for loss in losses]).sum()

    def run(self, *inputs: torch.Tensor) -> Result | None:
        """The entry's forward pass over the micro-batches, without
        gradients. On the last stage, each result joined along its first
        axis (a scalar result: one value per micro-batch); None elsewhere."""
        built = self._build(inputs)
        # The last stage hands each scalar result over as one value, so that
        # the schedule joins the micro-batches' results along the first axis.
        built.module.joining = True
        try:
            with torch.no_grad():
                self._hold(built, trains=False)
                merged: Result | None = built.schedule.eval(
                    kwarg_mbs=self._microbatches(built, inputs),
                    target_mbs=self._targets(),
                )
        finally:
            built.module.joining = False
            built.module.held = None
        return merged if self.is_last else None

    def _hold(self, built: _Built, trains: bool) -> Held | None:
        """With the stage's weights sharded, gathers each whole for the
        step's micro-batches to share (`linnet.torch.fsdp.Held`)."""
        if not self.module.fully_sharded:
            return None
        held = Held(trains)
        for (owner, leaf), dtype in zip(built.module.locations, built.module.dtypes, strict=True):
            held.whole(getattr(owner, leaf), dtype)
        built.module.held = held
        return held

    def _targets(self) -> list[torch.Tensor] | None:
        """What the schedule hands the loss on the last stage: nothing it
        reads (the entry's result is the loss), but a tensor it requires."""
        if not self.is_last:
            return None
        return [torch.zeros((), device=self.device)] * self.microbatches

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
        # The root's generics are the module's own; the entry's come from the inputs.
        bindings = self.module.bindings_for(self.entry, one, {})
        source = self.module.source_for(self.entry, bindings, trains=True, prepare=False)
        pieces = split(source, self.stage_of, self.stages)
        generated = self.module.import_source(
            pieces.source, f"{self.entry}_stages_{len(self._built)}"
        )
        if self.module.shard_group is not None and hasattr(generated, "_GROUP"):
            generated._GROUP = self.module.shard_group
        stage = pieces.stages[self.stage]
        with torch.no_grad():
            computed = generated.constants(self.device) if pieces.constants else ()
            constants = dict(zip(pieces.constants, computed, strict=True))
        function: StageFunction = getattr(generated, stage.name)
        if self.compile is not None:
            function = torch.compile(function, backend=self.compile)
        blocks = [
            self.module.root.get_submodule(unit)
            for unit, stage_index in zip(self.units, self.assigned, strict=True)
            if stage_index == self.stage
        ]
        locations: list[tuple[nn.Module, str]] = []
        dtypes: list[torch.dtype] = []
        for index in stage.parameters:
            path = pieces.parameters[index]
            owner, leaf = owner_of(self.module, path)
            locations.append((owner, leaf))
            parameter: torch.Tensor = getattr(owner, leaf)
            dtypes.append(self.module.gathered_dtypes.get(path, parameter.dtype))
        wrapped = _StageModule(
            function,
            locations,
            dtypes,
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
            grad = parameter.grad
            if isinstance(grad, DTensor):
                grad = grad.to_local()  # a sharded weight's: this process's part
            dist.all_reduce(grad, group=group)

    def _summed(self) -> list[nn.Parameter]:
        """This stage's parameters whose gradients a step sums with other
        processes': those shared with other stages, and those its sharding
        keeps whole."""
        found: dict[int, nn.Parameter] = {}
        for paths, _ in self.ties:
            mine = [p for p in paths if self.stage_of(p) == self.stage]
            if mine:
                owner, leaf = owner_of(self.module, mine[0])
                parameter = getattr(owner, leaf)
                if isinstance(parameter, nn.Parameter) and parameter.requires_grad:
                    found[id(parameter)] = parameter
        if self.data_parallel is not None:
            for _, parameter in self.named_parameters():
                if not sharded(parameter) and parameter.requires_grad:
                    found[id(parameter)] = parameter
        return list(found.values())

    def _sum_whole_gradients(self) -> None:
        """Sums, across the processes training this stage, the gradients of
        the parameters its sharding keeps whole (adapters)."""
        import torch.distributed as dist

        if self.data_parallel is None:
            return
        for _, parameter in self.named_parameters():
            if sharded(parameter) or not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, group=self.data_parallel)


@dataclass
class _Built:
    split: Split
    stage: Stage
    schedule: _Runner
    module: _StageModule


class _StageModule(nn.Module):
    """A stage function as the module `PipelineStage` runs: the values that
    cross into it positionally, the entry's inputs by keyword."""

    def __init__(
        self,
        function: StageFunction,
        parameters: list[tuple[nn.Module, str]],
        dtypes: list[torch.dtype],
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
        # The dtype each parameter is read in: a sharded one's whole.
        self.dtypes = dtypes
        self.constants = constants
        self.device = device
        self.last = last
        self.joining = False
        # Sharded weights gathered whole for the current step.
        self.held: Held | None = None

    def forward(self, *received: torch.Tensor, **inputs: torch.Tensor) -> Result:
        weights = [getattr(owner, leaf) for owner, leaf in self.locations]
        if self.held is not None:
            weights = [
                self.held.whole(weight, dtype)
                for weight, dtype in zip(weights, self.dtypes, strict=True)
            ]
        out = self.function(*received, *weights, *self.constants, _device=self.device, **inputs)
        if self.last and self.joining:
            out = tuple(value.reshape(1) if value.dim() == 0 else value for value in out)
        return out[0] if self.last and len(out) == 1 else out


def _result(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    group: ProcessGroup | None = None,
    tensor_parallel: DeviceMesh | None = None,
    data_parallel: DeviceMesh | None = None,
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
    `torch.compile` with that backend. The rest is as `linnet.torch.load`.

    `tensor_parallel`, a one-dimensional `DeviceMesh`, splits each stage
    over its processes, for a model that says how it splits (a `Shards`
    generic): each binds its own part of its stage's weights. `group` then
    names this process's pipeline, one process of each stage, as the
    `"pp"` dimension of a two-dimensional mesh does. Split stages train when
    the entry passes each split computation its input through
    `std.nn.parallel::shared`, as the zoo's decoders do; the processes of a
    stage then take the same micro-batches.

    `data_parallel`, a one-dimensional `DeviceMesh`, shards each stage's
    weights over the processes that train the same stage on batches of
    their own (`linnet.torch.fsdp.fully_shard`): their gradients are summed.
    `group` names this process's pipeline as with `tensor_parallel`."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise PlanError("a pipeline runs in a torch.distributed job: call init_process_group first")
    if microbatches < 1:
        raise PlanError("a pipeline needs at least one micro-batch")

    if schedule not in ("1f1b", "gpipe"):
        raise PlanError('schedule must be "1f1b" or "gpipe"')
    if (tensor_parallel is not None or data_parallel is not None) and group is None:
        raise PlanError("a pipeline over split or sharded stages needs group=, its own processes")
    if tensor_parallel is not None and data_parallel is not None:
        raise PlanError("a pipeline's stages are split or sharded, not both")
    # Set once the job is initialized, as it is here.
    group = group if group is not None else cast("ProcessGroup", dist.group.WORLD)
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    if microbatches < size:
        raise PlanError(
            f"a pipeline of {size} stages needs at least as many micro-batches, not {microbatches}"
        )
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    device = torch.device(device)
    plan = compile_plan(source, root=root, std_root=std_root, numerics=numerics)
    shard: tuple[int, int] | None = None
    if tensor_parallel is not None:
        if not any(g.name == "Shards" for g in plan.root.generics):
            raise PlanError(
                "a pipeline splits its stages only for a model that says how it splits "
                "(a `Shards` generic)"
            )
        generics = {**generics, "Shards": tensor_parallel.size()}
        shard = (tensor_parallel.get_local_rank(), tensor_parallel.size())
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
        assigned = layout_stages(names, [unit.bytes for unit in found], size, stages)
    except ValueError as error:
        raise PlanError(f"{error} (one per process of the group)") from None
    for name, _ in module.root.named_parameters(remove_duplicate=False):
        if stage_of_path(names, assigned, name) is None:
            raise PlanError(f"`{name}` belongs to no block; a pipeline splits the root's blocks")

    def mine(path: str) -> bool:
        return stage_of_path(names, assigned, path) == rank

    _materialize(module, mine, device)
    bound: dict[str, str] = {}
    if weights is not None:
        bound = bind_weights(
            module, weights, bindings, strict=strict, cast_dtype=cast_dtype, only=mine, shard=shard
        )
    # Parameters read from one checkpoint tensor: one parameter within a
    # stage, and summed gradients across stages.
    by_tensor = paths_by_tensor(bound)
    ties: list[tuple[list[str], ProcessGroup | None]] = []
    # Every process makes every group, in the same order: with the stages
    # split or sharded, each place in a stage has a pipeline of its own.
    pipelines = [[dist.get_global_rank(group, s) for s in range(size)]]
    if tensor_parallel is not None or data_parallel is not None:
        everyone: list[list[int] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(everyone, pipelines[0])
        # Every place now holds that process's pipeline.
        pipelines = [list(p) for p in sorted({tuple(p) for p in cast(list[list[int]], everyone)})]
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
            own: ProcessGroup | None = None
            for ranks_of in pipelines:
                ranks = [ranks_of[s] for s in holders]
                made = dist.new_group(ranks)
                if dist.get_rank() in ranks:
                    own = cast("ProcessGroup", made)  # a member's is the group itself
            ties.append((paths, own))
    module.forget_parameters()
    if tensor_parallel is not None:
        from .collectives import prepare

        module.shard_group = tensor_parallel.get_group()
        # A collective over the stage's processes, once.
        prepare(module.shard_group)
    if trainable:
        module.set_trainable(trainable)
        # Parameters of other stages hold no memory; never train them here.
        for name, parameter in module.root.named_parameters(remove_duplicate=False):
            if not mine(name):
                parameter.requires_grad_(False)
    # Sharded after what trains is known: those parts are kept in f32. A
    # weight two stages share is cut the same way on both, so each place in
    # a stage sums its part with its own pipeline's.
    if data_parallel is not None:
        from .fsdp import fully_shard

        fully_shard(module, data_parallel, only=mine)
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
        data_parallel=data_parallel.get_group() if data_parallel is not None else None,
    )


__all__ = ["Pipeline", "pipeline"]

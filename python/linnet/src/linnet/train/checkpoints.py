"""Checkpoints to resume training from: the parameters being trained, the
optimizer's state, the step count, and the learning-rate schedule.

`save_checkpoint(directory, model, optimizer, step=...)` writes them to
`directory/step-<step>` with `torch.distributed.checkpoint`; under
`torch.distributed` every process calls it and writes its own part, the
parts of a model split by `fully_shard` included. `load_checkpoint` reads
the latest complete one back into a model and optimizer made as before and
returns its step (0 when there is none). Weights that do not train (a LoRA
run's base model) are not saved: load the model as the run did.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportPrivateImportUsage=false

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

PREFIX = "step-"


def save_checkpoint(
    directory: str | Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    schedule: Any = None,
    keep: int | None = 2,
) -> Path:
    """Writes a checkpoint of `step` under `directory`, then removes all
    but the `keep` latest (None keeps every one). Returns its path."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    root = Path(directory)
    target = root / f"{PREFIX}{step:08d}"
    dcp.save(_state(model, optimizer, schedule, step), checkpoint_id=str(target))
    first = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    if first and keep is not None:
        for old in _complete(root)[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
    return target


def load_checkpoint(
    directory: str | Path, model: Any, optimizer: torch.optim.Optimizer, *, schedule: Any = None
) -> int:
    """Reads the latest complete checkpoint under `directory` into `model`,
    `optimizer` and `schedule`; returns its step, or 0 when there is none."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    found = _complete(Path(directory))
    if not found:
        return 0
    state = _state(model, optimizer, schedule, 0)
    dcp.load(state, checkpoint_id=str(found[-1]))
    set_optimizer_state_dict(model, optimizer, state["optimizer"])
    if schedule is not None:
        schedule.load_state_dict(state["progress"]["schedule"])
    return int(state["progress"]["step"])


def _state(
    model: Any, optimizer: torch.optim.Optimizer, schedule: Any, step: int
) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    if type(optimizer).__name__ == "ZeroRedundancyOptimizer":
        raise ValueError(
            "checkpoints take an optimizer whose state is whole or split with its parameters "
            "(fully_shard), not ZeroRedundancyOptimizer"
        )
    # Detached views: loading writes into the parameters' own memory.
    trained = {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}
    return {
        "model": trained,
        "optimizer": get_optimizer_state_dict(model, optimizer),
        "progress": {
            "step": step,
            "schedule": schedule.state_dict() if schedule is not None else None,
        },
    }


def _complete(root: Path) -> list[Path]:
    """The checkpoints under `root` that finished writing, oldest first."""
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.name.startswith(PREFIX) and (path / ".metadata").exists()
    )


__all__ = ["load_checkpoint", "save_checkpoint"]

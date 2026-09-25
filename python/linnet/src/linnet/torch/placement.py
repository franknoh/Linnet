"""Placing a model across devices: several GPUs, and the host when the GPUs
are not enough.

A model is divided into *units*: each sub-block of the root, and each
element of a sub-block array (`layers.0`, `layers.1`, ...). Every tensor's
size is known from the manifest before any weight is read, so the plan is
made up front: units fill the first GPU in the order they are declared, then
the next, and whatever does not fit stays on the host and is streamed to a
GPU one unit at a time ("offloaded").

The plan becomes `--place` and `--offload` arguments to `linnet torch`, which
compiles the placement into the generated source: every operation runs on
its unit's device, a value crossing to another device is transferred once,
and an offloaded unit's parameters are copied in when it starts and dropped
when it returns. Nothing is decided at run time, so the placed model is the
same straight-line code as an unplaced one, with a few `.to()` calls in it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from ..plan import PlanError
from .module import BlockModule, LinnetModule

# Room left on every GPU for what the manifest does not count: activations,
# attention workspace, the caching allocator's slack.
ACTIVATION_RESERVE = 0.10
MINIMUM_RESERVE = 1 << 30

_SIZE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGT]?i?B?)\s*$", re.IGNORECASE)
_UNITS = {
    "": 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 1 << 10,
    "mib": 1 << 20,
    "gib": 1 << 30,
    "tib": 1 << 40,
}


def parse_size(size: int | str) -> int:
    """Bytes from `12345`, `"20GiB"`, or `"20GB"`."""
    if isinstance(size, int):
        return size
    match = _SIZE.match(size)
    if match is None or match.group(2).lower() not in _UNITS:
        raise PlanError(f"cannot read the memory size {size!r}")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])


@dataclass(frozen=True)
class Unit:
    """A placeable piece of the model: its path, module, and weight bytes."""

    path: str
    module: nn.Module
    bytes: int


@dataclass(frozen=True)
class Placement:
    """Where each unit runs, and which units stream their weights in."""

    devices: tuple[torch.device, ...]  # slot -> the device operations run on
    slots: Mapping[str, int]  # unit path -> slot
    offloaded: tuple[str, ...]  # units whose parameters stay on the host

    @property
    def trivial(self) -> bool:
        """One device and nothing offloaded: an ordinary model."""
        return len(self.devices) <= 1 and not self.offloaded

    def flags(self) -> list[str]:
        """The `linnet torch` arguments that compile this placement."""
        flags: list[str] = []
        for path, slot in self.slots.items():
            if slot != 0:
                flags += ["--place", f"{path}={slot}"]
        for path in self.offloaded:
            flags += ["--offload", path]
        return flags

    def describe(self) -> str:
        """One line per device, for a person deciding whether it is right."""
        lines: list[str] = []
        for slot, device in enumerate(self.devices):
            here = [
                path for path, s in self.slots.items() if s == slot and path not in self.offloaded
            ]
            lines.append(f"{device}: {_ranges(here) or '(nothing resident)'}")
        if self.offloaded:
            lines.append(f"host, streamed in: {_ranges(list(self.offloaded))}")
        return "\n".join(lines)


def units_of(model: LinnetModule) -> list[Unit]:
    """The model's units in declaration order, which is execution order for
    every model in practice: embeddings, then the layers, then the head."""
    found: list[Unit] = []
    for name, child in model.root.named_children():
        if isinstance(child, BlockModule):
            found.append(Unit(name, child, _weight_bytes(child)))
        elif isinstance(child, nn.ModuleList):
            for index, element in enumerate(child):
                found.append(Unit(f"{name}.{index}", element, _weight_bytes(element)))
    return found


def plan(
    model: LinnetModule,
    *,
    max_memory: Mapping[int | str, int | str] | None = None,
    offload: bool = True,
) -> Placement:
    """Fills the visible GPUs in order and offloads what remains.

    `max_memory` caps what a device may hold, by index (`{0: "20GiB"}`); by
    default each GPU may use what is free on it now, less a reserve for
    activations. Offloaded units run on the last GPU, which keeps room for
    the largest of them to be streamed in.
    """
    count = torch.cuda.device_count()
    if count == 0:
        raise PlanError('device_map="auto" needs a CUDA device; use device="cpu" instead')
    budgets = [_budget(index, max_memory) for index in range(count)]
    found = units_of(model)
    # Parameters of the root itself run in the root's code, on slot 0.
    budgets[0] -= _weight_bytes(model.root, recurse=False)

    # The room kept for streaming depends on what is offloaded, and what is
    # offloaded depends on that room: iterate to the fixed point, which a
    # few rounds reach since each can only push more units out.
    offloaded: list[str] = []
    slots: dict[str, int] = {}
    for _ in range(len(found) + 1):
        slots, now = _fill(found, budgets, _streaming_room(found, offloaded))
        if now == offloaded:
            break
        offloaded = now
    if offloaded and not offload:
        total = sum(unit.bytes for unit in found)
        raise PlanError(
            f"the model's {total / 2**30:.1f} GiB of weights do not fit the GPUs, and offload=False"
        )
    used = sorted({slots[unit.path] for unit in found} | {0})
    devices = tuple(torch.device("cuda", index) for index in range(used[-1] + 1))
    return Placement(devices, slots, tuple(offloaded))


def from_map(model: LinnetModule, device_map: Mapping[str, str | int]) -> Placement:
    """A placement written by hand: unit path -> `"cuda:1"`, `1`, or `"cpu"`
    (offloaded, and run on the highest-numbered GPU in the map). Units the
    map leaves out run on the first GPU."""
    known = {unit.path for unit in units_of(model)}
    gpus: set[int] = {0}
    targets: dict[str, int | None] = {}
    for path, target in device_map.items():
        if path not in known:
            raise PlanError(f"device_map names `{path}`, which is not a unit of this model")
        if target == "cpu":
            targets[path] = None
            continue
        device = torch.device(f"cuda:{target}" if isinstance(target, int) else target)
        # The stubs say `int`, but a bare `"cuda"` has no index at run time.
        index = cast("int | None", device.index)
        if device.type != "cuda" or index is None:
            raise PlanError(f"device_map sends `{path}` to {target!r}; use `cuda:N` or `cpu`")
        gpus.add(index)
        targets[path] = index
    last = max(gpus)
    slots = {path: 0 for path in known}
    offloaded: list[str] = []
    for path, index in targets.items():
        if index is None:
            slots[path] = last
            offloaded.append(path)
        else:
            slots[path] = index
    devices = tuple(torch.device("cuda", index) for index in range(last + 1))
    return Placement(devices, slots, tuple(offloaded))


def apply(model: LinnetModule, placement: Placement) -> None:
    """Moves every unit's tensors where the placement says they live.

    A resident unit's parameters and state go to its device. An offloaded
    unit's parameters stay on the host, pinned so the copy in runs
    asynchronously; its state (a key/value cache) lives on the device the
    unit runs on, since it is read and written every step.
    """
    root_device = placement.devices[0]
    _move(model.root, root_device, host=False)
    offloaded = set(placement.offloaded)
    for unit in units_of(model):
        device = placement.devices[placement.slots[unit.path]]
        for module in unit.module.modules():
            _move(module, device, host=unit.path in offloaded)
    model.interpreter.device = root_device


def _move(module: nn.Module, device: torch.device, *, host: bool) -> None:
    """One module's own tensors: parameters to `device` (or pinned host
    memory when offloaded), buffers always to `device`."""
    for parameter in module.parameters(recurse=False):
        if host:
            on_host = parameter.data.to("cpu")
            parameter.data = on_host.pin_memory() if torch.cuda.is_available() else on_host
        else:
            parameter.data = parameter.data.to(device)
    for name, buffer in list(module.named_buffers(recurse=False)):
        setattr(module, name, buffer.to(device))


def _fill(
    units: list[Unit], budgets: list[int], streaming: int
) -> tuple[dict[str, int], list[str]]:
    """Units into GPUs in order, keeping `streaming` bytes free on the last
    GPU for an offloaded unit's copy; what is left over is offloaded."""
    last = len(budgets) - 1
    room = list(budgets)
    room[last] -= streaming
    slots: dict[str, int] = {}
    offloaded: list[str] = []
    gpu = 0
    for unit in units:
        while gpu <= last and unit.bytes > room[gpu]:
            gpu += 1
        if gpu <= last:
            slots[unit.path] = gpu
            room[gpu] -= unit.bytes
        else:
            slots[unit.path] = last
            offloaded.append(unit.path)
    return slots, offloaded


def _streaming_room(units: list[Unit], offloaded: list[str]) -> int:
    streamed = [unit.bytes for unit in units if unit.path in offloaded]
    return max(streamed, default=0)


def _budget(index: int, max_memory: Mapping[int | str, int | str] | None) -> int:
    free, total = torch.cuda.mem_get_info(index)
    if max_memory is not None and (index in max_memory or f"cuda:{index}" in max_memory):
        cap = max_memory.get(index, max_memory.get(f"cuda:{index}"))
        assert cap is not None
        return min(parse_size(cap), free)
    reserve = max(MINIMUM_RESERVE, int(total * ACTIVATION_RESERVE))
    return max(0, free - reserve)


def _weight_bytes(module: nn.Module, recurse: bool = True) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters(recurse=recurse))


def _ranges(paths: list[str]) -> str:
    """`layers.0, layers.1, layers.2, head` as `layers.0-2, head`."""
    out: list[str] = []
    run: tuple[str, int, int] | None = None
    for path in paths:
        stem, _, index = path.rpartition(".")
        if stem and index.isdigit():
            number = int(index)
            if run is not None and run[0] == stem and run[2] + 1 == number:
                run = (stem, run[1], number)
                continue
            if run is not None:
                out.append(_run_text(run))
            run = (stem, number, number)
            continue
        if run is not None:
            out.append(_run_text(run))
            run = None
        out.append(path)
    if run is not None:
        out.append(_run_text(run))
    return ", ".join(out)


def _run_text(run: tuple[str, int, int]) -> str:
    stem, first, last = run
    return f"{stem}.{first}" if first == last else f"{stem}.{first}-{last}"

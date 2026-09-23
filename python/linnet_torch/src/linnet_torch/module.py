"""A `torch.nn.Module` built from a plan.

The module hierarchy mirrors the block hierarchy of the Linnet source: every
`sub` is a child module (a `ModuleList` for arrays), every `param` an
`nn.Parameter`, every `buffer` a registered buffer, so that `state_dict()` uses
the canonical Linnet parameter paths. Entries become methods; `forward` calls
the entry named `forward`, or the only entry.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .interp import BlockInstance, Interpreter
from .plan import TORCH_DTYPES, Env, Plan, PlanError


class BlockModule(nn.Module):
    """One instantiated block. Parameters are created lazily as zeros of the
    right shape and dtype so that a checkpoint can replace them."""

    def __init__(self, plan: Plan, name: str, env: Env, device: torch.device) -> None:
        super().__init__()
        self.block_name = name
        self.env = env
        self.optional_params: set[str] = set()
        self.absent_params: set[str] = set()
        self.state_names: set[str] = set()
        definition = plan.blocks[name]
        for constraint in definition["constraints"]:
            if not env.relation_holds(constraint):
                raise PlanError(f"a `where` constraint of block `{name}` does not hold")
        for member in definition["members"]:
            member_type = member["type"]
            if member["kind"] == "sub":
                self.add_module(member["name"], _instantiate_sub(plan, member_type, env, device))
                continue
            is_optional = member_type["kind"] == "optional"
            tensor_type = member_type["inner"] if is_optional else member_type
            shape = env.shape(tensor_type["shape"])
            dtype = env.dtype(tensor_type["dtype"])
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            if is_optional:
                self.optional_params.add(member["name"])
                self.absent_params.add(member["name"])
            if member["kind"] == "param":
                self.register_parameter(member["name"], nn.Parameter(tensor, requires_grad=False))
            elif member["kind"] == "state":
                # Execution state: starts at zero, kept between calls, never
                # part of the weights.
                self.state_names.add(member["name"])
                self.register_buffer(member["name"], tensor, persistent=False)
            else:
                self.register_buffer(member["name"], tensor)

    def _get_name(self) -> str:
        return self.block_name  # `print(model)` shows the Linnet block names

    def extra_repr(self) -> str:
        """The block's own tensors, as `name=dtype[shape]`; `?` marks an
        optional parameter and `state` an execution-state member."""
        parts: list[str] = []
        for name, parameter in self.named_parameters(recurse=False):
            optional = "?" if name in self.optional_params else ""
            parts.append(f"{name}={_tensor_repr(parameter)}{optional}")
        for name, buffer in self.named_buffers(recurse=False):
            kind = " state" if name in self.state_names else ""
            parts.append(f"{name}={_tensor_repr(buffer)}{kind}")
        return ", ".join(parts)

    def instance(self) -> BlockInstance:
        """The interpreter's view of this module, sharing its tensors."""
        params: dict[str, torch.Tensor | None] = {}
        states: dict[str, torch.Tensor] = {}
        for name, parameter in self.named_parameters(recurse=False):
            params[name] = None if name in self.absent_params else parameter
        for name, buffer in self.named_buffers(recurse=False):
            if name in self.state_names:
                states[name] = buffer
            else:
                params[name] = buffer
        subs: dict[str, BlockInstance | list[BlockInstance]] = {}
        for name, child in self.named_children():
            if isinstance(child, BlockModule):
                subs[name] = child.instance()
            elif isinstance(child, nn.ModuleList):
                subs[name] = [
                    element.instance() for element in child if isinstance(element, BlockModule)
                ]
        return BlockInstance(self.block_name, self.env, params, subs, states, self._write_state)

    def _write_state(self, name: str, value: torch.Tensor) -> None:
        # Rebinding (not `copy_`) so values read earlier in the call keep
        # their contents.
        setattr(self, name, value.detach())

    def reset_state(self) -> None:
        """Zeroes this block's `state` members and those of its sub-blocks."""
        for name in self.state_names:
            setattr(self, name, torch.zeros_like(getattr(self, name)))
        for child in self.modules():
            if child is not self and isinstance(child, BlockModule):
                for name in child.state_names:
                    setattr(child, name, torch.zeros_like(getattr(child, name)))


def _tensor_repr(tensor: torch.Tensor) -> str:
    dtype = str(tensor.dtype).removeprefix("torch.")
    return f"{dtype}{list(tensor.shape)}"


def _block_env(plan: Plan, block_type: dict[str, Any], env: Env) -> Env:
    definition = plan.blocks[block_type["name"]]
    inner = Env()
    for generic, arg in zip(definition["generics"], block_type["args"], strict=True):
        if generic["kind"] == "dim":
            inner.dims[int(generic["sym"])] = env.dim(arg["dim"])
        elif generic["kind"] == "shape":
            inner.packs[int(generic["sym"])] = env.shape(arg["shape"])
        else:
            inner.dtypes[int(generic["var"])] = env.dtype_name(arg["dtype"])
    return inner


def _instantiate_sub(
    plan: Plan, member_type: dict[str, Any], env: Env, device: torch.device
) -> nn.Module:
    if member_type["kind"] == "array":
        count = env.dim(member_type["length"])
        element = member_type["element"]
        return nn.ModuleList(
            [
                BlockModule(plan, element["name"], _block_env(plan, element, env), device)
                for _ in range(count)
            ]
        )
    return BlockModule(plan, member_type["name"], _block_env(plan, member_type, env), device)


class LinnetModule(nn.Module):
    """The root block as a PyTorch module."""

    def __init__(self, plan: Plan, generics: Mapping[str, int | str], device: torch.device) -> None:
        super().__init__()
        self.plan = plan
        root = plan.root
        env = Env()
        for generic in root["generics"]:
            if generic["name"] not in generics:
                if "default" not in generic:
                    raise PlanError(
                        f"generic parameter `{generic['name']}` of `{root['name']}` needs a value"
                    )
                default = generic["default"]
                if generic["kind"] == "dim":
                    env.dims[int(generic["sym"])] = env.dim(default["dim"])
                elif generic["kind"] == "dtype":
                    env.dtypes[int(generic["var"])] = env.dtype_name(default["dtype"])
                continue
            value = generics[generic["name"]]
            if generic["kind"] == "dim":
                if not isinstance(value, int):
                    raise PlanError(f"`{generic['name']}` is a dimension; give an integer")
                env.dims[int(generic["sym"])] = value
            elif generic["kind"] == "dtype":
                if not isinstance(value, str) or value not in TORCH_DTYPES:
                    raise PlanError(
                        f"`{generic['name']}` is a dtype; give one of {sorted(TORCH_DTYPES)}"
                    )
                env.dtypes[int(generic["var"])] = value
            else:
                raise PlanError("shape-pack generics on a root block are not supported")
        for name in generics:
            if all(generic["name"] != name for generic in root["generics"]):
                raise PlanError(f"`{root['name']}` has no generic parameter `{name}`")
        self.root = BlockModule(plan, root["name"], env, device)
        self.interpreter = Interpreter(plan, device)
        self.entries = {
            function["name"].rsplit(".", 1)[1]: function
            for function in plan.entries_of(root["name"])
        }
        if not self.entries:
            raise PlanError(f"block `{root['name']}` has no entry")
        for name in self.entries:
            setattr(self, name, self._entry_callable(name))

    def _entry_callable(self, name: str) -> Callable[..., Any]:
        def run(*inputs: torch.Tensor) -> Any:
            return self.run_entry(name, list(inputs))

        run.__name__ = name
        return run

    def run_entry(
        self,
        name: str,
        inputs: list[torch.Tensor],
        generics: Mapping[str, int | str] | None = None,
    ) -> Any:
        """Runs the entry `name`. Its generic parameters are bound from the
        input shapes, or from `generics` by name for those the inputs do not
        determine (an output length such as `Steps`)."""
        function = self.entries[name]
        params = function["body"]["args"][1:]  # after `self`
        if len(params) != len(inputs):
            raise PlanError(f"entry `{name}` takes {len(params)} inputs, got {len(inputs)}")
        env = Env(dict(self.root.env.dims), dict(self.root.env.packs), dict(self.root.env.dtypes))
        bind_generics(env, function["generics"], generics or {})
        for param, value in zip(params, inputs, strict=True):
            bind_input(env, param, value)
        for generic in function["generics"]:
            key = int(generic.get("sym", generic.get("var", -1)))
            bound = (
                key in env.dims
                if generic["kind"] == "dim"
                else key in env.packs
                if generic["kind"] == "shape"
                else key in env.dtypes
            )
            if not bound:
                raise PlanError(
                    f"cannot determine `{generic['name']}` of entry `{name}` from its inputs"
                )
        return self.interpreter.call(function, env, [self.root.instance(), *inputs])

    def forward(self, *inputs: torch.Tensor) -> Any:
        name = "forward" if "forward" in self.entries else next(iter(self.entries))
        return self.run_entry(name, list(inputs))

    def reset_state(self) -> None:
        """Zeroes every `state` member (a KV cache, for instance) so the next
        entry call starts fresh."""
        self.root.reset_state()

    def state_paths(self) -> list[str]:
        """The `state` members by parameter path, in manifest order."""
        return [entry["path"] for entry in self.plan.manifest if entry["kind"] == "state"]


def bind_generics(env: Env, declared: list[dict[str, Any]], given: Mapping[str, int | str]) -> None:
    """Binds an entry's generics that are given explicitly by name."""
    names = {str(generic["name"]) for generic in declared}
    for name in given:
        if name not in names:
            raise PlanError(f"the entry has no generic parameter `{name}`")
    for generic in declared:
        name = str(generic["name"])
        if name not in given:
            continue
        value = given[name]
        if generic["kind"] == "dim":
            if not isinstance(value, int):
                raise PlanError(f"`{name}` is a dimension; give an integer")
            env.dims[int(generic["sym"])] = value
        elif generic["kind"] == "dtype":
            env.dtypes[int(generic["var"])] = str(value)
        else:
            raise PlanError(f"`{name}` is a shape pack and cannot be given by name")


def bind_input(env: Env, param: dict[str, Any], value: torch.Tensor) -> None:
    """Binds the generic dimensions of an entry from an input's shape and
    checks the rest."""
    param_type = param["type"]
    name = param["name"]
    if param_type["kind"] == "scalar":
        if value.dim() != 0:
            raise PlanError(f"input `{name}` must be a scalar")
        return
    if param_type["kind"] != "tensor":
        raise PlanError(f"input `{name}` has a type that cannot be passed from PyTorch")
    expected_dtype = env.dtype(param_type["dtype"])
    if value.dtype != expected_dtype:
        raise PlanError(f"input `{name}` has dtype {value.dtype}, expected {expected_dtype}")
    units: list[dict[str, Any] | int] = param_type["shape"]
    packs = [i for i, unit in enumerate(units) if isinstance(unit, dict) and "pack" in unit]
    if len(packs) > 1:
        raise PlanError(f"input `{name}` has more than one shape pack")
    fixed = len(units) - len(packs)
    if (packs and value.dim() < fixed) or (not packs and value.dim() != fixed):
        raise PlanError(
            f"input `{name}` has rank {value.dim()}, expected {'at least ' if packs else ''}{fixed}"
        )
    actual = list(value.shape)
    pack_width = value.dim() - fixed
    at = 0
    for unit in units:
        if isinstance(unit, dict) and "pack" in unit:
            symbol = int(unit["pack"])
            sizes = actual[at : at + pack_width]
            if env.packs.setdefault(symbol, sizes) != sizes:
                raise PlanError(f"input `{name}` disagrees on shape pack `{unit['name']}`")
            at += pack_width
            continue
        size = actual[at]
        at += 1
        if isinstance(unit, dict) and "sym" in unit:
            symbol = int(unit["sym"])
            if env.dims.setdefault(symbol, size) != size:
                raise PlanError(
                    f"input `{name}` has size {size} where `{unit['name']}` is {env.dims[symbol]}"
                )
            continue
        expected = env.dim(unit)
        if expected != size:
            raise PlanError(f"input `{name}` has size {size} on an axis that must be {expected}")


# ------------------------------------------------------------------ weights


def bind_weights(
    module: LinnetModule,
    weights: str | Path,
    bindings: str | Path | None = None,
    strict: bool = True,
) -> None:
    """Loads SafeTensors weights into the module.

    `weights` is a `.safetensors` file or a directory of them. `bindings` is an
    optional JSON file mapping Linnet parameter paths to checkpoint tensor
    names; paths that are not listed use their own name. Every tensor is
    checked against the plan's shape and dtype before anything is assigned,
    and with `strict` every required parameter must be present.
    """
    from safetensors import safe_open  # type: ignore[import-untyped]

    files = (
        sorted(Path(weights).glob("*.safetensors")) if Path(weights).is_dir() else [Path(weights)]
    )
    if not files:
        raise PlanError(f"no .safetensors files under {weights}")
    available: dict[str, tuple[Path, list[int], str]] = {}
    for file in files:
        # safetensors ships no type information; its handle is treated as Any.
        with cast(Any, safe_open(str(file), framework="pt")) as handle:
            for key in cast(list[str], list(handle.keys())):
                slice_ = handle.get_slice(key)
                shape = cast(list[int], list(slice_.get_shape()))
                available[key] = (file, shape, str(slice_.get_dtype()))

    mapping: dict[str, str] = {}
    if bindings is not None:
        loaded_mapping: object = json.loads(Path(bindings).read_text(encoding="utf-8"))
        if not isinstance(loaded_mapping, dict):
            raise PlanError(
                "bindings must be a JSON object mapping parameter paths to tensor names"
            )
        mapping = {str(k): str(v) for k, v in cast(dict[Any, Any], loaded_mapping).items()}

    safetensor_dtypes = {
        "BOOL": torch.bool,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
        "U16": torch.uint16,
        "U32": torch.uint32,
        "U64": torch.uint64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
    }

    problems: list[str] = []
    assignments: list[tuple[str, str, Path]] = []
    for path, tensor in _all_tensors(module):
        source = mapping.get(path, path)
        owner, leaf = owner_of(module, path)
        if source not in available:
            if leaf in owner.optional_params:
                continue
            if strict:
                problems.append(f"missing tensor `{source}` for `{path}`")
            continue
        _, shape, dtype_name = available[source]
        expected_dtype = tensor.dtype
        if shape != list(tensor.shape):
            problems.append(f"`{source}` has shape {shape}, `{path}` needs {list(tensor.shape)}")
        elif safetensor_dtypes.get(dtype_name) != expected_dtype:
            problems.append(f"`{source}` has dtype {dtype_name}, `{path}` needs {expected_dtype}")
        else:
            assignments.append((path, source, available[source][0]))
    if problems:
        raise PlanError("checkpoint does not match the model:\n  " + "\n  ".join(problems))

    with torch.no_grad():
        for path, source, file in assignments:
            with cast(Any, safe_open(str(file), framework="pt")) as handle:
                loaded = cast(torch.Tensor, handle.get_tensor(source))
            owner, leaf = owner_of(module, path)
            getattr(owner, leaf).copy_(loaded)
            owner.absent_params.discard(leaf)


def _all_tensors(module: LinnetModule) -> list[tuple[str, torch.Tensor]]:
    tensors: list[tuple[str, torch.Tensor]] = []
    prefix = "root."
    for name, parameter in module.named_parameters():
        tensors.append((name.removeprefix(prefix), parameter))
    for name, buffer in module.named_buffers():
        path = name.removeprefix(prefix)
        owner, leaf = owner_of(module, path)
        if leaf not in owner.state_names:  # state is never bound from weights
            tensors.append((path, buffer))
    return tensors


def owner_of(module: LinnetModule, path: str) -> tuple[BlockModule, str]:
    owner: nn.Module = module.root
    parts = path.split(".")
    for part in parts[:-1]:
        owner = owner[int(part)] if isinstance(owner, nn.ModuleList) else getattr(owner, part)
    assert isinstance(owner, BlockModule)
    return owner, parts[-1]

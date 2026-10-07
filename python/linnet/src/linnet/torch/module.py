"""A `torch.nn.Module` built from a plan.

The module hierarchy mirrors the block hierarchy of the Linnet source: every
`sub` is a child module (a `ModuleList` for arrays), every `param` an
`nn.Parameter`, every `buffer` a registered buffer, so `state_dict()` names
each tensor `root.` and its Linnet parameter path. Entries become methods;
`forward` calls the entry named `forward`, or the only entry.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .. import ir
from ..plan import Env, Plan, PlanError, align_shape
from ..weights import read_bindings, safetensors_index
from .dtypes import (
    FROM_SAFETENSORS,
    LINNET_DTYPES,
    SAFETENSORS_NAMES,
    TORCH_DTYPES,
    torch_dtype,
)
from .interp import BlockInstance, Interpreter


class BlockModule(nn.Module):
    """One instantiated block. Parameters are created lazily as zeros of the
    right shape and dtype so that a checkpoint can replace them."""

    def __init__(self, plan: Plan, name: str, env: Env, device: torch.device) -> None:
        super().__init__()
        self.block_name = name
        self.env = env
        self.optional_params: set[str] = set()
        self.absent_params: set[str] = set()
        # Optional sub-blocks, present or absent as a whole: absent until
        # weights bind every parameter one requires.
        self.optional_subs: set[str] = set()
        self.absent_subs: set[str] = set()
        self.state_names: set[str] = set()
        definition = plan.blocks[name]
        for constraint in definition["constraints"]:
            if not env.relation_holds(constraint):
                raise PlanError(f"a `where` constraint of block `{name}` does not hold")
        for member in definition["members"]:
            member_type = member["type"]
            if member["kind"] == "sub":
                if member_type["kind"] == "optional":
                    member_type = member_type["inner"]
                    self.optional_subs.add(member["name"])
                    self.absent_subs.add(member["name"])
                self.add_module(member["name"], _instantiate_sub(plan, member_type, env, device))
                continue
            is_optional = member_type["kind"] == "optional"
            tensor_type = member_type["inner"] if is_optional else member_type
            shape = env.shape(tensor_type["shape"])
            dtype = torch_dtype(env, tensor_type["dtype"])
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
        subs: dict[str, BlockInstance | list[BlockInstance] | None] = {}
        for name, child in self.named_children():
            if name in self.absent_subs:
                subs[name] = None
            elif isinstance(child, BlockModule):
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
        self.generics = dict(generics)  # the root block's, as given
        # Mixed precision: the dtype entries run in under `torch.autocast`,
        # the weights staying in theirs (`load(amp=...)`).
        self.amp: torch.dtype | None = None
        # The DeviceMesh weights are split over (`load(tensor_parallel=...)`).
        self.tensor_parallel: Any = None
        # The checkpoint tensor each parameter path was bound from
        # (`bind_weights`), for writing the weights back under those names.
        self.weight_names: dict[str, str] = {}
        # The paths bound to one part of their tensor, split across processes
        # (`bind_weights(shard=...)`): the axis and the part's extent.
        self.shard_parts: dict[str, tuple[int, int]] = {}
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
        # The block's `where` clause under the values given: the checker
        # proves it at every call inside the program, not for the caller's.
        for constraint in root.get("constraints", []):
            if not env.relation_holds(constraint):
                raise PlanError(
                    f"the generics given to `{root['name']}` break its `where` clause "
                    f"({constraint['relation']} does not hold)"
                )
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
        **options: Any,
    ) -> Any:
        """Runs the entry `name`. Its generic parameters are bound from the
        input shapes, or from `generics` by name for those the inputs do not
        determine (an output length such as `Steps`). With `amp` set (see
        `load`), the entry runs under `torch.autocast` in that dtype."""
        if self.tensor_parallel is not None:
            from torch.distributed.tensor.experimental import implicit_replication

            from .parallel import whole

            # Plain tensors the entry makes (masks, tables, its inputs) meet
            # split weights as replicated values; results come back whole.
            with implicit_replication():
                return whole(self._run_entry_precision(name, inputs, generics, **options))
        return self._run_entry_precision(name, inputs, generics, **options)

    def _run_entry_precision(
        self,
        name: str,
        inputs: list[torch.Tensor],
        generics: Mapping[str, int | str] | None,
        **options: Any,
    ) -> Any:
        if self.amp is None:
            return self._run_entry(name, inputs, generics, **options)
        with torch.autocast(self.interpreter.device.type, dtype=self.amp):
            return self._run_entry(name, inputs, generics, **options)

    def _run_entry(
        self,
        name: str,
        inputs: list[torch.Tensor],
        generics: Mapping[str, int | str] | None = None,
    ) -> Any:
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

    def set_trainable(self, trainable: bool | str | Sequence[str]) -> list[str]:
        """Chooses the parameters that require gradients and returns their
        paths. `True` is every floating-point parameter, `False` none, and a
        glob pattern or a list of them (`"layers.*.mlp.*"`) those whose path
        matches. A parameter the weights left out never trains. Parameters
        tied to one checkpoint tensor are one tensor: it trains if any of its
        paths matches."""
        named = list(self.root.named_parameters(remove_duplicate=False))
        picked = set(ir.chosen_paths([p for p, _ in named], trainable, ir.shared_paths(named)))
        wanted: dict[int, bool] = {}
        tensors: dict[int, torch.nn.Parameter] = {}
        for path, parameter in named:
            owner, leaf = owner_of(self, path)
            use = path in picked and parameter.is_floating_point()
            use = use and leaf not in owner.absent_params
            wanted[id(parameter)] = wanted.get(id(parameter), False) or use
            tensors[id(parameter)] = parameter
        for key, parameter in tensors.items():
            parameter.requires_grad_(wanted[key])
        return [
            path
            for path, parameter in self.root.named_parameters(remove_duplicate=False)
            if wanted[id(parameter)]
        ]

    def add_lora(
        self,
        patterns: str | Sequence[str],
        *,
        rank: int = 16,
        alpha: float = 32.0,
        seed: int = 0,
    ) -> list[str]:
        """Low-rank adapters: see `CompiledLinnetModule.add_lora`. The
        interpreter has none; load with `compile=True`."""
        raise PlanError("adapters need generated code: load with compile=True")

    def merge_lora(self) -> list[str]:
        """Adds the adapters into their weights: see
        `CompiledLinnetModule.merge_lora`. The interpreter has none."""
        return []

    def copy_weights(self, source: LinnetModule) -> None:
        """Copies `source`'s weights into this model's, in place: a policy
        being trained into the copy a serving engine samples from. Every
        parameter and bound buffer goes by path, cast to this model's dtype.
        A weight `source` adapts (`add_lora`) arrives with its adapter's
        product added in, so this model needs no adapters. A source split by
        `fully_shard` is gathered a tensor at a time: every process calls
        this."""
        adapted: set[str] = set()
        scale = 0.0
        lora = getattr(source, "lora", None)
        if lora is not None:
            _, rank, alpha = lora
            adapted = set(getattr(source, "lora_paths", []))
            scale = alpha / rank
        from .parallel import whole

        theirs = dict(_all_tensors(source))
        with torch.no_grad():
            for path, tensor in _all_tensors(self):
                if path not in theirs:
                    raise PlanError(f"`{path}` is not among the source model's weights")
                value = whole(theirs[path])
                owner, leaf = owner_of(self, path)
                their_owner, _ = owner_of(source, path)
                absent = leaf in owner.absent_params
                if absent != (leaf in their_owner.absent_params):
                    raise PlanError(f"`{path}` is bound in only one of the two models")
                if absent:
                    continue
                if path in adapted:
                    down = their_owner.get_parameter("lora_a").float()
                    up = their_owner.get_parameter("lora_b").float()
                    value = value.float() + (up @ down) * scale
                if value.data_ptr() != tensor.data_ptr():
                    tensor.copy_(value)

    def save_weights(
        self,
        path: str | Path,
        *,
        names: str = "checkpoint",
        dtype: torch.dtype | None = None,
        include: Sequence[str] | None = None,
    ) -> Path:
        """Writes the weights to one SafeTensors file: every parameter and
        `buffer` the weights supplied, never `state`, and no optional
        parameter they left out. `names="checkpoint"` names each tensor as the
        checkpoint it was bound from did (through its bindings), so the file
        takes that checkpoint's place: `linnet.hf.export(card, weights=...)`
        reads it. `names="linnet"` names each by its parameter path. Paths
        bound to one checkpoint tensor (a tied embedding) are written once.
        `dtype` converts floating-point tensors as they are written. `include`
        keeps only the paths matching one of its glob patterns: adapters
        alone are `include=["*.lora_a", "*.lora_b"]`. A model split by
        `fully_shard` is gathered a tensor at a time: every process calls
        this, and the first writes the file, whatever it holds."""
        from ..weights import LazyBytes, write_safetensors
        from .parallel import whole

        if names not in ("checkpoint", "linnet"):
            raise PlanError('names must be "checkpoint" or "linnet"')
        if self.tensor_parallel is not None or getattr(self, "shard_group", None) is not None:
            raise PlanError("a model split across processes cannot be saved as one file")
        absent_subs = [
            prefix
            for prefix, (parent, sub) in _optional_subs(self).items()
            if sub in parent.absent_subs
        ]
        written: dict[str, tuple[str, torch.Tensor]] = {}
        entries: list[tuple[str, str, tuple[int, ...], Any]] = []
        split: list[torch.Tensor] = []
        for tensor_path, tensor in _all_tensors(self):
            owner, leaf = owner_of(self, tensor_path)
            if leaf in owner.absent_params or any(
                tensor_path.startswith(prefix + ".") for prefix in absent_subs
            ):
                continue
            if include is not None and not any(
                fnmatch.fnmatchcase(tensor_path, pattern) for pattern in include
            ):
                continue
            name = (
                self.weight_names.get(tensor_path, tensor_path)
                if names == "checkpoint"
                else tensor_path
            )
            if name in written:
                first, kept = written[name]
                if kept is not tensor and not torch.equal(kept, tensor):
                    raise PlanError(
                        f"`{first}` and `{tensor_path}` both come from `{name}` but now differ"
                    )
                continue
            written[name] = (tensor_path, tensor)
            if callable(getattr(tensor, "full_tensor", None)):
                split.append(tensor)
            target = dtype if dtype is not None and tensor.is_floating_point() else tensor.dtype
            entries.append(
                (
                    name,
                    SAFETENSORS_NAMES[target],
                    tuple(tensor.shape),
                    LazyBytes(tensor.numel() * target.itemsize, _bytes_of(tensor, target)),
                )
            )
        if split or getattr(self, "fully_sharded", ()):
            import torch.distributed as dist

            if dist.get_rank() != 0:
                # The first process gathers each as it writes; the others
                # join every gather in the same order.
                for tensor in split:
                    whole(tensor)
                return Path(path)
        return write_safetensors(path, entries, metadata={"format": "pt"})


def _bytes_of(tensor: torch.Tensor, dtype: torch.dtype) -> Callable[[], bytes]:
    """Reads `tensor` as `dtype` into host bytes, when called."""

    def read() -> bytes:
        from .parallel import whole

        value = whole(tensor).detach().to(device="cpu", dtype=dtype).contiguous()
        return value.reshape(-1).view(torch.uint8).numpy().tobytes()

    return read


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
    spec: str | dict[str, Any] = param_type["dtype"]
    if isinstance(spec, dict) and value.dtype in LINNET_DTYPES:
        # A dtype generic of the entry's own (a function's `T`), bound by the
        # first input that carries it; the rest must agree.
        env.dtypes.setdefault(int(spec["var"]), LINNET_DTYPES[value.dtype])
    expected_dtype = torch_dtype(env, spec)
    if value.dtype != expected_dtype:
        raise PlanError(f"input `{name}` has dtype {value.dtype}, expected {expected_dtype}")
    dims, pack = align_shape(param_type["shape"], list(value.shape), name)
    if pack is not None:
        unit, sizes = pack
        if env.packs.setdefault(int(unit["pack"]), sizes) != sizes:
            raise PlanError(f"input `{name}` disagrees on shape pack `{unit['name']}`")
    for unit, size in dims:
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
    cast_dtype: bool = False,
    shard: tuple[int, int] | None = None,
    tie: bool = False,
    only: Callable[[str], bool] | None = None,
) -> dict[str, str]:
    """Loads SafeTensors weights into the module and returns the checkpoint
    tensor each parameter path is bound to.

    `weights` is a `.safetensors` file or a directory of them. `bindings` is an
    optional JSON file mapping Linnet parameter paths to checkpoint tensor
    names; paths that are not listed use their own name. Every tensor is
    checked against the plan's shape and dtype before anything is assigned,
    and with `strict` every required parameter must be present.

    `cast_dtype=True` accepts a floating-point tensor of another width and
    converts it on the way in, one tensor at a time: an f32 checkpoint run in
    bf16, or a bf16 one checked in f32, without writing a converted copy.
    Any other mismatch (an integer where a float is expected) still means the
    binding is wrong, and is still an error.

    `shard=(index, count)` binds one shard of a model split `count` ways
    (its `Shards` generic): a tensor the checkpoint holds `count` times over
    along one axis is read as its `index`-th part along that axis, from the
    file, without loading the rest.

    `tie=True` makes parameters bound to one checkpoint tensor (an embedding
    and an output head the checkpoint ties) one `nn.Parameter`, so training
    updates them together and the weights are held once.

    `only` reads just the paths it accepts (a pipeline stage's blocks); the
    rest are still checked, and still decide which optional parameters are
    present, so every stage compiles the same model.
    """
    from safetensors import safe_open  # type: ignore[import-untyped]

    available = safetensors_index(weights)
    mapping = read_bindings(bindings) if bindings is not None else {}

    problems: list[str] = []
    assignments: list[tuple[str, str, Path]] = []
    parts: dict[str, tuple[int, int]] = {}  # path -> (axis, extent) of its shard
    optional_subs = _optional_subs(module)
    for path, tensor in _all_tensors(module):
        source = mapping.get(path, path)
        owner, leaf = owner_of(module, path)
        if source not in available:
            if leaf in owner.optional_params or any(
                path.startswith(prefix + ".") for prefix in optional_subs
            ):
                continue
            if strict:
                problems.append(f"missing tensor `{source}` for `{path}`")
            continue
        shape, dtype_name = list(available[source].shape), available[source].dtype
        expected_dtype = tensor.dtype
        axis = _shard_axis(shape, list(tensor.shape), shard[1]) if shard is not None else None
        if axis is not None:
            parts[path] = (axis, tensor.shape[axis])
            shape = list(tensor.shape)
        if shape != list(tensor.shape):
            problems.append(f"`{source}` has shape {shape}, `{path}` needs {list(tensor.shape)}")
        elif FROM_SAFETENSORS.get(dtype_name) != expected_dtype and not (
            cast_dtype
            and expected_dtype.is_floating_point
            and (found := FROM_SAFETENSORS.get(dtype_name)) is not None
            and found.is_floating_point
        ):
            problems.append(f"`{source}` has dtype {dtype_name}, `{path}` needs {expected_dtype}")
        else:
            assignments.append((path, source, available[source].file))
    if problems:
        raise PlanError("checkpoint does not match the model:\n  " + "\n  ".join(problems))

    with torch.no_grad():
        for path, source, file in assignments:
            owner, leaf = owner_of(module, path)
            owner.absent_params.discard(leaf)
            if only is not None and not only(path):
                continue
            with cast(Any, safe_open(str(file), framework="pt")) as handle:
                if path in parts:
                    assert shard is not None
                    axis, extent = parts[path]
                    index = (slice(None),) * axis + (
                        slice(shard[0] * extent, (shard[0] + 1) * extent),
                    )
                    loaded = cast(torch.Tensor, handle.get_slice(source)[index])
                else:
                    loaded = cast(torch.Tensor, handle.get_tensor(source))
            getattr(owner, leaf).copy_(loaded)
            module.weight_names[path] = source
    # The paths bound to a part of their tensor: split across processes.
    module.shard_parts.update(parts)
    if tie:
        _tie(module, [(path, source, parts.get(path)) for path, source, _ in assignments])
    # An optional sub-block is present when every parameter it requires was
    # bound (and something was): a checkpoint without its weights leaves it out.
    bound = {path for path, _, _ in assignments}
    every = [path for path, _ in _all_tensors(module)]
    for prefix, (parent, name) in optional_subs.items():
        inside = [path for path in every if path.startswith(prefix + ".")]
        nested = [other + "." for other in optional_subs if other.startswith(prefix + ".")]
        required = [
            path
            for path in inside
            if not any(path.startswith(other) for other in nested)
            and owner_of(module, path)[1] not in owner_of(module, path)[0].optional_params
        ]
        if all(path in bound for path in required) and any(path in bound for path in inside):
            parent.absent_subs.discard(name)
        else:
            parent.absent_subs.add(name)
    # Which optional parameters are bound decides what a compiled entry
    # computes; entries prepared before this binding are prepared again.
    for cache in ("_fast", "_prepared"):
        entries = getattr(module, cache, None)
        if isinstance(entries, dict):
            entries.clear()
    forget = getattr(module, "forget_parameters", None)
    if callable(forget):
        forget()  # tying replaced parameters
    return {path: source for path, source, _ in assignments}


def _tie(module: LinnetModule, bound: list[tuple[str, str, tuple[int, int] | None]]) -> None:
    """Makes the parameters bound from the same checkpoint tensor, and the
    same part of it, one parameter: the first path's."""
    first: dict[tuple[str, tuple[int, int] | None], torch.nn.Parameter] = {}
    for path, source, part in bound:
        owner, leaf = owner_of(module, path)
        tensor = getattr(owner, leaf)
        if not isinstance(tensor, torch.nn.Parameter):
            continue
        kept = first.setdefault((source, part), tensor)
        if kept is not tensor and kept.shape == tensor.shape and kept.dtype == tensor.dtype:
            setattr(owner, leaf, kept)


def _shard_axis(full: list[int], local: list[int], count: int) -> int | None:
    """The one axis along which a checkpoint tensor of shape `full` holds
    `count` shards of shape `local`, or None."""
    if count <= 1 or len(full) != len(local):
        return None
    differ = [axis for axis, (f, n) in enumerate(zip(full, local, strict=True)) if f != n]
    if len(differ) == 1 and full[differ[0]] == local[differ[0]] * count:
        return differ[0]
    return None


def _optional_subs(module: LinnetModule) -> dict[str, tuple[BlockModule, str]]:
    """Every optional sub-block by path, with the block that holds it and its
    name there."""
    found: dict[str, tuple[BlockModule, str]] = {}
    named = cast("Iterable[tuple[str, nn.Module]]", module.named_modules())
    for name, block in named:
        if isinstance(block, BlockModule):
            path = name.removeprefix("root").removeprefix(".")
            for sub in block.optional_subs:
                found[f"{path}.{sub}" if path else sub] = (block, sub)
    return found


def _all_tensors(module: LinnetModule) -> list[tuple[str, torch.Tensor]]:
    """Every parameter and bound buffer by path, a tied tensor under each of
    its paths."""
    tensors: list[tuple[str, torch.Tensor]] = []
    prefix = "root."
    for name, parameter in module.named_parameters(remove_duplicate=False):
        tensors.append((name.removeprefix(prefix), parameter))
    for name, buffer in module.named_buffers(remove_duplicate=False):
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

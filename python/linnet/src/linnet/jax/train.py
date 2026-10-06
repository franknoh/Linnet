"""Training a Linnet model in JAX: packed batches, an optax optimizer, and
one compiled step.

```python
import optax
from linnet import nest
from linnet.jax.train import train
from linnet.packing import Example, pack

model = nest.load("llama-3.1-8b-instruct", backend="jax_source", entry="loss_packed",
                  generics={"Batch": 1, "MaxSeq": 4096, "T": "bf16"}, cast_dtype=True)
examples = [Example.prompted(prompt, completion) for prompt, completion in data]
params, history = train(model, pack(examples, tokens=4096), optimizer=optax.adamw(1e-5),
                        steps=1000, accumulate=8)
```

`model` is the entry `loss_packed` as generated JAX (`load_source`, or
`nest.load(..., backend="jax_source")`): `(tokens [P] i32, positions [P]
i32, segments [P] i32, targets [P] i64, weights [P] f32) -> f32`, the
weighted sum of each position's cross-entropy for its target.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

import contextlib
import fnmatch
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..dtypes import BY_NUMPY
from ..packing import Packed
from ..weights import read_arrays, write_safetensors


@dataclass
class Step:
    """What one optimizer step did."""

    step: int
    loss: float  # the mean over the step's counted positions
    tokens: int  # positions processed, padding included
    seconds: float
    grad_norm: float | None = None


@dataclass
class History:
    steps: list[Step] = field(default_factory=lambda: list[Step]())

    @property
    def losses(self) -> list[float]:
        return [step.loss for step in self.steps]


def train(
    model: Any,
    batches: Iterable[Packed],
    *,
    optimizer: Any,
    steps: int | None = None,
    accumulate: int = 1,
    clip: float | None = 1.0,
    trainable: bool | str | Sequence[str] | None = None,
    master: Any = jnp.float32,
    parameters: dict[str, Any] | None = None,
    mesh: Any = None,
    remat: bool = False,
    checkpoint: str | Path | None = None,
    checkpoint_every: int | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> tuple[dict[str, Any], History]:
    """Fits `model`'s parameters on `batches` (`linnet.packing.pack`) with
    `optimizer` (an optax transformation); returns them, path -> array, with
    the steps taken.

    Each step sums `accumulate` batches' gradients, every batch's loss
    weighed by the step's total count of learned positions, so the step
    follows the mean over all of them; the batches run in one compiled
    step, a `lax.scan` over them. `clip` rescales the gradients to that
    global norm at most. It stops after `steps` steps or when the batches
    run out.

    `trainable` picks the floating-point parameters that train: all, or
    those whose path matches a glob pattern (`"layers.*.mlp.*"`); by
    default the adapters alone when the model has them (`add_lora`), and
    otherwise all. They are
    kept in `master` (f32) and cast to the dtype the model computes in on
    every call, so gradients and optimizer state are in f32; the rest stay
    as loaded. Paths bound to one checkpoint tensor (a tied embedding and
    output head) are one parameter, their gradients summed. `parameters`
    starts from given arrays instead of the loaded weights.

    With `mesh` (a `jax.sharding.Mesh`), training is data-parallel and fully
    sharded over the mesh's first axis of `N` devices. Each step takes `N`
    times `accumulate` batches, one per device at a time (`shard_map`).
    Every parameter, its gradient and its optimizer state are split along
    an axis `N` divides (copied when none does); each device gathers a
    block's weights whole where the block runs (`linnet.jax.fsdp`). Set the
    mesh before the model's first call: the weights then load straight into
    their parts.

    `remat` has the backward pass compute each layer (each element of the
    root's lists) again instead of keeping its values, and, over a mesh,
    gather its weights again: the step keeps a layer's inputs alone.

    With `checkpoint`, a directory, training resumes from the latest
    checkpoint there (`load_checkpoint`), skipping the batches its steps
    took, and writes one every `checkpoint_every` steps and at the end
    (`save_checkpoint`, one host)."""
    from jax.sharding import NamedSharding, PartitionSpec

    width = 1
    split: Any = None
    if mesh is not None:
        axis = mesh.axis_names[0]
        width = int(mesh.shape[axis])
        split = NamedSharding(mesh, PartitionSpec(None, axis))
    prepare_blocks(model, mesh=mesh, remat=remat, parameters=parameters)
    per_step = accumulate * width
    iterator = iter(batches)

    def take() -> list[Packed]:
        return [batch for _, batch in zip(range(per_step), iterator, strict=False)]

    group = take()
    if not group:
        return dict(parameters or {}), History()
    weights = (
        dict(parameters) if parameters is not None else model.parameters_for(*group[0].arrays(1.0))
    )
    learner = Learner(model, weights, optimizer, trainable=trainable, master=master, mesh=mesh)
    trained, frozen, state = learner.trained, learner.frozen, learner.state
    start = 0
    if checkpoint is not None:
        found = load_checkpoint(checkpoint, trained, state)
        if found is not None:
            start, trained, state = found
            for _ in range(start * per_step - len(group)):
                next(iterator, None)
            group = take()

    def loss_of(trained: Any, frozen: Any, inputs: Any) -> Any:
        return model.apply(learner.values(trained, frozen), *inputs)

    def micro(trained: Any, frozen: Any, inputs: Any) -> Any:
        if mesh is None:
            return loss_of(trained, frozen, inputs)
        return spread(trained, frozen, inputs)

    spread: Any = None
    if mesh is not None:
        spread = _spread(mesh, trained, frozen, loss_of, learner.dtypes, learner.gathered)

    def update(trained: Any, state: Any, frozen: Any, stacked: Any) -> Any:
        def one(carry: Any, inputs: Any) -> Any:
            grads, total = carry
            loss, more = jax.value_and_grad(micro)(trained, frozen, inputs)
            return (jax.tree.map(jnp.add, grads, more), total + loss), None

        zeros = jax.tree.map(jnp.zeros_like, trained)
        (grads, loss), _ = jax.lax.scan(one, (zeros, jnp.zeros((), jnp.float32)), stacked)
        norm = _global_norm(grads)
        if clip is not None:
            scale = jnp.minimum(1.0, clip / (norm + 1e-6))
            grads = jax.tree.map(lambda g: g * scale.astype(g.dtype), grads)
        updates, state = optimizer.update(grads, state, trained)
        trained = jax.tree.map(lambda p, u: p + u.astype(p.dtype), trained, updates)
        return trained, state, loss, norm

    if mesh is None:
        compiled = jax.jit(update, donate_argnums=(0, 1))
    else:
        # The parts stay where they are, step after step.
        whole = NamedSharding(mesh, PartitionSpec())
        compiled = jax.jit(
            update,
            donate_argnums=(0, 1),
            out_shardings=(
                jax.tree.map(lambda x: x.sharding, trained),
                jax.tree.map(lambda x: x.sharding, state),
                whole,
                whole,
            ),
        )
    history = History()
    step = start
    while len(group) == per_step and (steps is None or step < steps):
        count = max(1, sum(batch.count for batch in group))
        stacked = [
            np.stack(values) for values in zip(*(b.arrays(count) for b in group), strict=True)
        ]
        if mesh is not None:
            stacked = [
                jax.device_put(value.reshape(accumulate, width, *value.shape[1:]), split)
                for value in stacked
            ]
        begin = time.perf_counter()
        with _partitioner(mesh):
            trained, state, loss, norm = compiled(trained, state, frozen, stacked)
        step += 1
        record = Step(
            step=step,
            loss=float(loss),
            tokens=sum(batch.tokens.size for batch in group),
            seconds=time.perf_counter() - begin,
            grad_norm=float(norm),
        )
        history.steps.append(record)
        if on_step is not None:
            on_step(record)
        if checkpoint is not None and checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(checkpoint, step, trained, state)
        group = take()
    if (
        checkpoint is not None
        and history.steps
        and not (checkpoint_every and step % checkpoint_every == 0)
    ):
        save_checkpoint(checkpoint, step, trained, state)
    learner.trained, learner.state = trained, state
    return learner.parameters(), history


class Learner:
    """What a run trains and how: copies of the chosen parameters in
    `master` (the step donates them, and the model's own arrays must stay),
    the rest of the model's as loaded, and `optimizer`'s state over the
    copies. Paths sharing one array (a tied embedding and output head) are
    one parameter, their gradients summed. With `mesh`, everything is split
    as `placement` says.

    `trainable` picks floating-point parameters: all, or glob patterns; by
    default the adapters alone on a model with them (`add_lora`), and
    otherwise all."""

    def __init__(
        self,
        model: Any,
        weights: dict[str, Any],
        optimizer: Any,
        *,
        trainable: bool | str | Sequence[str] | None = None,
        master: Any = jnp.float32,
        mesh: Any = None,
    ) -> None:
        self.optimizer = optimizer
        self.mesh = mesh
        self.width = 1 if mesh is None else int(mesh.shape[mesh.axis_names[0]])
        self._model = model
        # On the devices, as `mesh` splits them; one array per tied tensor.
        moved: dict[int, Any] = {}
        placed: dict[str, Any] = {}
        for path, array in weights.items():
            if id(array) not in moved:
                moved[id(array)] = _placed(array, path, mesh)
            placed[path] = moved[id(array)]
        self.ties = _ties(placed)
        if trainable is None:
            adapted = getattr(model, "lora", None) is not None
            trainable = ["*.lora_a", "*.lora_b"] if adapted else True
        chosen = _chosen([path for path in placed if path not in self.ties], placed, trainable)
        self.dtypes = {path: placed[path].dtype for path in chosen}
        shardings = {path: placed[path].sharding for path in chosen}
        self.trained: dict[str, Any] = jax.jit(
            lambda tree: {p: jnp.copy(v.astype(master or v.dtype)) for p, v in tree.items()},
            out_shardings=shardings,
        )({path: placed[path] for path in chosen})
        self.frozen = {
            path: value
            for path, value in placed.items()
            if path not in chosen and path not in self.ties
        }
        self.gathered = _gathered_by_code(model, placed, self.ties)
        if mesh is None:
            self.state = jax.jit(optimizer.init)(self.trained)
        else:
            # Zeros depend on no input: their parts are said, as the weights'.
            shapes = jax.eval_shape(optimizer.init, self.trained)
            self.state = jax.jit(
                optimizer.init,
                out_shardings=jax.tree.map(lambda s: _split_as(mesh, s.shape), shapes),
            )(self.trained)
        self._apply: Any = None

    def values(
        self, trained: dict[str, Any], frozen: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Every parameter the model takes, the trained ones cast to the
        dtype it computes in (traceable). One the model's code gathers from
        its parts stays in the master's dtype: the code casts it, and its
        gradient is reduced in the master's dtype first."""
        values = {
            **(self.frozen if frozen is None else frozen),
            **{
                path: value if path in self.gathered else value.astype(self.dtypes[path])
                for path, value in trained.items()
            },
        }
        values.update({path: values[tie] for path, tie in self.ties.items()})
        return values

    def parameters(self, *, copy: bool = False) -> dict[str, Any]:
        """The parameters now, path -> array: the trained ones in `master`.
        The next `step` donates the trained arrays; `copy` hands out copies
        of them that outlive it (for an engine to keep)."""
        trained = self.trained
        if copy:
            trained = _copied(trained)
        final = {**self.frozen, **trained}
        final.update({path: final[tie] for path, tie in self.ties.items()})
        return final

    def gradient(self, loss: Callable[..., Any]) -> Callable[..., Any]:
        """`loss(values, *args) -> (scalar, aux)` as a compiled function of
        `(trained, frozen, *args)` returning `((scalar, aux), gradients)`.

        Over a mesh, each of `args` is `width` devices' arguments stacked
        (`stack`): every device computes the loss of its own, and the
        scalars, aux values and gradients come back summed over them."""

        def of(trained: Any, frozen: Any, *args: Any) -> Any:
            return loss(self.values(trained, frozen), *args)

        if self.mesh is None:
            return jax.jit(jax.value_and_grad(of, has_aux=True))
        spread = _spread(
            self.mesh,
            self.trained,
            self.frozen,
            lambda trained, frozen, args: of(trained, frozen, *args),
            self.dtypes,
            self.gathered,
        )
        compiled = jax.jit(
            jax.value_and_grad(
                lambda trained, frozen, *args: spread(trained, frozen, args), has_aux=True
            )
        )
        mesh = self.mesh

        def call(*arguments: Any) -> Any:
            with _partitioner(mesh):
                return compiled(*arguments)

        return call

    def forward(self, function: Callable[..., Any], parameters: Any = None) -> Callable[..., Any]:
        """`function(values, *args)` as a compiled function of `*args`, over
        the trained parameters as they are when called, or over
        `parameters` (path -> array, all of them, placed as `placement`
        says). Over a mesh, `args` are stacked (`stack`) and so is the
        result: each device's own."""
        if self.mesh is None:
            compiled = jax.jit(function)
            if parameters is not None:
                return lambda *args: compiled(parameters, *args)
            return lambda *args: compiled(self.values(self.trained), *args)
        mesh = self.mesh
        if parameters is not None:
            # Arguments, not closed over: XLA would hold them as constants.
            given = {path: _placed(value, path, mesh) for path, value in parameters.items()}
            spread = _spread(
                mesh,
                {},
                given,
                lambda _, frozen, args: function(frozen, *args),
                {},
                _gathered_by_code(self._model, given, {}),
                reduce=False,
            )
            jitted = jax.jit(lambda values, *args: spread({}, values, args))

            def call_given(*args: Any) -> Any:
                with _partitioner(mesh):
                    return jitted(given, *args)

            return call_given
        spread = _spread(
            mesh,
            self.trained,
            self.frozen,
            lambda trained, frozen, args: function(self.values(trained, frozen), *args),
            self.dtypes,
            self.gathered,
            reduce=False,
        )
        own = jax.jit(lambda trained, frozen, *args: spread(trained, frozen, args))

        def call(*args: Any) -> Any:
            with _partitioner(mesh):
                return own(self.trained, self.frozen, *args)

        return call

    def stack(self, rows: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
        """One call's arguments from `width` devices' own (`rows`, each a
        sequence of arrays or lists of arrays): as they are on one device,
        stacked along a new first axis and split over the mesh on more."""
        if self.mesh is None:
            (row,) = rows
            return tuple(row)
        from jax.sharding import NamedSharding, PartitionSpec

        split = NamedSharding(self.mesh, PartitionSpec(self.mesh.axis_names[0]))
        return jax.tree.map(
            lambda *values: jax.device_put(np.stack([np.asarray(v) for v in values]), split),
            *[tuple(row) for row in rows],
        )

    def step(self, grads: dict[str, Any], clip: float | None) -> float:
        """Clips `grads` to that global norm at most and applies the
        optimizer to the trained parameters; returns the norm before."""
        if self._apply is None:

            def apply(trained: Any, state: Any, grads: Any) -> Any:
                norm = _global_norm(grads)
                if clip is not None:
                    scale = jnp.minimum(1.0, clip / (norm + 1e-6))
                    grads = jax.tree.map(lambda g: g * scale.astype(g.dtype), grads)
                updates, state = self.optimizer.update(grads, state, trained)
                trained = jax.tree.map(lambda p, u: p + u.astype(p.dtype), trained, updates)
                return trained, state, norm

            self._apply = jax.jit(apply, donate_argnums=(0, 1))
        self.trained, self.state, norm = self._apply(self.trained, self.state, grads)
        return float(norm)


@jax.jit
def _copied(tree: Any) -> Any:
    return jax.tree.map(jnp.copy, tree)


@jax.jit
def add(total: Any, more: Any) -> Any:
    """Two gradient trees summed."""
    return jax.tree.map(jnp.add, total, more)


def _spread(
    mesh: Any,
    trained: Any,
    frozen: Any,
    function: Callable[..., Any],
    dtypes: dict[str, Any],
    gathered: set[str],
    *,
    reduce: bool = True,
) -> Any:
    """`function(trained, frozen, args)` on every device of the mesh, each on
    its own arguments (`args`, stacked `[N, ...]`, split by the first axis),
    with every parameter whole: summed over the devices (`reduce`), or each
    device's result stacked. Inside, every device computes alone, with no
    partitioner to split activations along the weights' axes.

    A parameter in `gathered` reaches the model as this device's part: the
    generated code gathers it where its block runs (`linnet.jax.fsdp`). Any
    other is gathered here; a trained one in `dtypes[path]`, the dtype the
    model computes in, with its gradient reduced in the part's own (the f32
    master's)."""
    from jax.sharding import PartitionSpec

    from . import fsdp

    try:
        from jax import shard_map
    except ImportError:  # JAX before 0.6
        from jax.experimental.shard_map import shard_map  # type: ignore[no-redef]

    axis = mesh.axis_names[0]
    trained_specs = {path: value.sharding.spec for path, value in trained.items()}
    frozen_specs = {path: value.sharding.spec for path, value in frozen.items()}

    def split_at(spec: Any) -> int | None:
        return next((dim for dim, name in enumerate(spec) if name == axis), None)

    def whole(value: Any, spec: Any) -> Any:
        dim = split_at(spec)
        return value if dim is None else jax.lax.all_gather(value, axis, axis=dim, tiled=True)

    def whole_as(value: Any, spec: Any, dtype: Any) -> Any:
        dim = split_at(spec)
        if dim is None:
            return value.astype(dtype)
        return fsdp.gather_as(axis, dim, dtype)(value)

    def local(trained: Any, frozen: Any, args: Any) -> Any:
        mine = jax.tree.map(lambda value: value[0], args)
        with fsdp.gathering(axis):
            out = function(
                {
                    p: v if p in gathered else whole_as(v, trained_specs[p], dtypes[p])
                    for p, v in trained.items()
                },
                {p: v if p in gathered else whole(v, frozen_specs[p]) for p, v in frozen.items()},
                mine,
            )
        if reduce:
            return jax.tree.map(lambda value: jax.lax.psum(value, axis), out)
        return jax.tree.map(lambda value: value[None], out)

    # Unchecked: the blockwise loss's scans start from zeros every device
    # shares and carry values each device has its own of.
    import inspect

    unchecked: dict[str, Any] = (
        {"check_vma": False}
        if "check_vma" in inspect.signature(shard_map).parameters
        else {"check_rep": False}
    )
    spread: Any = shard_map
    return spread(
        local,
        mesh=mesh,
        in_specs=(trained_specs, frozen_specs, PartitionSpec(axis)),
        out_specs=PartitionSpec() if reduce else PartitionSpec(axis),
        **unchecked,
    )


def _gathered_by_code(model: Any, placed: dict[str, Any], ties: dict[str, str]) -> set[str]:
    """The parameters the model's generated code gathers from their parts
    itself (`SourceFunction.blocks`). A tied pair is gathered by the code
    only when both sides are: passed whole, a weight is only cast."""
    from . import fsdp

    sharded = getattr(model, "sharded", ())
    chosen = {path for path in placed if fsdp.gathered_by_code(path, sharded)}
    for path, tie in ties.items():
        if (path in chosen) != (tie in chosen):
            chosen.discard(path)
            chosen.discard(tie)
    return chosen


def prepare_blocks(
    model: Any, *, mesh: Any = None, remat: bool = False, parameters: Any = None
) -> None:
    """Before the model's first call: over `mesh`, its weights load split
    (`placement`) and its blocks gather them where they run; with `remat`,
    the backward pass computes each layer again (`SourceFunction.blocks`)."""
    from ..compiler import LinnetError
    from . import fsdp

    if mesh is not None and parameters is None:
        model.placement = placement(mesh)
    if mesh is None and not remat:
        return
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        if remat:
            raise LinnetError("recomputing layers needs the entry as generated JAX (load_source)")
        return
    paths = list(parameters) if parameters is not None else list(model.weights)
    every, listed = fsdp.units(paths)
    blocks(sharded=every if mesh is not None else (), remat=listed if remat else ())


def _partitioner(mesh: Any) -> Any:
    """Where the step is compiled over `mesh`: with GSPMD. Shardy, JAX's
    default, crashes XLA compiling cuDNN attention inside `shard_map`
    (JAX 0.11, H100)."""
    if mesh is None:
        return contextlib.nullcontext()
    from jax._src import config as jax_config

    shardy = getattr(jax_config, "use_shardy_partitioner", None)
    return contextlib.nullcontext() if shardy is None else shardy(False)


def merge_lora(model: Any, parameters: dict[str, Any]) -> dict[str, Any]:
    """`parameters` with each adapter's product added into its weight,
    `W + B @ A * alpha / rank` in f32, and the adapters left out: plain
    weights for the model loaded again without adapters, or for serving."""
    if getattr(model, "lora", None) is None:
        return dict(parameters)
    _, rank, alpha = model.lora
    merged = {
        path: value
        for path, value in parameters.items()
        if not path.endswith((".lora_a", ".lora_b"))
    }
    for path, down in parameters.items():
        if not path.endswith(".lora_a"):
            continue
        block = path.removesuffix(".lora_a")
        weight = merged[block + ".weight"]
        up = parameters[block + ".lora_b"]
        product = up.astype(jnp.float32) @ down.astype(jnp.float32)
        merged[block + ".weight"] = (weight.astype(jnp.float32) + product * (alpha / rank)).astype(
            weight.dtype
        )
    return merged


def placement(mesh: Any) -> Callable[[str, Any], Any]:
    """How a weight goes onto `mesh` for fully sharded training: split along
    its largest axis the mesh's first axis divides, or copied to every
    device when none does."""

    def place(path: str, value: Any) -> Any:
        return jax.device_put(value, _split_as(mesh, tuple(np.shape(value))))

    return place


def _split_as(mesh: Any, shape: tuple[int, ...]) -> Any:
    from jax.sharding import NamedSharding, PartitionSpec

    axis = mesh.axis_names[0]
    width = int(mesh.shape[axis])
    spec: list[Any] = [None] * len(shape)
    fits = [i for i, extent in enumerate(shape) if extent % width == 0 and extent >= width]
    if fits:
        spec[max(fits, key=lambda i: shape[i])] = axis
    return NamedSharding(mesh, PartitionSpec(*spec))


def _placed(array: Any, path: str, mesh: Any) -> Any:
    """`array` on the devices: as it is when already where `mesh` wants it."""
    if mesh is None:
        return array if isinstance(array, jax.Array) else jnp.asarray(array)
    wanted = _split_as(mesh, tuple(np.shape(array)))
    if isinstance(array, jax.Array) and array.sharding == wanted:
        return array
    return jax.device_put(array, wanted)


# ------------------------------------------------------------------ files

PREFIX = "step-"


def _write(path: Path, arrays: dict[str, Any]) -> None:
    entries: list[tuple[str, str, tuple[int, ...], Any]] = []
    for name, array in arrays.items():
        host = np.ascontiguousarray(np.asarray(jax.device_get(array)))
        code = BY_NUMPY[host.dtype.name].safetensors
        entries.append((name, code, tuple(host.shape), host.tobytes()))
    write_safetensors(path, entries)


def save_weights(parameters: dict[str, Any], path: str | Path, *, dtype: Any = None) -> Path:
    """Writes `parameters` (path -> array, as `train` returns them) to one
    SafeTensors file under their Linnet paths, which `load_source` reads
    back; `dtype` converts floating ones as they are written."""
    arrays = {
        name: array.astype(dtype)
        if dtype is not None and jnp.issubdtype(array.dtype, jnp.floating)
        else array
        for name, array in parameters.items()
    }
    target = Path(path)
    _write(target, arrays)
    return target


def save_checkpoint(
    directory: str | Path, step: int, trained: dict[str, Any], state: Any, *, keep: int | None = 2
) -> Path:
    """Writes the parameters being trained and the optimizer state at
    `step` to `directory/step-<step>` (from one host), then removes all but
    the `keep` latest."""
    import json
    import shutil

    root = Path(directory)
    target = root / f"{PREFIX}{step:08d}"
    target.mkdir(parents=True, exist_ok=True)
    _write(target / "parameters.safetensors", trained)
    leaves = jax.tree.leaves(state)
    _write(target / "state.safetensors", {f"{i:06d}": leaf for i, leaf in enumerate(leaves)})
    # Written last: a checkpoint without it did not finish.
    (target / "progress.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    if keep is not None:
        for old in _complete(root)[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
    return target


def load_checkpoint(
    directory: str | Path, trained: dict[str, Any], state: Any
) -> tuple[int, dict[str, Any], Any] | None:
    """The latest complete checkpoint under `directory`: its step, and the
    parameters and optimizer state shaped and placed as `trained` and
    `state` are. None when there is none."""
    import json

    found = _complete(Path(directory))
    if not found:
        return None
    latest = found[-1]
    stored = read_arrays(latest / "parameters.safetensors")
    trained = {path: jax.device_put(stored[path], like.sharding) for path, like in trained.items()}
    leaves, structure = jax.tree.flatten(state)
    kept = read_arrays(latest / "state.safetensors")
    leaves = [jax.device_put(kept[f"{i:06d}"], like.sharding) for i, like in enumerate(leaves)]
    step = int(json.loads((latest / "progress.json").read_text(encoding="utf-8"))["step"])
    return step, trained, jax.tree.unflatten(structure, leaves)


def _complete(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.name.startswith(PREFIX) and (path / "progress.json").exists()
    )


def _ties(weights: dict[str, Any]) -> dict[str, str]:
    """Each path whose array another path already holds, to that path."""
    first: dict[int, str] = {}
    ties: dict[str, str] = {}
    for path, array in weights.items():
        owner = first.setdefault(id(array), path)
        if owner != path:
            ties[path] = owner
    return ties


def _chosen(
    paths: list[str], weights: dict[str, Any], trainable: bool | str | Sequence[str]
) -> list[str]:
    floating = [p for p in paths if jnp.issubdtype(weights[p].dtype, jnp.floating)]
    if trainable is True:
        return floating
    if trainable is False:
        return []
    patterns = [trainable] if isinstance(trainable, str) else list(trainable)
    return [p for p in floating if any(fnmatch.fnmatchcase(p, pattern) for pattern in patterns)]


def _global_norm(tree: Any) -> Any:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.zeros((), jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves))


__all__ = [
    "History",
    "Learner",
    "Step",
    "add",
    "load_checkpoint",
    "merge_lora",
    "placement",
    "save_checkpoint",
    "save_weights",
    "train",
]

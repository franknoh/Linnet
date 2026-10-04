"""Training in JAX: the blockwise output-head losses match the dense ones in
value and gradient, the Llama example's `loss_packed` gives what PyTorch
gives, and `linnet.jax.train` lowers the loss, follows the mean over
accumulated batches, trains tied paths as one, and leaves frozen ones."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false, reportUnknownLambdaType=false

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import-untyped]
import pytest
import torch
from safetensors.numpy import save_file  # type: ignore[import-untyped]

from linnet.jax import load_source
from linnet.jax import loss as blockwise
from linnet.jax.train import train
from linnet.packing import Example, Packed, pack

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"
LLAMA = REPO / "examples/05-llama/src/lib.linnet"
GENERICS: dict[str, int | str] = {
    "Vocab": 11,
    "H": 8,
    "Heads": 4,
    "KvHeads": 2,
    "Inner": 16,
    "Layers": 2,
    "Batch": 1,
    "MaxSeq": 8,
    "T": "f32",
}


def _dense_log_probs(hidden: Any, weight: Any, targets: Any) -> Any:
    logits = (hidden @ weight.T).astype(jnp.float32)
    picked = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
    return picked - jax.nn.logsumexp(logits, axis=-1)


def test_blockwise_losses_match_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    # Three rows a block: ten rows run as four, the last padded.
    monkeypatch.setattr(blockwise, "BLOCK_BYTES", 4 * 7 * 3)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    # f32 throughout: a generated module turns on 64-bit defaults process-wide.
    hidden = jax.random.normal(keys[0], (10, 4), dtype=jnp.float32)
    weight = jax.random.normal(keys[1], (7, 4), dtype=jnp.float32)
    targets = jax.random.randint(keys[2], (10,), 0, 7, dtype=jnp.int32)
    weights = jax.random.uniform(keys[3], (10,), dtype=jnp.float32)

    def dense(h: Any, w: Any, m: Any) -> Any:
        return -jnp.sum(m * _dense_log_probs(h, w, targets))

    def mine(h: Any, w: Any, m: Any) -> Any:
        return blockwise.linear_cross_entropy(h, w, targets, m)

    for got, want in zip(
        jax.value_and_grad(mine, argnums=(0, 1, 2))(hidden, weight, weights),
        jax.value_and_grad(dense, argnums=(0, 1, 2))(hidden, weight, weights),
        strict=True,
    ):
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    cotangent = jax.random.normal(keys[3], (10,), dtype=jnp.float32)
    out, back = jax.vjp(
        lambda h, w: blockwise.linear_token_log_probs(h, w, targets), hidden, weight
    )
    ref, ref_back = jax.vjp(lambda h, w: _dense_log_probs(h, w, targets), hidden, weight)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    for a, b in zip(back(cotangent), ref_back(cotangent), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    from linnet.torch import load as load_torch

    skeleton = load_torch(LLAMA, generics=GENERICS, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): (torch.randn(p.shape, generator=generator) * 0.3).numpy()
        for name, p in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    directory = tmp_path / "weights"
    directory.mkdir()
    save_file(tensors, str(directory / "model.safetensors"))
    return directory


def _model(weights: Path | dict[str, Any]) -> Any:
    return load_source(
        LLAMA, generics=GENERICS, weights=weights, entry="loss_packed", std_root=STDLIB
    )


def _examples() -> list[Example]:
    generator = torch.Generator().manual_seed(1)
    return [
        Example.prompted(
            torch.randint(0, 11, (2,), generator=generator).tolist(),
            torch.randint(0, 11, (n - 2,), generator=generator).tolist(),
        )
        for n in [5, 6, 4, 7]
    ]


def test_loss_packed_matches_pytorch(weights: Path) -> None:
    from linnet.torch import load as load_torch
    from linnet.train import Batch

    batch = next(pack(_examples(), tokens=24))
    arrays = batch.arrays(batch.count)
    model = _model(weights)
    params = model.parameters_for(*arrays)
    loss, grads = jax.value_and_grad(lambda p: model.apply(p, *arrays))(params)

    theirs = load_torch(
        LLAMA, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True, trainable=True
    )
    expected = theirs.run_entry("loss_packed", Batch.of(batch).inputs(batch.count))
    expected.backward()
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5, atol=1e-5)
    for name, parameter in theirs.named_parameters():
        path = name.removeprefix("root.")
        if parameter.grad is not None and path in grads:
            np.testing.assert_allclose(
                np.asarray(grads[path]), parameter.grad.numpy(), rtol=1e-4, atol=1e-5
            )


def test_training_lowers_the_loss(weights: Path) -> None:
    batch = next(pack(_examples(), tokens=24))
    params, history = train(
        _model(weights),
        itertools.repeat(batch),
        optimizer=optax.adamw(3e-2),
        steps=12,
        accumulate=2,
    )
    assert len(history.steps) == 12
    assert history.losses[-1] < history.losses[0] * 0.8
    assert all(step.tokens == 48 and step.grad_norm is not None for step in history.steps)
    assert params["embedding.weight"].dtype == jnp.float32


def _one_step(model: Any, batches: list[Packed], **options: Any) -> dict[str, Any]:
    params, _ = train(model, iter(batches), optimizer=optax.sgd(0.1), steps=1, clip=None, **options)
    return params


def test_accumulated_batches_follow_the_mean_over_all_of_them(weights: Path) -> None:
    examples = _examples()
    halves = [next(pack(examples[:2], tokens=12)), next(pack(examples[2:], tokens=12))]
    whole = [next(pack(examples, tokens=24))]
    accumulated = _one_step(_model(weights), halves, accumulate=2)
    together = _one_step(_model(weights), whole)
    for path, value in together.items():
        np.testing.assert_allclose(accumulated[path], value, rtol=1e-4, atol=1e-6)


def test_tied_paths_train_as_one_and_frozen_ones_stay(weights: Path) -> None:
    from safetensors.numpy import load_file  # type: ignore[import-untyped]

    loaded = load_file(str(weights / "model.safetensors"))
    loaded["lm_head.weight"] = loaded["embedding.weight"]  # one tensor, two paths
    batch = [next(pack(_examples(), tokens=24))]
    model = _model(loaded)
    start = model.parameters_for(*batch[0].arrays(1.0))
    assert start["lm_head.weight"] is start["embedding.weight"]
    params = _one_step(model, batch, trainable=["embedding.*", "lm_head.*", "norm.*"])
    assert params["lm_head.weight"] is params["embedding.weight"]

    # The shared step is the sum of both paths' gradients.
    arrays = batch[0].arrays(batch[0].count)
    untied = {**start, "lm_head.weight": jnp.array(start["embedding.weight"])}
    grads = jax.grad(lambda p: model.apply(p, *arrays))(untied)
    expected = start["embedding.weight"] - 0.1 * (
        grads["embedding.weight"] + grads["lm_head.weight"]
    )
    np.testing.assert_allclose(params["embedding.weight"], expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(
        params["layers.0.mlp.gate.weight"], start["layers.0.mlp.gate.weight"]
    )


def _four_batches() -> list[Packed]:
    examples = _examples() * 2
    return [next(pack(examples[i : i + 2], tokens=12)) for i in range(0, 8, 2)]


def test_a_resumed_run_ends_where_an_unbroken_one_does(weights: Path, tmp_path: Path) -> None:
    def run(steps: int, checkpoint: Path | None) -> tuple[dict[str, Any], list[int]]:
        params, history = train(
            _model(weights),
            iter(_four_batches()),
            optimizer=optax.adamw(3e-2),
            steps=steps,
            checkpoint=checkpoint,
            checkpoint_every=1 if checkpoint is not None else None,
        )
        return params, [step.step for step in history.steps]

    unbroken, numbers = run(4, None)
    assert numbers == [1, 2, 3, 4]
    assert run(2, tmp_path / "run")[1] == [1, 2]
    resumed, numbers = run(4, tmp_path / "run")
    assert numbers == [3, 4]
    for path, value in unbroken.items():
        np.testing.assert_allclose(resumed[path], value, rtol=1e-5, atol=1e-6)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == [
        "step-00000003",
        "step-00000004",
    ]


def test_saved_weights_load_back(weights: Path, tmp_path: Path) -> None:
    from linnet.jax.train import save_weights

    params, _ = train(
        _model(weights), iter(_four_batches()), optimizer=optax.sgd(0.1), steps=1, accumulate=2
    )
    save_weights(params, tmp_path / "trained.safetensors")
    again = _model(tmp_path / "trained.safetensors").parameters_for(*_four_batches()[0].arrays(1.0))
    for path, value in again.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(params[path]))


MESH_SCRIPT = """
import sys
from pathlib import Path

import jax
import numpy as np
import optax
from jax.sharding import Mesh

sys.path.insert(0, sys.argv[2])
from test_train import GENERICS, LLAMA, STDLIB, _four_batches, _model  # noqa: E402

from linnet.jax import load_source  # noqa: E402
from linnet.jax.train import train  # noqa: E402

weights = Path(sys.argv[1])
assert len(jax.devices()) == 2
mesh = Mesh(np.array(jax.devices()), ("data",))
split, _ = train(_model(weights), iter(_four_batches()), optimizer=optax.sgd(0.1), steps=2,
                 clip=None, mesh=mesh)
one, _ = train(_model(weights), iter(_four_batches()), optimizer=optax.sgd(0.1), steps=2,
               clip=None, accumulate=2)
gate = split["layers.0.mlp.gate.weight"]
assert len(gate.sharding.device_set) == 2 and gate.sharding.spec[0] == "data", gate.sharding
for path, value in one.items():
    np.testing.assert_allclose(np.asarray(split[path]), np.asarray(value), rtol=1e-4, atol=1e-6)


# Computing in bf16 from f32 masters: the parts are gathered in bf16, the
# gradients reduced in f32, as one device casts and accumulates.
def bf16():
    return load_source(LLAMA, generics={**GENERICS, "T": "bf16"}, weights=weights,
                       entry="loss_packed", std_root=STDLIB, cast_dtype=True)


split, _ = train(bf16(), iter(_four_batches()), optimizer=optax.sgd(0.1), steps=2, clip=None,
                 mesh=mesh)
one, _ = train(bf16(), iter(_four_batches()), optimizer=optax.sgd(0.1), steps=2, clip=None,
               accumulate=2)
for path, value in one.items():
    assert split[path].dtype == np.float32, (path, split[path].dtype)
    np.testing.assert_allclose(np.asarray(split[path]), np.asarray(value), rtol=1e-3, atol=1e-5)
print("same")
"""


def test_sharded_training_over_a_mesh_matches_one_device(weights: Path, tmp_path: Path) -> None:
    """Two host devices, each with its own batch and part of every weight,
    take the steps one device takes accumulating both batches."""
    import os
    import subprocess
    import sys

    script = tmp_path / "mesh.py"
    script.write_text(MESH_SCRIPT, encoding="utf-8")
    environment = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    completed = subprocess.run(
        [sys.executable, str(script), str(weights), str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0 and "same" in completed.stdout, completed.stderr[-3000:]


ADAPTED = "layers.*.attention.*_proj.weight"


def test_adapters_start_as_the_base_and_train_alone(weights: Path) -> None:
    from linnet.jax.train import merge_lora

    batch = next(pack(_examples(), tokens=24))
    arrays = batch.arrays(batch.count)
    base = _model(weights)
    base_loss = float(base(*arrays))
    model = _model(weights)
    model.add_lora(ADAPTED, rank=2, alpha=4)
    start = model.parameters_for(*arrays)
    adapters = sorted(path for path in start if "lora" in path)
    assert len(adapters) == 16
    np.testing.assert_allclose(float(model.apply(start, *arrays)), base_loss, rtol=1e-6)

    params, history = train(model, itertools.repeat(batch), optimizer=optax.adamw(5e-2), steps=10)
    assert history.losses[-1] < history.losses[0]
    for path in start:
        if "lora" not in path:
            np.testing.assert_array_equal(np.asarray(params[path]), np.asarray(start[path]))

    # Merged into the weights, the plain model computes what the adapted one does.
    merged = merge_lora(model, params)
    assert not any("lora" in path for path in merged)
    np.testing.assert_allclose(
        float(base.apply(merged, *arrays)), float(model.apply(params, *arrays)), rtol=1e-5
    )


def test_adapters_match_pytorch(weights: Path) -> None:
    from linnet.torch import load as load_torch
    from linnet.train import Batch

    batch = next(pack(_examples(), tokens=24))
    arrays = batch.arrays(batch.count)
    model = _model(weights)
    model.add_lora(ADAPTED, rank=2, alpha=4)
    params = dict(model.parameters_for(*arrays))
    # A nonzero B, so both adapters get gradients.
    for path in params:
        if path.endswith("lora_b"):
            params[path] = jnp.full(params[path].shape, 0.05, jnp.float32)
    loss, grads = jax.value_and_grad(lambda p: model.apply(p, *arrays))(params)

    theirs = load_torch(LLAMA, generics=GENERICS, std_root=STDLIB, weights=weights, compile=True)
    theirs.add_lora(ADAPTED, rank=2, alpha=4)
    with torch.no_grad():
        for name, parameter in theirs.named_parameters():
            path = name.removeprefix("root.")
            if "lora" in path:
                parameter.copy_(torch.from_numpy(np.asarray(params[path])))
    expected = theirs.run_entry("loss_packed", Batch.of(batch).inputs(batch.count))
    expected.backward()
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5, atol=1e-6)
    for name, parameter in theirs.named_parameters():
        path = name.removeprefix("root.")
        if "lora" in path:
            assert parameter.grad is not None
            np.testing.assert_allclose(
                np.asarray(grads[path]), parameter.grad.numpy(), rtol=1e-4, atol=1e-6
            )

"""A Linnet model as a Flax NNX module: the block hierarchy becomes module
state under the parameter paths, and calls read the module's arrays."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linnet.jax import export_linnet, load, load_nnx

from .test_round_trip import STDLIB, forward, init_params

nnx = pytest.importorskip("flax.nnx")


def test_exported_model_as_nnx_module(tmp_path: Path) -> None:
    params = init_params(jax.random.PRNGKey(0))
    tokens = jnp.array([[1, 4, 7, 2, 9], [3, 3, 0, 10, 5]], dtype=jnp.int32)
    result = export_linnet(
        forward,
        params,
        (tokens,),
        output=tmp_path / "src/model.linnet",
        weights=tmp_path / "weights",
        std_root=STDLIB,
    )
    model = load_nnx(result.source, generics={}, weights=tmp_path / "weights", std_root=STDLIB)

    # The hierarchy: a sub array of layers, parameters at their paths.
    assert type(model).__name__ == "Model"
    assert isinstance(model.layers, nnx.List) and len(model.layers) == 2
    assert isinstance(model.layers[1].q, nnx.Param) and model.layers[1].q.shape == (8, 8)
    flat = {"/".join(map(str, k)) for k, _ in nnx.to_flat_state(nnx.state(model))}
    assert {"embedding", "head", "layers/0/q", "layers/1/ob"} <= flat

    expected = forward(params, tokens)
    np.testing.assert_allclose(np.asarray(model(tokens)), np.asarray(expected), atol=1e-5)

    # The call reads the module's arrays: change one, and the result follows.
    model.head[...] = model.head[...] * 2
    params["head"] = params["head"] * 2
    np.testing.assert_allclose(
        np.asarray(model(tokens)), np.asarray(forward(params, tokens)), atol=1e-5
    )

    # Split and merge, the functional NNX round trip, gives a working module.
    graphdef, state = nnx.split(model)
    merged = nnx.merge(graphdef, state)
    np.testing.assert_allclose(
        np.asarray(merged(tokens)), np.asarray(forward(params, tokens)), atol=1e-5
    )


def test_apply_with_new_parameters(tmp_path: Path) -> None:
    """`LinnetFunction.apply` runs the entry with arrays from the caller."""
    params = init_params(jax.random.PRNGKey(1))
    tokens = jnp.array([[2, 5, 8]], dtype=jnp.int32)
    result = export_linnet(
        forward, params, (tokens,), output=tmp_path / "model.linnet", std_root=STDLIB
    )
    weights = {
        "embedding": params["embedding"],
        "head": params["head"],
        **{
            f"layers.{i}.{name}": array
            for i, layer in enumerate(params["layers"])
            for name, array in layer.items()
        },
    }
    function = load(result.source, generics={}, weights=weights, std_root=STDLIB)
    scaled = {path: array * 0.5 for path, array in weights.items()}

    def half(array: jax.Array) -> jax.Array:
        return array * 0.5

    halved = jax.tree_util.tree_map(half, params)
    np.testing.assert_allclose(
        np.asarray(function.apply(scaled, tokens)), np.asarray(forward(halved, tokens)), atol=1e-5
    )


def test_nnx_module_trains_over_generated_source(tmp_path: Path) -> None:
    """An NNX module built on `load_source` differentiates with `nnx.grad`."""
    from linnet.jax import load_source, to_nnx

    source = tmp_path / "mlp.linnet"
    source.write_text(
        "module regression\n\nuse std.nn.linear::{Linear}\n\n"
        "pub block Mlp<In: Dim, Out: Dim> {\n    sub layer: Linear<In, Out, f32>\n\n"
        "    pub entry forward<B: Dim>(x: Tensor[B, In; f32]) -> Tensor[B, Out; f32] {\n"
        "        return layer.forward(x)\n    }\n}\n"
    )
    weights = {
        "layer.weight": jnp.zeros((1, 3), jnp.float32),
        "layer.bias": jnp.zeros((1,), jnp.float32),
    }
    model = to_nnx(
        load_source(source, generics={"In": 3, "Out": 1}, weights=weights, std_root=STDLIB)
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (128, 3), jnp.float32)
    y = x @ jnp.array([[1.0, -2.0, 0.5]], jnp.float32).T

    def loss(module: object) -> jax.Array:
        return jnp.mean((module(x) - y) ** 2)  # type: ignore[operator]

    first = float(loss(model))
    for _ in range(100):
        grads = nnx.grad(loss)(model)
        flat = {"/".join(map(str, k)): v for k, v in nnx.to_flat_state(grads)}
        model.layer.weight[...] = model.layer.weight[...] - 0.1 * flat["layer/weight"]
        model.layer.bias[...] = model.layer.bias[...] - 0.1 * flat["layer/bias"]
    assert float(loss(model)) < first * 0.05

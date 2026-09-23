"""`load_source` runs entries as generated JAX code: it agrees with the
StableHLO-compiled `load`, threads state, loops with `lax.while_loop`, and
differentiates under `jax.grad`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from test_round_trip import REPO, STDLIB, forward, init_params

from linnet_jax import export_linnet, load, load_source

MLP = """\
module regression

use std.nn.activations::{silu}
use std.nn.linear::{Linear}

pub block Mlp<In: Dim, Hidden: Dim, Out: Dim> {
    sub up: Linear<In, Hidden, f32>
    sub down: Linear<Hidden, Out, f32>

    pub entry forward<B: Dim>(x: Tensor[B, In; f32]) -> Tensor[B, Out; f32] {
        return down.forward(silu(up.forward(x)))
    }
}
"""


def test_source_matches_compiled_stablehlo(tmp_path: Path) -> None:
    params = init_params(jax.random.PRNGKey(0))
    tokens = jnp.array([[1, 4, 7, 2, 9], [3, 3, 0, 10, 5]], dtype=jnp.int32)
    result = export_linnet(
        forward,
        params,
        (tokens,),
        output=tmp_path / "model.linnet",
        weights=tmp_path / "w",
        std_root=STDLIB,
    )
    compiled = load(result.source, generics={}, weights=tmp_path / "w", std_root=STDLIB)
    generated = load_source(result.source, generics={}, weights=tmp_path / "w", std_root=STDLIB)
    np.testing.assert_allclose(
        np.asarray(generated(tokens)), np.asarray(compiled(tokens)), atol=1e-5, rtol=1e-5
    )
    assert "def main(" in generated.generated_source()


def test_source_threads_state_and_loops() -> None:
    fixture = REPO / "spec-tests/valid/020_state.linnet"
    step = load_source(fixture, generics={"B": 1, "Max": 4, "D": 3}, weights={}, std_root=STDLIB)
    reference = load(fixture, generics={"B": 1, "Max": 4, "D": 3}, weights={}, std_root=STDLIB)
    key = jnp.arange(3, dtype=jnp.float32).reshape(1, 1, 3)
    out, state = step(key, jnp.int32(0))
    expected, expected_state = reference(key, jnp.int32(0))
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected))
    for path in expected_state:
        np.testing.assert_allclose(np.asarray(state[path]), np.asarray(expected_state[path]))

    loop = REPO / "spec-tests/valid/023_while.linnet"
    run = load_source(
        loop,
        generics={"N": 3},
        weights={"scale": np.array([2.0, 3.0, 1.5], np.float32)},
        std_root=STDLIB,
    )
    out = run(jnp.ones(3, jnp.float32), jnp.int32(100))
    np.testing.assert_allclose(np.asarray(out), np.array([128.0, 2187.0, 17.0859375], np.float32))
    assert "jax.lax.while_loop" in run.generated_source()


def test_jax_grad_trains_a_linnet_mlp(tmp_path: Path) -> None:
    source = tmp_path / "mlp.linnet"
    source.write_text(MLP)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)
    weights = {
        "up.weight": jax.random.normal(keys[0], (16, 3), jnp.float32) * 0.3,
        "up.bias": jnp.zeros((16,), jnp.float32),
        "down.weight": jax.random.normal(keys[1], (1, 16), jnp.float32) * 0.3,
        "down.bias": jnp.zeros((1,), jnp.float32),
    }
    f = load_source(
        source, generics={"In": 3, "Hidden": 16, "Out": 1}, weights=weights, std_root=STDLIB
    )
    x = jax.random.normal(keys[2], (256, 3), jnp.float32)
    y = x @ jnp.array([[1.0, -2.0, 0.5]], jnp.float32).T + 0.1
    f(x)  # compiles for this shape; `f.parameters` now holds the device arrays

    def loss(params: dict[str, jax.Array]) -> jax.Array:
        return jnp.mean((f.apply(params, x) - y) ** 2)

    grad = jax.jit(jax.grad(loss))
    params = dict(f.parameters)
    first = float(loss(params))
    for _ in range(200):
        grads = grad(params)
        params = {name: value - 0.05 * grads[name] for name, value in params.items()}
    last = float(loss(params))
    assert last < first * 0.05, (first, last)


def test_fast_numerics_skips_f32_accumulation() -> None:
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 16,
        "Layers": 1,
        "Batch": 2,
        "MaxSeq": 6,
        "T": "bf16",
    }
    llama = REPO / "examples/05-llama/src/lib.linnet"
    shapes = {"embedding.weight": (11, 8), "norm.weight": (8,), "lm_head.weight": (11, 8)}
    for name, shape in (
        ("q_proj", (8, 8)),
        ("k_proj", (4, 8)),
        ("v_proj", (4, 8)),
        ("o_proj", (8, 8)),
    ):
        shapes[f"layers.0.attention.{name}.weight"] = shape
    shapes["layers.0.attention_norm.weight"] = (8,)
    shapes["layers.0.mlp_norm.weight"] = (8,)
    shapes["layers.0.mlp.gate.weight"] = (16, 8)
    shapes["layers.0.mlp.up.weight"] = (16, 8)
    shapes["layers.0.mlp.down.weight"] = (8, 16)
    key = jax.random.PRNGKey(0)
    weights = {
        name: (jax.random.normal(jax.random.fold_in(key, i), shape) * 0.3).astype(jnp.bfloat16)
        for i, (name, shape) in enumerate(shapes.items())
    }
    tokens = jnp.array([[1, 4, 7, 2, 9], [3, 3, 0, 10, 5]], dtype=jnp.int32)

    def make(loader: Any, numerics: str) -> Any:
        return loader(
            llama,
            generics=generics,
            weights=weights,
            std_root=STDLIB,
            entry="forward",
            numerics=numerics,
        )

    exact = make(load_source, "exact")
    fast = make(load_source, "fast")
    compiled = make(load, "fast")
    reference = np.asarray(exact(tokens), np.float32)
    np.testing.assert_allclose(np.asarray(fast(tokens), np.float32), reference, atol=0.1, rtol=0.1)
    np.testing.assert_allclose(
        np.asarray(compiled(tokens), np.float32), reference, atol=0.1, rtol=0.1
    )
    source = fast.generated_source()
    assert "jax.nn.softmax(" in source
    # Only the RoPE index arithmetic casts to f32; attention and the norms do not.
    assert ".astype(jnp.float32)" not in source.split("jnp.einsum(", 1)[1]

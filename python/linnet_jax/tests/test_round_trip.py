"""JAX -> Linnet -> JAX: a function exported to source checks, formats, loads
back with the same weights, and agrees with the original; Linnet examples run
in JAX through the same `load`."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linnet_jax import ExportError, export_linnet, find_compiler, load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
EXAMPLES = REPO / "examples"


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def _compiler_ok(*args: str) -> None:
    completed = subprocess.run(
        [find_compiler(), *args], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def rms_norm(x: Any, weight: Any) -> Any:
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-5) * weight


def layer(params: dict[str, Any], x: Any) -> Any:
    batch, seq, width = x.shape
    heads = 2
    h = rms_norm(x, params["norm"])
    q = (h @ params["q"]).reshape(batch, seq, heads, width // heads).transpose(0, 2, 1, 3)
    k = (h @ params["k"]).reshape(batch, seq, heads, width // heads).transpose(0, 2, 1, 3)
    v = (h @ params["v"]).reshape(batch, seq, heads, width // heads).transpose(0, 2, 1, 3)
    score = jnp.einsum("bhqd,bhkd->bhqk", q, k) * 0.5
    mask = jnp.tril(jnp.ones((seq, seq), dtype=bool))
    score = jnp.where(mask, score, -1e30)
    mixed = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(score, axis=-1), v)
    return x + mixed.transpose(0, 2, 1, 3).reshape(batch, seq, width) @ params["o"] + params["ob"]


def forward(params: dict[str, Any], tokens: Any) -> Any:
    x = jnp.take(params["embedding"], tokens, axis=0)
    for block in params["layers"]:
        x = layer(block, x)
    return jax.nn.silu(x @ params["head"])


def _init(key: Any) -> dict[str, Any]:
    keys = jax.random.split(key, 8)
    scale = 0.3

    def dense(k: Any, shape: tuple[int, ...]) -> Any:
        return jax.random.normal(k, shape, dtype=jnp.float32) * scale

    return {
        "embedding": dense(keys[0], (11, 8)),
        "layers": [
            {
                "norm": jnp.ones((8,), jnp.float32),
                "q": dense(keys[1 + i], (8, 8)),
                "k": dense(keys[3 + i], (8, 8)),
                "v": dense(keys[5 + i], (8, 8)),
                "o": dense(keys[7], (8, 8)) * (i + 1),
                "ob": jnp.zeros((8,), jnp.float32),
            }
            for i in range(2)
        ],
        "head": dense(keys[6], (8, 11)),
    }


def test_transformer_round_trip(tmp_path: Path) -> None:
    params = _init(jax.random.PRNGKey(0))
    tokens = jnp.array([[1, 4, 7, 2, 9], [3, 3, 0, 10, 5]], dtype=jnp.int32)
    result = export_linnet(
        forward,
        params,
        (tokens,),
        output=tmp_path / "src/model.linnet",
        weights=tmp_path / "weights",
        std_root=STDLIB,
    )
    source = result.source.read_text()
    assert "sub layers: [Layer; 2]" in source
    assert "param embedding: Tensor[11, 8; f32]" in source
    assert result.bindings is None
    _compiler_ok("lint", "--std", str(STDLIB), str(result.source))
    _compiler_ok("fmt", "--check", str(result.source))

    model = load(result.source, generics={}, weights=tmp_path / "weights", std_root=STDLIB)
    expected = forward(params, tokens)
    actual = jax.jit(model)(tokens)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)

    again = export_linnet(
        forward, params, (tokens,), output=tmp_path / "again/model.linnet", std_root=STDLIB
    )
    assert again.source.read_text() == source


def test_linnet_example_runs_in_jax(tmp_path: Path) -> None:
    """The hand-written transformer example, materialized through JAX, agrees
    with the same model's `jnp` reference for its weights."""
    source = EXAMPLES / "09-tiny-transformer/src/lib.linnet"
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 2,
        "Inner": 16,
        "Layers": 2,
        "T": "f32",
    }
    key = jax.random.PRNGKey(1)
    shapes = {
        "embedding.weight": (11, 8),
        "norm.weight": (8,),
        "head.weight": (11, 8),
    }
    for i in range(2):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes[f"layers.{i}.attention.{name}.weight"] = (8, 8)
        shapes[f"layers.{i}.attention_norm.weight"] = (8,)
        shapes[f"layers.{i}.mlp_norm.weight"] = (8,)
        shapes[f"layers.{i}.mlp.gate.weight"] = (16, 8)
        shapes[f"layers.{i}.mlp.up.weight"] = (16, 8)
        shapes[f"layers.{i}.mlp.down.weight"] = (8, 16)
    weights: dict[str, Any] = {}
    for name, shape in shapes.items():
        key, sub = jax.random.split(key)
        weights[name] = np.asarray(jax.random.normal(sub, shape, dtype=jnp.float32) * 0.3)
    model = load(source, generics=generics, weights=weights, std_root=STDLIB)

    seq, head = 5, 4
    tokens = jnp.array([[1, 4, 7, 2, 9], [3, 3, 0, 10, 5]], dtype=jnp.int32)
    positions = jnp.arange(seq, dtype=jnp.float32)[:, None]
    frequencies = 1.0 / (10000 ** (jnp.arange(0, head, 2, dtype=jnp.float32) / head))
    angles = positions * frequencies
    cos_table = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)
    sin_table = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)
    out = model(tokens, cos_table, sin_table)
    assert out.shape == (2, seq, 11)

    def rope(x: Any) -> Any:
        first, second = x[..., : head // 2], x[..., head // 2 :]
        return x * cos_table + jnp.concatenate([-second, first], axis=-1) * sin_table

    def split(x: Any) -> Any:
        return x.reshape(2, seq, 2, head).transpose(0, 2, 1, 3)

    x = weights["embedding.weight"][tokens]
    mask = jnp.tril(jnp.ones((seq, seq), dtype=bool))
    for i in range(2):
        p = f"layers.{i}."
        h = rms_norm(x, weights[p + "attention_norm.weight"])
        q, k, v = (split(h @ weights[p + f"attention.{n}_proj.weight"].T) for n in ("q", "k", "v"))
        score = jnp.einsum("bhqd,bhkd->bhqk", rope(q), rope(k)) * 0.125
        score = jnp.where(mask, score, -1e30)
        mixed = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(score, axis=-1), v)
        merged = mixed.transpose(0, 2, 1, 3).reshape(2, seq, 8)
        x = x + merged @ weights[p + "attention.o_proj.weight"].T
        h = rms_norm(x, weights[p + "mlp_norm.weight"])
        gate, up = h @ weights[p + "mlp.gate.weight"].T, h @ weights[p + "mlp.up.weight"].T
        x = x + (jax.nn.silu(gate) * up) @ weights[p + "mlp.down.weight"].T
    reference = rms_norm(x, weights["norm.weight"]) @ weights["head.weight"].T
    np.testing.assert_allclose(np.asarray(out), np.asarray(reference), atol=1e-4, rtol=1e-4)


def test_unsupported_operations_are_named(tmp_path: Path) -> None:
    def odd(params: dict[str, Any], x: Any) -> Any:
        return jax.lax.erf(x * params["w"])

    with pytest.raises(ExportError, match="erf"):
        export_linnet(
            odd, {"w": jnp.ones((4,))}, (jnp.ones((2, 4)),), output=tmp_path / "odd.linnet"
        )

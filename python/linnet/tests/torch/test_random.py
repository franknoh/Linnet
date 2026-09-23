"""`std.random` draws the same numbers as JAX's Threefry PRNG, running as
ordinary Linnet arithmetic in PyTorch: keys, bits, and uniforms match bit for
bit; normals and categorical draws have the right statistics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from linnet.torch import load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

jax = pytest.importorskip("jax")

SOURCE = """\
module rng_check

use std.random::{bits, categorical, fold_in, normal, split, uniform}

pub block Rng<N: Dim> {
    param unused: Tensor[1; f32]

    pub entry split_keys(key: Tensor[2; i64]) -> Tensor[N, 2; i64] {
        return split<N>(key)
    }

    pub entry random_bits(key: Tensor[2; i64]) -> Tensor[N; i64] {
        return bits<N>(key)
    }

    pub entry uniforms(key: Tensor[2; i64]) -> Tensor[N; f32] {
        return uniform<N, f32>(key)
    }

    pub entry normals(key: Tensor[2; i64]) -> Tensor[N; f32] {
        return normal<N, f32>(key)
    }

    pub entry folded(key: Tensor[2; i64], data: i32) -> Tensor[2; i64] {
        return fold_in(key, data)
    }

    pub entry draw(key: Tensor[2; i64], logits: Tensor[N, 4; f32]) -> Tensor[N; i32] {
        return categorical(key, logits)
    }
}
"""


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def _key(seed: int) -> torch.Tensor:
    data = np.asarray(jax.random.key_data(jax.random.key(seed))).astype(np.int64)
    return torch.tensor(data)


def test_keys_bits_and_uniforms_match_jax(tmp_path: Path) -> None:
    source = tmp_path / "rng.linnet"
    source.write_text(SOURCE)
    n = 64
    model = load(source, generics={"N": n}, std_root=STDLIB)
    for seed in (0, 42, 123456):
        key = jax.random.key(seed)
        keys = model.run_entry("split_keys", [_key(seed)]).numpy()
        expected = np.asarray(jax.random.key_data(jax.random.split(key, n))).astype(np.int64)
        np.testing.assert_array_equal(keys, expected)

        words = model.run_entry("random_bits", [_key(seed)]).numpy()
        expected_bits = np.asarray(jax.random.bits(key, (n,), dtype=jax.numpy.uint32))
        np.testing.assert_array_equal(words, expected_bits.astype(np.int64))

        uniforms = model.run_entry("uniforms", [_key(seed)]).numpy()
        expected_uniform = np.asarray(jax.random.uniform(key, (n,), dtype=jax.numpy.float32))
        np.testing.assert_array_equal(uniforms, expected_uniform)

        folded = model.run_entry("folded", [_key(seed), torch.tensor(7, dtype=torch.int32)])
        expected_fold = np.asarray(jax.random.key_data(jax.random.fold_in(key, 7)))
        np.testing.assert_array_equal(folded.numpy(), expected_fold.astype(np.int64))


def test_normal_and_categorical_statistics(tmp_path: Path) -> None:
    source = tmp_path / "rng.linnet"
    source.write_text(SOURCE)
    n = 4096
    model = load(source, generics={"N": n}, std_root=STDLIB)
    normals = model.run_entry("normals", [_key(3)])
    assert abs(float(normals.mean())) < 0.05 and abs(float(normals.std()) - 1.0) < 0.05

    # Category 2 has probability e^2 / (1 + e + e^2 + e^3) ≈ 0.24 in every row.
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat(n, 1)
    draws = model.run_entry("draw", [_key(5), logits])
    counts = torch.bincount(draws.long(), minlength=4).float() / n
    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0, 3.0]), 0)
    torch.testing.assert_close(counts, expected, atol=0.03, rtol=0)

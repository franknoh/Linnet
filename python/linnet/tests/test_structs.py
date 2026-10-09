"""Struct values: built by calling the struct's name with every field, read
by field, passed through functions, and returned from an entry as the tuple
of their fields. Every backend computes the same numbers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from linnet.torch import load_function

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.structs

struct Moments<N: Dim, T: Float> {
    mean: Tensor[N; T]
    scale: Tensor[N; T]
}

fn moments<N: Dim, T: Float>(x: Tensor[N; T]) -> Moments<N, T> {
    let total = sum[n] x[n]
    let mean[n] = total / cast<T>(N) + x[n] * 0.0
    return Moments(scale = x * x, mean = mean)
}

pub entry normalized<N: Dim>(x: Tensor[N; f32]) -> Tensor[N; f32] {
    let m = moments(x)
    let flipped = Moments(mean = m.scale, scale = m.mean)
    return (x - m.mean) * flipped.mean
}

pub entry stats<N: Dim>(x: Tensor[N; f32]) -> Moments<N, f32> {
    return moments(x)
}
"""


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "structs.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


X = np.array([1.0, 2.0, 4.0, 5.0], dtype=np.float32)
MEAN = np.full(4, X.mean(), dtype=np.float32)
NORMALIZED = (X - MEAN) * X * X


@pytest.mark.parametrize("compile", [False, True])
def test_torch(source: Path, compile: bool) -> None:
    normalized = load_function(source, "normalized", std_root=STDLIB, compile=compile)
    np.testing.assert_allclose(normalized(torch.tensor(X)).numpy(), NORMALIZED, rtol=1e-6)
    stats = load_function(source, "stats", std_root=STDLIB, compile=compile)
    mean, scale = stats(torch.tensor(X))
    np.testing.assert_allclose(mean.numpy(), MEAN, rtol=1e-6)
    np.testing.assert_allclose(scale.numpy(), X * X, rtol=1e-6)


def test_jax(source: Path) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load_function as load_jax

    normalized = load_jax(source, "normalized", std_root=STDLIB)
    np.testing.assert_allclose(np.asarray(normalized(X)), NORMALIZED, rtol=1e-6)
    mean, scale = load_jax(source, "stats", std_root=STDLIB)(X)
    np.testing.assert_allclose(np.asarray(mean), MEAN, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(scale), X * X, rtol=1e-6)


def test_onnx(source: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_function

    for name, expected in (("normalized", [NORMALIZED]), ("stats", [MEAN, X * X])):
        exported = export_function(source, name, generics={"N": 4}, std_root=STDLIB)
        session = onnxruntime.InferenceSession(
            exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = session.run(None, {session.get_inputs()[0].name: X})
        assert len(got) == len(expected)
        for value, want in zip(got, expected, strict=True):
            np.testing.assert_allclose(value, want, rtol=1e-6)

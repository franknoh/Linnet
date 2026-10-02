"""A runtime integer index outside index notation -- `x[i]`, `x[i, :]`,
`x[:, j]`, `x[i, 1:3]` for an integer input -- reads the position it holds
in every backend: interpreted and generated PyTorch (with gradients through
it), generated JAX, ONNX Runtime, and StableHLO through JAX."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from linnet.torch import load, load_function

SOURCE = """\
module tests.runtime_index

pub entry element<N: Dim>(x: Tensor[N; f32], i: i64) -> f32 {
    return x[i]
}

pub entry row<N: Dim, M: Dim>(x: Tensor[N, M; f32], i: i64) -> Tensor[M; f32] {
    return x[i, :]
}

pub entry column<N: Dim, M: Dim>(x: Tensor[N, M; f32], j: i32) -> Tensor[N; f32] {
    return x[:, j]
}

pub entry window<N: Dim, M: Dim>(x: Tensor[N, M; f32], i: i64) -> Tensor[2; f32]
where M >= 3 {
    return x[i, 1:3]
}

pub entry plane<A: Dim, B: Dim, C: Dim>(x: Tensor[A, B, C; f32], i: i64, k: i64) -> Tensor[B; f32] {
    return x[i, ..., k]
}

pub block Table<N: Dim, M: Dim> {
    param weight: Tensor[N, M; f32]

    pub entry lookup(i: i64) -> Tensor[M; f32] {
        return weight[i]
    }
}
"""

X2 = np.arange(20, dtype=np.float32).reshape(4, 5) / 7
X3 = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 5
# Each case: the function, its inputs, and what it must return.
CASES: list[tuple[str, tuple[Any, ...], np.ndarray]] = [
    ("element", (X2[1], 3), X2[1][3]),
    ("row", (X2, 2), X2[2]),
    ("column", (X2, 4), X2[:, 4]),
    ("window", (X2, 3), X2[3, 1:3]),
    ("plane", (X3, 1, 2), X3[1, :, 2]),
]


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "runtime_index.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize(("name", "inputs", "expected"), CASES)
def test_pytorch_reads_the_position(
    source: Path, compile: bool, name: str, inputs: tuple[Any, ...], expected: np.ndarray
) -> None:
    function = load_function(source, name, compile=compile)
    x = torch.tensor(inputs[0], requires_grad=True)
    got = function(x, *inputs[1:])
    np.testing.assert_allclose(got.detach().numpy(), expected, rtol=0, atol=0)
    # The gradient reaches exactly the positions read.
    got.sum().backward()
    assert x.grad is not None
    marked = np.zeros_like(inputs[0])
    np.put(marked, np.flatnonzero(np.isin(inputs[0], expected)), 1.0)
    np.testing.assert_array_equal(x.grad.numpy(), marked)


@pytest.mark.parametrize(("name", "inputs", "expected"), CASES)
def test_jax_reads_the_position(
    source: Path, name: str, inputs: tuple[Any, ...], expected: np.ndarray
) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load_function as load_jax_function

    function = load_jax_function(source, name)
    np.testing.assert_allclose(np.asarray(function(*inputs)), expected, rtol=0, atol=0)


@pytest.mark.parametrize(("name", "inputs", "expected"), CASES)
def test_onnx_runtime_reads_the_position(
    source: Path, name: str, inputs: tuple[Any, ...], expected: np.ndarray
) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_function

    shape = inputs[0].shape
    generics = dict(zip({"element": "N", "plane": "ABC"}.get(name, "NM"), shape, strict=True))
    exported = export_function(source, name, generics=generics)
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds: dict[str, Any] = {"x": inputs[0]}
    for port, value in zip(exported.inputs[1:], inputs[1:], strict=True):
        feeds[port.name] = np.array(value, dtype=np.int32 if port.dtype == "i32" else np.int64)
    (got,) = session.run(None, feeds)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=0, atol=0)


@pytest.mark.parametrize("compile", [False, True])
def test_a_parameter_is_read_at_a_runtime_row(source: Path, compile: bool) -> None:
    model = load(source, generics={"N": 4, "M": 5}, root="Table", compile=compile)
    with torch.no_grad():
        dict(model.named_parameters())["root.weight"].copy_(torch.tensor(X2))
    got = model.run_entry("lookup", [torch.tensor(2)])
    np.testing.assert_allclose(got.numpy(), X2[2], rtol=0, atol=0)


def test_stablehlo_reads_a_parameter_at_a_runtime_row(source: Path) -> None:
    jax = pytest.importorskip("jax")
    from linnet.jax import load as load_jax

    jax.config.update("jax_enable_x64", True)
    lookup = load_jax(source, generics={"N": 4, "M": 5}, weights={"weight": X2}, root="Table")
    got = lookup(jax.numpy.asarray(3, dtype=jax.numpy.int64))
    np.testing.assert_allclose(np.asarray(got), X2[3], rtol=0, atol=0)

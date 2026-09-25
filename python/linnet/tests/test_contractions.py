"""A sum of products written in index notation runs as a contraction
(`einsum`, `dot_general`, ONNX `Einsum`) in every export, never as the whole
product built and then summed -- and computes the same numbers.

Taken literally, `sum[c] q[b, h, i, j, c] * r[i, k, c]` builds a tensor with
every axis of both operands before summing one away: 51 GB for SAM's
relative-position term, 85 GiB for a mixture-of-experts projection. Two
out-of-memory kills of the whole machine traced back to it."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.compiler import find_compiler
from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.contract

pub block Model<T: Float = f32> {
    param a: Tensor[3, 4; T]
    param b: Tensor[4, 5; T]
    param x: Tensor[2, 3, 6; T]
    param w: Tensor[4, 5, 6; T]
    param q: Tensor[2, 3, 4, 5, 6; T]
    param r: Tensor[4, 7, 6; T]

    // A matrix product.
    pub entry plain() -> Tensor[3, 5; f32] {
        let c[i, j] = sum<f32>[k] cast<f32>(a[i, k]) * cast<f32>(b[k, j])
        return c
    }

    // The same, with the result's axes the other way round.
    pub entry swapped() -> Tensor[5, 3; f32] {
        let c[j, i] = sum<f32>[k] cast<f32>(a[i, k]) * cast<f32>(b[k, j])
        return c
    }

    // A mixture-of-experts projection: every token against every expert.
    pub entry experts() -> Tensor[2, 3, 4, 5; f32] {
        let y[n, s, e, o] = sum<f32>[h] cast<f32>(x[n, s, h]) * cast<f32>(w[e, o, h])
        return y
    }

    // SAM's decomposed relative position: `i` is shared and kept (a batch
    // axis of the contraction), `c` is shared and summed.
    pub entry relative() -> Tensor[2, 3, 4, 5, 7; f32] {
        let rel[n, h, i, j, k] = sum<f32>[c] cast<f32>(q[n, h, i, j, c]) * cast<f32>(r[i, k, c])
        return rel
    }

    // An axis only one operand has, summed away with the shared one.
    pub entry lopsided() -> Tensor[3; f32] {
        let s[i] = sum<f32>[k, j] cast<f32>(a[i, k]) * cast<f32>(b[k, j])
        return s
    }

    // A factor read through a computed index (a gather, as a mixture of
    // experts reads its chosen experts) is not a plain contraction.
    pub entry gathered() -> Tensor[5; f32] {
        let rows[k] = (iota<i64>(4)[k] + 1) % 4
        let s[j] = sum<f32>[k] cast<f32>(a[0, k]) * cast<f32>(b[rows[k], j])
        return s
    }

    // A product in a narrower type than its sum rounds before it adds, which a
    // contraction would not: this one must stay a product and a sum.
    pub entry narrow() -> Tensor[3, 5; f32] {
        let c[i, j] = sum<f32>[k] cast<f32>(a[i, k] * b[k, j])
        return c
    }
}
"""

ENTRIES = ["plain", "swapped", "experts", "relative", "lopsided", "gathered", "narrow"]


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def model(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    arrays = {
        "a": rng.standard_normal((3, 4)),
        "b": rng.standard_normal((4, 5)),
        "x": rng.standard_normal((2, 3, 6)),
        "w": rng.standard_normal((4, 5, 6)),
        "q": rng.standard_normal((2, 3, 4, 5, 6)),
        "r": rng.standard_normal((4, 7, 6)),
    }
    arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
    source = tmp_path / "contract.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    weights = tmp_path / "model.safetensors"
    save_file({k: torch.tensor(v) for k, v in arrays.items()}, str(weights))
    return source, weights, arrays


def _expected(entry: str, v: dict[str, np.ndarray]) -> np.ndarray:
    if entry in ("plain", "narrow"):
        return np.einsum("ik,kj->ij", v["a"], v["b"])
    if entry == "swapped":
        return np.einsum("ik,kj->ji", v["a"], v["b"])
    if entry == "experts":
        return np.einsum("nsh,eoh->nseo", v["x"], v["w"])
    if entry == "gathered":
        return np.einsum("k,kj->j", v["a"][0], v["b"][[1, 2, 3, 0]])
    if entry == "relative":
        return np.einsum("nhijc,ikc->nhijk", v["q"], v["r"])
    return np.einsum("ik,kj->i", v["a"], v["b"])


@pytest.mark.parametrize("entry", ENTRIES)
def test_compiled_torch(model: tuple[Path, Path, dict[str, np.ndarray]], entry: str) -> None:
    source, weights, arrays = model
    compiled = load(source, generics={}, std_root=STDLIB, weights=weights, compile=True)
    got = compiled.run_entry(entry, []).numpy()
    np.testing.assert_allclose(got, _expected(entry, arrays), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("entry", ENTRIES)
def test_xla(model: tuple[Path, Path, dict[str, np.ndarray]], entry: str) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load as load_jax

    source, weights, arrays = model
    function = load_jax(source, generics={}, weights=weights, std_root=STDLIB, entry=entry)
    got = np.asarray(function())
    np.testing.assert_allclose(got, _expected(entry, arrays), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("entry", ENTRIES)
def test_onnx(model: tuple[Path, Path, dict[str, np.ndarray]], entry: str) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model

    source, weights, arrays = model
    exported = export_model(source, generics={}, weights=weights, std_root=STDLIB, entry=entry)
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = session.run(None, {})
    np.testing.assert_allclose(got, _expected(entry, arrays), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("entry", "contracted"),
    [("plain", True), ("experts", True), ("relative", True), ("narrow", False)],
)
def test_the_product_is_never_built(tmp_path: Path, entry: str, contracted: bool) -> None:
    """The generated source says `einsum` where the body is a plain
    contraction, and keeps the product where rounding forbids it."""
    source = tmp_path / "contract.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    command = [find_compiler(), "torch", "--root", "Model", "--entry", entry]
    command += ["--std", str(STDLIB), str(source)]
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    assert ("torch.einsum(" in text) is contracted

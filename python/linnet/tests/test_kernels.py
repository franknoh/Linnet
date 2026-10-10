"""Kernels: an op with a `kernel` launches it -- a Triton program from
generated PyTorch, a Pallas kernel from generated JAX, run here in their
interpreters -- and computes what the op's body computes. Its gradient is
the op's `grad`, or else its body's."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import importlib.util
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

from linnet.compiler import find_compiler

REPO = Path(__file__).resolve().parents[3]
ATOMICS = REPO / "spec-tests/valid/031_kernel_atomics.linnet"
SCANS = REPO / "spec-tests/valid/032_cumsum.linnet"
PRODUCTS = REPO / "spec-tests/valid/033_kernel_prod.linnet"
# Set by conftest.py without a GPU, before Triton loaded: Triton then runs
# kernels in its interpreter.
TRITON_INTERPRETS = os.environ.get("TRITON_INTERPRET") == "1"
KERNELS = REPO / "spec-tests/valid/030_kernels.linnet"

GRAD = """\
module tests.kernel_grad

pub kernel twice<N: Dim, B: Dim>(x: Tensor[N; f32]) -> y: Tensor[N; f32] grid((N + B - 1) / B)
where B > 0 {
    let i = program_id(0) * B + iota<i32>(B)
    let inside = i < N
    store(y[i], load(x[i], inside) * 2.0, inside)
}

// Its gradient scaled by three, to tell it from the body's.
pub op double<N: Dim>(x: Tensor[N; f32]) -> Tensor[N; f32] kernel twice<N, 16> {
    return x * 2.0
} grad(y, dy) {
    return dy * 3.0
}

pub entry run<N: Dim>(x: Tensor[N; f32]) -> f32 {
    let y = double(x)
    return sum[n] y[n]
}
"""


@pytest.fixture(autouse=True)
def _interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                monkeypatch.setenv("LINNET_BIN", str(REPO / candidate))
                break
    # Generated code reads them: kernels run on the CPU, in interpreters.
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    monkeypatch.setenv("LINNET_PALLAS_INTERPRET", "1")


def _module(
    source: Path, numerics: str, *binds: str, target: str = "torch", entry: str = "run"
) -> tuple[str, Callable[..., tuple[object, ...]]]:
    command = [find_compiler(), target, "--entry", entry, "--numerics", numerics]
    for bind in binds:
        command += ["--bind", bind]
    completed = subprocess.run([*command, str(source)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    namespace: dict[str, object] = {}
    exec(compile(completed.stdout, "<generated>", "exec"), namespace)
    main = cast(Callable[..., tuple[object, ...]], namespace["main"])
    make = namespace.get("constants")
    hoisted = cast(Callable[[str], tuple[object, ...]], make)("cpu") if make else ()
    return completed.stdout, lambda *inputs: main(*inputs, *hoisted)


def test_kernels_compute_what_bodies_do() -> None:
    pytest.importorskip("triton")
    if not TRITON_INTERPRETS:
        pytest.skip("Triton's interpreter was not on as Triton loaded")
    # A tiled product over two K tiles, then a row softmax: tiles past the
    # edges are masked.
    binds = ("R=5", "C=40")
    text, kernels = _module(KERNELS, "equivalent", *binds)
    assert "@triton.jit" in text
    exact_text, bodies = _module(KERNELS, "exact", *binds)
    assert "triton" not in exact_text
    torch.manual_seed(0)
    x = torch.randn(5, 40)
    w = torch.randn(40, 40) * 0.2
    (got,) = kernels(x, w)
    (want,) = bodies(x, w)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    # Without a `grad`, the backward pass runs the bodies again.
    x_kernel = x.clone().requires_grad_()
    (out,) = cast(tuple[torch.Tensor], kernels(x_kernel, w))
    out.pow(2).sum().backward()
    x_body = x.clone().requires_grad_()
    (out,) = cast(tuple[torch.Tensor], bodies(x_body, w))
    out.pow(2).sum().backward()
    torch.testing.assert_close(x_kernel.grad, x_body.grad, rtol=1e-5, atol=1e-5)


def test_a_kernel_takes_the_ops_grad(tmp_path: Path) -> None:
    pytest.importorskip("triton")
    if not TRITON_INTERPRETS:
        pytest.skip("Triton's interpreter was not on as Triton loaded")
    source = tmp_path / "kernel_grad.linnet"
    source.write_text(GRAD, encoding="utf-8")
    text, run = _module(source, "equivalent", "N=37")
    assert "@triton.jit" in text
    x = torch.randn(37, requires_grad=True)
    (loss,) = cast(tuple[torch.Tensor], run(x))
    torch.testing.assert_close(loss, 2.0 * x.detach().sum())
    loss.backward()
    torch.testing.assert_close(x.grad, torch.full((37,), 3.0))


def test_pallas_kernels_compute_what_bodies_do() -> None:
    pytest.importorskip("jax")
    import jax
    import numpy as np
    from jax import Array

    binds = ("R=5", "C=40")
    text, kernels = _module(KERNELS, "equivalent", *binds, target="jax")
    assert "pl.pallas_call" in text
    exact_text, bodies = _module(KERNELS, "exact", *binds, target="jax")
    assert "pallas" not in exact_text
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 40)).astype(np.float32)
    w = (rng.standard_normal((40, 40)) * 0.2).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(kernels(x, w)[0]), np.asarray(bodies(x, w)[0]), rtol=1e-5, atol=1e-5
    )

    # Without a `grad`, the backward pass differentiates the bodies.
    def loss(run: Callable[..., tuple[object, ...]]) -> Callable[[Array], Array]:
        def squared(x: Array) -> Array:
            return (cast(Array, run(x, w)[0]) ** 2).sum()

        return squared

    np.testing.assert_allclose(
        np.asarray(jax.grad(loss(kernels))(x)),
        np.asarray(jax.grad(loss(bodies))(x)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_a_pallas_kernel_takes_the_ops_grad(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    import jax
    import numpy as np
    from jax import Array

    source = tmp_path / "kernel_grad.linnet"
    source.write_text(GRAD, encoding="utf-8")
    text, run = _module(source, "equivalent", "N=37", target="jax")
    assert "pl.pallas_call" in text
    x = np.random.default_rng(1).standard_normal(37).astype(np.float32)

    def total(x: Array) -> Array:
        return cast(Array, run(x)[0])

    loss, grad = jax.value_and_grad(total)(x)
    np.testing.assert_allclose(float(loss), 2.0 * float(x.sum()), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(grad), np.full(37, 3.0), rtol=1e-6)


def test_atomic_writes() -> None:
    # Column sums by `atomic_add` and maxima by `atomic_max`, one row a
    # program; the results start from the operations' identities.
    pytest.importorskip("triton")
    if not TRITON_INTERPRETS:
        pytest.skip("Triton's interpreter was not on as Triton loaded")
    text, kernels = _module(ATOMICS, "equivalent", "R=37", "C=40")
    assert "tl.atomic_add" in text
    assert "num_warps=4" in text
    _, bodies = _module(ATOMICS, "exact", "R=37", "C=40")
    x = torch.randn(37, 40)
    for got, want in zip(kernels(x), bodies(x), strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_pallas_atomic_writes() -> None:
    pytest.importorskip("jax")
    import numpy as np

    text, kernels = _module(ATOMICS, "equivalent", "R=37", "C=40", target="jax")
    assert "plgpu.atomic_add" in text
    assert "input_output_aliases" in text
    _, bodies = _module(ATOMICS, "exact", "R=37", "C=40", target="jax")
    x = np.random.default_rng(4).standard_normal((37, 40)).astype(np.float32)
    for got, want in zip(kernels(x), bodies(x), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_tile_scans() -> None:
    # A row's running sums in one tile, `tl.cumsum` and `jnp.cumsum`.
    import numpy as np

    x = np.random.default_rng(5).standard_normal((3, 50)).astype(np.float32)
    want = np.cumsum(x, axis=1)
    if TRITON_INTERPRETS and importlib.util.find_spec("triton") is not None:
        text, run = _module(SCANS, "equivalent", "R=3", "C=50", entry="scanned")
        assert "tl.cumsum" in text
        (got,) = run(torch.tensor(x))
        np.testing.assert_allclose(cast(torch.Tensor, got).numpy(), want, rtol=1e-5, atol=1e-5)
    if importlib.util.find_spec("jax") is not None:
        text, run = _module(SCANS, "equivalent", "R=3", "C=50", entry="scanned", target="jax")
        assert "pl.pallas_call" in text
        (got,) = run(x)
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-5)


def test_tile_products() -> None:
    # Each row's product within the tile: `tl.reduce` with a multiplication
    # in Triton, `jnp.prod` in Pallas; rows and columns past the edges masked.
    import numpy as np

    x = (1 + 0.1 * np.random.default_rng(6).standard_normal((7, 40))).astype(np.float32)
    want = np.prod(x, axis=1)
    if TRITON_INTERPRETS and importlib.util.find_spec("triton") is not None:
        text, run = _module(PRODUCTS, "equivalent", "R=7", "C=40")
        assert "tl.reduce(" in text
        (got,) = run(torch.tensor(x))
        np.testing.assert_allclose(cast(torch.Tensor, got).numpy(), want, rtol=1e-5)
    if importlib.util.find_spec("jax") is not None:
        text, run = _module(PRODUCTS, "equivalent", "R=7", "C=40", target="jax")
        assert "pl.pallas_call" in text
        (got,) = run(x)
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5)

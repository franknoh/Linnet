"""Kernels in generated PyTorch: an op with a `kernel` launches it as a
Triton program, run here in Triton's interpreter, and computes what the op's
body computes. Its gradient is the op's `grad`, or else its body's."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

from linnet.compiler import find_compiler

pytest.importorskip("triton")
# Set by conftest.py without a GPU, before Triton loaded.
if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("Triton's interpreter was not on as Triton loaded", allow_module_level=True)

REPO = Path(__file__).resolve().parents[3]
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
    # Generated code reads it: kernels run on the CPU, in the interpreter.
    monkeypatch.setenv("TRITON_INTERPRET", "1")


def _module(
    source: Path, numerics: str, *binds: str
) -> tuple[str, Callable[..., tuple[object, ...]]]:
    command = [find_compiler(), "torch", "--entry", "run", "--numerics", numerics]
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
    source = tmp_path / "kernel_grad.linnet"
    source.write_text(GRAD, encoding="utf-8")
    text, run = _module(source, "equivalent", "N=37")
    assert "@triton.jit" in text
    x = torch.randn(37, requires_grad=True)
    (loss,) = cast(tuple[torch.Tensor], run(x))
    torch.testing.assert_close(loss, 2.0 * x.detach().sum())
    loss.backward()
    torch.testing.assert_close(x.grad, torch.full((37,), 3.0))

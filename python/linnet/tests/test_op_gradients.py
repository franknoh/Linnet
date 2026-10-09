"""An op's `grad` clause: wherever an entry is differentiated -- PyTorch
autograd over the interpreter or generated code, `jax.grad` over generated
JAX, the compiler's own `--grad` exports -- the clause stands for the backward
pass of the op's body."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import importlib
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from linnet.compiler import find_compiler
from linnet.torch import load_function

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
SOURCE = REPO / "spec-tests/valid/029_op_gradients.linnet"

X = np.array([1.7, -2.3, 0.4, 3.9, -0.6], dtype=np.float32)
W = np.array([0.5, -1.0, 2.0, 0.1, -0.3], dtype=np.float32)


def _expected() -> tuple[float, NDArray[np.float32], NDArray[np.float32]]:
    """`sum(y * y)` for `y = trunc(x) * softplus(w)`, with the ops' own
    gradients: straight through `truncate`, clipped for `scaled`'s `w`."""
    truncated = np.trunc(X)
    soft = np.log1p(np.exp(W))
    y = truncated * soft
    dy = 2.0 * y
    dx = dy * soft
    dsoft = np.clip(dy * truncated, -1.0, 1.0)
    dw = dsoft / (1.0 + np.exp(-W))
    return float((y * y).sum()), dx.astype(np.float32), dw.astype(np.float32)


def _check(loss: object, dx: object, dw: object) -> None:
    want_loss, want_dx, want_dw = _expected()
    np.testing.assert_allclose(float(np.asarray(loss)), want_loss, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dx), want_dx, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(dw), want_dw, rtol=1e-5, atol=1e-6)


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.mark.parametrize("compile", [False, True])
def test_torch_autograd(compile: bool) -> None:
    loss_of = load_function(SOURCE, "loss", std_root=STDLIB, compile=compile)
    x = torch.tensor(X, requires_grad=True)
    w = torch.tensor(W, requires_grad=True)
    loss = cast(torch.Tensor, loss_of(x, w))
    loss.backward()
    _check(
        loss.detach().numpy(),
        cast(torch.Tensor, x.grad).numpy(),
        cast(torch.Tensor, w.grad).numpy(),
    )


def test_jax_grad() -> None:
    jax = pytest.importorskip("jax")
    from linnet.jax import load_function as load_jax

    loss_of = cast(Callable[..., object], load_jax(SOURCE, "loss", std_root=STDLIB))
    loss, (dx, dw) = jax.value_and_grad(loss_of, argnums=(0, 1))(X, W)
    _check(loss, dx, dw)


def _export(target: str) -> str:
    command = [find_compiler(), target, "--grad", "--entry", "loss", "--bind", "N=5", str(SOURCE)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _python(text: str, arguments: Sequence[object]) -> tuple[object, ...]:
    namespace: dict[str, object] = {}
    exec(compile(text, "<generated>", "exec"), namespace)
    assert namespace["GRADIENTS"] == ["x", "w"]
    main = cast(Callable[..., tuple[object, ...]], namespace["main"])
    make = namespace.get("constants")
    hoisted = cast(Callable[[str], tuple[object, ...]], make)("cpu") if make else ()
    return main(*arguments, *hoisted)


def test_grad_export_torch() -> None:
    loss, dx, dw = _python(_export("torch"), [torch.tensor(X), torch.tensor(W)])
    _check(*(cast(torch.Tensor, value).numpy() for value in (loss, dx, dw)))


def test_grad_export_jax() -> None:
    pytest.importorskip("jax")
    _check(*_python(_export("jax"), [X, W]))


def test_grad_export_onnx() -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    import onnx.parser

    model = onnx.parser.parse_model(_export("onnx"))
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    _check(*session.run(None, {"x": X, "w": W}))


def test_grad_export_stablehlo() -> None:
    jax = pytest.importorskip("jax")
    import jax.extend as jex

    text = _export("stablehlo")
    assert re.search(r'linnet\.gradients = \["x", "w"\]', text)
    options = importlib.import_module("jaxlib._jax")
    backend = jex.backend.get_backend()
    device = backend.local_devices()[0]
    executable = backend.compile_and_load(
        text, options.DeviceList((device,)), options.CompileOptions()
    )
    _check(
        *(np.asarray(o) for o in executable.execute([jax.device_put(a, device) for a in (X, W)]))
    )

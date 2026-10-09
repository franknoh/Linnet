"""`for` loops: a counted runtime loop carries its `var` locals, and a `for`
bound to a name stacks each iteration's `yield` along a new first axis.
Every backend computes what a Python loop does."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from linnet.torch import load_function

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
SOURCE = REPO / "spec-tests/valid/028_for_loops.linnet"

XS = np.array([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5], [0.25, 0.75, 1.0]], dtype=np.float32)
H0 = np.array([0.1, -0.2, 0.3], dtype=np.float32)
X = np.array([1.0, -2.0], dtype=np.float32)


def _recurrent() -> NDArray[np.float32]:
    h = H0
    out: list[NDArray[np.float32]] = []
    for x in XS:
        h = np.tanh(h * 0.5 + x).astype(np.float32)
        out.append(h)
    return np.stack(out)


def _counted() -> list[NDArray[np.number]]:
    ys = np.stack([X * 2.0**k for k in range(1, 5)])
    return [X * 4, np.array([0, 2, 4, 6], dtype=np.int64), ys.sum(axis=1), ys]


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.mark.parametrize("compile", [False, True])
def test_torch(compile: bool) -> None:
    recurrent = load_function(SOURCE, "recurrent", std_root=STDLIB, compile=compile)
    got = recurrent(torch.tensor(XS), torch.tensor(H0)).numpy()
    np.testing.assert_allclose(got, _recurrent(), rtol=1e-6)
    counted = load_function(SOURCE, "counted", std_root=STDLIB, compile=compile)
    for value, want in zip(counted(torch.tensor(X)), _counted(), strict=True):
        np.testing.assert_allclose(value.numpy(), want, rtol=1e-6)


def test_jax() -> None:
    pytest.importorskip("jax")
    from linnet.jax import load_function as load_jax

    recurrent = load_jax(SOURCE, "recurrent", std_root=STDLIB)
    np.testing.assert_allclose(np.asarray(recurrent(XS, H0)), _recurrent(), rtol=1e-6)
    counted = load_jax(SOURCE, "counted", std_root=STDLIB)
    for value, want in zip(counted(X), _counted(), strict=True):
        np.testing.assert_allclose(np.asarray(value), want, rtol=1e-6)


def test_onnx() -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_function

    cases = (
        ("recurrent", {"T": 3, "H": 3}, [XS, H0], [_recurrent()]),
        ("counted", {"N": 2}, [X], _counted()),
    )
    for name, generics, inputs, expected in cases:
        exported = export_function(SOURCE, name, generics=generics, std_root=STDLIB)
        session = onnxruntime.InferenceSession(
            exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = {arg.name: value for arg, value in zip(session.get_inputs(), inputs, strict=True)}
        got = session.run(None, feeds)
        for value, want in zip(got, expected, strict=True):
            np.testing.assert_allclose(value, want, rtol=1e-6)

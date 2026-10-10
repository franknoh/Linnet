"""`cumsum(x, axis = k)`: running sums along one axis, in the tensor's own
dtype, the same on every backend as NumPy's."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from linnet.compiler import find_compiler
from linnet.torch import load_function

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "spec-tests/valid/032_cumsum.linnet"

X = np.random.default_rng(0).standard_normal((3, 5)).astype(np.float32)
COUNTS = np.random.default_rng(1).integers(-4, 5, (3, 5)).astype(np.int32)


def _expected() -> list[NDArray[np.number]]:
    return [np.cumsum(X, axis=-1), np.cumsum(COUNTS, axis=0, dtype=np.int32)]


def _check(outputs: object) -> None:
    for got, want in zip(cast(tuple[object, ...], outputs), _expected(), strict=True):
        array = np.asarray(got)
        assert array.dtype == want.dtype
        np.testing.assert_allclose(array, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("compile", [False, True])
def test_torch(compile: bool) -> None:
    run = load_function(SOURCE, "run", compile=compile)
    outputs = cast(tuple[torch.Tensor, ...], run(torch.tensor(X), torch.tensor(COUNTS)))
    _check(tuple(output.numpy() for output in outputs))


def test_jax() -> None:
    pytest.importorskip("jax")
    from linnet.jax import load_function as load_jax

    _check(load_jax(SOURCE, "run")(X, COUNTS))


def test_onnx() -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_function

    exported = export_function(SOURCE, "run", generics={"R": 3, "C": 5})
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [argument.name for argument in session.get_inputs()]
    _check(tuple(session.run(None, dict(zip(names, (X, COUNTS), strict=True)))))


def test_stablehlo() -> None:
    jax = pytest.importorskip("jax")
    import jax.extend as jex

    command = [find_compiler(), "stablehlo", "--entry", "run", "--bind", "R=3", "--bind", "C=5"]
    text = subprocess.run(
        [*command, str(SOURCE)], capture_output=True, text=True, check=True
    ).stdout
    options = importlib.import_module("jaxlib._jax")
    backend = jex.backend.get_backend()
    device = backend.local_devices()[0]
    executable = backend.compile_and_load(
        text, options.DeviceList((device,)), options.CompileOptions()
    )
    outputs = executable.execute([jax.device_put(a, device) for a in (X, COUNTS)])
    _check(tuple(np.asarray(output) for output in outputs))

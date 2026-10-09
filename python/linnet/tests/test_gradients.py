"""`--grad` exports: an entry's scalar loss followed by its gradient with
respect to every floating parameter (or, for a module-level entry, every
floating input), emitted by the compiler as a backward pass of the exported
operations. Each format's gradient matches PyTorch's autograd run over the
same entry's generated forward code."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import importlib
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from linnet.compiler import find_compiler

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.gradients

use std.nn.norm::{rms_norm}
use std.nn.softmax::{softmax}

pub block Model<V: Dim, H: Dim, C: Dim> {
    param embedding: Tensor[V, H; f32]
    param norm: Tensor[H; f32]
    param w: Tensor[H, C; f32]
    param b: Tensor[C; f32]

    // Embedding rows by token, an RMS norm, a linear layer, log-softmax, and
    // the mean negative log-likelihood of the labels.
    pub entry classify<N: Dim>(tokens: Tensor[N; i64], labels: Tensor[N; i64]) -> f32 {
        let x[n, h] = embedding[tokens[n], h]
        let normed = rms_norm(x, norm, 1e-5)
        let logits[n, c] = (sum[h] normed[n, h] * w[h, c]) + b[c]
        let top[n] = max[c] logits[n, c]
        let shifted[n, c] = logits[n, c] - top[n]
        let total[n] = sum[c] exp(shifted[n, c])
        let picked[n] = shifted[n, labels[n]] - log(total[n])
        return -(sum[n] picked[n]) / cast<f32>(N)
    }

    // Elementwise functions, a softmax, slices with a step, a concatenation
    // and a transposition, all summed to one number.
    pub entry shapes<N: Dim>(x: Tensor[N, H; f32]) -> f32 {
        let y[n, c] = sum[h] tanh(x[n, h]) * w[h, c]
        let z = sqrt(y * y + 1.0) + rsqrt(abs(y) + 2.0) + sin(y) * cos(y)
        let p = softmax(z / (b * b + 1.0))
        let halves = concat(p[:, 0::2], max(p[:, 1::2], 0.1), axis = 1)
        let turned = permute(halves, [1, 0])
        let pick[c, n] = min(turned[c, n], 0.9) * norm[0]
        return sum[c, n] pick[c, n]
    }
}

// A module-level entry: its gradient is with respect to its inputs.
pub entry mse<N: Dim>(predicted: Tensor[N; f32], target: Tensor[N; f32]) -> f32 {
    let error[n] = predicted[n] - target[n]
    return (sum[n] error[n] * error[n]) / cast<f32>(N)
}
"""

GENERICS = {"V": 7, "H": 4, "C": 6, "N": 5}
SHAPES = {"embedding": (7, 4), "norm": (4,), "w": (4, 6), "b": (6,)}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("gradients") / "gradients.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _export(source: Path, target: str, entry: str, gradient: bool) -> str:
    command = [find_compiler(), target, "--std", str(STDLIB), "--entry", entry]
    command += ["--numerics", "exact"]
    for name, value in GENERICS.items():
        command += ["--bind", f"{name}={value}"]
    if gradient:
        command.append("--grad")
    completed = subprocess.run([*command, str(source)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _python(text: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(compile(text, "<generated>", "exec"), namespace)
    return namespace


def _run_python(namespace: dict[str, object], arguments: Sequence[object]) -> tuple[object, ...]:
    """`main` of a generated module with its hoisted constants, if any."""
    main = cast(Callable[..., tuple[object, ...]], namespace["main"])
    make = namespace.get("constants")
    hoisted = cast(Callable[[str], tuple[object, ...]], make)("cpu") if make else ()
    return main(*arguments, *hoisted)


CASES = {
    "classify": [
        np.array([1, 3, 3, 0, 6], dtype=np.int64),
        np.array([2, 0, 5, 1, 2], dtype=np.int64),
    ],
    "shapes": [np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)],
    "mse": [
        np.array([0.5, -1.0, 2.0, 0.0, 1.5], dtype=np.float32),
        np.array([1.0, -1.0, 0.0, 0.5, 1.0], dtype=np.float32),
    ],
}


def _parameters() -> dict[str, NDArray[np.float32]]:
    rng = np.random.default_rng(0)
    return {k: rng.standard_normal(v).astype(np.float32) * 0.5 for k, v in SHAPES.items()}


def _reference(source: Path, entry: str) -> tuple[float, list[NDArray[np.float32]]]:
    """The loss and PyTorch autograd's gradient of the generated forward code."""
    module = _python(_export(source, "torch", entry, gradient=False))
    inputs = [torch.tensor(value) for value in CASES[entry]]
    if entry == "mse":
        wrt = inputs
        arguments: list[torch.Tensor] = inputs
    else:
        parameters = _parameters()
        paths = cast(list[str], module["PARAMETERS"])
        wrt = [torch.tensor(parameters[path]) for path in paths]
        arguments = [*inputs, *wrt]
    for value in wrt:
        value.requires_grad_(True)
    (loss,) = cast(tuple[torch.Tensor], _run_python(module, arguments))
    # An input the loss does not read has no gradient: None.
    grads = cast(Sequence[torch.Tensor | None], torch.autograd.grad(loss, wrt, allow_unused=True))
    return float(loss), [
        np.zeros(tuple(value.shape), np.float32) if grad is None else grad.detach().numpy()
        for value, grad in zip(wrt, grads, strict=True)
    ]


def _arguments(entry: str, paths: Sequence[str]) -> list[NDArray[np.generic]]:
    parameters = _parameters()
    return [*CASES[entry], *(parameters[path] for path in paths)]


def _check(outputs: Sequence[object], expected: tuple[float, list[NDArray[np.float32]]]) -> None:
    loss, grads = expected
    assert len(outputs) == 1 + len(grads)
    np.testing.assert_allclose(float(np.asarray(outputs[0])), loss, rtol=1e-5, atol=1e-6)
    for got, want in zip(outputs[1:], grads, strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("entry", list(CASES))
def test_torch(source: Path, entry: str) -> None:
    expected = _reference(source, entry)
    module = _python(_export(source, "torch", entry, gradient=True))
    gradients = cast(list[str], module["GRADIENTS"])
    if entry == "mse":
        assert gradients == ["predicted", "target"]
    else:
        assert gradients == cast(list[str], module["PARAMETERS"])
    arguments = [torch.tensor(a) for a in _arguments(entry, gradients if entry != "mse" else [])]
    outputs = _run_python(module, arguments)
    _check([cast(torch.Tensor, o).detach().numpy() for o in outputs], expected)


@pytest.mark.parametrize("entry", list(CASES))
def test_jax(source: Path, entry: str) -> None:
    pytest.importorskip("jax")
    expected = _reference(source, entry)
    module = _python(_export(source, "jax", entry, gradient=True))
    paths = [] if entry == "mse" else cast(list[str], module["PARAMETERS"])
    _check(_run_python(module, _arguments(entry, paths)), expected)


@pytest.mark.parametrize("entry", list(CASES))
def test_onnx(source: Path, entry: str) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    import onnx.parser

    expected = _reference(source, entry)
    model = onnx.parser.parse_model(_export(source, "onnx", entry, gradient=True))
    metadata = {p.key: p.value for p in model.metadata_props}
    names = [graph_input.name for graph_input in model.graph.input]
    inputs = dict(zip(names, CASES[entry], strict=False))
    parameters = _parameters()
    for name in names[len(CASES[entry]) :]:
        inputs[name] = parameters[metadata[f"linnet.path.{name}"]]
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    _check(session.run(None, inputs), expected)


@pytest.mark.parametrize("entry", list(CASES))
def test_stablehlo(source: Path, entry: str) -> None:
    jax = pytest.importorskip("jax")
    import jax.extend as jex

    expected = _reference(source, entry)
    text = _export(source, "stablehlo", entry, gradient=True)
    assert "linnet.gradients" in text
    import re

    paths = re.findall(r'linnet\.path = "([^"]+)"', text)
    options = importlib.import_module("jaxlib._jax")
    backend = jex.backend.get_backend()
    device = backend.local_devices()[0]
    executable = backend.compile_and_load(
        text, options.DeviceList((device,)), options.CompileOptions()
    )
    buffers = [jax.device_put(a, device) for a in _arguments(entry, paths)]
    _check([np.asarray(o) for o in executable.execute(buffers)], expected)


REFUSED = """\
module tests.refused

pub block Model<N: Dim> {
    param w: Tensor[N; f32]
    state count: Tensor[N; f32]

    pub entry vector(x: Tensor[N; f32]) -> Tensor[N; f32] {
        return x * w
    }

    pub entry counted(x: Tensor[N; f32]) -> f32 {
        count = count + x
        return sum[n] x[n] * w[n]
    }

    pub entry looped(x: Tensor[N; f32]) -> f32 {
        var y = x
        for _i in 0..3 {
            y = y * w
        }
        return sum[n] y[n]
    }
}
"""


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("vector", "one result is a floating scalar"),
        ("counted", "assigns `state`"),
        ("looped", "runtime loop"),
    ],
)
def test_what_has_no_gradient_is_refused(tmp_path: Path, entry: str, message: str) -> None:
    source = tmp_path / "refused.linnet"
    source.write_text(REFUSED, encoding="utf-8")
    command = [find_compiler(), "torch", "--grad", "--entry", entry, "--bind", "N=3"]
    completed = subprocess.run([*command, str(source)], capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert message in completed.stderr

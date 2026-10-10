"""FP8 E4M3 weights (`std.quant::Fp8Linear`): every byte decodes to the value
PyTorch's `float8_e4m3fn` gives it, on every backend, and a checkpoint
stored as `compressed-tensors` stores it (an `F8_E4M3` weight, a
`weight_scale` of `[Out, 1]`) binds as it is and multiplies as its
dequantized weight. On a GPU with FP8 products, `numerics="fast"` rounds
the input to FP8 too."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.compiler import find_compiler
from linnet.torch import load
from linnet.weights import write_bindings

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.fp8

use std.quant::{Fp8Linear, decode_fp8}

pub block Model<In: Dim, Out: Dim, T: Float = f32> {
    sub proj: Fp8Linear<In, Out, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return proj.forward(x)
    }
}

// Every byte's value.
pub entry values(bits: Tensor[256; u8]) -> Tensor[256; f32] {
    return decode_fp8(bits)
}
"""

IN, OUT, ROWS = 512, 32, 3
BYTES = np.arange(256, dtype=np.uint8)
# The two NaN bytes: PyTorch's NaN, the body's 480.
VALID = (BYTES & 0x7F) != 0x7F


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def _expected() -> NDArray[np.float32]:
    return torch.from_numpy(BYTES).view(torch.float8_e4m3fn).float().numpy()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "fp8.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _export(source: Path, target: str, numerics: str) -> str:
    command = [find_compiler(), target, "--std", str(STDLIB), "--entry", "values"]
    completed = subprocess.run(
        [*command, "--numerics", numerics, str(source)], capture_output=True, text=True, check=True
    )
    return completed.stdout


@pytest.mark.parametrize("target", ["torch", "jax", "onnx", "stablehlo"])
@pytest.mark.parametrize("numerics", ["exact", "fast"])
def test_every_byte_decodes(source: Path, target: str, numerics: str) -> None:
    """The body (`exact`), or a backend's own FP8 type where it has one."""
    text = _export(source, target, numerics)
    if target == "torch":
        namespace: dict[str, object] = {}
        exec(compile(text, "<generated>", "exec"), namespace)
        (got,) = namespace["main"](torch.from_numpy(BYTES))  # type: ignore[operator]
        values = np.asarray(got)
    elif target == "jax":
        pytest.importorskip("jax")
        namespace = {}
        exec(compile(text, "<generated>", "exec"), namespace)
        (got,) = namespace["main"](BYTES)  # type: ignore[operator]
        values = np.asarray(got)
    elif target == "onnx":
        onnxruntime = pytest.importorskip("onnxruntime")
        import onnx.parser

        model = onnx.parser.parse_model(text)
        session = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (values,) = session.run(None, {model.graph.input[0].name: BYTES})
    else:
        jax = pytest.importorskip("jax")
        import jax.extend as jex

        options = importlib.import_module("jaxlib._jax")
        backend = jex.backend.get_backend()
        device = backend.local_devices()[0]
        executable = backend.compile_and_load(
            text, options.DeviceList((device,)), options.CompileOptions()
        )
        (got,) = executable.execute([jax.device_put(BYTES, device)])
        values = np.asarray(got)
    np.testing.assert_array_equal(np.asarray(values)[VALID], _expected()[VALID])


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, Path, NDArray[np.float32]]:
    """FP8 weights under `compressed-tensors` names, their bindings, and
    the weight they stand for."""
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(OUT, IN, generator=generator)
    scale = full.abs().amax(dim=1, keepdim=True) / 448.0
    weight = (full / scale).to(torch.float8_e4m3fn)
    save_file(
        {"model.proj.weight": weight, "model.proj.weight_scale": scale},
        str(tmp_path / "fp8.safetensors"),
    )
    bindings = write_bindings(
        tmp_path / "bindings.json",
        {"proj.weight": "model.proj.weight", "proj.scale": "model.proj.weight_scale"},
    )
    dequantized = (weight.float() * scale).numpy()
    return tmp_path / "fp8.safetensors", bindings, dequantized


def _input() -> NDArray[np.float32]:
    return np.random.default_rng(1).standard_normal((ROWS, IN)).astype(np.float32)


@pytest.mark.parametrize("compile", [False, True])
def test_torch_binds_and_multiplies(
    source: Path, checkpoint: tuple[Path, Path, NDArray[np.float32]], compile: bool
) -> None:
    weights, bindings, dequantized = checkpoint
    model = load(
        source,
        generics={"In": IN, "Out": OUT},
        std_root=STDLIB,
        weights=weights,
        bindings=bindings,
        compile=compile,
    )
    got = model.run_entry("forward", [torch.from_numpy(_input())]).detach().numpy()
    np.testing.assert_allclose(got, _input() @ dequantized.T, rtol=1e-5, atol=1e-5)


def test_jax_binds_and_multiplies(
    source: Path, checkpoint: tuple[Path, Path, NDArray[np.float32]]
) -> None:
    pytest.importorskip("jax")
    from linnet.jax import load as load_jax
    from linnet.jax import load_source

    weights, bindings, dequantized = checkpoint
    for make in (load_jax, load_source):
        function = make(
            source,
            generics={"In": IN, "Out": OUT},
            weights=weights,
            bindings=bindings,
            std_root=STDLIB,
            entry="forward",
        )
        np.testing.assert_allclose(
            np.asarray(function(_input())), _input() @ dequantized.T, rtol=1e-5, atol=1e-5
        )


def test_onnx_embeds_and_multiplies(
    source: Path, checkpoint: tuple[Path, Path, NDArray[np.float32]]
) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model

    weights, bindings, dequantized = checkpoint
    exported = export_model(
        source,
        generics={"In": IN, "Out": OUT, "B": ROWS},
        weights=weights,
        bindings=bindings,
        std_root=STDLIB,
        entry="forward",
    )
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = session.run(None, {session.get_inputs()[0].name: _input()})
    np.testing.assert_allclose(got, _input() @ dequantized.T, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FP8 products need compute capability 9 or later",
)
@pytest.mark.parametrize("rows", [1, ROWS])
def test_cuda_fast_multiplies_in_fp8(
    source: Path, checkpoint: tuple[Path, Path, NDArray[np.float32]], rows: int
) -> None:
    """One row widens the weight inside Linnet's kernel; more round the
    input to FP8 a row at a time. Both within FP8's rounding of the product
    with the dequantized weight."""
    weights, bindings, dequantized = checkpoint
    model = load(
        source,
        generics={"In": IN, "Out": OUT, "T": "bf16"},
        std_root=STDLIB,
        weights=weights,
        bindings=bindings,
        cast_dtype=True,  # the checkpoint's scales are f32
        device="cuda",
        compile=True,
        numerics="fast",
    )
    inputs = np.random.default_rng(2).standard_normal((rows, IN)).astype(np.float32)
    x = torch.from_numpy(inputs).to("cuda", torch.bfloat16)
    got = model.run_entry("forward", [x]).float().cpu().numpy()
    want = x.float().cpu().numpy() @ dequantized.T
    assert np.abs(got - want).max() <= 0.06 * np.abs(want).max()


SIBLINGS = """\
module tests.fp8_siblings

use std.quant::{Fp8Linear}

pub block Model<In: Dim, Out: Dim, T: Float = f32> {
    sub q: Fp8Linear<In, Out, T>
    sub k: Fp8Linear<In, Out, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return q.forward(x) * k.forward(x)
    }
}
"""


def test_siblings_run_as_one_product(tmp_path: Path) -> None:
    """Two FP8 layers reading one input multiply by their weights and
    scales side by side, as `F.linear`'s siblings do."""
    from linnet.torch import CompiledLinnetModule

    source = tmp_path / "siblings.linnet"
    source.write_text(SIBLINGS, encoding="utf-8")
    generator = torch.Generator().manual_seed(3)
    tensors: dict[str, torch.Tensor] = {}
    dequantized: dict[str, NDArray[np.float32]] = {}
    for name in ("q", "k"):
        full = torch.randn(OUT, IN, generator=generator)
        scale = full.abs().amax(dim=1, keepdim=True) / 448.0
        weight = (full / scale).to(torch.float8_e4m3fn)
        tensors[f"{name}.weight"], tensors[f"{name}.scale"] = weight, scale
        dequantized[name] = (weight.float() * scale).numpy()
    save_file(tensors, str(tmp_path / "siblings.safetensors"))
    model = load(
        source,
        generics={"In": IN, "Out": OUT},
        std_root=STDLIB,
        weights=tmp_path / "siblings.safetensors",
        compile=True,
    )
    got = model.run_entry("forward", [torch.from_numpy(_input())]).detach().numpy()
    want = (_input() @ dequantized["q"].T) * (_input() @ dequantized["k"].T)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    assert isinstance(model, CompiledLinnetModule)
    generated = model.generated_source("forward")
    assert generated.count("_fp8_linear(") == 2  # the helper and one call
    assert "_adjacent(" in generated

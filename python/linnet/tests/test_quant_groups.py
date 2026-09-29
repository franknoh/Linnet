"""Group-wise 4-bit linears (`std.quant::Int4GroupLinear`): the rounding
`linnet.quant` does, and the same numbers from the canonical body, the
interpreter, generated PyTorch (tinygemm on CUDA), and ONNX Runtime's
`MatMulNBits`."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.quant import dequantize_int4_groups, quantize_checkpoint, quantize_int4_groups
from linnet.torch import CompiledLinnetModule, load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.q4

use std.quant::{Int4GroupLinear}

pub block Model<In: Dim, Out: Dim, Group: Dim, T: Float = f32>
where
    Group > 0,
    Group % 2 == 0,
    In % Group == 0
{
    sub proj: Int4GroupLinear<In, Out, Group, T>

    pub entry forward<B: Dim>(x: Tensor[B, In; T]) -> Tensor[B, Out; T] {
        return proj.forward(x)
    }
}
"""

IN, OUT, GROUP = 256, 64, 128


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


def test_rounding_stays_within_half_a_step() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((OUT, IN)).astype(np.float32)
    packed, scale, zero = quantize_int4_groups(weight, GROUP)
    assert packed.shape == (OUT, IN // GROUP, GROUP // 2) and packed.dtype == np.uint8
    assert scale.shape == zero.shape == (OUT, IN // GROUP)
    error = np.abs(dequantize_int4_groups(packed, scale, zero) - weight)
    step = np.repeat(scale, GROUP, axis=1)
    assert np.all(error <= step / 2 + 1e-6)


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    """The source, a quantized checkpoint written by `quantize_checkpoint`
    from a float one under other names, and the weight it computes with."""
    source = tmp_path / "q4.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    weight = torch.randn(OUT, IN)
    save_file({"model.proj.weight": weight}, str(tmp_path / "float.safetensors"))
    quantized = quantize_checkpoint(
        tmp_path / "float.safetensors",
        tmp_path / "int4.safetensors",
        patterns=["proj.weight"],
        group=GROUP,
        dtype="f32",
        bindings={"proj.weight": "model.proj.weight"},
    )
    assert quantized == ["proj.weight"]
    packed, scale, zero = quantize_int4_groups(weight.numpy(), GROUP)
    return source, tmp_path / "int4.safetensors", dequantize_int4_groups(packed, scale, zero)


@pytest.mark.parametrize("compile", [False, True])
def test_torch_computes_with_the_dequantized_weight(
    model_files: tuple[Path, Path, np.ndarray], compile: bool
) -> None:
    source, weights, weight = model_files
    generics: dict[str, int | str] = {"In": IN, "Out": OUT, "Group": GROUP}
    model = load(source, generics=generics, std_root=STDLIB, weights=weights, compile=compile)
    x = torch.randn(3, IN)
    got = model.run_entry("forward", [x]).detach().numpy()
    np.testing.assert_allclose(got, x.numpy() @ weight.T, rtol=1e-4, atol=1e-4)


def test_onnx_runs_matmulnbits(model_files: tuple[Path, Path, np.ndarray]) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from linnet.onnx import export_model

    source, weights, weight = model_files
    exported = export_model(
        source,
        generics={"In": IN, "Out": OUT, "Group": GROUP, "B": 3},
        weights=weights,
        std_root=STDLIB,
        numerics="fast",
    )
    assert any(node.op_type == "MatMulNBits" for node in exported.model.graph.node)
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.random.default_rng(1).standard_normal((3, IN)).astype(np.float32)
    (got,) = session.run(None, {session.get_inputs()[0].name: x})
    np.testing.assert_allclose(got, x @ weight.T, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tinygemm runs on CUDA")
def test_cuda_bf16_runs_tinygemm(
    model_files: tuple[Path, Path, np.ndarray], tmp_path: Path
) -> None:
    source, _, weight = model_files
    quantize_checkpoint(
        tmp_path / "float.safetensors",
        tmp_path / "int4-bf16.safetensors",
        patterns=["proj.weight"],
        group=GROUP,
        dtype="bf16",
        bindings={"proj.weight": "model.proj.weight"},
    )
    generics: dict[str, int | str] = {"In": IN, "Out": OUT, "Group": GROUP, "T": "bf16"}
    model = load(
        source,
        generics=generics,
        std_root=STDLIB,
        weights=tmp_path / "int4-bf16.safetensors",
        device="cuda",
        compile=True,
    )
    x = torch.randn(3, IN, device="cuda", dtype=torch.bfloat16)
    got = model.run_entry("forward", [x]).float().cpu().numpy()
    # The packed weights went to tinygemm: its tuple has three members.
    assert isinstance(model, CompiledLinnetModule)
    prepared = next(iter(model._prepared.values()))  # pyright: ignore[reportPrivateUsage]
    assert isinstance(prepared, tuple) and len(prepared) == 3
    expected = x.float().cpu().numpy() @ weight.T
    np.testing.assert_allclose(got, expected, rtol=3e-2, atol=3e-1)

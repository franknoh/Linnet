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

from linnet.quant import (
    dequantize_int4_groups,
    import_quantized,
    quantize_checkpoint,
    quantize_int4_groups,
)
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
    # A prompt's many rows take the dequantized weight instead: the same numbers.
    many = torch.randn(40, IN, device="cuda", dtype=torch.bfloat16)
    got = model.run_entry("forward", [many]).float().cpu().numpy()
    expected = many.float().cpu().numpy() @ weight.T
    np.testing.assert_allclose(got, expected, rtol=3e-2, atol=3e-1)


# GPTQ's and AWQ's int32 packings, written from their definitions rather
# than from the importer: `qweight[i // 8, o]` holds GPTQ input `i` at bits
# `4 * (i % 8)` and `qzeros[g, o // 8]` group `g`'s zero point, less one, at
# bits `4 * (o % 8)`; AWQ's `qweight[i, c]` and `qzeros[g, c]` hold output
# `8c + ORDER[k]` at bits `4k`, zero points as they are.
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_gptq(q: np.ndarray, zero: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qweight = np.zeros((q.shape[0] // 8, q.shape[1]), np.uint32)
    qzeros = np.zeros((zero.shape[0], zero.shape[1] // 8), np.uint32)
    for k in range(8):
        qweight |= q[k::8].astype(np.uint32) << (4 * k)
        qzeros |= (zero[:, k::8] - 1).astype(np.uint32) << (4 * k)
    return qweight.view(np.int32), qzeros.view(np.int32)


def _pack_awq(q: np.ndarray, zero: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qweight = np.zeros((q.shape[0], q.shape[1] // 8), np.uint32)
    qzeros = np.zeros((zero.shape[0], zero.shape[1] // 8), np.uint32)
    for k, column in enumerate(AWQ_ORDER):
        qweight |= q[:, column::8].astype(np.uint32) << (4 * k)
        qzeros |= zero[:, column::8].astype(np.uint32) << (4 * k)
    return qweight.view(np.int32), qzeros.view(np.int32)


@pytest.mark.parametrize(
    ("method", "act_order", "compile"),
    [("gptq", False, True), ("gptq", True, True), ("gptq", True, False), ("awq", False, True)],
)
def test_calibrated_checkpoints_repack(
    tmp_path: Path, method: str, act_order: bool, compile: bool
) -> None:
    """A GPTQ or AWQ checkpoint's layers compute what its own format says,
    `W[i, o] = (q[i, o] - zero[g(i), o]) * scale[g(i), o]`; in GPTQ's
    activation order the group `g(i)` of input `i` is `g_idx[i]`, not
    `i // group`."""
    source = tmp_path / "q4.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    rng = np.random.default_rng(3)
    groups = IN // GROUP
    q = rng.integers(0, 16, (IN, OUT))
    zero = rng.integers(1, 16, (groups, OUT))
    scale = rng.uniform(0.01, 0.1, (groups, OUT)).astype(np.float16)
    group_of = np.arange(IN) // GROUP
    if act_order:
        group_of = group_of[rng.permutation(IN)]
    weight = (q - zero[group_of]) * scale[group_of].astype(np.float32)  # [In, Out]
    qweight, qzeros = (_pack_gptq if method == "gptq" else _pack_awq)(q, zero)
    tensors = {
        "model.proj.qweight": torch.from_numpy(qweight),
        "model.proj.qzeros": torch.from_numpy(qzeros),
        "model.proj.scales": torch.from_numpy(scale),
    }
    if method == "gptq":
        tensors["model.proj.g_idx"] = torch.from_numpy(group_of.astype(np.int32))
    save_file(tensors, str(tmp_path / "checkpoint.safetensors"))
    quantized = import_quantized(
        tmp_path / "checkpoint.safetensors",
        tmp_path / "int4.safetensors",
        bindings={"proj.weight": "model.proj.weight"},
        dtype="f32",
        config={"quant_method": method, "bits": 4, "group_size": GROUP, "desc_act": act_order},
    )
    assert quantized == ["proj.weight"]
    generics: dict[str, int | str] = {"In": IN, "Out": OUT, "Group": GROUP}
    model = load(
        source,
        generics=generics,
        std_root=STDLIB,
        weights=tmp_path / "int4.safetensors",
        compile=compile,
    )
    x = torch.randn(3, IN)
    got = model.run_entry("forward", [x]).detach().numpy()
    np.testing.assert_allclose(got, x.numpy() @ weight, rtol=1e-4, atol=1e-4)
    if act_order:
        onnxruntime = pytest.importorskip("onnxruntime")
        from linnet.onnx import export_model

        exported = export_model(
            source,
            generics={**generics, "B": 3},
            weights=tmp_path / "int4.safetensors",
            std_root=STDLIB,
            numerics="fast",
        )
        session = onnxruntime.InferenceSession(
            exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (onnx,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
        np.testing.assert_allclose(onnx, x.numpy() @ weight, rtol=1e-3, atol=1e-3)

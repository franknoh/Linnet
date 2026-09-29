"""`export_model` packages `linnet onnx` output with the checkpoint inside, so
it runs in onnxruntime with nothing else and agrees with the interpreter."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet import LinnetError
from linnet.onnx import export_model
from linnet.torch import load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"
SOURCE = REPO / "examples/06-gpt2/gpt2.linnet"
GENERICS: dict[str, int | str] = {
    "Vocab": 11,
    "MaxPositions": 16,
    "H": 8,
    "Heads": 2,
    "Layers": 2,
    "T": "f32",
}


def _checkpoint(tmp_path: Path) -> tuple[Path, torch.nn.Module]:
    torch.manual_seed(0)
    reference = load(SOURCE, generics=GENERICS, std_root=STDLIB)
    weights = {
        name.removeprefix("root."): torch.randn(parameter.shape) * 0.3
        for name, parameter in reference.named_parameters()
    }
    path = tmp_path / "model.safetensors"
    save_file(weights, str(path))
    return path, load(SOURCE, generics=GENERICS, std_root=STDLIB, weights=path)


def test_exported_model_is_self_contained(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    checkpoint, reference = _checkpoint(tmp_path)
    exported = export_model(
        SOURCE, generics={**GENERICS, "B": 2, "S": 5}, weights=checkpoint, std_root=STDLIB
    )
    assert [p.name for p in exported.inputs] == ["tokens"]
    assert exported.inputs[0].dtype == "i32" and exported.inputs[0].shape == (2, 5)
    assert exported.outputs[0].shape == (2, 5, 11) and exported.outputs[0].dtype == "f32"
    assert "blocks.1.mlp.down.weight" in exported.parameters
    assert len(exported.model.graph.initializer) == len(exported.parameters)

    saved = exported.save(tmp_path / "out/model.onnx")
    session = onnxruntime.InferenceSession(str(saved), providers=["CPUExecutionProvider"])
    tokens = torch.randint(0, 11, (2, 5), dtype=torch.int32)
    (actual,) = session.run(None, {"tokens": tokens.numpy()})
    torch.testing.assert_close(
        torch.from_numpy(np.array(actual)), reference(tokens), atol=1e-4, rtol=1e-4
    )


def test_export_rejects_a_mismatched_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    save_file({"wte": torch.zeros(11, 4)}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(LinnetError, match="`wte` has shape \\[11, 4\\]"):
        export_model(
            SOURCE,
            generics={**GENERICS, "B": 1, "S": 2},
            weights=tmp_path / "bad.safetensors",
            std_root=STDLIB,
        )


TIED = """\
module tests.tied

pub block Model {
    param embedding: Tensor[3, 4; f32]
    param head: Tensor[3, 4; f32]

    pub entry forward(x: Tensor[4; f32]) -> Tensor[3; f32] {
        let y[v] = sum[d] (embedding[v, d] + 2.0 * head[v, d]) * x[d]
        return y
    }
}
"""


def test_one_tensor_serves_two_parameters(tmp_path: Path) -> None:
    """An output head tied to the embedding: both parameters read the one
    checkpoint tensor, and each gets an initializer of it."""
    onnxruntime = pytest.importorskip("onnxruntime")
    source = tmp_path / "tied.linnet"
    source.write_text(TIED, encoding="utf-8")
    shared = torch.randn(3, 4)
    save_file({"shared": shared}, str(tmp_path / "model.safetensors"))
    bindings = tmp_path / "bindings.json"
    bindings.write_text('{"embedding": "shared", "head": "shared"}', encoding="utf-8")
    exported = export_model(
        source, generics={}, weights=tmp_path / "model.safetensors", bindings=bindings
    )
    assert len(exported.model.graph.initializer) == 2
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.arange(4, dtype=np.float32)
    (got,) = session.run(None, {session.get_inputs()[0].name: x})
    np.testing.assert_allclose(got, (3.0 * shared.numpy()) @ x, rtol=1e-5, atol=1e-5)


SIXTEEN = """\
module tests.sixteen

pub block Model<T: Float = bf16> {
    param levels: Tensor[4; u8]
    param weight: Tensor[4; T]

    pub entry forward(x: Tensor[4; T]) -> Tensor[4; T] {
        let shifted = shr(levels, 1) + cast<u8>(1)
        let lifted[i] = cast<T>(shifted[i])
        return select(x > weight, x, lifted)
    }
}
"""


def test_types_onnx_runtime_lacks_kernels_for_are_widened(tmp_path: Path) -> None:
    """ONNX Runtime has no bf16 comparisons and no arithmetic on 8-bit
    integers: those compute in f32 and i32 in the export."""
    pytest.importorskip("onnx")
    import onnx

    source = tmp_path / "sixteen.linnet"
    source.write_text(SIXTEEN, encoding="utf-8")
    save_file(
        {
            "levels": torch.tensor([0, 7, 200, 255], dtype=torch.uint8),
            "weight": torch.zeros(4, dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    exported = export_model(source, generics={}, weights=tmp_path / "model.safetensors")
    typed = onnx.shape_inference.infer_shapes(exported.model)
    types = {v.name: v.type.tensor_type.elem_type for v in typed.graph.value_info}
    types.update({i.name: i.type.tensor_type.elem_type for i in typed.graph.input})
    for node in typed.graph.node:
        if node.op_type in ("Greater", "Less"):
            assert all(types[i] == onnx.TensorProto.FLOAT for i in node.input)
        if node.op_type in ("Add", "Sub", "Where"):
            assert all(types.get(i) != onnx.TensorProto.UINT8 for i in node.input if i in types)


PARTS = """\
module tests.parts

use std.nn.linear::{Linear}

pub block Model<H: Dim, T: Float = f32> {
    sub first: Linear<H, H, T>
    sub second: Linear<H, H, T>

    state memory: Tensor[1, H; T]

    pub entry remember(x: Tensor[1, H; T]) -> Tensor[1, H; T] {
        memory = first.forward(x)
        return memory
    }

    pub entry recall(x: Tensor[1, H; T]) -> Tensor[1, H; T] {
        return second.forward(x) + memory
    }
}
"""


def test_entries_with_state_read_their_own_weights(tmp_path: Path) -> None:
    """An entry with state that reads only some weights (an encoder filling
    a decoder's caches) runs beside one that reads the others, over the
    same state, on ONNX Runtime as in the interpreter."""
    pytest.importorskip("onnxruntime")
    from linnet.onnx import load_model

    source = tmp_path / "parts.linnet"
    source.write_text(PARTS, encoding="utf-8")
    generics: dict[str, int | str] = {"H": 4}
    torch.manual_seed(0)
    skeleton = load(source, generics=generics, std_root=STDLIB)
    save_file(
        {
            name.removeprefix("root."): torch.randn(parameter.shape)
            for name, parameter in skeleton.named_parameters()
            if not name.endswith(".bias")
        },
        str(tmp_path / "parts.safetensors"),
    )
    reference = load(
        source, generics=generics, std_root=STDLIB, weights=tmp_path / "parts.safetensors"
    )
    model = load_model(
        source, generics=generics, weights=tmp_path / "parts.safetensors", std_root=STDLIB
    )
    x = np.random.default_rng(0).standard_normal((1, 4)).astype(np.float32)
    y = np.random.default_rng(1).standard_normal((1, 4)).astype(np.float32)
    reference.run_entry("remember", [torch.from_numpy(x)])
    model.run_entry("remember", [x])
    expected = reference.run_entry("recall", [torch.from_numpy(y)]).detach().numpy()
    np.testing.assert_allclose(model.run_entry("recall", [y]), expected, rtol=1e-5, atol=1e-5)

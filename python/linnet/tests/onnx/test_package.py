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

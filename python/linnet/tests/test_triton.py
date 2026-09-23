"""`linnet.triton.export` writes model repositories Triton loads: a config
with the entry's static interface and an ONNX model or a Python backend."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet import LinnetError, triton
from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
SOURCE = REPO / "examples/06-gpt2/gpt2.linnet"
LLAMA = REPO / "examples/05-llama/src/lib.linnet"
GENERICS: dict[str, int | str] = {
    "Vocab": 11,
    "MaxPositions": 16,
    "H": 8,
    "Heads": 2,
    "Layers": 2,
    "T": "f32",
}


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    reference = load(SOURCE, generics=GENERICS, std_root=STDLIB)
    weights = {
        name.removeprefix("root."): torch.randn(parameter.shape) * 0.3
        for name, parameter in reference.named_parameters()
    }
    path = tmp_path / "gpt2.safetensors"
    save_file(weights, str(path))
    return path


def test_onnx_repository(tmp_path: Path, checkpoint: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    repository = triton.export(
        SOURCE,
        tmp_path / "repo",
        name="gpt2-tiny",
        generics={**GENERICS, "B": 1, "S": 4},
        weights=checkpoint,
        std_root=STDLIB,
    )
    assert repository.directory == tmp_path / "repo/gpt2-tiny"
    config = repository.config.read_text(encoding="utf-8")
    assert 'name: "gpt2-tiny"' in config and 'platform: "onnxruntime_onnx"' in config
    assert re.search(r'name: "tokens"\s+data_type: TYPE_INT32\s+dims: \[ 1, 4 \]', config)
    assert re.search(r'name: "output0"\s+data_type: TYPE_FP32\s+dims: \[ 1, 4, 11 \]', config)
    model = repository.directory / "1/model.onnx"
    session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    (out,) = session.run(None, {"tokens": np.array([[1, 2, 3, 4]], np.int32)})
    reference = load(SOURCE, generics=GENERICS, std_root=STDLIB, weights=checkpoint)
    expected = reference(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
    torch.testing.assert_close(torch.from_numpy(np.array(out)), expected, atol=1e-4, rtol=1e-4)


def test_python_repository_keeps_source_and_weights(tmp_path: Path, checkpoint: Path) -> None:
    repository = triton.export(
        SOURCE,
        tmp_path / "repo",
        generics={**GENERICS, "B": 2, "S": 3},
        weights=checkpoint,
        std_root=STDLIB,
        backend="python",
        numerics="fast",
    )
    assert repository.name == "gpt2"
    config = repository.config.read_text(encoding="utf-8")
    assert 'backend: "python"' in config and "max_batch_size: 0" in config
    version = repository.directory / "1"
    assert (version / "model/gpt2.linnet").exists()
    assert (version / "weights/gpt2.safetensors").exists()
    script = (version / "model.py").read_text(encoding="utf-8")
    assert "ENTRY = 'forward'" in script and "numerics='fast'" in script
    assert "INPUTS = [('tokens', 'i32')]" in script
    assert "'Vocab': 11" in script and "ENTRY_GENERICS = {'B': 2, 'S': 3}" in script
    compile(script, "model.py", "exec")  # the template renders to valid Python


def test_stateful_entries_need_the_python_backend(tmp_path: Path) -> None:
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 4,
        "KvHeads": 2,
        "Inner": 16,
        "Layers": 1,
        "Batch": 1,
        "MaxSeq": 6,
        "T": "f32",
    }
    with pytest.raises(LinnetError, match="uses state"):
        triton.export(LLAMA, tmp_path / "repo", entry="decode", generics=generics, std_root=STDLIB)
    repository = triton.export(
        LLAMA,
        tmp_path / "repo",
        entry="decode",
        generics=generics,
        std_root=STDLIB,
        backend="python",
    )
    assert [t.name for t in repository.inputs] == ["token", "pos"]
    assert repository.inputs[1].dims == ()
    config = repository.config.read_text(encoding="utf-8")
    assert "reshape: { shape: [ ] }" in config  # the scalar `pos`


def test_cli(tmp_path: Path, checkpoint: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("onnx")
    code = triton.main(
        [
            "export",
            str(SOURCE),
            "-o",
            str(tmp_path / "repo"),
            "--std",
            str(STDLIB),
            "--weights",
            str(checkpoint),
            *[f"--bind={k}={v}" for k, v in {**GENERICS, "B": 1, "S": 2}.items()],
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "wrote" in out and "input  tokens: i32[1, 2]" in out

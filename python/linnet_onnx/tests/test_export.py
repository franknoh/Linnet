"""`linnet onnx` output parses as an ONNX model, runs under onnxruntime with
the same weights as the PyTorch materializer, agrees with it, and imports
back into Linnet."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnx  # type: ignore[import-untyped]
import pytest
import torch
from linnet_torch import load
from onnx import parser  # type: ignore[import-untyped]
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet_onnx import find_compiler, import_onnx

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
EXAMPLES = REPO / "examples"


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


def _export_onnx(source: Path, bindings: dict[str, int | str], entry: str | None = None) -> Any:
    command = [find_compiler(), "onnx", "--std", str(STDLIB)]
    if entry is not None:
        command += ["--entry", entry]
    for name, value in bindings.items():
        command += ["--bind", f"{name}={value}"]
    completed = subprocess.run([*command, str(source)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    model = parser.parse_model(completed.stdout)
    onnx.checker.check_model(model, full_check=True)
    return model


def test_tiny_transformer_runs_under_onnxruntime(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    generics: dict[str, int | str] = {
        "Vocab": 11,
        "H": 8,
        "Heads": 2,
        "Inner": 16,
        "Layers": 2,
        "T": "f32",
    }
    source = EXAMPLES / "04-tiny-transformer/src/lib.linnet"
    reference = load(source, generics=generics, std_root=STDLIB)
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in reference.named_parameters():
        path = name.removeprefix("root.")
        if not path.endswith(".bias"):
            weights[path] = torch.randn(parameter.shape) * 0.3
    save_file(weights, str(tmp_path / "model.safetensors"))
    reference = load(source, generics=generics, std_root=STDLIB, weights=tmp_path)

    seq, head = 5, 4
    tokens = torch.randint(0, 11, (2, seq), dtype=torch.int32)
    positions = torch.arange(seq, dtype=torch.float32)[:, None]
    frequencies = 1.0 / (10000 ** (torch.arange(0, head, 2, dtype=torch.float32) / head))
    angles = positions * frequencies
    cos_table = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin_table = torch.cat([angles.sin(), angles.sin()], dim=-1)
    expected = reference(tokens, cos_table, sin_table)

    model = _export_onnx(source, {**generics, "B": 2, "S": seq})
    paths = {p.key.removeprefix("linnet.path."): p.value for p in model.metadata_props}
    feeds: dict[str, Any] = {
        "tokens": tokens.numpy(),
        "cos_table": cos_table.numpy(),
        "sin_table": sin_table.numpy(),
    }
    for name, path in paths.items():
        feeds[name] = weights[path].numpy()
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(None, feeds)
    torch.testing.assert_close(torch.from_numpy(np.array(actual)), expected, atol=1e-4, rtol=1e-4)

    # And back: the ONNX model imports as Linnet that computes the same thing.
    onnx.save(model, str(tmp_path / "model.onnx"))
    # The exported model carries no weights (its parameters are inputs named
    # in the metadata), so the original SafeTensors bind by path.
    result = import_onnx(
        tmp_path / "model.onnx", output=tmp_path / "back/model.linnet", std_root=STDLIB
    )
    imported = load(
        result.source, generics={}, std_root=STDLIB, weights=tmp_path, bindings=result.bindings
    )
    torch.testing.assert_close(
        imported(tokens, cos_table, sin_table), expected, atol=1e-4, rtol=1e-4
    )


def test_while_loop_runs_under_onnxruntime() -> None:
    """A `while` exports as an ONNX `Loop` whose body recomputes the condition,
    and iterates as the interpreter does."""
    onnxruntime = pytest.importorskip("onnxruntime")
    source = REPO / "spec-tests/valid/023_while.linnet"
    model = _export_onnx(source, {"N": 3})
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    scale = np.array([2.0, 3.0, 1.5], np.float32)
    for limit, expected in ((3, [8.0, 27.0, 3.375]), (100, [128.0, 2187.0, 17.0859375])):
        feeds = {"x": np.ones(3, np.float32), "limit": np.array(limit, np.int32), "param0": scale}
        (actual,) = session.run(None, feeds)
        np.testing.assert_allclose(np.asarray(actual), np.array(expected, np.float32))

"""FlashInfer's decode attention as a backend, in a plan replayed as one
CUDA graph."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from linnet.backends import Context
from linnet.torch import load
from linnet.torch.compiled import CompiledLinnetModule
from linnet.torch.cublaslt import CublasLt
from linnet.torch.flashinfer import FlashInfer
from linnet.torch.planned import Captured
from tests.helpers import LLAMA, STDLIB, requires_sm90

pytest.importorskip("flashinfer")

# Heads 64 wide, as FlashInfer's decode kernels take.
LLAMA_64: dict[str, int | str] = {
    "Vocab": 11,
    "H": 128,
    "Heads": 2,
    "KvHeads": 1,
    "Inner": 64,
    "Layers": 2,
    "Batch": 3,
    "MaxSeq": 16,
    "T": "bf16",
}


def _model(weights: Path) -> CompiledLinnetModule:
    model = load(
        LLAMA, generics=LLAMA_64, std_root=STDLIB, weights=weights, compile=True, device="cuda"
    )
    assert isinstance(model, CompiledLinnetModule)
    return model


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

    skeleton = load(LLAMA, generics=LLAMA_64, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): (torch.randn(p.shape, generator=generator) * 0.3).bfloat16()
        for name, p in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return path


@requires_sm90
def test_flashinfer_attends_as_the_entry_does(weights: Path) -> None:
    planned, direct = _model(weights), _model(weights)
    tokens = torch.tensor([[3], [5], [7]], dtype=torch.int32, device="cuda")

    def positions(step: int) -> torch.Tensor:
        return torch.tensor(step, dtype=torch.int32, device="cuda")

    plan = planned.plan_for(
        "decode", [tokens, positions(0)], backends=[FlashInfer(), CublasLt()], measure=False
    )
    assert [step.backend for step in plan.steps].count("flashinfer") == 2
    parameters = {
        name.removeprefix("root."): parameter.detach()
        for name, parameter in planned.named_parameters()
    }
    runner = Captured(plan, Context(plan.target, parameters))
    with torch.no_grad():
        for step in range(5):  # as it is, captured, then replayed
            (logits,) = runner([tokens, positions(step)])
            expected: object = direct.run_entry("decode", [tokens, positions(step)])
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(logits, expected, rtol=3e-2, atol=3e-2)

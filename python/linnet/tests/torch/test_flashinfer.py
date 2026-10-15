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
    pytest.importorskip("flashinfer")
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


# The serving test's model in bf16, paged with the cards' 64 positions a page.
PAGED: dict[str, int | str] = {
    "Vocab": 64,
    "H": 256,
    "Heads": 4,
    "Batch": 1,
    "MaxSeq": 256,
    "PageSize": 64,
    "T": "bf16",
}


def _paged_model(tmp_path: Path) -> tuple[Path, Path]:
    from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

    from tests.test_serve import SOURCE

    source = tmp_path / "serve.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    skeleton = load(source, generics=PAGED, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): (torch.randn(p.shape, generator=generator) * 0.3).bfloat16()
        for name, p in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    weights = tmp_path / "serve.safetensors"
    save_file(tensors, str(weights))
    return source, weights


def test_flashinfer_takes_paged_decode(tmp_path: Path) -> None:
    import subprocess

    from linnet.backends import Graph, Target
    from linnet.compiler import find_compiler

    source, _ = _paged_model(tmp_path)
    binds = [p for k, v in {**PAGED, "Rows": 2, "Pages": 1}.items() for p in ("--bind", f"{k}={v}")]
    command = [find_compiler(), "graph", "--std", str(STDLIB), "--entry", "decode_paged"]
    text = subprocess.run(
        [*command, "--numerics", "fast", *binds, str(source)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    hopper = Target("cuda", 0, "H100", (9, 0))
    assert len(list(FlashInfer().match(Graph.parse(text), hopper))) == 1


@requires_sm90
def test_flashinfer_attends_over_pages_as_the_entry_does(tmp_path: Path) -> None:
    pytest.importorskip("flashinfer")
    source, weights = _paged_model(tmp_path)

    def model() -> CompiledLinnetModule:
        made = load(
            source, generics=PAGED, std_root=STDLIB, weights=weights, compile=True, device="cuda"
        )
        assert isinstance(made, CompiledLinnetModule)
        return made

    planned, direct = model(), model()

    def ints(values: list[int], *shape: int) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.int32, device="cuda")
        return tensor.reshape(shape) if shape else tensor

    # Rows on pages 1 and 2 (page 0 takes the writes nobody reads): prompts
    # of 6 and 8 tokens, then a step of each.
    table = ints([1, 2], 2, 1)
    prompts = [
        ints([7, 1, 2, 3, 4, 5, 9, 8, 7, 6, 5, 4, 3, 2, 0, 0]),
        ints([*range(6), *range(8), 0, 0]),
        ints([0] * 6 + [1] * 8 + [0, 0]),
        ints([64 + p for p in range(6)] + [128 + p for p in range(8)] + [0, 0]),
        ints([5, 13]),
        table,
    ]
    step = [ints([1, 2], 2, 1), ints([6, 8]), table]
    with torch.no_grad():
        for module in (planned, direct):
            module.run_entry("prefill_paged", prompts)
        plan = planned.plan_for("decode_paged", step, backends=[FlashInfer()], measure=False)
        assert [s.backend for s in plan.steps].count("flashinfer") == 1
        (got,) = plan.run(step, planned.context_for(plan))
        expected: object = direct.run_entry("decode_paged", step)
    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(got.float(), expected.float(), rtol=5e-2, atol=5e-2)

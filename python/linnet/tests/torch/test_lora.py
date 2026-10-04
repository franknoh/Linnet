"""Low-rank adapters (`add_lora`): the adapted model starts as the base
model, trains its adapters alone, merges them into its weights with the same
outputs, and saves and loads them on their own."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open  # type: ignore[import-untyped]
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.plan import PlanError
from linnet.torch import LinnetModule, bind_weights, load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"
LLAMA = REPO / "examples/01-llama/src/lib.linnet"
GENERICS: dict[str, int | str] = {
    "Vocab": 11,
    "H": 8,
    "Heads": 4,
    "KvHeads": 2,
    "Inner": 16,
    "Layers": 2,
    "Batch": 1,
    "MaxSeq": 8,
    "T": "f32",
}
PATTERN = "layers.*.attention.*_proj.weight"
ADAPTERS = ["*.lora_a", "*.lora_b"]


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    skeleton = load(LLAMA, generics=GENERICS, std_root=STDLIB)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name.removeprefix("root."): torch.randn(parameter.shape, generator=generator) * 0.3
        for name, parameter in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return path


def _model(weights: Path, **options: object) -> LinnetModule:
    return load(LLAMA, generics=GENERICS, std_root=STDLIB, weights=weights, **options)  # type: ignore[arg-type]


TOKENS = torch.tensor([[3, 1, 4, 1, 5, 9, 2]], dtype=torch.int32)
PACKED = [
    torch.tensor([3, 1, 4, 1, 5, 9, 2, 6], dtype=torch.int32),
    torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32),
    torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
    torch.tensor([1, 4, 1, 5, 9, 2, 6, 5]),
    torch.full((8,), 1 / 8),
]


def _train(model: LinnetModule, steps: int) -> list[float]:
    trained = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=5e-2)
    losses: list[float] = []
    for _ in range(steps):
        loss = model.run_entry("loss_packed", PACKED)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach()))
    return losses


def test_adapters_start_as_the_base_model_and_train_alone(weights: Path) -> None:
    model = _model(weights, compile=True)
    with torch.no_grad():
        base = model.run_entry("forward", [TOKENS])
    q_proj = model.get_parameter("root.layers.0.attention.q_proj.weight").detach().clone()

    adapted = model.add_lora(PATTERN, rank=4, alpha=8)
    assert len(adapted) == 2 * 4
    trained = {n.removeprefix("root.") for n, p in model.named_parameters() if p.requires_grad}
    assert trained == {
        f"layers.{i}.attention.{proj}.{part}"
        for i in range(2)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        for part in ("lora_a", "lora_b")
    }
    with torch.no_grad():
        torch.testing.assert_close(model.run_entry("forward", [TOKENS]), base)

    losses = _train(model, 20)
    assert losses[-1] < losses[0]
    torch.testing.assert_close(model.get_parameter("root.layers.0.attention.q_proj.weight"), q_proj)


def test_merged_adapters_compute_the_same(weights: Path) -> None:
    model = _model(weights, compile=True)
    model.add_lora(PATTERN, rank=4, alpha=8)
    _train(model, 5)
    with torch.no_grad():
        adapted = model.run_entry("forward", [TOKENS])
    merged = model.merge_lora()
    assert len(merged) == 8
    assert not any("lora" in name for name, _ in model.named_parameters())
    with torch.no_grad():
        torch.testing.assert_close(model.run_entry("forward", [TOKENS]), adapted)


def test_adapters_save_and_load_alone(weights: Path, tmp_path: Path) -> None:
    model = _model(weights, compile=True)
    model.add_lora(PATTERN, rank=4, alpha=8)
    _train(model, 5)
    with torch.no_grad():
        expected = model.run_entry("forward", [TOKENS])
    saved = model.save_weights(tmp_path / "adapters.safetensors", names="linnet", include=ADAPTERS)
    with safe_open(str(saved), framework="pt") as handle:  # type: ignore[no-untyped-call]
        keys = set(handle.keys())
    assert len(keys) == 16 and all(key.split(".")[-1] in ("lora_a", "lora_b") for key in keys)

    again = _model(weights, compile=True)
    again.add_lora(PATTERN, rank=4, alpha=8)
    bind_weights(again, saved, strict=False)
    with torch.no_grad():
        torch.testing.assert_close(again.run_entry("forward", [TOKENS]), expected)


def test_the_interpreter_has_no_adapters(weights: Path) -> None:
    with pytest.raises(PlanError, match="generated code"):
        _model(weights, compile=False).add_lora(PATTERN)


def test_adapters_refuse_prepared_weights() -> None:
    """A prepared (joined) weight is no longer the parameter a pattern names."""
    import subprocess

    from linnet.compiler import find_compiler

    command = [find_compiler(), "torch", "--std", str(STDLIB), "--entry", "forward"]
    for name, value in {**GENERICS, "B": 1, "S": 4}.items():
        command += ["--bind", f"{name}={value}"]
    command += ["--prepare", "--lora", PATTERN, "--lora-rank", "4"]
    completed = subprocess.run([*command, str(LLAMA)], capture_output=True, text=True, check=False)
    assert completed.returncode != 0 and "unprepared" in completed.stderr

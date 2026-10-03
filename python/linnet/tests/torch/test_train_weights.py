"""Training a loaded model: which parameters train, gradients reaching every
one even after the model ran without them, a tied checkpoint tensor trained
as one parameter, cache state starting each call detached, and the trained
weights written back under the checkpoint's names or Linnet's."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open  # type: ignore[import-untyped]
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet.torch import LinnetModule, load

from ..test_serve import GENERICS, SOURCE

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The serving test's model, its output head tied to the token
    embedding: `bindings.json` binds both paths to one tensor."""
    source = tmp_path / "model.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "embedding.weight": torch.randn(64, 32, generator=generator) * 0.3,
        "positions.weight": torch.randn(48, 32, generator=generator) * 0.3,
    }
    for name in ("q_proj", "k_proj", "v_proj"):
        tensors[f"{name}.weight"] = torch.randn(32, 32, generator=generator) * 0.3
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps({"head.weight": "embedding.weight"}), encoding="utf-8")
    return source, weights, bindings


def _load(files: tuple[Path, Path, Path], **options: object) -> LinnetModule:
    source, weights, bindings = files
    return load(
        source,
        generics=GENERICS,
        weights=weights,
        bindings=bindings,
        std_root=STDLIB,
        device="cpu",
        **options,  # type: ignore[arg-type]
    )


TOKENS = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=torch.int32)


@pytest.mark.parametrize("compile", [False, True])
def test_patterns_choose_the_parameters_that_train(
    files: tuple[Path, Path, Path], compile: bool
) -> None:
    model = _load(files, compile=compile, trainable=["q_proj.*", "k_proj.weight"])
    trained = {
        name.removeprefix("root.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    # The projections' biases are optional and the checkpoint has none.
    assert trained == {"q_proj.weight", "k_proj.weight"}
    assert model.set_trainable(False) == []
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_gradients_reach_every_weight_after_running_without_them(
    files: tuple[Path, Path, Path],
) -> None:
    """Generated code compiled for inference does weight-only work ahead and
    may join sibling weights into one buffer. Once the model trains, its
    entries compile again: every gradient arrives, on the parameters an
    optimizer holds."""
    model = _load(files, compile=True)
    with torch.no_grad():
        before = model.run_entry("forward", [TOKENS])
    trained = model.set_trainable(True)
    assert set(trained) >= {"q_proj.weight", "k_proj.weight", "v_proj.weight", "head.weight"}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    loss = model.run_entry("forward", [TOKENS]).square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    optimizer.step()
    with torch.no_grad():
        after = model.run_entry("forward", [TOKENS])
    assert not torch.allclose(before, after)
    assert model.root.q_proj.weight in [
        p for group in optimizer.param_groups for p in group["params"]
    ]


@pytest.mark.parametrize("compile", [False, True])
def test_a_tied_tensor_trains_as_one_parameter(
    files: tuple[Path, Path, Path], compile: bool
) -> None:
    model = _load(files, compile=compile, trainable=True)
    assert model.root.head.weight is model.root.embedding.weight
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.run_entry("forward", [TOKENS]).square().mean().backward()
    optimizer.step()
    assert model.root.head.weight is model.root.embedding.weight


def test_written_cache_starts_each_call_detached(files: tuple[Path, Path, Path]) -> None:
    """Generated code writes a KV cache in place. Trained through it, the
    cache would carry the call's graph into the next one."""
    model = _load(files, compile=True, trainable=True)
    prompts = [
        torch.tensor([[3, 1, 4, 1], [5, 9, 2, 6]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([4, 3], dtype=torch.int32),
    ]
    for _ in range(2):
        model.run_entry("prefill_slots", prompts).square().mean().backward()
        assert not model.root.cache_k.requires_grad
        assert not model.root.cache_v.requires_grad


def test_trained_weights_are_written_back(files: tuple[Path, Path, Path], tmp_path: Path) -> None:
    source, _, bindings = files
    model = _load(files, compile=True, trainable=True)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.run_entry("forward", [TOKENS]).square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        expected = model.run_entry("forward", [TOKENS])

    # Under the checkpoint's names: the tied tensor once, no absent bias.
    saved = model.save_weights(tmp_path / "trained.safetensors")
    with safe_open(str(saved), framework="pt") as handle:  # type: ignore[no-untyped-call]
        assert set(handle.keys()) == {
            "embedding.weight",
            "positions.weight",
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
        }
    again = load(
        source,
        generics=GENERICS,
        weights=saved,
        bindings=bindings,
        std_root=STDLIB,
        device="cpu",
    )
    with torch.no_grad():
        torch.testing.assert_close(again.run_entry("forward", [TOKENS]), expected)

    # Under Linnet's paths, converted as written.
    linnet = model.save_weights(
        tmp_path / "linnet.safetensors", names="linnet", dtype=torch.bfloat16
    )
    with safe_open(str(linnet), framework="pt") as handle:  # type: ignore[no-untyped-call]
        keys = set(handle.keys())
        assert {"head.weight", "embedding.weight"} <= keys
        head = handle.get_tensor("head.weight")
    assert head.dtype == torch.bfloat16
    embedding = model.get_parameter("root.embedding.weight")
    torch.testing.assert_close(head, embedding.detach().to(torch.bfloat16))

"""End-to-end checks: Linnet sources materialize into modules whose outputs
match hand-written PyTorch references for the same weights."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from linnet_torch import PlanError, load

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


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_block_model_matches_reference(tmp_path: Path, dtype: str) -> None:
    generics: dict[str, int | str] = {"H": 8, "Inner": 16, "Layers": 2, "Vocab": 12, "T": dtype}
    model = load(EXAMPLES / "05-block-and-weights/model.linnet", generics=generics, std_root=STDLIB)
    # Random weights in the checkpoint layout, then bound through SafeTensors.
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        path = name.removeprefix("root.")
        if path.endswith(".bias"):
            continue  # optional; left absent
        weights[path] = torch.randn(parameter.shape).to(parameter.dtype)
    save_file(weights, str(tmp_path / "model.safetensors"))
    model = load(
        EXAMPLES / "05-block-and-weights/model.linnet",
        generics=generics,
        std_root=STDLIB,
        weights=tmp_path,
    )
    assert isinstance(model, torch.nn.Module)
    assert sorted(k.removeprefix("root.") for k in model.state_dict()) == sorted(
        list(weights)
        + [
            k
            for k in (n.removeprefix("root.") for n, _ in model.named_parameters())
            if k.endswith(".bias")
        ]
    )

    tokens = torch.randint(0, 12, (2, 3), dtype=torch.int32)
    out = model(tokens)
    assert out.shape == (2, 3, 12)
    assert out.dtype == (torch.float32 if dtype == "f32" else torch.bfloat16)

    # Reference, written directly against the checkpoint tensors.
    def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        ms = (xf * xf).sum(-1, keepdim=True) / xf.shape[-1]
        return (xf * torch.rsqrt(ms + 1e-5)).to(x.dtype) * w

    x = weights["embedding"][tokens.long()]
    for i in range(2):
        prefix = f"layers.{i}."
        h = rms_norm(x, weights[prefix + "norm_weight"])
        h = (h.float() @ weights[prefix + "up.weight"].float().T).to(x.dtype)
        h = torch.maximum(h, torch.zeros((), dtype=h.dtype))
        x = x + (h.float() @ weights[prefix + "down.weight"].float().T).to(x.dtype)
    reference = (x.float() @ weights["head.weight"].float().T).to(x.dtype)
    # bf16 rounds every product before the sum; the reference rounds only the
    # result, so it is compared loosely.
    tolerance = 1e-5 if dtype == "f32" else 0.5
    torch.testing.assert_close(out.float(), reference.float(), atol=tolerance, rtol=tolerance)


def test_checkpoint_mismatch_is_rejected_before_running(tmp_path: Path) -> None:
    generics = {"H": 8, "Inner": 16, "Layers": 1, "Vocab": 12}
    save_file(
        {"embedding": torch.zeros(12, 4, dtype=torch.bfloat16)}, str(tmp_path / "w.safetensors")
    )
    with pytest.raises(PlanError, match="shape"):
        load(
            EXAMPLES / "05-block-and-weights/model.linnet",
            generics=generics,
            std_root=STDLIB,
            weights=tmp_path,
        )
    with pytest.raises(PlanError, match="missing tensor"):
        save_file(
            {"embedding": torch.zeros(12, 8, dtype=torch.bfloat16)}, str(tmp_path / "w.safetensors")
        )
        load(
            EXAMPLES / "05-block-and-weights/model.linnet",
            generics=generics,
            std_root=STDLIB,
            weights=tmp_path,
        )


def test_wrong_generics_and_inputs_are_rejected() -> None:
    with pytest.raises(PlanError, match="needs a value"):
        load(EXAMPLES / "05-block-and-weights/model.linnet", generics={"H": 8}, std_root=STDLIB)
    model = load(
        EXAMPLES / "05-block-and-weights/model.linnet",
        generics={"H": 8, "Inner": 16, "Layers": 1, "Vocab": 12},
        std_root=STDLIB,
    )
    with pytest.raises(PlanError, match="dtype"):
        model(torch.zeros(2, 3, dtype=torch.int64))
    with pytest.raises(PlanError, match="rank"):
        model(torch.zeros(2, dtype=torch.int32))


def test_tiny_transformer_matches_reference(tmp_path: Path) -> None:
    """The transformer example, built only from the standard library, against
    a straightforward PyTorch implementation of the same architecture."""
    H, heads, inner, layers, vocab = 8, 2, 16, 2, 11  # noqa: N806
    generics: dict[str, int | str] = {
        "Vocab": vocab,
        "H": H,
        "Heads": heads,
        "Inner": inner,
        "Layers": layers,
        "T": "f32",
    }
    source = EXAMPLES / "09-tiny-transformer/src/lib.linnet"
    skeleton = load(source, generics=generics, std_root=STDLIB)
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in skeleton.named_parameters():
        path = name.removeprefix("root.")
        if not path.endswith(".bias"):
            weights[path] = torch.randn(parameter.shape) * 0.3
    save_file(weights, str(tmp_path / "model.safetensors"))
    model = load(source, generics=generics, std_root=STDLIB, weights=tmp_path)

    B, S, D = 2, 5, H // heads  # noqa: N806
    tokens = torch.randint(0, vocab, (B, S), dtype=torch.int32)
    positions = torch.arange(S, dtype=torch.float32)[:, None]
    frequencies = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    angles = positions * frequencies  # [S, D/2]
    cos_table = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin_table = torch.cat([angles.sin(), angles.sin()], dim=-1)
    out = model(tokens, cos_table, sin_table)
    assert out.shape == (B, S, vocab)

    def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w

    def linear(x: torch.Tensor, prefix: str) -> torch.Tensor:
        return x @ weights[prefix + ".weight"].T

    def rope(x: torch.Tensor) -> torch.Tensor:
        first, second = x[..., : D // 2], x[..., D // 2 :]
        return x * cos_table + torch.cat([-second, first], dim=-1) * sin_table

    def split(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(B, S, heads, D).permute(0, 2, 1, 3)

    x = weights["embedding.weight"][tokens.long()]
    mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
    for i in range(layers):
        p = f"layers.{i}."
        h = rms_norm(x, weights[p + "attention_norm.weight"])
        q, k, v = (split(linear(h, p + f"attention.{n}_proj")) for n in ("q", "k", "v"))
        score = (rope(q) @ rope(k).transpose(-1, -2)) * 0.125
        score = torch.where(mask, score, torch.full_like(score, -1e30))
        mixed = torch.softmax(score, dim=-1) @ v
        merged = mixed.permute(0, 2, 1, 3).reshape(B, S, H)
        x = x + linear(merged, p + "attention.o_proj")
        h = rms_norm(x, weights[p + "mlp_norm.weight"])
        gate, up = linear(h, p + "mlp.gate"), linear(h, p + "mlp.up")
        x = x + linear(torch.nn.functional.silu(gate) * up, p + "mlp.down")
    reference = linear(rms_norm(x, weights["norm.weight"]), "head")
    torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)
    assert next(iter(model.state_dict())) == "root.embedding.weight"

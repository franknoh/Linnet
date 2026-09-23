"""Torch -> Linnet -> Torch round trips: a PyTorch model exported to source
checks, formats, materializes back with the same weights, and agrees with the
original on several input shapes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.export import Dim

from linnet import find_compiler
from linnet.torch import ExportError, export_linnet, load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * self.weight


class Layer(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.norm = RMSNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, width = x.shape
        h = self.norm(x)
        head = width // self.heads

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, seq, self.heads, head).permute(0, 2, 1, 3)

        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        score = (q @ k.transpose(-1, -2)) * 0.5
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        score = torch.where(mask, score, torch.full_like(score, -1e30))
        mixed = torch.softmax(score, dim=-1) @ v
        return x + self.o(mixed.permute(0, 2, 1, 3).reshape(batch, seq, width))


class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.embedding = nn.Embedding(11, 8)
        self.layers = nn.ModuleList([Layer(8, 2) for _ in range(2)])
        self.head = nn.Linear(8, 11)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return nn.functional.silu(self.head(x))


class Mlp(nn.Module):
    """A non-uniform `Sequential` (its activations own nothing) exercises the
    renamed-member path and the bindings file; `LayerNorm` maps to the
    standard library's `layer_norm`."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.norm = nn.LayerNorm(6)
        self.net = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 3), nn.GELU("tanh"))
        self.register_buffer("scale", torch.full((3,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)) * self.scale


def _compiler_ok(*args: str) -> None:
    completed = subprocess.run(
        [find_compiler(), *args], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_transformer_round_trip(tmp_path: Path) -> None:
    model = Transformer()
    tokens = torch.randint(0, 11, (2, 5))
    result = export_linnet(
        model,
        (tokens,),
        output=tmp_path / "src/model.linnet",
        dynamic_shapes={"tokens": {0: Dim("batch"), 1: Dim("seq", min=2)}},
        weights=tmp_path / "weights",
        std_root=STDLIB,
    )
    source = result.source.read_text()
    assert "pub entry forward<batch: Dim, seq: Dim>" in source
    assert "sub layers: [Layer; 2]" in source
    assert result.bindings is None  # every parameter path is spelled as in PyTorch
    _compiler_ok("lint", "--std", str(STDLIB), str(result.source))
    _compiler_ok("fmt", "--check", str(result.source))

    copy = load(
        result.source,
        generics={},
        std_root=STDLIB,
        weights=tmp_path / "weights",
    )
    assert sorted(k.removeprefix("root.") for k in copy.state_dict()) == sorted(model.state_dict())
    for batch, seq in [(2, 5), (1, 2), (3, 7)]:
        inputs = torch.randint(0, 11, (batch, seq))
        torch.testing.assert_close(copy(inputs), model(inputs), atol=1e-5, rtol=1e-5)

    # Exporting again yields the same text.
    again = export_linnet(
        model,
        (tokens,),
        output=tmp_path / "again/model.linnet",
        dynamic_shapes={"tokens": {0: Dim("batch"), 1: Dim("seq", min=2)}},
        std_root=STDLIB,
    )
    assert again.source.read_text() == source


def test_static_mlp_with_bindings(tmp_path: Path) -> None:
    model = Mlp()
    x = torch.randn(4, 6)
    result = export_linnet(
        model, (x,), output=tmp_path / "mlp.linnet", weights=tmp_path / "w", std_root=STDLIB
    )
    source = result.source.read_text()
    assert "buffer scale: Tensor[3; f32]" in source
    assert "sub _0: Linear" in source and "sub _2: Linear" in source
    assert "layer_norm<[4], 6, f32>(x, norm.weight, some(norm.bias), 1e-05)" in source
    assert result.bindings is not None
    copy = load(
        result.source,
        generics={},
        std_root=STDLIB,
        weights=tmp_path / "w",
        bindings=result.bindings,
    )
    torch.testing.assert_close(copy(x), model(x), atol=1e-5, rtol=1e-5)


def test_unsupported_operations_are_named() -> None:
    class Odd(nn.Module):
        def __init__(self) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.weight = nn.Parameter(torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.erf(x * self.weight)

    with pytest.raises(ExportError, match=r"aten\.erf\.default"):
        export_linnet(Odd(), (torch.randn(2, 4),), output="/nonexistent/odd.linnet")

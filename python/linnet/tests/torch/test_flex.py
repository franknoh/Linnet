"""Masked attention under fast numerics: the generated PyTorch runs it as
FlexAttention over the key blocks its mask reaches when compiled for CUDA
(`_attend`), and as plain arithmetic otherwise. Both must give the same
numbers -- a serving step's rows at their own lengths, and prompts packed
end to end that each see only themselves."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

from linnet.compiler import find_compiler
from linnet.torch import load

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.flex

use std.nn.attention::{grouped_attention, grouped_attention_rows}

pub block Model<B: Dim, H: Dim, Hk: Dim, K: Dim, P: Dim, D: Dim, T: Float = bf16>
where
    Hk > 0,
    H % Hk == 0
{
    // One query per row against a cache of `K`, row `b` seeing positions up
    // to `at[b]`.
    pub entry rows(
        query: Tensor[B, H, 1, D; T],
        keys: Tensor[B, Hk, K, D; T],
        values: Tensor[B, Hk, K, D; T],
        at: Tensor[B; i32],
    ) -> Tensor[B, H, 1, D; T] {
        let slots = iota<i32>(K)
        let seen[b, s] = slots[s] <= at[b]
        return grouped_attention_rows(query, keys, values, 0.25, reshape(seen, [B, 1, K]))
    }

    // `P` tokens of prompts packed end to end, each seeing its own prompt up
    // to itself.
    pub entry packed(
        query: Tensor[1, H, P, D; T],
        keys: Tensor[1, Hk, P, D; T],
        values: Tensor[1, Hk, P, D; T],
        segments: Tensor[P; i32],
        positions: Tensor[P; i32],
    ) -> Tensor[1, H, P, D; T] {
        let own[i, j] = segments[i] == segments[j] && positions[j] <= positions[i]
        return grouped_attention(query, keys, values, 0.25, some(own))
    }

    // `B` prompts of up to `P` tokens, each causal up to its own length: a
    // mask per sequence over many queries.
    pub entry prompts(
        query: Tensor[B, H, P, D; T],
        keys: Tensor[B, Hk, P, D; T],
        values: Tensor[B, Hk, P, D; T],
        lengths: Tensor[B; i32],
    ) -> Tensor[B, H, P, D; T] {
        let slots = iota<i32>(P)
        let seen[b, i, j] = slots[j] <= slots[i] && slots[j] < lengths[b]
        return grouped_attention_rows(query, keys, values, 0.25, seen)
    }
}
"""

GENERICS: dict[str, int | str] = {"B": 3, "H": 4, "Hk": 2, "K": 128, "P": 256, "D": 16}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "flex.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _generated(source: Path, entry: str, numerics: str, **generics: int) -> str:
    command = [find_compiler(), "torch", "--root", "Model", "--entry", entry, "--numerics"]
    command += [numerics, "--std", str(STDLIB)]
    bound = {**GENERICS, **generics}
    command += [arg for name, value in bound.items() for arg in ("--bind", f"{name}={value}")]
    return subprocess.run(
        [*command, str(source)], capture_output=True, text=True, check=True
    ).stdout


def test_only_fast_numerics_attend_by_blocks(source: Path) -> None:
    """`_attend` is the fast lowering; `equivalent` keeps its f32 arithmetic,
    and one mask's blocks are listed once."""
    for entry in ("rows", "packed"):
        fast = _generated(source, entry, "fast")
        assert fast.count("_flex_blocks(v") == 1 and "_attend(" in fast
        assert "_attend(" not in _generated(source, entry, "equivalent")


def test_a_step_attends_by_blocks_for_whole_groups_only(source: Path) -> None:
    """FlexAttention's decoding kernel goes wrong unless each key head serves
    a power of two of query heads; seven a head (Qwen2.5) keeps the two
    products, while a packed pass, which the kernel for many queries runs,
    keeps FlexAttention."""
    assert "_attend(" not in _generated(source, "rows", "fast", H=14, Hk=2)
    assert "_attend(" in _generated(source, "packed", "fast", H=14, Hk=2)


def test_prompts_with_a_mask_each_keep_the_arithmetic(source: Path) -> None:
    """FlexAttention's kernel for many queries gives wrong results when each
    sequence of a batch has its own mask -- its block lists computed in the
    same compiled graph (PyTorch 2.14) -- so such prompts keep the
    arithmetic, while one query a row and one mask for all keep
    FlexAttention."""
    assert "_attend(" not in _generated(source, "prompts", "fast")
    assert "_attend(" in _generated(source, "rows", "fast")
    assert "_attend(" in _generated(source, "packed", "fast")


def _packed_inputs(device: str) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    lengths = [100, 37, 90, 29]  # 256 tokens, the last prompt ending the pack
    segments = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])
    positions = torch.cat([torch.arange(n) for n in lengths])
    shapes = [(1, 4, 256, 16), (1, 2, 256, 16), (1, 2, 256, 16)]
    tensors = [torch.randn(*shape, generator=generator).to(torch.bfloat16) for shape in shapes]
    return [t.to(device) for t in [*tensors, segments.int(), positions.int()]]


def _rows_inputs(device: str) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    shapes = [(3, 4, 1, 16), (3, 2, 128, 16), (3, 2, 128, 16)]
    tensors = [torch.randn(*shape, generator=generator).to(torch.bfloat16) for shape in shapes]
    return [t.to(device) for t in [*tensors, torch.tensor([5, 127, 64], dtype=torch.int32)]]


def _prompts_inputs(device: str) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    shapes = [(3, 4, 256, 16), (3, 2, 256, 16), (3, 2, 256, 16)]
    tensors = [torch.randn(*shape, generator=generator).to(torch.bfloat16) for shape in shapes]
    return [t.to(device) for t in [*tensors, torch.tensor([100, 256, 37], dtype=torch.int32)]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention runs on CUDA")
@pytest.mark.parametrize("entry", ["rows", "packed", "prompts"])
def test_flex_attention_agrees_with_the_arithmetic(source: Path, entry: str) -> None:
    inputs = {"rows": _rows_inputs, "packed": _packed_inputs, "prompts": _prompts_inputs}[entry](
        "cuda"
    )
    eager = load(source, generics=GENERICS, std_root=STDLIB, device="cuda", compile=True)
    compiled = load(source, generics=GENERICS, std_root=STDLIB, device="cuda", compile="inductor")
    expected = eager.run_entry(entry, inputs).float()
    got = compiled.run_entry(entry, inputs).float()
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
    # The packed prompts saw nothing of each other: prompt 1 alone gives the
    # same rows.
    if entry == "packed":
        query, keys, values, segments, positions = inputs
        alone = slice(100, 137)
        solo = torch.nn.functional.scaled_dot_product_attention(
            query[:, :, alone].float(),
            keys[:, :, alone].float().repeat_interleave(2, dim=1),
            values[:, :, alone].float().repeat_interleave(2, dim=1),
            is_causal=True,
            scale=0.25,
        )
        torch.testing.assert_close(got[:, :, alone], solo, atol=2e-2, rtol=2e-2)
        assert bool((segments[alone] == 1).all()) and int(positions[alone][0]) == 0

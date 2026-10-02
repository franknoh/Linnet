"""`std.nn.attention::sink_attention` (gpt-oss's attention with one sink
logit per query head) against attention over the keys and a sink column,
that column then dropped: the canonical body, the interpreter's fast path,
the generated PyTorch, and on CUDA its FlexAttention path under
`torch.compile`."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from linnet.torch import load_function

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.sinks

use std.nn.attention::{sink_attention}

pub entry attend<B: Dim, Hq: Dim, Hk: Dim, Q: Dim, K: Dim, D: Dim, T: Float>(
    query: Tensor[B, Hq, Q, D; T],
    key: Tensor[B, Hk, K, D; T],
    value: Tensor[B, Hk, K, D; T],
    sinks: Tensor[Hq; T],
    mask: Tensor[B, Q, K; bool],
) -> Tensor[B, Hq, Q, D; T]
where
    Hk > 0,
    Hq % Hk == 0
{
    return sink_attention(query, key, value, sinks, 0.125, mask)
}
"""


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Each query head's keys and its sink as one softmax, in f64; the sink
    column dropped before the values are mixed."""
    group = query.shape[1] // key.shape[1]
    keys = key.double().repeat_interleave(group, dim=1)
    values = value.double().repeat_interleave(group, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", query.double(), keys) * 0.125
    scores = scores.masked_fill(~mask[:, None], float("-inf"))
    sink = sinks.double()[None, :, None, None].expand(*scores.shape[:-1], 1)
    weights = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bhqk,bhkd->bhqd", weights, values)


def _inputs(
    device: str, queries: int, keys: int, dtype: torch.dtype = torch.float32, batch: int = 2
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(0)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device).to(dtype)

    query, key, value = (
        normal(batch, 8, queries, 64),
        normal(batch, 2, keys, 64),
        normal(batch, 2, keys, 64),
    )
    sinks = normal(8) * 2
    # Each sequence at its own length, causal within it, and one query of the
    # second that sees nothing: its output is all sink, so zero.
    ends = torch.tensor([keys - queries, 3][:batch], device=device)
    at = torch.arange(queries, device=device)[:, None] + ends[:, None, None]
    mask = torch.arange(keys, device=device)[None, None, :] <= at
    if batch > 1:
        mask[1, 0] = False
    return query, key, value, sinks, mask


@pytest.mark.parametrize(
    ("numerics", "compile"), [("exact", False), ("fast", False), ("fast", True)]
)
def test_the_sink_takes_its_share(tmp_path: Path, numerics: str, compile: bool) -> None:
    source = tmp_path / "sinks.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    attend = load_function(source, "attend", std_root=STDLIB, numerics=numerics, compile=compile)
    inputs = _inputs("cpu", 5, 7)
    got = attend(*inputs)
    torch.testing.assert_close(got.double(), _reference(*inputs), atol=1e-5, rtol=1e-5)
    assert bool((got[1, :, 0] == 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention runs on CUDA")
@pytest.mark.parametrize(("queries", "batch"), [(128, 1), (1, 2), (128, 2)])
def test_flex_attention_folds_the_sink_in(tmp_path: Path, queries: int, batch: int) -> None:
    """Under `torch.compile` on CUDA, FlexAttention over the blocks the mask
    reaches -- a prompt's 128 queries under one mask, or one query per row
    of a serving step -- the sink folded in from its log-sum-exp. Many
    queries with a mask per sequence take the two products instead."""
    source = tmp_path / "sinks.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    attend = load_function(source, "attend", std_root=STDLIB, numerics="fast", compile="inductor")
    inputs = _inputs("cuda", queries, 256, torch.bfloat16, batch)
    got = attend(*inputs)
    torch.testing.assert_close(got.double(), _reference(*inputs), atol=2e-2, rtol=2e-2)

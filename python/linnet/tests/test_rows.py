"""A batch whose sequences sit at different positions: a cache row written
at each sequence's own position (`write_rows`), a prompt written into one
row of a shared cache (`write_slot`), and attention masked per sequence
(`grouped_attention_rows`) -- what serving several requests at once needs --
must compute the same numbers on every path, and the generated PyTorch must
write its caches in place rather than copy them."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from linnet.compiler import find_compiler
from linnet.torch import load

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

SOURCE = """\
module tests.rows

use std.nn.attention::{grouped_attention_rows}
use std.nn.cache::{write_rows, write_slot, write_slots}

pub block Model<B: Dim, Hk: Dim, H: Dim, S: Dim, D: Dim, N: Dim, T: Float = f32>
where
    S > 0,
    N > 0,
    Hk > 0,
    H % Hk == 0
{
    state cache: Tensor[B, Hk, S, D; T]

    pub entry step(
        value: Tensor[B, Hk, 1, D; T],
        at: Tensor[B; i32],
        query: Tensor[B, H, 1, D; T],
    ) -> Tensor[B, H, 1, D; T] {
        cache = write_rows(cache, value, at)
        let positions = iota<i32>(S)
        let seen[b, s] = positions[s] <= at[b]
        return grouped_attention_rows(query, cache, cache, 0.5, reshape(seen, [B, 1, S]))
    }

    pub entry store(value: Tensor[1, Hk, N, D; T], slot: i32, at: i32) -> Tensor[B, Hk, S, D; T] {
        cache = write_slot(cache, value, slot, at)
        return cache
    }

    pub entry store_many(
        value: Tensor[2, Hk, N, D; T],
        slots: Tensor[2; i32],
        at: i32,
    ) -> Tensor[B, Hk, S, D; T] {
        cache = write_slots(cache, value, slots, at)
        return cache
    }
}
"""

GENERICS: dict[str, int | str] = {"B": 3, "Hk": 2, "H": 4, "S": 8, "D": 4, "N": 3}


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "rows.linnet"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _case() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "prompt": rng.standard_normal((1, 2, 3, 4)).astype(np.float32),
        "value": rng.standard_normal((3, 2, 1, 4)).astype(np.float32),
        "at": np.array([3, 0, 6], dtype=np.int32),
        "query": rng.standard_normal((3, 4, 1, 4)).astype(np.float32),
    }


def _expected(case: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """`store(prompt, slot=2, at=1)` into a zero cache, then `step`."""
    cache = np.zeros((3, 2, 8, 4), np.float32)
    cache[2, :, 1:4] = case["prompt"][0]
    stored = cache.copy()
    for b, at in enumerate(case["at"]):
        cache[b, :, at] = case["value"][b, :, 0]
    keys = np.repeat(cache, 2, axis=1)  # each key/value head serves two queries
    scores = np.einsum("bhqd,bhkd->bhqk", case["query"], keys) * 0.5
    seen = np.arange(8)[None, :] <= case["at"][:, None]
    scores = np.where(seen[:, None, None, :], scores, -1e30)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return stored, np.einsum("bhqk,bhkd->bhqd", weights, keys)


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("numerics", ["exact", "equivalent"])
def test_torch(source: Path, compile: bool, numerics: str) -> None:
    case = _case()
    stored, mixed = _expected(case)
    model = load(source, generics=GENERICS, std_root=STDLIB, compile=compile, numerics=numerics)
    got_store = model.run_entry(
        "store",
        [
            torch.tensor(case["prompt"]),
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(1, dtype=torch.int32),
        ],
    )
    np.testing.assert_allclose(got_store.numpy(), stored, atol=1e-6)
    inputs = [torch.tensor(case[k]) for k in ("value", "at", "query")]
    got = model.run_entry("step", inputs)
    np.testing.assert_allclose(got.numpy(), mixed, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("compile", [False, True])
def test_several_rows_at_once(source: Path, compile: bool) -> None:
    """`write_slots`: two prompts into rows 2 and 0 of a cache, one pass."""
    rng = np.random.default_rng(1)
    value = rng.standard_normal((2, 2, 3, 4)).astype(np.float32)
    expected = np.zeros((3, 2, 8, 4), np.float32)
    expected[2, :, 1:4] = value[0]
    expected[0, :, 1:4] = value[1]
    model = load(source, generics=GENERICS, std_root=STDLIB, compile=compile)
    got = model.run_entry(
        "store_many",
        [
            torch.tensor(value),
            torch.tensor([2, 0], dtype=torch.int32),
            torch.tensor(1, dtype=torch.int32),
        ],
    )
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-6)
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from linnet.jax import load as load_jax
    from linnet.jax import load_source

    for loader in (load_jax, load_source):
        many = loader(source, generics=GENERICS, weights={}, std_root=STDLIB, entry="store_many")
        out, _ = many(jnp.asarray(value), jnp.asarray([2, 0], dtype=jnp.int32), jnp.int32(1))
        np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6)


def test_the_cache_is_written_in_place(source: Path) -> None:
    """The generated source writes the state tensor itself, and the loaded
    model keeps that tensor rather than a copy of it."""
    command = [find_compiler(), "torch", "--root", "Model", "--entry", "step"]
    command += [arg for name, value in GENERICS.items() for arg in ("--bind", f"{name}={value}")]
    command += ["--std", str(STDLIB), str(source)]
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    assert ".index_put_(" in text
    model = load(source, generics=GENERICS, std_root=STDLIB, compile=True)
    case = _case()
    model.run_entry("step", [torch.tensor(case[k]) for k in ("value", "at", "query")])
    before = dict(model.named_buffers())["root.cache"]
    model.run_entry("step", [torch.tensor(case[k]) for k in ("value", "at", "query")])
    assert dict(model.named_buffers())["root.cache"] is before


@pytest.mark.parametrize("target", ["stablehlo", "jax"])
def test_xla(source: Path, target: str) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from linnet.jax import load as load_jax
    from linnet.jax import load_source

    loader = load_source if target == "jax" else load_jax
    case = _case()
    stored, mixed = _expected(case)
    store = loader(source, generics=GENERICS, weights={}, std_root=STDLIB, entry="store")
    got_store, state = store(jnp.asarray(case["prompt"]), jnp.int32(2), jnp.int32(1))
    np.testing.assert_allclose(np.asarray(got_store), stored, atol=1e-6)
    step = loader(source, generics=GENERICS, weights={}, std_root=STDLIB, entry="step")
    got, _ = step(*(jnp.asarray(case[k]) for k in ("value", "at", "query")), state=state)
    np.testing.assert_allclose(np.asarray(got), mixed, atol=1e-5, rtol=1e-5)


def test_onnx(source: Path, tmp_path: Path) -> None:
    """Cache writes become ONNX scatters (`ScatterElements` for a row per
    sequence, `ScatterND` for a slot's span), and attention with a mask per
    sequence two `MatMul`s around a `Softmax`. Both must be right."""
    onnxruntime = pytest.importorskip("onnxruntime")
    from safetensors.numpy import save_file  # type: ignore[import-untyped]

    from linnet.onnx import export_model

    case = _case()
    _, mixed = _expected(case)
    weights = tmp_path / "none.safetensors"  # the model has state and no parameters
    save_file({}, str(weights))
    exported = export_model(
        source, generics=GENERICS, weights=weights, std_root=STDLIB, entry="step"
    )
    session = onnxruntime.InferenceSession(
        exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [i.name for i in session.get_inputs()]
    cache = np.zeros((3, 2, 8, 4), np.float32)
    cache[2, :, 1:4] = case["prompt"][0]
    feeds = dict(zip(names, [case["value"], case["at"], case["query"], cache], strict=True))
    got = session.run(None, feeds)[0]
    np.testing.assert_allclose(got, mixed, atol=1e-5, rtol=1e-5)
    # Every cache write is one scatter, not a pass over the whole cache.
    ops = [node.op_type for node in exported.model.graph.node]
    assert ops.count("ScatterElements") == 1 and ops.count("MatMul") == 2
    rng = np.random.default_rng(2)
    one = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    two = rng.standard_normal((2, 2, 3, 4)).astype(np.float32)
    for entry, inputs, expected in [
        ("store", [one, np.array(2, np.int32), np.array(1, np.int32)], {2: one[0]}),
        (
            "store_many",
            [two, np.array([2, 0], np.int32), np.array(1, np.int32)],
            {2: two[0], 0: two[1]},
        ),
    ]:
        exported = export_model(
            source, generics=GENERICS, weights=weights, std_root=STDLIB, entry=entry
        )
        session = onnxruntime.InferenceSession(
            exported.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        names = [i.name for i in session.get_inputs()]
        feeds = dict(zip(names, [*inputs, np.zeros((3, 2, 8, 4), np.float32)], strict=True))
        want = np.zeros((3, 2, 8, 4), np.float32)
        for row, values in expected.items():
            want[row, :, 1:4] = values
        np.testing.assert_allclose(session.run(None, feeds)[0], want, atol=1e-6)

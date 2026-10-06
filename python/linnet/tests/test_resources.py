"""Static memory analysis on graphs whose peaks are worked out by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

from linnet.resources import expr as ex
from linnet.resources import graph
from linnet.resources.analysis import MemoryModel
from linnet.resources.config import ExecutionConfig
from linnet.resources.graph import Category, Confidence
from linnet.resources.kvcache import PagedLayout
from linnet.resources.planner import ExecutionPlanner, ResourceConstraint
from linnet.resources.storage import tensor_bytes
from linnet.resources.training import OPTIMIZERS, CheckpointPolicy, TrainingConfig, timeline

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"

CHAIN = """\
module chain

pub block Model<N: Dim, T: Float = f32> {
    param scale: Tensor[N; T]

    pub entry forward<B: Dim>(x: Tensor[B, N; T]) -> Tensor[B, N; T] {
        let a = exp(x)
        let b = a * scale
        return b + 1.0
    }

    pub entry views<B: Dim>(x: Tensor[B, N; T]) -> Tensor[B * N; T] {
        return reshape(x, [B * N])
    }

    pub entry copies<B: Dim>(x: Tensor[B, N; T]) -> Tensor[N * B; T] {
        return reshape(permute(x, [1, 0]), [N * B])
    }
}
"""

CACHE = """\
module kv

use std.nn.cache::{write_at}

pub block Layer<Batch: Dim, KvHeads: Dim, MaxSeq: Dim, D: Dim, T: Float = bf16>
where MaxSeq > 0 {
    state keys: Tensor[Batch, KvHeads, MaxSeq, D; T]
    state values: Tensor[Batch, KvHeads, MaxSeq, D; T]

    pub fn write(k: Tensor[Batch, KvHeads, 1, D; T], pos: i32) -> Tensor[Batch, KvHeads, 1, D; T] {
        keys = write_at(keys, k, pos)
        values = write_at(values, k, pos)
        return k
    }
}

pub block Model<Layers: Dim, Batch: Dim, KvHeads: Dim, MaxSeq: Dim, D: Dim, T: Float = bf16>
where MaxSeq > 0 {
    sub layers: [Layer<Batch, KvHeads, MaxSeq, D, T>; Layers]

    pub entry decode(
        k: Tensor[Batch, KvHeads, 1, D; T],
        pos: i32,
    ) -> Tensor[Batch, KvHeads, 1, D; T] {
        var x = k
        static for layer in layers {
            x = layer.write(x, pos)
        }
        return x
    }
}
"""

MLP = """\
module mlp

use std.nn.activations::{relu}
use std.nn.linear::{Linear}

pub block Layer<H: Dim, T: Float = f32> {
    sub up: Linear<H, H, T>

    pub fn forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return relu(up.forward(x))
    }
}

pub block Model<H: Dim, Layers: Dim, T: Float = f32> {
    sub layers: [Layer<H, T>; Layers]
    sub head: Linear<H, H, T>

    pub entry loss<B: Dim>(x: Tensor[B, H; T]) -> f32 {
        var h = x
        static for layer in layers {
            h = layer.forward(h)
        }
        let y = head.forward(h)
        return sum<f32>[b, i] cast<f32>(y[b, i])
    }
}
"""


SQUARES = """\
module squares

pub block Model<N: Dim, T: Float = f32> {
    param scale: Tensor[N; T]

    pub entry halved<B: Dim>(x: Tensor[B, N; T]) -> f32 {
        let a = x * scale
        let b = a * 0.5
        return sum<f32>[i, j] cast<f32>(b[i, j])
    }

    pub entry squared<B: Dim>(x: Tensor[B, N; T]) -> f32 {
        let a = x * scale
        let b = a * a
        return sum<f32>[i, j] cast<f32>(b[i, j])
    }
}
"""


def source(tmp_path: Path, text: str, name: str = "model") -> Path:
    path = tmp_path / f"{name}.linnet"
    path.write_text(text, encoding="utf-8")
    return path


def memory(model: Path, config: ExecutionConfig, **kwargs: object) -> MemoryModel:
    return MemoryModel(model, config, std_root=STDLIB, **kwargs)  # type: ignore[arg-type]


def config(**kwargs: object) -> ExecutionConfig:
    return ExecutionConfig(backend="generic", **kwargs)  # type: ignore[arg-type]


# ---- expressions and storage


def test_expressions_fold_and_evaluate() -> None:
    b, s = ex.sym("B"), ex.sym("S")
    total = ex.add(ex.mul(ex.const(4), b, s), ex.mul(ex.const(4), b, s), ex.const(0))
    assert ex.format_expr(total) == "8 * B * S"
    assert ex.evaluate(total, {"B": 2, "S": 3}) == 48
    assert ex.evaluate(ex.maximum(b, ex.const(5)), {"B": 2}) == 5
    assert ex.evaluate(ex.ceildiv(b, ex.const(16)), {"B": 17}) == 2
    assert ex.symbols(total) == {"B", "S"}
    with pytest.raises(ex.ExprError):
        ex.evaluate(b, {})


def test_tensor_bytes_follow_the_dtype() -> None:
    n = ex.const(10)
    sizes = {d: ex.evaluate(tensor_bytes(d, n), {}) for d in ("bool", "u8", "bf16", "f32", "i64")}
    assert sizes == {"bool": 10, "u8": 10, "bf16": 20, "f32": 40, "i64": 80}


def test_quantized_weights_count_their_metadata(tmp_path: Path) -> None:
    text = """\
module q

use std.quant::{Int4GroupLinear}

pub block Model<T: Float = bf16> {
    sub proj: Int4GroupLinear<256, 64, 128, T>

    pub entry forward<B: Dim>(x: Tensor[B, 256; T]) -> Tensor[B, 64; T] {
        return proj.forward(x)
    }
}
"""
    model = memory(source(tmp_path, text), config(batch=1))
    weights = model.analyze().component("Weights")
    # u8 pairs [64, 2, 64], bf16 scales [64, 2], u8 zero points [64, 2]
    assert weights.nbytes == 64 * 2 * 64 + 64 * 2 * 2 + 64 * 2
    assert weights.confidence == Confidence.EXACT


# ---- liveness, views and in-place writes


def test_peak_counts_only_what_is_live(tmp_path: Path) -> None:
    model = memory(source(tmp_path, CHAIN), config(batch=8, bindings={"N": 16}))
    result = model.analyze()
    unit = 8 * 16 * 4
    # `x` (held by the caller), `a`, then `b` while `a` is read: three at once.
    assert result.component("Peak activations").nbytes == 3 * unit
    assert result.component("Weights").nbytes == 16 * 4
    assert result.allocation is not None
    assert result.allocation.lower_bound == 3 * unit
    assert result.allocation.lower_bound <= result.allocation.planned <= result.allocation.naive


def test_views_share_storage_and_copies_do_not(tmp_path: Path) -> None:
    built = source(tmp_path, CHAIN)
    unit = 4 * 16 * 4
    views = memory(built, config(entry="views", batch=4, bindings={"N": 16}))
    copies = memory(built, config(entry="copies", batch=4, bindings={"N": 16}))
    assert views.analyze().component("Peak activations").nbytes == unit
    # The permuted view is not contiguous, so the reshape copies it.
    assert copies.analyze().component("Peak activations").nbytes == 2 * unit


def test_tied_parameters_are_counted_once(tmp_path: Path) -> None:
    text = """\
module tied

use std.nn.linear::{Linear}

pub block Model<T: Float = f32> {
    sub a: Linear<8, 8, T>
    sub b: Linear<8, 8, T>

    pub entry forward<B: Dim>(x: Tensor[B, 8; T]) -> Tensor[B, 8; T] {
        return b.forward(a.forward(x))
    }
}
"""
    built = source(tmp_path, text)
    # A card whose bindings read both weights from one checkpoint tensor.
    card = tmp_path / "card"
    card.mkdir()
    (card / "model.linnet").write_text(text, encoding="utf-8")
    (card / "nest.toml").write_text(
        '[model]\nname = "tied"\n\n[source]\npath = "model.linnet"\nroot = "Model"\n\n'
        '[weights]\nbindings = "bindings.json"\n',
        encoding="utf-8",
    )
    (card / "bindings.json").write_text(
        '{"a.weight": "shared", "b.weight": "shared"}',
        encoding="utf-8",
    )
    alone = memory(built, config(batch=1)).analyze().component("Weights").nbytes
    once = memory(card, config(batch=1)).analyze().component("Weights").nbytes
    untied = memory(card, config(batch=1), bindings=False).analyze().component("Weights").nbytes
    assert (alone, once, untied) == (2 * 8 * 8 * 4, 8 * 8 * 4, 2 * 8 * 8 * 4)


@pytest.mark.parametrize(("kv_heads", "label"), [(8, "MHA"), (2, "GQA"), (1, "MQA")])
def test_kv_cache_follows_the_declared_heads(tmp_path: Path, kv_heads: int, label: str) -> None:
    bindings = {"Layers": 3, "KvHeads": kv_heads, "D": 64}
    model = memory(source(tmp_path, CACHE), config(batch=2, cache=100, bindings=bindings))
    result = model.analyze()
    cache = result.component("KV cache")
    assert cache.nbytes == 3 * 2 * 2 * kv_heads * 100 * 64 * 2, label
    assert cache.confidence == Confidence.EXACT
    # Each write updates its cache in place: no second copy of a cache.
    one_cache = 2 * kv_heads * 100 * 64 * 2
    assert (result.component("Peak activations").nbytes or 0) < one_cache


def test_a_paged_layout_rounds_up_to_blocks(tmp_path: Path) -> None:
    bindings = {"Layers": 1, "KvHeads": 2, "D": 64}
    paged = config(batch=1, cache=20, bindings=bindings, kv_layout=PagedLayout(16))
    cache = memory(source(tmp_path, CACHE), paged).analyze().component("KV cache")
    per_token = 2 * 64 * 2
    assert cache.nbytes == 2 * 2 * 16 * per_token  # 20 positions take 2 blocks of 16, twice
    assert cache.confidence == Confidence.MODELED


def test_sizes_stay_functions_of_the_batch(tmp_path: Path) -> None:
    model = memory(source(tmp_path, CHAIN), config(batch=1, bindings={"N": 16}))
    assert model.graph.objects[model.graph.inputs[0]].nbytes == ex.mul(
        ex.const(64), ex.sym("batch")
    )
    one = model.analyze(batch=1).component("Peak activations").nbytes
    four = model.analyze(batch=4).component("Peak activations").nbytes
    assert four == 4 * (one or 0)


def test_lifetimes_and_buffer_plan_on_a_trace(tmp_path: Path) -> None:
    model = memory(source(tmp_path, CHAIN), config(batch=2, bindings={"N": 4}))
    spans = graph.lifetimes(model.graph)
    env = {"batch": 2}
    plan = graph.plan_buffers(model.graph, env, spans)
    found = graph.peak(model.graph, env, spans)
    assert plan.lower_bound == found.nbytes == 3 * 2 * 4 * 4
    assert ex.evaluate(graph.peak_expr(model.graph, spans), env) == found.nbytes


# ---- unknowns


def test_an_unmodeled_backend_reports_unknowns_rather_than_zero(tmp_path: Path) -> None:
    bindings = {"Layers": 1, "KvHeads": 1, "D": 8}
    result = memory(source(tmp_path, CACHE), config(batch=1, cache=8, bindings=bindings)).analyze()
    assert any("index_copy" in item for item in result.unknown)
    assert any(item.startswith("Runtime overhead") for item in result.unknown)
    assert all(c.nbytes is not None for c in result.components)
    assert "Backend workspace" not in {c.name for c in result.components}


def test_the_cuda_model_says_how_sure_it_is(tmp_path: Path) -> None:
    model = memory(
        source(tmp_path, CHAIN), ExecutionConfig(batch=2, bindings={"N": 4}, context_bytes=1 << 20)
    )
    result = model.analyze()
    assert result.component("CUDA context").confidence == Confidence.MODELED
    assert result.component("Weights").confidence == Confidence.EXACT
    assert result.expected_peak > result.graph_peak
    assert result.to_dict()["peak"]["graph"] == result.graph_peak


# ---- training


def training(tmp_path: Path, batch: int = 4, **kwargs: object) -> MemoryModel:
    train = TrainingConfig(**kwargs)  # type: ignore[arg-type]
    return memory(
        source(tmp_path, MLP),
        config(batch=batch, bindings={"H": 32, "Layers": 3}, training=train),
    )


def test_optimizer_states_follow_the_optimizer(tmp_path: Path) -> None:
    numel = 4 * 32 * 32
    states = {}
    for name in ("sgd", "sgd-momentum", "adamw"):
        result = training(tmp_path, optimizer=OPTIMIZERS[name]).analyze()
        states[name] = result.total(Category.OPTIMIZER)
        assert result.component("Weights").nbytes == numel * 4
    assert states == {"sgd": 0, "sgd-momentum": numel * 4, "adamw": 2 * numel * 4}


def test_master_weights_and_gradient_dtypes(tmp_path: Path) -> None:
    numel = 4 * 32 * 32
    model = memory(
        source(tmp_path, MLP),
        config(
            batch=4,
            dtype="bf16",
            bindings={"H": 32, "Layers": 3},
            training=TrainingConfig(master_dtype="f32"),
        ),
    )
    result = model.analyze()
    assert result.component("Weights").nbytes == numel * 2
    assert result.total(Category.MASTER) == numel * 4
    assert result.total(Category.OPTIMIZER) == 2 * numel * 4
    steps = timeline(
        model.graph, model.env(), TrainingConfig(master_dtype="f32"), model.backend, model.arrays
    )
    gradients = sum(i.nbytes for i in steps.intervals if i.category == Category.GRADIENT)
    assert gradients == numel * 4  # f32, as the master weights they update


def test_checkpointing_keeps_less_and_recomputes(tmp_path: Path) -> None:
    # A large batch, so that activations rather than weights set the peak.
    plain = training(tmp_path, batch=1024).analyze()
    blocks = training(tmp_path, batch=1024, checkpoint=CheckpointPolicy("blocks")).analyze()
    assert blocks.total(Category.ACTIVATION) < plain.total(Category.ACTIVATION)
    assert blocks.graph_peak < plain.graph_peak
    assert plain.recompute is None
    assert blocks.recompute is not None and 0 < blocks.recompute < 1 / 3


def test_only_trainable_parameters_get_gradients(tmp_path: Path) -> None:
    model = training(tmp_path, trainable=("head.*",))
    result = model.analyze()
    assert result.total(Category.OPTIMIZER) == 2 * 32 * 32 * 4


def test_autograd_keeps_a_square_but_not_a_scaled_tensor(tmp_path: Path) -> None:
    built = source(tmp_path, SQUARES)
    peaks = {
        entry: memory(
            built, config(entry=entry, batch=64, bindings={"N": 8}, training=TrainingConfig())
        )
        .analyze()
        .graph_peak
        for entry in ("halved", "squared")
    }
    # `a * a` keeps `a` for its backward; `a * 0.5` does not.
    assert peaks["squared"] > peaks["halved"]


# ---- fitting


def test_fit_finds_the_largest_batch(tmp_path: Path) -> None:
    built = source(tmp_path, CACHE)
    bindings = {"Layers": 2, "KvHeads": 2, "D": 16}
    base = config(cache=64, bindings=bindings)
    planner = ExecutionPlanner(built, std_root=STDLIB)
    at = {
        b: planner.evaluate(config(batch=b, cache=64, bindings=bindings)).expected_peak
        for b in (10, 11)
    }
    found = planner.maximize(base, ResourceConstraint(at[10] + (at[11] - at[10]) // 2), "batch")
    assert found.value == 10
    assert found.headroom is not None and 0 <= found.headroom < at[11] - at[10]
    assert found.limited_by == "memory"


def test_a_reserve_shrinks_the_budget(tmp_path: Path) -> None:
    built = source(tmp_path, CACHE)
    bindings = {"Layers": 2, "KvHeads": 2, "D": 16}
    planner = ExecutionPlanner(built, std_root=STDLIB)
    memory = planner.evaluate(config(batch=10, cache=64, bindings=bindings)).expected_peak
    plain = ResourceConstraint(memory)
    reserved = ResourceConstraint(memory, reserve=memory // 2)
    percent = ResourceConstraint(memory, reserve_percent=50)
    base = config(cache=64, bindings=bindings)
    assert planner.maximize(base, plain, "batch").value == 10
    assert reserved.budget == percent.budget == memory - memory // 2
    smaller = planner.maximize(base, reserved, "batch").value
    assert smaller is not None and smaller < 10


def test_fit_reports_when_nothing_fits(tmp_path: Path) -> None:
    built = source(tmp_path, CACHE)
    found = ExecutionPlanner(built, std_root=STDLIB).maximize(
        config(cache=64, bindings={"Layers": 2, "KvHeads": 2, "D": 16}),
        ResourceConstraint(10),
        "batch",
    )
    assert found.value is None and found.limited_by == "memory"


def test_fit_maximizes_the_cache(tmp_path: Path) -> None:
    built = source(tmp_path, CACHE)
    bindings = {"Layers": 1, "KvHeads": 1, "D": 8}
    planner = ExecutionPlanner(built, std_root=STDLIB)
    memory = planner.evaluate(config(batch=1, cache=1000, bindings=bindings)).expected_peak
    found = planner.maximize(
        config(batch=1, bindings=bindings), ResourceConstraint(memory), "kv-cache"
    )
    assert found.value == 1000


def test_unsupported_parallelism_is_refused(tmp_path: Path) -> None:
    from linnet.resources.trace import TraceError

    with pytest.raises(TraceError, match="tensor and pipeline"):
        memory(source(tmp_path, CHAIN), config(batch=1, bindings={"N": 4}, tensor_parallel=2))

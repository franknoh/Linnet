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


SHARDED = """\
module sharded

use std.nn.linear::{Linear}
use std.nn.parallel::{all_gather, all_reduce}

pub block Model<H: Dim, Out: Dim, T: Float = f32, Shards: Dim = 1>
where
    Shards > 0,
    H % Shards == 0,
    Out % Shards == 0
{
    sub up: Linear<H, H / Shards, T>
    sub down: Linear<H / Shards, H, T>
    sub head: Linear<H, Out / Shards, T>

    pub entry forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, Out; T] {
        let y = all_reduce(down.forward(up.forward(x)))
        return all_gather<B, Out / Shards, Shards, T>(head.forward(y))
    }
}
"""

SPLIT = """\
module split

use std.nn.linear::{Linear}

pub block Model<H: Dim, T: Float = f32> {
    sub q_proj: Linear<H, H, T>
    sub o_proj: Linear<H, H, T>
    sub norm: Linear<H, H, T>

    pub entry forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return norm.forward(o_proj.forward(q_proj.forward(x)))
    }
}
"""


def test_a_sharded_model_is_one_process_part(tmp_path: Path) -> None:
    built = source(tmp_path, SHARDED)
    bindings = {"H": 16, "Out": 32}
    whole = memory(built, ExecutionConfig(batch=4, bindings=bindings)).analyze()
    model = memory(built, ExecutionConfig(batch=4, bindings=bindings, tensor_parallel=2))
    part = model.analyze()
    weights = 16 * 16 + 16 * 16 + 32 * 16
    assert whole.component("Weights").nbytes == weights * 4
    assert part.component("Weights").nbytes == weights * 4 // 2
    # Across processes the sum is a tensor of its own, and the gather
    # stacks the parts before laying them side by side.
    by = {s.implementation: s for s in model.graph.steps if s.implementation}
    gather = by["torch.distributed.all_gather"]
    assert model.backend.workspace(gather, model.graph, model.env()).nbytes == 4 * 32 * 4
    out = model.graph.objects[by["torch.distributed.all_reduce"].outputs[0]]
    assert out.storage == out.id
    assert (part.component("NCCL communicator").nbytes or 0) > 0
    assert any("tensor parallelism" in a for a in part.assumptions)


def test_dtensor_splits_weights_by_the_rules(tmp_path: Path) -> None:
    built = source(tmp_path, SPLIT)
    whole = memory(built, config(batch=2, bindings={"H": 16})).analyze()
    part = memory(built, config(batch=2, bindings={"H": 16}, tensor_parallel=2)).analyze()
    # q_proj by output and o_proj by input split in two; `norm` matches no
    # rule and is copied.
    assert whole.component("Weights").nbytes == 3 * 16 * 16 * 4
    assert part.component("Weights").nbytes == (16 * 16 // 2 * 2 + 16 * 16) * 4
    assert part.component("Peak activations").confidence == Confidence.ESTIMATED


def test_dtensor_splits_values_made_from_split_weights(tmp_path: Path) -> None:
    part = memory(source(tmp_path, SPLIT), config(batch=2, bindings={"H": 16}, tensor_parallel=2))
    graph, env = part.graph, part.env()
    first = next(s for s in graph.steps if s.implementation == "torch.nn.functional.linear")
    # q_proj's rows are split, so its result is too; the input is whole.
    assert ex.evaluate(graph.objects[first.outputs[0]].nbytes, env) == 2 * 16 * 4 // 2
    assert ex.evaluate(graph.objects[graph.inputs[0]].nbytes, env) == 2 * 16 * 4


STATEFUL = """\
module stateful

use std.nn.linear::{Linear}

pub block Layer<H: Dim, T: Float = f32> {
    sub up: Linear<H, H, T>
    state seen: Tensor[4, H; T]

    pub fn forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return up.forward(x)
    }
}

pub block Model<H: Dim, Layers: Dim, T: Float = f32> {
    sub layers: [Layer<H, T>; Layers]

    pub entry forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        var h = x
        static for layer in layers {
            h = layer.forward(h)
        }
        return h
    }
}
"""


def test_a_stage_holds_its_state_whether_the_entry_reads_it_or_not(tmp_path: Path) -> None:
    model = memory(
        source(tmp_path, STATEFUL),
        config(batch=8, bindings={"H": 16, "Layers": 2}, pipeline_parallel=2, microbatches=2),
    )
    for part in model.stages:
        held = {o.path for o in part.graph.objects if o.category == Category.STATE}
        assert held == {f"{unit}.seen" for unit in part.units}


def _pipeline(tmp_path: Path, **kwargs: object) -> MemoryModel:
    return memory(
        source(tmp_path, MLP),
        config(batch=64, bindings={"H": 32, "Layers": 3}, pipeline_parallel=2, **kwargs),
    )


def test_a_pipeline_splits_blocks_as_the_runtime_does(tmp_path: Path) -> None:
    model = _pipeline(tmp_path, microbatches=4)
    assert [part.units for part in model.stages] == [("layers.0", "layers.1"), ("layers.2", "head")]
    # Only the hidden rows cross; each stage holds half the weights.
    assert len(model.stages[1].receives) == 1
    result = model.analyze()
    assert [d.name for d in result.devices] == ["stage 0", "stage 1"]
    assert result.component("Weights").nbytes == 2 * 32 * 32 * 4
    # The three other micro-batches' hidden rows (16 of 64) stay to the end.
    held = result.component("Micro-batches held").nbytes or 0
    assert held >= 3 * 16 * 32 * 4
    explicit = _pipeline(tmp_path, microbatches=4, stages=("layers.1",))
    units = [part.units for part in explicit.stages]
    assert units == [("layers.0",), ("layers.1", "layers.2", "head")]


SPLIT_LAYERS = """\
module split_layers

use std.nn.activations::{silu}
use std.nn.linear::{Linear}
use std.nn.parallel::{all_reduce}

pub block Layer<H: Dim, Inner: Dim, T: Float, Shards: Dim>
where
    Shards > 0,
    Inner % Shards == 0
{
    sub up: Linear<H, Inner / Shards, T>
    sub down: Linear<Inner / Shards, H, T>

    pub fn forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        return x + all_reduce(down.forward(silu(up.forward(x))))
    }
}

pub block Model<H: Dim, Inner: Dim, Layers: Dim, T: Float = f32, Shards: Dim = 1>
where
    Shards > 0,
    Inner % Shards == 0
{
    sub layers: [Layer<H, Inner, T, Shards>; Layers]

    pub entry forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, H; T] {
        var h = x
        static for layer in layers {
            h = layer.forward(h)
        }
        return h
    }
}
"""


def test_a_pipeline_of_split_stages_holds_a_part_of_each_stage(tmp_path: Path) -> None:
    model = memory(
        source(tmp_path, SPLIT_LAYERS),
        ExecutionConfig(
            batch=8,
            bindings={"H": 16, "Inner": 32, "Layers": 2},
            tensor_parallel=2,
            pipeline_parallel=2,
            microbatches=2,
        ),
    )
    assert [part.units for part in model.stages] == [("layers.0",), ("layers.1",)]
    result = model.analyze()
    # One layer a stage, half of each split weight: up's rows and down's
    # columns.
    assert (result.component("Weights").nbytes or 0) <= (16 * 16 + 16 * 16 + 16 + 16) * 4
    # A communicator for the stage's processes, and one for the pipeline's.
    single = memory(
        source(tmp_path, SPLIT_LAYERS),
        ExecutionConfig(batch=8, bindings={"H": 16, "Inner": 32, "Layers": 2}, tensor_parallel=2),
    ).analyze()
    both = result.component("NCCL communicator").nbytes or 0
    assert both == 2 * (single.component("NCCL communicator").nbytes or 0)


def test_sharded_weights_are_kept_as_the_runtime_keeps_them(tmp_path: Path) -> None:
    bindings: dict[str, int | str] = {"H": 32, "Layers": 3, "T": "bf16"}
    whole = memory(
        source(tmp_path, MLP), config(batch=8, bindings=bindings, training=TrainingConfig())
    ).analyze()
    sharded = memory(
        source(tmp_path, MLP),
        config(batch=8, bindings=bindings, training=TrainingConfig(shards=2)),
    ).analyze()
    # Each process's part of a trained weight is kept in f32: half the
    # elements at twice the bytes.
    assert sharded.component("Weights").nbytes == whole.component("Weights").nbytes
    # Stages shard too.
    staged = _pipeline(tmp_path, microbatches=4, training=TrainingConfig(shards=2)).analyze()
    assert [d.name for d in staged.devices] == ["stage 0", "stage 1"]


def test_gpipe_keeps_more_micro_batches_than_1f1b(tmp_path: Path) -> None:
    def peak(schedule: str) -> int:
        model = _pipeline(tmp_path, microbatches=4, schedule=schedule, training=TrainingConfig())
        stage0 = model.analyze().devices[0]
        return stage0.expected_peak

    assert peak("gpipe") > peak("1f1b")


def test_each_call_runs_at_its_rate(tmp_path: Path) -> None:
    from linnet.resources.performance import DeviceSpec

    device = DeviceSpec("toy", 1 << 30, {"f32": 1e9}, 1e9, 1e9, 0.0)
    model = memory(source(tmp_path, MLP), config(batch=8, bindings={"H": 32, "Layers": 3}))
    flops = sum(ex.evaluate(step.flops, model.env()) for step in model.graph.steps)
    predicted = model.throughput(device)
    # No host time: the kernels run back to back, each at its arithmetic or
    # its bytes, whichever is longer.
    assert predicted.host == 0
    assert predicted.seconds == pytest.approx(predicted.compute + predicted.memory)
    assert predicted.seconds >= flops / 1e9
    assert predicted.tokens == 8  # samples: the input is not token ids
    # Two replicas take twice the tokens in the same time.
    assert model.throughput(device, replicas=2).tokens_per_second == pytest.approx(
        2 * predicted.tokens_per_second
    )


def test_a_slow_host_sets_the_step(tmp_path: Path) -> None:
    from dataclasses import replace

    from linnet.resources.performance import DeviceSpec

    device = DeviceSpec("toy", 1 << 30, {"f32": 1e12}, 1e12, 1e12, 0.0, dispatch=1.0)
    model = memory(source(tmp_path, MLP), config(batch=8, bindings={"H": 32, "Layers": 3}))
    predicted = model.throughput(device)
    calls = len(model.graph.steps)
    assert predicted.limit == "host"
    assert calls <= predicted.seconds < calls + 1
    # Compiled, the host issues nothing per call.
    compiled = memory(
        source(tmp_path, MLP), config(batch=8, bindings={"H": 32, "Layers": 3}, compiled=True)
    )
    assert compiled.throughput(replace(device)).seconds < 1


def test_point_to_point_channels_follow_what_a_process_does(tmp_path: Path) -> None:
    from linnet.resources.backends import Channels, CudaTorchBackend

    backend = CudaTorchBackend(processes=2)

    def channels(training: bool, which: Channels) -> int | None:
        items = {i.name: i.estimate.nbytes for i in backend.runtime(training, which)}
        return items.get("NCCL send and receive")

    assert channels(False, "none") is None
    # A pipeline's first stage only sends, its last only receives.
    sends, receives = channels(False, "send") or 0, channels(False, "receive") or 0
    assert 0 < sends < receives < (channels(True, "both") or 0)
    # DTensor's redistributions open them when it splits a weight.
    split = memory(
        source(tmp_path, SPLIT), ExecutionConfig(batch=2, bindings={"H": 16}, tensor_parallel=2)
    ).analyze()
    assert (split.component("NCCL send and receive").nbytes or 0) > 0
    sharded = memory(
        source(tmp_path, SHARDED),
        ExecutionConfig(batch=4, bindings={"H": 16, "Out": 32}, tensor_parallel=2),
    ).analyze()
    assert all(c.name != "NCCL send and receive" for c in sharded.components)


def test_a_calibrated_profile_loads(tmp_path: Path) -> None:
    import json

    from linnet.resources.performance import device

    profile = {
        "name": "mine",
        "source": "measured",
        "memory": 1 << 30,
        "flops": {"bf16": 100.0, "f16": 100.0, "f32": 10.0},
        "products": {
            "bf16": [[128, 128, 10.0], [128, 512, 20.0], [512, 128, 30.0], [512, 512, 50.0]]
        },
        "host": {"view": 2e-7},
        "bandwidth": 5.0,
        "attention": {"causal/64": 30.0, "causal/128": 60.0, "causal/64/f32": 3.0},
        "dispatch": 1e-6,
        "kernel": 2e-6,
        "link": 0.0,
        "latency": 0.0,
        "reduce": 0.0,
    }
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    loaded = device(str(path))
    assert loaded.name == "mine" and loaded.dispatch == 1e-6
    # Between the measured shapes on a log scale; below them in proportion;
    # past them at the nearest edge.
    assert loaded.product("bf16", 256, 256) == pytest.approx(27.5)
    assert loaded.product("bf16", 64, 512) == pytest.approx(10.0)
    assert loaded.product("bf16", 1 << 20, 1 << 20) == 50.0
    assert loaded.product("f32", 256, 256) == 10.0
    # A kind measured costs its own; any other the mix's average.
    assert loaded.host_of("view", "product") == pytest.approx(2e-7 + 1e-6)
    assert loaded.attention_rate("causal", 80) == 30.0
    assert loaded.attention_rate("causal", 120) == 60.0
    assert loaded.attention_rate("causal", 64, "f32") == 3.0
    assert loaded.link > 0  # one process measures no link: the H100's
    with pytest.raises(ValueError, match="unknown device"):
        device("no-such-device")


ATTEND = """\
module attend

use std.nn.attention::{attention, causal_mask}
use std.nn.linear::{Linear}
use std.nn.norm::{RmsNorm}

pub block Model<H: Dim, Heads: Dim, T: Float = bf16>
where
    Heads > 0,
    H % Heads == 0
{
    sub norm: RmsNorm<H, T>
    sub q_proj: Linear<H, H, T>
    sub k_proj: Linear<H, H, T>
    sub v_proj: Linear<H, H, T>
    sub o_proj: Linear<H, H, T>

    pub entry forward<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, S, H; T] {
        let h = norm.forward(x)
        let q = permute(reshape(q_proj.forward(h), [B, S, Heads, H / Heads]), [0, 2, 1, 3])
        let k = permute(reshape(k_proj.forward(h), [B, S, Heads, H / Heads]), [0, 2, 1, 3])
        let v = permute(reshape(v_proj.forward(h), [B, S, Heads, H / Heads]), [0, 2, 1, 3])
        let mixed = attention(q, k, v, 0.125, some(causal_mask<S, S>()))
        return o_proj.forward(reshape(permute(mixed, [0, 2, 1, 3]), [B, S, H]))
    }
}
"""


def _attend(tmp_path: Path, **kwargs: object) -> MemoryModel:
    return memory(
        source(tmp_path, ATTEND),
        ExecutionConfig(batch=2, context=16, bindings={"H": 32, "Heads": 4}, **kwargs),  # type: ignore[arg-type]
    )


def test_fused_attention_results_lie_as_the_kernel_writes_them(tmp_path: Path) -> None:
    model = _attend(tmp_path)
    graph = model.graph
    attention = next(
        s for s in graph.steps if (s.implementation or "").startswith("torch.nn.functional.scaled")
    )
    readers = {i: s for s in graph.steps for i in s.inputs}
    permuted = readers[attention.outputs[0]]
    merged = readers[permuted.outputs[0]]
    # [B, S, Heads, D] in memory: its heads merge without a copy.
    assert merged.kind == "reshape"
    assert not graph.objects[merged.outputs[0]].owns_storage


def test_the_step_costs_follow_the_generated_calls(tmp_path: Path) -> None:
    from linnet.resources import performance

    model = _attend(tmp_path)
    graph = model.graph
    # The three projections of one input run as one product.
    fused = performance._fused(graph)  # pyright: ignore[reportPrivateUsage]
    assert [len(members) for members in fused.values()] == [3]
    # The causal mask is made once per shape, not per call.
    hoisted = performance._hoisted(graph)  # pyright: ignore[reportPrivateUsage]
    assert any(graph.steps[i].label == "causal_mask" for i in hoisted)
    assert all(graph.steps[i].implementation != "torch.nn.functional.linear" for i in hoisted)


def test_normalizations_and_collectives_keep_what_autograd_keeps(tmp_path: Path) -> None:
    model = _attend(tmp_path, training=TrainingConfig())
    graph, env = model.graph, model.env()
    norm = next(s for s in graph.steps if (s.implementation or "").startswith("torch.rms_norm"))
    every = {o.id: True for o in graph.objects}
    saved = model.backend.saved(norm, graph, env, every)
    assert saved is not None
    rows, width = 2 * 16, 32
    # f32 statistics, and the normalized rows the weight's gradient reads.
    assert saved.extra.nbytes == rows * 4 + rows * width * 2
    frozen = model.backend.saved(norm, graph, env, {**every, norm.inputs[1]: False})
    assert frozen is not None and frozen.extra.nbytes == rows * 4
    sharded = memory(
        source(tmp_path, SHARDED),
        ExecutionConfig(batch=4, bindings={"H": 16, "Out": 32}, tensor_parallel=2),
    )
    reduce = next(
        s for s in sharded.graph.steps if s.implementation == "torch.distributed.all_reduce"
    )
    kept = sharded.backend.saved(reduce, sharded.graph, sharded.env(), every)
    assert kept is not None and kept.objects == ()


def test_the_planner_ranks_layouts_by_throughput(tmp_path: Path) -> None:
    from linnet.resources.performance import DEVICES

    planner = ExecutionPlanner(source(tmp_path, MLP), std_root=STDLIB)
    base = config(bindings={"H": 32, "Layers": 3})
    plan = planner.plan(base, ResourceConstraint(1 << 30), DEVICES["h100-80gb"], 2)
    ranked = [c.throughput.tokens_per_second for c in plan.candidates if c.throughput]
    assert plan.best is not None and ranked == sorted(ranked, reverse=True)
    assert plan.target == "batch" and plan.best.size is not None
    layouts = {(c.config.tensor_parallel, c.config.pipeline_parallel) for c in plan.candidates}
    assert {(1, 1), (1, 2), (2, 1)} <= layouts


def test_unsupported_parallelism_is_refused(tmp_path: Path) -> None:
    from linnet.resources.trace import TraceError

    with pytest.raises(TraceError, match="Shards"):
        memory(
            source(tmp_path, CHAIN),
            config(batch=1, bindings={"N": 4}, pipeline_parallel=2, tensor_parallel=2),
        )
    with pytest.raises(TraceError, match="without gradients"):
        memory(
            source(tmp_path, SPLIT_LAYERS),
            ExecutionConfig(
                batch=8,
                bindings={"H": 16, "Inner": 32, "Layers": 2},
                tensor_parallel=2,
                pipeline_parallel=2,
                microbatches=2,
                training=TrainingConfig(),
            ),
        )
    with pytest.raises(TraceError, match="divide"):
        _pipeline(tmp_path, microbatches=5).analyze()
    with pytest.raises(TraceError, match="DTensor"):
        memory(
            source(tmp_path, MLP),
            config(
                batch=4,
                bindings={"H": 32, "Layers": 3},
                tensor_parallel=2,
                training=TrainingConfig(),
            ),
        )

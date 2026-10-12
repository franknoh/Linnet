"""The graph `linnet graph` records, the planner over it, and a plan run
through the generated source (docs/design/execution.md)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pytest
import torch

from linnet.backends import (
    Benchmarks,
    Capabilities,
    Context,
    Cost,
    ExecutableMeta,
    Graph,
    Measured,
    Options,
    Planner,
    Region,
    Target,
    runs_of,
    splittable,
)
from linnet.compiler import find_compiler
from linnet.torch import load
from linnet.torch.compiled import CompiledLinnetModule
from tests.helpers import LLAMA, REPO, STDLIB, TINY_LLAMA, random_llama


def _graph(*args: str) -> Graph:
    completed = subprocess.run(
        [find_compiler(), "graph", "--std", str(STDLIB), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return Graph.parse(completed.stdout)


def _decode() -> Graph:
    binds = [arg for key, value in TINY_LLAMA.items() for arg in ("--bind", f"{key}={value}")]
    return _graph("--entry", "decode", "--numerics", "fast", *binds, str(LLAMA))


def test_the_graph_keeps_each_call_around_the_steps_it_made() -> None:
    graph = _decode()
    assert len(graph.inputs) == 2
    assert len(graph.outputs) == 1
    assert "layers.1.mlp.down.weight" in graph.parameters
    linear = list(graph.semantic_calls("std.nn.linear::linear"))
    # Seven projections a layer and the head.
    assert len(linear) == 2 * 7 + 1
    # The target took each as a library call, the call's one step.
    assert all([s.op for s in graph.steps[c.start : c.stop]] == ["native_call"] for c in linear)
    paths = {call.path for call in graph.calls}
    assert {"layers.0.attention", "layers.1.mlp"} <= paths
    for step in graph.steps:
        for outer, inner in zip(step.calls, step.calls[1:], strict=False):
            assert graph.calls[inner].parent == outer
            assert graph.calls[outer].start <= step.index < graph.calls[outer].stop


def test_a_region_reads_what_came_before_and_hands_on_what_comes_after() -> None:
    graph = _decode()
    mlp = next(call for call in graph.calls if call.path == "layers.0.mlp")
    region = Region.of(graph, mlp.start, mlp.stop, mlp.index)
    assert set(region.parameters) == {
        "layers.0.mlp.gate.weight",
        "layers.0.mlp.up.weight",
        "layers.0.mlp.down.weight",
    }
    assert region.inputs == tuple(a.value for a in mlp.arguments if a.value is not None)
    assert region.outputs == tuple(r.value for r in mlp.results)


class _Executable:
    def __init__(self, region: Region) -> None:
        self.region = region

    @property
    def meta(self) -> ExecutableMeta:
        return ExecutableMeta((), ())

    def prepare(self, context: Context[object]) -> None:
        return None

    def execute(
        self, inputs: Sequence[object], outputs: Sequence[object], context: Context[object]
    ) -> Sequence[object]:
        return [object() for _ in self.region.outputs]


class _Backend:
    """Takes every call of `callee` (or, without one, any region)."""

    def __init__(self, name: str, callee: str | None = None, seconds: float | None = None) -> None:
        self._name = name
        self._callee = callee
        self._seconds = seconds

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "0"

    def probe(self, target: Target) -> Capabilities | None:
        return Capabilities()

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        if self._callee is None:
            return []
        return [
            Region.of(graph, call.start, call.stop, call.index)
            for call in graph.semantic_calls(self._callee)
        ]

    def compile(self, region: Region, target: Target, options: Options) -> _Executable:
        return _Executable(region)

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None if self._seconds is None else Cost(self._seconds)


def test_the_planner_gives_matched_calls_to_their_backend_and_the_rest_to_the_fallback() -> None:
    graph = _decode()
    plan = Planner([_Backend("linear", "std.nn.linear::linear")], _Backend("fallback")).plan(
        graph, Target("cpu")
    )
    backends = [step.backend for step in plan.steps]
    assert backends.count("linear") == 2 * 7 + 1
    assert "fallback" in backends
    # The regions follow each other, and what lies between them computes nothing.
    covered = set[int]()
    for before, after in zip(plan.steps, plan.steps[1:], strict=False):
        assert before.region.stop <= after.region.start
    for step in plan.steps:
        covered.update(range(step.region.start, step.region.stop))
    assert all(
        step.index in covered
        for step in graph.steps
        if step.results and step.op not in ("input", "parameter", "state")
    )
    # Each region reads only what the entry or an earlier region made.
    made = {*graph.inputs, *graph.parameters.values(), *graph.states.values()}
    for step in plan.steps:
        assert set(step.region.inputs) <= made
        made |= set(step.region.outputs)
    assert set(graph.outputs) <= made
    assert plan.run([object(), object()], Context(plan.target))


def test_the_cheaper_of_two_backends_takes_a_call() -> None:
    graph = _decode()
    slow = _Backend("slow", "std.nn.linear::linear", seconds=2.0)
    fast = _Backend("fast", "std.nn.linear::linear", seconds=1.0)
    plan = Planner([slow, fast], _Backend("fallback")).plan(graph, Target("cpu"))
    assert "slow" not in {step.backend for step in plan.steps}
    assert "fast" in {step.backend for step in plan.steps}


def test_no_region_cuts_through_a_loop() -> None:
    graph = _graph(
        "--entry", "run", "--bind", "N=4", str(REPO / "spec-tests/valid/023_while.linnet")
    )
    begin = next(graph.find("begin_while")).index
    end = next(graph.find("end_while")).index
    assert splittable(graph, 0, len(graph.steps))
    assert splittable(graph, begin, end + 1)
    assert not splittable(graph, begin + 1, len(graph.steps))
    assert not splittable(graph, 0, end)


def test_a_plan_through_the_generated_source_gives_the_entry_results(tmp_path: Path) -> None:
    weights = random_llama(tmp_path / "model.safetensors")
    planned, direct = (
        load(LLAMA, generics=TINY_LLAMA, std_root=STDLIB, weights=weights, compile=True)
        for _ in range(2)
    )
    assert isinstance(planned, CompiledLinnetModule)
    tokens = [torch.tensor([[3]], dtype=torch.int32), torch.tensor([[5]], dtype=torch.int32)]
    plan = planned.plan_for("decode", [tokens[0], torch.tensor(0, dtype=torch.int32)])
    assert [step.backend for step in plan.steps] == ["torch-source"]
    for pos, token in enumerate(tokens):
        step = torch.tensor(pos, dtype=torch.int32)
        (logits,) = plan.run([token, step], Context(plan.target))
        expected: object = direct.run_entry("decode", [token, step])
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(logits, expected)


class _Linear:
    """A backend that takes every `linear` call and runs it with PyTorch,
    reading the weight from the context by path."""

    @property
    def name(self) -> str:
        return "linear"

    @property
    def version(self) -> str:
        return "0"

    def probe(self, target: Target) -> Capabilities | None:
        return Capabilities()

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        return [
            Region.of(graph, call.start, call.stop, call.index)
            for call in graph.semantic_calls("std.nn.linear::linear")
        ]

    def compile(self, region: Region, target: Target, options: Options) -> _LinearCall:
        return _LinearCall(region)

    def estimate_cost(self, region: Region, target: Target) -> Cost | None:
        return None


class _LinearCall:
    def __init__(self, region: Region) -> None:
        (self._weight,) = region.parameters

    @property
    def meta(self) -> ExecutableMeta:
        return ExecutableMeta((), ())

    def prepare(self, context: Context[torch.Tensor]) -> None:
        return None

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        return [torch.nn.functional.linear(inputs[0], context.parameters[self._weight])]


def test_the_generated_source_runs_what_another_backend_leaves(tmp_path: Path) -> None:
    weights = random_llama(tmp_path / "model.safetensors")
    planned, direct = (
        load(LLAMA, generics=TINY_LLAMA, std_root=STDLIB, weights=weights, compile=True)
        for _ in range(2)
    )
    assert isinstance(planned, CompiledLinnetModule)
    tokens = [torch.tensor([[3]], dtype=torch.int32), torch.tensor([[5]], dtype=torch.int32)]
    plan = planned.plan_for(
        "decode",
        [tokens[0], torch.tensor(0, dtype=torch.int32)],
        backends=[_Linear()],
        measure=False,
    )
    backends = [step.backend for step in plan.steps]
    assert backends.count("linear") == 2 * 7 + 1
    assert "torch-source" in backends
    parameters = {
        name.removeprefix("root."): parameter.detach()
        for name, parameter in planned.named_parameters()
    }
    with torch.no_grad():
        for pos, token in enumerate(tokens):
            step = torch.tensor(pos, dtype=torch.int32)
            (logits,) = plan.run([token, step], Context(plan.target, parameters))
            expected: object = direct.run_entry("decode", [token, step])
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(logits, expected)


def test_the_region_with_the_entrys_finish_hands_on_its_results() -> None:
    graph = _decode()
    head = list(graph.semantic_calls("std.nn.linear::linear"))[-1]
    region = Region.of(graph, head.start, len(graph.steps))
    assert set(graph.outputs) <= set(region.outputs)


class _Joined(_Linear):
    """Takes the `linear` calls that read the same input together, as one
    region of separate steps, and runs each with PyTorch."""

    @property
    def name(self) -> str:
        return "joined"

    def match(self, graph: Graph, target: Target) -> Iterable[Region]:
        siblings: dict[int | None, list[int]] = {}
        for call in graph.semantic_calls("std.nn.linear::linear"):
            siblings.setdefault(call.arguments[0].value, []).append(call.start)
        return [Region.of_steps(graph, steps) for steps in siblings.values() if len(steps) > 1]

    def compile(self, region: Region, target: Target, options: Options) -> _LinearCall:
        return _JoinedCall(region)


class _JoinedCall(_LinearCall):
    def __init__(self, region: Region) -> None:
        self._weights = list(region.parameters)

    def execute(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        context: Context[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        return [
            torch.nn.functional.linear(inputs[0], context.parameters[path])
            for path in self._weights
        ]


def test_sibling_calls_run_together_where_the_first_stands(tmp_path: Path) -> None:
    weights = random_llama(tmp_path / "model.safetensors")
    planned, direct = (
        load(LLAMA, generics=TINY_LLAMA, std_root=STDLIB, weights=weights, compile=True)
        for _ in range(2)
    )
    assert isinstance(planned, CompiledLinnetModule)
    token = torch.tensor([[3]], dtype=torch.int32)
    plan = planned.plan_for(
        "decode", [token, torch.tensor(0, dtype=torch.int32)], backends=[_Joined()], measure=False
    )
    joined = [step.region for step in plan.steps if step.backend == "joined"]
    # q, k and v; gate and up: in each layer.
    assert sorted(len(region.members) for region in joined) == [2, 2, 3, 3]
    parameters = {
        name.removeprefix("root."): parameter.detach()
        for name, parameter in planned.named_parameters()
    }
    with torch.no_grad():
        for pos in range(2):
            step = torch.tensor(pos, dtype=torch.int32)
            (logits,) = plan.run([token, step], Context(plan.target, parameters))
            expected: object = direct.run_entry("decode", [token, step])
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(logits, expected)


LINEAR = "std.nn.linear::linear"


class _Clock:
    """What the fake executables take: each says how long it ran, and the
    timer returns that."""

    def __init__(self) -> None:
        self.last = 0.0
        self.calls = 0

    def time(self, run: Callable[[], object], meta: ExecutableMeta) -> float:
        self.calls += 1
        run()
        return self.last


class _TimedCall(_Executable):
    def __init__(self, region: Region, clock: _Clock, seconds: float, holds: int) -> None:
        super().__init__(region)
        self._clock = clock
        self._seconds = seconds
        self._holds = holds

    @property
    def meta(self) -> ExecutableMeta:
        return ExecutableMeta((), (), holds=self._holds)

    def execute(
        self, inputs: Sequence[object], outputs: Sequence[object], context: Context[object]
    ) -> Sequence[object]:
        self._clock.last = self._seconds
        return super().execute(inputs, outputs, context)


class _Timed(_Backend):
    """Takes every call of `callee` and runs each in `seconds`, keeping
    `holds` bytes; without a callee, runs any region in `seconds` a step
    and `overhead` more (what a region it is cut into costs it)."""

    def __init__(
        self,
        name: str,
        clock: _Clock,
        callee: str | None = None,
        seconds: float = 1.0,
        holds: int = 0,
        overhead: float = 0.0,
    ) -> None:
        super().__init__(name, callee)
        self._clock = clock
        self._each = seconds
        self._holds = holds
        self._overhead = overhead

    def compile(self, region: Region, target: Target, options: Options) -> _TimedCall:
        seconds = self._each if self._callee else self._each * len(region.indices)
        if not self._callee:
            seconds += self._overhead
        return _TimedCall(region, self._clock, seconds, self._holds)


def _measured(clock: _Clock, benchmarks: Benchmarks | None = None) -> Measured[object]:
    reference = _Timed("fallback", clock)
    return Measured([object(), object()], Context(Target("cpu")), reference, clock.time, benchmarks)


def test_a_backend_takes_only_the_regions_it_runs_faster() -> None:
    graph = _decode()
    clock = _Clock()
    fallback = _Timed("fallback", clock)
    fast = _Timed("fast", clock, LINEAR, seconds=0.5)
    slow = _Timed("slow", clock, LINEAR, seconds=2.0)
    close = _Timed("close", clock, LINEAR, seconds=0.99)  # saves less than the margin
    plan = Planner([slow, close, fast], fallback, cost_model=_measured(clock)).plan(
        graph, Target("cpu")
    )
    taken = [step for step in plan.steps if step.backend != "fallback"]
    assert [step.backend for step in taken] == ["fast"] * (2 * 7 + 1)
    assert all(step.saving == pytest.approx(0.5) for step in taken)
    plan = Planner([slow, close], fallback, cost_model=_measured(clock)).plan(graph, Target("cpu"))
    assert {step.backend for step in plan.steps} == {"fallback"}


def test_each_kind_of_region_is_timed_once_and_kept(tmp_path: Path) -> None:
    graph = _decode()
    clock = _Clock()
    fallback = _Timed("fallback", clock)
    fast = _Timed("fast", clock, LINEAR, seconds=0.5)
    kept = tmp_path / "benchmarks.json"
    Planner([fast], fallback, cost_model=_measured(clock, Benchmarks(kept))).plan(
        graph, Target("cpu")
    )
    # Each key timed once and kept: the five kinds of linear call (q and o,
    # k and v, gate and up, down, the head) on `fast` and on the fallback,
    # and the stretches around them that `confirm` prices.
    benchmarks = Benchmarks(kept)
    assert clock.calls == len(benchmarks) > 2 * 5
    clock.calls = 0
    plan = Planner([fast], fallback, cost_model=_measured(clock, benchmarks)).plan(
        graph, Target("cpu")
    )
    assert clock.calls == 0
    assert [step.backend for step in plan.steps].count("fast") == 2 * 7 + 1


def test_regions_keep_no_more_bytes_than_the_budget() -> None:
    graph = _decode()
    clock = _Clock()
    big = _Timed("big", clock, LINEAR, seconds=0.1, holds=10)
    fast = _Timed("fast", clock, LINEAR, seconds=0.5)
    plan = Planner([big, fast], _Timed("fallback", clock), cost_model=_measured(clock)).plan(
        graph, Target("cpu"), Options(memory=25)
    )
    used = [step.backend for step in plan.steps]
    assert used.count("big") == 2
    assert used.count("fast") == 2 * 7 + 1 - 2


def test_the_fallback_runs_steps_apart_as_ranges_of_their_own() -> None:
    graph = _decode()
    q, k, v = list(graph.semantic_calls(LINEAR))[:3]
    joined = Region.of_steps(graph, [q.start, k.start, v.start])
    runs = runs_of(joined)
    assert [(run.start, run.stop) for run in runs] == [(c.start, c.stop) for c in (q, k, v)]
    # The layers' regions of one kind are one kind.
    calls = list(graph.semantic_calls(LINEAR))
    other = Region.of_steps(graph, [calls[7].start, calls[8].start, calls[9].start])
    assert other.kind == joined.kind
    assert Region.of(graph, q.start, q.stop).kind != joined.kind


def test_a_measured_plan_gives_the_entry_results(tmp_path: Path) -> None:
    weights = random_llama(tmp_path / "model.safetensors")
    planned, direct = (
        load(LLAMA, generics=TINY_LLAMA, std_root=STDLIB, weights=weights, compile=True)
        for _ in range(2)
    )
    assert isinstance(planned, CompiledLinnetModule)
    token = torch.tensor([[3]], dtype=torch.int32)
    benchmarks = Benchmarks(tmp_path / "benchmarks.json")
    plan = planned.plan_for(
        "decode",
        [token, torch.tensor(0, dtype=torch.int32)],
        backends=[_Linear()],
        benchmarks=benchmarks,
    )
    # Each kind timed on both; whichever was faster runs it.
    assert len(Benchmarks(tmp_path / "benchmarks.json")) > 2 * 5
    with torch.no_grad():
        for pos in range(2):
            step = torch.tensor(pos, dtype=torch.int32)
            (logits,) = plan.run([token, step], planned.context_for(plan))
            expected: object = direct.run_entry("decode", [token, step])
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(logits, expected)


def test_a_region_makes_again_what_it_computes_from_constants() -> None:
    graph = _decode()
    attention = next(graph.semantic_calls("std.nn.attention::grouped_attention"))
    region = Region.of(graph, attention.start, attention.stop, attention.index)
    query, key, value, scale, mask = attention.arguments
    # The scale, computed from the head width, is no input: the region
    # makes it again.
    assert scale.value in graph.constants
    assert region.inputs == (query.value, key.value, value.value, mask.value)


def test_the_context_joins_parameters_without_a_copy(tmp_path: Path) -> None:
    weights = random_llama(tmp_path / "model.safetensors")
    planned, direct = (
        load(LLAMA, generics=TINY_LLAMA, std_root=STDLIB, weights=weights, compile=True)
        for _ in range(2)
    )
    assert isinstance(planned, CompiledLinnetModule)
    token = torch.tensor([[3]], dtype=torch.int32)
    plan = planned.plan_for("decode", [token, torch.tensor(0, dtype=torch.int32)], measure=False)
    context = planned.context_for(plan)
    assert context.join is not None
    paths = [f"layers.0.attention.{name}_proj.weight" for name in ("q", "k", "v")]
    apart = torch.cat([context.parameters[path].clone() for path in paths])
    joined = context.join(paths)
    assert joined is not None
    torch.testing.assert_close(joined, apart)
    # The model's parameters, and the context's, are now views of it.
    storage = joined.untyped_storage().data_ptr()
    assert all(context.parameters[p].untyped_storage().data_ptr() == storage for p in paths)
    named = dict(planned.named_parameters())
    assert all(named[f"root.{p}"].untyped_storage().data_ptr() == storage for p in paths)
    with torch.no_grad():
        step = torch.tensor(0, dtype=torch.int32)
        (logits,) = plan.run([token, step], planned.context_for(plan))
        expected: object = direct.run_entry("decode", [token, step])
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(logits, expected)


def test_a_region_that_pays_alone_but_not_in_place_is_dropped() -> None:
    graph = _decode()
    clock = _Clock()
    fast = _Timed("fast", clock, LINEAR, seconds=0.5)
    # Every region the fallback runs costs it 5 more: each linear call cut
    # out of its code adds a region, which costs more than the call saves.
    fallback = _Timed("fallback", clock, overhead=5.0)
    plan = Planner([fast], fallback, cost_model=_measured(clock)).plan(graph, Target("cpu"))
    assert {step.backend for step in plan.steps} == {"fallback"}
    # Without that cost, each call saves its half second in place too.
    fallback = _Timed("fallback", clock)
    plan = Planner([fast], fallback, cost_model=_measured(clock)).plan(graph, Target("cpu"))
    taken = [step for step in plan.steps if step.backend == "fast"]
    assert len(taken) == 2 * 7 + 1
    assert all(step.saving == pytest.approx(0.5) for step in taken)

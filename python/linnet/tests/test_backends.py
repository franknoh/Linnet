"""The graph `linnet graph` records, the planner over it, and a plan run
through the generated source (docs/design/execution.md)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from linnet.backends import (
    Capabilities,
    Context,
    Cost,
    ExecutableMeta,
    Graph,
    Options,
    Planner,
    Region,
    Target,
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
        "decode", [tokens[0], torch.tensor(0, dtype=torch.int32)], backends=[_Linear()]
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
        "decode", [token, torch.tensor(0, dtype=torch.int32)], backends=[_Joined()]
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

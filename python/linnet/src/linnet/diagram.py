"""Architecture diagrams from a typed program.

`build` turns one function (an entry by default) into a graph: inputs, the
calls and operations of its body, loops as nested groups, and outputs, with
every edge labelled by the exact tensor type the compiler inferred. `to_svg`
and `to_tikz` draw that graph with a small layered layout; `to_dot` hands it
to Graphviz instead.

    python -m linnet.diagram src/lib.linnet --entry forward -o forward.svg
"""

from __future__ import annotations

import argparse
import html
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from . import ir
from .compiler import LinnetError

NodeKind = Literal["input", "output", "param", "call", "op", "state"]


@dataclass(slots=True)
class Node:
    id: str
    kind: NodeKind
    label: str
    sublabel: str = ""
    group: str | None = None


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    label: str = ""


@dataclass(slots=True)
class Group:
    """A box around the nodes of a loop body or an expanded call."""

    id: str
    label: str
    parent: str | None = None


@dataclass(slots=True)
class Graph:
    title: str
    nodes: list[Node] = field(default_factory=list[Node])
    edges: list[Edge] = field(default_factory=list[Edge])
    groups: list[Group] = field(default_factory=list[Group])

    def node(self, id: str) -> Node:
        for node in self.nodes:
            if node.id == id:
                return node
        raise KeyError(id)


# ------------------------------------------------------------------ building

_ARITHMETIC = {
    "add": "+",
    "sub": "\u2212",
    "mul": "\u00d7",
    "div": "\u00f7",
    "neg": "\u2212x",
    "max": "max",
    "min": "min",
    "and": "and",
    "or": "or",
    "not": "not",
    "bitand": "&",
    "bitor": "|",
    "bitxor": "^",
    "shl": "<<",
    "shr": ">>",
    "compare": "compare",
    "select": "select",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "sin": "sin",
    "cos": "cos",
    "tanh": "tanh",
}
_HIDDEN = {
    "block.sub",
    "block.param",
    "return",
    "yield",
    "const.bool",
    "const.int",
    "const.float",
    "const.dim",
}


class _Builder:
    def __init__(self, program: ir.Program, expand: int, params: bool, arithmetic: bool) -> None:
        self.program = program
        self.expand = expand
        self.params = params
        self.arithmetic = arithmetic
        self.graph = Graph(title="")
        self.counter = 0
        # value id -> node that produces it (None for folded constants)
        self.producer: dict[int, str | None] = {}
        self.types: dict[int, ir.Type] = {}
        # Substitutions of the expanded calls we are inside, outermost first.
        self.substitutions: list[ir.Substitution] = []
        self.constants: dict[int, str] = {}
        # value id -> member path such as `layers[i].attention`
        self.members: dict[int, str] = {}
        # member paths of the expanded sub-block calls we are inside
        self.prefixes: list[str] = []

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def add(self, kind: NodeKind, label: str, sublabel: str = "", group: str | None = None) -> str:
        node_id = self.fresh("n")
        self.graph.nodes.append(Node(node_id, kind, label, sublabel, group))
        return node_id

    def typed(self, type: ir.Type) -> ir.Type:
        """A callee type in the caller's terms, through every enclosing expansion."""
        for substitution in reversed(self.substitutions):
            type = ir.substitute(type, substitution)
        return type

    def connect(self, value: int, dst: str) -> None:
        src = self.producer.get(value)
        if src is None:
            return
        label = ""
        if value in self.types:
            label = _edge_label(self.types[value])
        self.graph.edges.append(Edge(src, dst, label))

    def function(self, function: ir.Function) -> Graph:
        self.graph.title = function.short_name
        for param in function.params:
            node_id = self.add("input", param.name, ir.format_type(param.type))
            self.producer[param.id] = node_id
            self.types[param.id] = param.type
        if function.body.args and function.body.args[0].name == "self":
            self.members[function.body.args[0].id] = ""
        outputs = self.region(function.body, None, self.expand)
        for index, value in enumerate(outputs):
            label = "result" if len(outputs) == 1 else f"result {index}"
            sublabel = ir.format_type(self.types[value]) if value in self.types else ""
            node_id = self.add("output", label, sublabel)
            self.connect(value, node_id)
        return self.graph

    def region(self, region: ir.Region, group: str | None, expand: int) -> list[int]:
        """Adds a region's operations; returns the ids of the values it returns or yields."""
        for op in region.ops:
            for result in op.results:
                self.types[result.id] = self.typed(result.type)
            if op.kind in ("return", "yield"):
                return list(op.operands)
            self.op(op, group, expand)
        return []

    def op(self, op: ir.Op, group: str | None, expand: int) -> None:
        kind = op.kind
        if kind == "block.sub":
            owner = self.members.get(op.operands[0], "")
            name = str(op.attrs.get("name", ""))
            self.members[op.results[0].id] = f"{owner}.{name}" if owner else name
            return
        if kind == "block.param" or kind == "state.read":
            owner = self.members.get(op.operands[0], "")
            name = str(op.attrs.get("name", ""))
            path = f"{owner}.{name}" if owner else name
            if kind == "state.read" or self.params:
                node_kind: NodeKind = "state" if kind == "state.read" else "param"
                self.producer[op.results[0].id] = self.add(
                    node_kind, path, ir.format_type(self.types[op.results[0].id]), group
                )
            else:
                self.producer[op.results[0].id] = None
            return
        if kind == "state.write":
            owner = self.members.get(op.operands[0], "")
            name = str(op.attrs.get("name", ""))
            node_id = self.add("state", f"{owner}.{name}" if owner else name, "write", group)
            for operand in op.operands[1:]:
                self.connect(operand, node_id)
            return
        if kind.startswith("const."):
            value = op.attrs.get("value")
            self.constants[op.results[0].id] = _format_constant(value)
            self.producer[op.results[0].id] = None
            return
        if kind == "tensor.element" and op.operands and op.operands[0] in self.members:
            # `layers[i]`: an element of a sub array keeps the member path.
            array, *rest = op.operands
            index = self.constants.get(rest[0], "i") if rest else "i"
            self.members[op.results[0].id] = f"{self.members[array]}[{index}]"
            return
        if kind == "option.some" and len(op.operands) == 1 and len(op.results) == 1:
            # Wrapping a value in an optional changes its type, not the flow.
            self.producer[op.results[0].id] = self.producer.get(op.operands[0])
            if op.operands[0] in self.constants:
                self.constants[op.results[0].id] = self.constants[op.operands[0]]
            return
        if kind in ("static_for", "static_range", "while"):
            self.loop(op, group, expand)
            return
        if kind in ("call", "semantic.call"):
            self.call(op, group, expand)
            return
        label, sublabel = self.op_label(op)
        node_id = self.add("op", label, sublabel, group)
        for operand in op.operands:
            self.connect(operand, node_id)
        for result in op.results:
            self.producer[result.id] = node_id

    def op_label(self, op: ir.Op) -> tuple[str, str]:
        kind = op.kind
        folded = [self.constants[v] for v in op.operands if v in self.constants]
        if kind in _ARITHMETIC:
            symbol = _ARITHMETIC[kind]
            if kind == "compare":
                symbol = str(op.attrs.get("predicate", op.attrs.get("op", "compare")))
            return (
                f"{symbol} {folded[0]}"
                if len(folded) == 1 and kind in ("add", "sub", "mul", "div")
                else symbol
            ), ""
        if kind == "cast":
            target = self.types.get(op.results[0].id) if op.results else None
            return "cast", ir.format_type(target) if target is not None else ""
        if kind == "reduce":
            return f"reduce {op.attrs.get('reduction', op.attrs.get('op', ''))}".rstrip(), ""
        if kind == "comprehension":
            return "index notation", ""
        if kind in ("reshape", "permute", "broadcast", "concat", "slice", "fill", "iota"):
            return kind, ""
        if kind.startswith("tuple.") or kind.startswith("option."):
            return kind.split(".", 1)[1], ""
        return kind, ""

    def call(self, op: ir.Op, group: str | None, expand: int) -> None:
        callee = str(op.attrs.get("callee", op.attrs.get("name", "call")))
        function = self.program.functions.get(callee)
        receiver = op.operands[0] if op.operands else None
        member = self.members.get(receiver) if receiver is not None else None
        if member is not None and function is not None:
            shown = member
            for prefix in reversed(self.prefixes):
                if shown.startswith(prefix + "."):
                    shown = shown[len(prefix) + 1 :]
                    break
            label = f"{shown}.{function.short_name}" if shown else function.short_name
            sublabel = _block_label(self.types.get(receiver if receiver is not None else -1))
            operands = op.operands[1:]
        else:
            label = _callee_label(callee)
            sublabel = callee.rsplit("::", 1)[0] if "::" in callee else ""
            operands = op.operands
        if function is not None and function.block is not None:
            # A helper on the same block is inlined silently; a sub-block's
            # method becomes a group when expansion depth remains.
            if member == "":
                self.expand_call(op, function, None, group, expand)
                return
            if expand > 0:
                self.prefixes.append(member or "")
                self.expand_call(op, function, label, group, expand - 1)
                self.prefixes.pop()
                return
        node_id = self.add("call", label, sublabel, group)
        for operand in operands:
            self.connect(operand, node_id)
        for result in op.results:
            self.producer[result.id] = node_id

    def expand_call(
        self, op: ir.Op, function: ir.Function, label: str | None, group: str | None, expand: int
    ) -> None:
        inner = group
        if label is not None:
            inner = self.fresh("g")
            self.graph.groups.append(Group(inner, label, group))
        self.substitutions.append(ir.call_substitution(op))
        for arg, operand in zip(function.body.args, op.operands, strict=False):
            if arg.name == "self" and isinstance(arg.type, ir.NamedType):
                self.members[arg.id] = self.members.get(operand, "")
                continue
            self.producer[arg.id] = self.producer.get(operand)
            self.types[arg.id] = self.types.get(operand, self.typed(arg.type))
            if operand in self.constants:
                self.constants[arg.id] = self.constants[operand]
        outputs = self.region(function.body, inner, expand)
        self.substitutions.pop()
        for result, value in zip(op.results, outputs, strict=False):
            self.producer[result.id] = self.producer.get(value)
            if value in self.constants:
                self.constants[result.id] = self.constants[value]

    def loop(self, op: ir.Op, group: str | None, expand: int) -> None:
        body = op.regions[-1]
        if op.kind == "while":
            label = "while"
            carried = list(op.operands)
            args = list(body.args)
        elif op.kind == "static_for":
            # Iterates a sub array: the element keeps the member path.
            array = self.members.get(op.operands[0], "")
            label = f"for {body.args[0].name} in {array or 'array'}"
            carried = list(op.operands[1:])
            args = list(body.args[1:])
            self.members[body.args[0].id] = (
                f"{array}[{body.args[0].name}]" if array else body.args[0].name
            )
        else:
            bounds = op.operands[: len(op.operands) - (len(body.args) - 1)]
            label = (
                "for "
                + body.args[0].name
                + " in "
                + " .. ".join(
                    self.constants.get(b, ir.format_type(self.types[b]) if b in self.types else "?")
                    for b in bounds
                )
            )
            carried = list(op.operands[len(bounds) :])
            args = list(body.args[1:])
        group_id = self.fresh("g")
        self.graph.groups.append(Group(group_id, label, group))
        if op.kind == "static_range":
            self.constants[body.args[0].id] = body.args[0].name
            self.producer[body.args[0].id] = None
        for arg, operand in zip(args, carried, strict=False):
            self.producer[arg.id] = self.producer.get(operand)
            self.types[arg.id] = self.typed(arg.type)
        if op.kind == "while" and len(op.regions) == 2:
            for arg, operand in zip(op.regions[0].args, carried, strict=False):
                self.producer[arg.id] = self.producer.get(operand)
                self.types[arg.id] = self.typed(arg.type)
        outputs = self.region(body, group_id, expand)
        for result, value in zip(op.results, outputs, strict=False):
            self.producer[result.id] = self.producer.get(value)


def _format_constant(value: object) -> str:
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text if "." in text or "e" in text else text + ".0"
    return str(value)


def _callee_label(callee: str) -> str:
    tail = callee.rsplit("::", 1)[-1]
    return tail.rsplit(".", 1)[-1]


def _block_label(type: ir.Type | None) -> str:
    return ir.format_type(type) if isinstance(type, ir.NamedType) else ""


def _edge_label(type: ir.Type) -> str:
    if isinstance(type, ir.TensorType):
        return f"{ir.format_dtype(type.dtype)}[{ir.format_shape(type.shape)}]"
    return ir.format_type(type)


def build(
    program: ir.Program,
    function: str | ir.Function | None = None,
    *,
    expand: int = 0,
    params: bool = False,
    arithmetic: bool = True,
) -> Graph:
    """The dataflow graph of one function.

    `function` is an entry name of the root block (the only entry by default)
    or any `Function` of the program. `expand` inlines calls to block methods
    that many levels deep as nested groups; `params` adds a node per parameter
    read; `arithmetic` keeps elementwise operations as nodes.
    """
    target = function if isinstance(function, ir.Function) else program.entry(function)
    graph = _Builder(program, expand, params, arithmetic).function(target)
    if not arithmetic:
        _fold_arithmetic(graph)
    return graph


def _fold_arithmetic(graph: Graph) -> None:
    """Removes elementwise nodes, connecting their inputs to their consumers."""
    removable = {
        n.id
        for n in graph.nodes
        if n.kind == "op" and n.label.split(" ")[0] in set(_ARITHMETIC.values())
    }
    for node_id in removable:
        incoming = [e for e in graph.edges if e.dst == node_id]
        outgoing = [e for e in graph.edges if e.src == node_id]
        graph.edges = [e for e in graph.edges if e.src != node_id and e.dst != node_id]
        for out in outgoing:
            for inc in incoming:
                if not any(e.src == inc.src and e.dst == out.dst for e in graph.edges):
                    graph.edges.append(Edge(inc.src, out.dst, out.label or inc.label))
    graph.nodes = [n for n in graph.nodes if n.id not in removable]


# -------------------------------------------------------------------- layout


@dataclass(slots=True)
class Placed:
    node: Node
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class Box:
    group: Group
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class Layout:
    placed: dict[str, Placed]
    boxes: list[Box]
    width: float
    height: float


_CHAR = 7.2  # approximate glyph advance at the 12px label size
_SMALL = 6.2
_NODE_HEIGHT = 30.0
_NODE_HEIGHT_TWO = 44.0
_LAYER_GAP = 64.0
_NODE_GAP = 28.0
_GROUP_PAD = 18.0
_MARGIN = 24.0


def layout(graph: Graph) -> Layout:
    """A top-to-bottom layered layout: longest-path layers, barycentre ordering."""
    ids = [n.id for n in graph.nodes]
    index = {node_id: i for i, node_id in enumerate(ids)}
    incoming: dict[str, list[str]] = {i: [] for i in ids}
    outgoing: dict[str, list[str]] = {i: [] for i in ids}
    for edge in graph.edges:
        if edge.src in index and edge.dst in index:
            outgoing[edge.src].append(edge.dst)
            incoming[edge.dst].append(edge.src)

    layer: dict[str, int] = {}
    order = _topological(ids, incoming, outgoing)
    for node_id in order:
        layer[node_id] = max((layer[p] + 1 for p in incoming[node_id]), default=0)
    # Outputs sit on the last layer. Nodes with no inputs other than the
    # function's own inputs (constants, parameter reads, `iota`) move down
    # to just above their first consumer, so they sit next to what uses them.
    last = max(layer.values(), default=0)
    for node in graph.nodes:
        if node.kind == "output":
            layer[node.id] = last if last > 0 else 0
    for node_id in reversed(order):
        node = graph.node(node_id)
        if node.kind != "input" and not incoming[node_id] and outgoing[node_id]:
            layer[node_id] = max(0, min(layer[c] for c in outgoing[node_id]) - 1)
    depth: dict[str, int] = {}
    parents = {g.id: g.parent for g in graph.groups}
    for node in graph.nodes:
        d = 0
        g = node.group
        while g is not None:
            d += 1
            g = parents.get(g)
        depth[node.id] = d

    layers: dict[int, list[str]] = {}
    for node_id in ids:
        layers.setdefault(layer[node_id], []).append(node_id)
    position: dict[str, float] = {i: float(k) for k, i in enumerate(ids)}
    for _ in range(4):
        for level in sorted(layers):
            row = layers[level]

            def key(node_id: str) -> tuple[str, float]:
                sources = incoming[node_id]
                bary = (
                    sum(position[s] for s in sources) / len(sources)
                    if sources
                    else position[node_id]
                )
                return (_group_key(graph, node_id), bary)

            row.sort(key=key)
            for k, node_id in enumerate(row):
                position[node_id] = float(k)

    placed: dict[str, Placed] = {}
    y = _MARGIN
    max_width = 0.0
    for level in sorted(layers):
        row = layers[level]
        sizes = [_size(graph.node(i)) for i in row]
        row_height = max(h for _, h in sizes)
        total = sum(w for w, _ in sizes) + _NODE_GAP * (len(row) - 1)
        total += _GROUP_PAD * 2 * max(depth[i] for i in row)
        max_width = max(max_width, total)
        placed_row: list[tuple[str, float, float]] = []
        x = 0.0
        previous_group: str | None = None
        for node_id, (w, h) in zip(row, sizes, strict=True):
            node = graph.node(node_id)
            if previous_group is not None and node.group != previous_group:
                x += _GROUP_PAD
            placed_row.append((node_id, x, w))
            x += w + _NODE_GAP
            previous_group = node.group
            placed[node_id] = Placed(node, 0.0, y, w, h)
        row_width = x - _NODE_GAP
        for node_id, nx, _ in placed_row:
            placed[node_id].x = nx - row_width / 2
        y += row_height + _LAYER_GAP
    total_width = max(max_width, 200.0) + 2 * _MARGIN
    for p in placed.values():
        p.x += total_width / 2
    boxes = _group_boxes(graph, placed)
    height = y - _LAYER_GAP + _MARGIN
    for box in boxes:
        height = max(height, box.y + box.height + _MARGIN)
        total_width = max(total_width, box.x + box.width + _MARGIN)
        if box.x < _MARGIN:
            shift = _MARGIN - box.x
            for p in placed.values():
                p.x += shift
            for b in boxes:
                b.x += shift
            total_width += shift
    return Layout(placed, boxes, total_width, height)


def _topological(
    ids: Sequence[str], incoming: Mapping[str, list[str]], outgoing: Mapping[str, list[str]]
) -> list[str]:
    remaining = {i: len(incoming[i]) for i in ids}
    ready = [i for i in ids if remaining[i] == 0]
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for nxt in outgoing[node_id]:
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                ready.append(nxt)
    for node_id in ids:  # cycles never happen in a region, but stay total
        if node_id not in order:
            order.append(node_id)
    return order


def _group_key(graph: Graph, node_id: str) -> str:
    node = graph.node(node_id)
    parents = {g.id: g.parent for g in graph.groups}
    chain: list[str] = []
    g = node.group
    while g is not None:
        chain.append(g)
        g = parents.get(g)
    return "/".join(reversed(chain))


def _size(node: Node) -> tuple[float, float]:
    width = max(len(node.label) * _CHAR, len(node.sublabel) * _SMALL) + 24
    height = _NODE_HEIGHT_TWO if node.sublabel else _NODE_HEIGHT
    return max(width, 56.0), height


def _group_boxes(graph: Graph, placed: Mapping[str, Placed]) -> list[Box]:
    parents = {g.id: g.parent for g in graph.groups}
    members: dict[str, list[Placed]] = {g.id: [] for g in graph.groups}
    for p in placed.values():
        g = p.node.group
        while g is not None:
            members[g].append(p)
            g = parents.get(g)
    boxes: list[Box] = []
    depth_of: dict[str, int] = {}
    for group in graph.groups:
        d = 0
        g = group.parent
        while g is not None:
            d += 1
            g = parents.get(g)
        depth_of[group.id] = d
    for group in sorted(graph.groups, key=lambda g: -depth_of[g.id]):
        inside = members[group.id]
        if not inside:
            continue
        pad = _GROUP_PAD * 0.75
        x0 = min(p.x for p in inside) - pad
        y0 = min(p.y for p in inside) - pad - 14
        x1 = max(p.x + p.width for p in inside) + pad
        y1 = max(p.y + p.height for p in inside) + pad
        for child in boxes:
            if child.group.parent == group.id:
                x0 = min(x0, child.x - pad)
                y0 = min(y0, child.y - pad - 14)
                x1 = max(x1, child.x + child.width + pad)
                y1 = max(y1, child.y + child.height + pad)
        boxes.append(Box(group, x0, y0, x1 - x0, y1 - y0))
    boxes.sort(key=lambda b: depth_of[b.group.id])
    return boxes


# ----------------------------------------------------------------- renderers

_LIGHT = {
    "bg": "#ffffff",
    "ink": "#111111",
    "muted": "#666666",
    "line": "#444444",
    "node": "#ffffff",
    "node_border": "#222222",
    "accent": "hsl(358, 85%, 52%)",
    "group": "#f6f6f6",
    "group_border": "#bbbbbb",
}
_DARK = {
    "bg": "hsl(0, 0%, 2%)",
    "ink": "#f2f2f2",
    "muted": "#9a9a9a",
    "line": "#8a8a8a",
    "node": "#161616",
    "node_border": "#3a3a3a",
    "accent": "hsl(358, 85%, 52%)",
    "group": "#0e0e0e",
    "group_border": "#333333",
}


def _attrs(**values: object) -> str:
    """SVG attributes; numbers are written with one decimal, `class_` as `class`."""
    parts: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        name = "class" if key == "class_" else key.replace("_", "-")
        text = f"{value:.1f}" if isinstance(value, float) else str(value)
        parts.append(f'{name}="{html.escape(text, quote=True)}"')
    return " ".join(parts)


def _rect(**values: object) -> str:
    return f"<rect {_attrs(**values)}/>"


def _text(content: str, **values: object) -> str:
    return f"<text {_attrs(**values)}>{html.escape(content)}</text>"


def to_svg(graph: Graph, *, theme: Literal["light", "dark"] = "light", title: bool = True) -> str:
    """The graph as a standalone SVG document."""
    colors = _DARK if theme == "dark" else _LIGHT
    lay = layout(graph)
    top = 28.0 if title else 0.0
    mono = "'JetBrains Mono',ui-monospace,monospace"
    size = f"{lay.width:.0f} {lay.height + top:.0f}"
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{lay.width:.0f}" '
        f'height="{lay.height + top:.0f}" viewBox="0 0 {size}" '
        'font-family="Inter, system-ui, sans-serif">',
        "<style>",
        f".label{{font-size:12px;fill:{colors['ink']}}}",
        f".sub{{font-size:10.5px;fill:{colors['muted']};font-family:{mono}}}",
        f".edge{{font-size:10px;fill:{colors['muted']};font-family:{mono}}}",
        f".group{{font-size:11px;fill:{colors['muted']}}}",
        f".title{{font-size:13px;font-weight:600;fill:{colors['ink']}}}",
        "</style>",
        _rect(width="100%", height="100%", fill=colors["bg"]),
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colors["line"]}"/></marker></defs>',
    ]
    if title:
        out.append(_text(graph.title, class_="title", x=_MARGIN, y=19))
    for box in lay.boxes:
        out.append(
            _rect(
                x=box.x,
                y=box.y + top,
                width=box.width,
                height=box.height,
                rx=4,
                fill=colors["group"],
                stroke=colors["group_border"],
                stroke_dasharray="4 3",
            )
        )
        out.append(_text(box.group.label, class_="group", x=box.x + 8, y=box.y + top + 13))
    for edge in graph.edges:
        if edge.src not in lay.placed or edge.dst not in lay.placed:
            continue
        a = lay.placed[edge.src]
        b = lay.placed[edge.dst]
        x0, y0 = a.x + a.width / 2, a.y + a.height + top
        x1, y1 = b.x + b.width / 2, b.y + top
        mid = (y0 + y1) / 2
        path = (
            f"M {x0:.1f} {y0:.1f} C {x0:.1f} {mid:.1f}, {x1:.1f} {mid:.1f}, {x1:.1f} {y1 - 1:.1f}"
        )
        out.append(
            f'<path d="{path}" fill="none" stroke="{colors["line"]}" stroke-width="1.1" '
            'marker-end="url(#arrow)"/>'
        )
        if edge.label:
            w = len(edge.label) * _SMALL + 8
            lx = (x0 + x1) / 2
            out.append(
                _rect(
                    x=lx - w / 2,
                    y=mid - 8,
                    width=w,
                    height=15,
                    rx=2,
                    fill=colors["bg"],
                    opacity=0.92,
                )
            )
            out.append(_text(edge.label, class_="edge", x=lx, y=mid + 3.5, text_anchor="middle"))
    for p in lay.placed.values():
        node = p.node
        border = colors["accent"] if node.kind == "call" else colors["node_border"]
        out.append(
            _rect(
                x=p.x,
                y=p.y + top,
                width=p.width,
                height=p.height,
                rx=14 if node.kind in ("input", "output") else 4,
                fill=colors["node"],
                stroke=border,
                stroke_width=1.2,
                stroke_dasharray="3 2" if node.kind in ("param", "state") else None,
            )
        )
        cx = p.x + p.width / 2
        if node.sublabel:
            out.append(
                _text(node.label, class_="label", x=cx, y=p.y + top + 18, text_anchor="middle")
            )
            out.append(
                _text(node.sublabel, class_="sub", x=cx, y=p.y + top + 34, text_anchor="middle")
            )
        else:
            out.append(
                _text(node.label, class_="label", x=cx, y=p.y + top + 19, text_anchor="middle")
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def to_tikz(graph: Graph, *, scale: float = 0.028) -> str:
    """The graph as a TikZ picture (needs `\\usetikzlibrary{fit, arrows.meta}`)."""
    lay = layout(graph)
    out = [
        "% \\usetikzlibrary{fit, arrows.meta}",
        "\\begin{tikzpicture}[",
        "  every node/.style={font=\\small},",
        "  op/.style={draw, rounded corners=1pt, inner sep=4pt, align=center},",
        "  call/.style={op, draw=red!80!black, thick},",
        "  io/.style={draw, rounded corners=8pt, inner sep=4pt, align=center},",
        "  mem/.style={op, dashed},",
        "  grp/.style={draw, dashed, rounded corners=2pt, inner sep=6pt},",
        "  flow/.style={-{Stealth[length=5pt]}},",
        "  lbl/.style={font=\\scriptsize\\ttfamily, fill=white, inner sep=1pt},",
        "]",
    ]
    styles = {
        "input": "io",
        "output": "io",
        "call": "call",
        "op": "op",
        "param": "mem",
        "state": "mem",
    }
    for p in lay.placed.values():
        node = p.node
        x = (p.x + p.width / 2) * scale
        y = -(p.y + p.height / 2) * scale
        text = _tex(node.label)
        if node.sublabel:
            text += " \\\\ {\\scriptsize\\ttfamily " + _tex(node.sublabel) + "}"
        out.append(f"  \\node[{styles[node.kind]}] ({node.id}) at ({x:.2f}, {y:.2f}) {{{text}}};")
    members: dict[str, list[str]] = {g.id: [] for g in graph.groups}
    parents = {g.id: g.parent for g in graph.groups}
    for node in graph.nodes:
        g = node.group
        while g is not None:
            members[g].append(node.id)
            g = parents.get(g)
    for box in lay.boxes:
        inside = members[box.group.id]
        if not inside:
            continue
        fit = "".join(f"({i})" for i in inside)
        caption = f"[anchor=north west, font=\\scriptsize]north west:{_tex(box.group.label)}"
        out.append(f"  \\node[grp, fit={fit}, label={{{caption}}}] ({box.group.id}) {{}};")
    for edge in graph.edges:
        if edge.src not in lay.placed or edge.dst not in lay.placed:
            continue
        label = f" node[lbl, midway] {{{_tex(edge.label)}}}" if edge.label else ""
        out.append(f"  \\draw[flow] ({edge.src}) -- ({edge.dst}){label};")
    out.append("\\end{tikzpicture}")
    return "\n".join(out) + "\n"


def to_dot(graph: Graph) -> str:
    """The graph in Graphviz DOT, groups as clusters."""
    out = [
        "digraph linnet {",
        "  rankdir=TB;",
        '  node [shape=box, style=rounded, fontname="Inter", fontsize=11];',
        '  edge [fontname="JetBrains Mono", fontsize=9, color="#444444"];',
    ]
    children: dict[str | None, list[Group]] = {}
    for group in graph.groups:
        children.setdefault(group.parent, []).append(group)

    def emit_group(group: Group, indent: str) -> None:
        head = f'{indent}subgraph cluster_{group.id} {{ label="{_dot(group.label)}";'
        out.append(head + ' style=dashed; color="#999999";')
        for node in graph.nodes:
            if node.group == group.id:
                out.append(indent + "  " + _dot_node(node))
        for child in children.get(group.id, []):
            emit_group(child, indent + "  ")
        out.append(indent + "}")

    for node in graph.nodes:
        if node.group is None:
            out.append("  " + _dot_node(node))
    for group in children.get(None, []):
        emit_group(group, "  ")
    for edge in graph.edges:
        label = f' [label="{_dot(edge.label)}"]' if edge.label else ""
        out.append(f"  {edge.src} -> {edge.dst}{label};")
    out.append("}")
    return "\n".join(out) + "\n"


def _dot_node(node: Node) -> str:
    label = _dot(node.label) + (f"\\n{_dot(node.sublabel)}" if node.sublabel else "")
    extra = {
        "input": ', shape=box, style="rounded,filled", fillcolor="#f2f2f2"',
        "output": ', shape=box, style="rounded,filled", fillcolor="#f2f2f2"',
        "call": ', color="#d0252c", penwidth=1.4',
        "param": ", style=dashed",
        "state": ", style=dashed",
        "op": "",
    }[node.kind]
    return f'{node.id} [label="{label}"{extra}];'


def _dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _tex(text: str) -> str:
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("<", "$<$"),
        (">", "$>$"),
        ("\u00d7", "$\\times$"),
        ("\u00f7", "$\\div$"),
        ("\u2212", "$-$"),
        ("^", "\\^{}"),
        ("~", "\\~{}"),
        ("*", "$*$"),
        ("|", "$|$"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# ----------------------------------------------------------------------- CLI


def render(
    program: ir.Program,
    function: str | None = None,
    *,
    format: Literal["svg", "tikz", "dot"] = "svg",
    expand: int = 0,
    params: bool = False,
    arithmetic: bool = True,
    theme: Literal["light", "dark"] = "light",
) -> str:
    graph = build(program, function, expand=expand, params=params, arithmetic=arithmetic)
    if format == "svg":
        return to_svg(graph, theme=theme)
    if format == "tikz":
        return to_tikz(graph)
    return to_dot(graph)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m linnet.diagram", description="Draw a Linnet entry."
    )
    parser.add_argument("source", help="a .linnet file or package")
    parser.add_argument("--entry", help="the root entry to draw (the only one by default)")
    parser.add_argument("--root", help="the root block")
    parser.add_argument("--std", help="the standard library directory")
    parser.add_argument("--format", choices=["svg", "tikz", "dot"], default=None)
    parser.add_argument("--expand", type=int, default=0, help="inline block method calls this deep")
    parser.add_argument("--params", action="store_true", help="show parameter reads")
    parser.add_argument("--no-arithmetic", action="store_true", help="fold elementwise operations")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("-o", "--output", help="output file (format from its extension)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    fmt: Literal["svg", "tikz", "dot"] = "svg"
    if args.format in ("svg", "tikz", "dot"):
        fmt = args.format
    elif args.output:
        suffix = Path(args.output).suffix.lower()
        fmt = (
            "tikz" if suffix in (".tex", ".tikz") else "dot" if suffix in (".dot", ".gv") else "svg"
        )
    try:
        program = ir.load_program(args.source, root=args.root, std_root=args.std)
        text = render(
            program,
            args.entry,
            format=fmt,
            expand=args.expand,
            params=args.params,
            arithmetic=not args.no_arithmetic,
            theme=args.theme,
        )
    except LinnetError as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

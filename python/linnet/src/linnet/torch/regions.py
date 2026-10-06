"""A generated `main`'s repeated blocks as one function each, for
`torch.compile` to compile a layer once rather than every layer of a step.

`linnet torch` prints an entry as one straight-line function in which every
element of a block array (`layers.0`, `layers.1`, ...) appears again. Its
statements are grouped by the element whose parameters, states or prepared
weights they read; a statement reading only the entry's inputs and
constants (a mask, the positions) runs in the element that reads it alone,
and once in `main` when several read it. Each element becomes a
function of the values it receives and its own weights, and elements whose
functions are the same but for their names share one. `main` then calls
those functions, releasing what it no longer needs after each call.

Compiling the shared functions compiles one layer per kind of layer: the
first call compiles, the others reuse it, as long as their arguments have
the same shapes and dtypes. Entries with control flow, or whose blocks
differ, are left as they are (`regional` returns None).
"""

from __future__ import annotations

import ast
from collections.abc import Container, Sequence
from dataclasses import dataclass

from ..parallel import units
from .stages import _delete, _list_constant, _stores  # pyright: ignore[reportPrivateUsage]

_OUTER = -1  # `main`'s own statements: what no block array holds
_FREE = -2  # statements of the inputs and constants alone


@dataclass(frozen=True, slots=True)
class Regions:
    """The rewritten module source, the names of the shared functions, and
    how many calls of them one step makes."""

    source: str
    functions: tuple[str, ...]
    calls: int


def regional(source: str) -> Regions | None:
    """`source` with its block-array elements as shared functions, or None
    when the entry has none, has control flow, or its elements differ."""
    module = ast.parse(source)
    main = next(
        (n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None
    )
    if main is None:
        return None
    parameters = _list_constant(module, "PARAMETERS")
    states = _list_constant(module, "STATES")
    _, elements = units([*parameters, *states])
    if len(elements) < 2:
        return None
    index = {unit: i for i, unit in enumerate(elements)}

    def unit_of(path: str) -> int:
        parts = path.split(".")
        if len(parts) > 2 and parts[1].isdigit():
            return index.get(f"{parts[0]}.{parts[1]}", _OUTER)
        return _OUTER

    # Each argument of `main`: the element it belongs to, or free.
    owner: dict[str, int] = {}
    for arg in main.args.args:
        name = arg.arg
        if name.startswith("p") and name[1:].isdigit() and int(name[1:]) < len(parameters):
            owner[name] = unit_of(parameters[int(name[1:])])
        elif name.startswith("s") and name[1:].isdigit() and int(name[1:]) < len(states):
            owner[name] = unit_of(states[int(name[1:])])
        else:
            owner[name] = _FREE
    # Prepared weights belong where the weights they are made of do.
    prepare = next(
        (n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "prepare"), None
    )
    if prepare is not None:
        for node in prepare.body:
            for name in _stores(node):
                if name in owner:
                    made_of = [owner[read] for read in _reads(node, owner) if read != name]
                    anchored = [u for u in made_of if u != _FREE]
                    owner[name] = max(anchored) if anchored else _FREE
    arguments = set(owner)

    statements: list[ast.stmt] = []
    returned: list[str] = []
    for node in main.body:
        if isinstance(node, ast.Delete):
            continue
        if isinstance(node, ast.Return):
            if node.value is not None:
                values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
                if not all(isinstance(v, ast.Name) for v in values):
                    return None
                returned = [v.id for v in values if isinstance(v, ast.Name)]
            break
        if not isinstance(node, (ast.Assign, ast.Expr)):
            return None  # control flow
        statements.append(node)

    # Each statement's element: the latest one whose weights or values it
    # reads; `main`'s own when it reads a weight no element holds.
    known = dict(owner)
    reads: list[list[str]] = []
    writes: list[list[str]] = []
    place: list[int] = []
    for node in statements:
        read = _reads(node, known)
        direct = {owner[name] for name in read if name in arguments and owner[name] != _FREE}
        anchors = {known[name] for name in read if known[name] >= 0}
        if _OUTER in direct:
            where = _OUTER
        elif anchors:
            where = max(anchors)
        elif any(known[name] == _OUTER for name in read):
            where = _OUTER
        else:
            where = _FREE
        written = _stores(node)
        for name in written:
            known[name] = where
        reads.append(read)
        writes.append(written)
        place.append(where)

    # A free statement read by one element runs in it, as the generated code
    # computes it there; one read by several runs once in `main`, which
    # hands it to each.
    readers: dict[str, set[int]] = {}
    for name in returned:
        readers.setdefault(name, set()).add(_OUTER)
    runs: list[set[int]] = [set() for _ in statements]
    for i in range(len(statements) - 1, -1, -1):
        if place[i] != _FREE:
            runs[i] = {place[i]}
        else:
            for name in writes[i]:
                runs[i] |= readers.get(name, set())
            if len(runs[i]) > 1:
                runs[i] = {_OUTER}
        for name in reads[i]:
            readers.setdefault(name, set()).update(runs[i])

    # Each element's function: what it receives, its body, what it sends.
    defined_in: dict[str, int] = {}
    for i, where in enumerate(place):
        if where >= 0:
            for name in writes[i]:
                defined_in[name] = where
    read_after: dict[int, set[str]] = {}
    for i, where in enumerate(runs):
        for unit in where:
            for name in reads[i]:
                if defined_in.get(name, unit) != unit:
                    read_after.setdefault(unit, set()).add(name)
    outside: set[str] = set(returned)
    for i, where in enumerate(runs):
        if _OUTER in where:
            outside.update(reads[i])

    bodies: dict[int, list[int]] = {}
    for i, where in enumerate(runs):
        for unit in where:
            if unit >= 0:
                bodies.setdefault(unit, []).append(i)
    if sorted(bodies) != list(range(len(elements))):
        return None
    functions: dict[int, ast.FunctionDef] = {}
    signatures: dict[int, tuple[list[str], list[str]]] = {}
    for unit, members in bodies.items():
        body = [statements[i] for i in members]
        local = {name for i in members for name in writes[i]}
        wanted: dict[str, None] = {}
        for i in members:
            for name in reads[i]:
                if name not in local:
                    wanted[name] = None
        sends = [
            name
            for i in members
            if place[i] == unit
            for name in writes[i]
            if name in outside or any(name in read_after.get(u, ()) for u in bodies if u > unit)
        ]
        signatures[unit] = (list(wanted), sends)
        functions[unit] = _function(body, list(wanted), sends)

    # Elements the same but for their names share a function.
    shared: dict[str, str] = {}
    kinds: list[ast.FunctionDef] = []
    name_of: dict[int, str] = {}
    for unit in sorted(functions):
        key = _normalized(functions[unit])
        if key not in shared:
            name = f"_region_{len(kinds)}"
            shared[key] = name
            function = functions[unit]
            function.name = name
            kinds.append(_renamed(function))
        name_of[unit] = shared[key]
    if len(kinds) == len(functions):
        return None  # nothing repeats

    # `main` again: its own statements in order, and each element's call
    # where it began, or later once what it receives exists.
    first = {unit: next(i for i in members if place[i] == unit) for unit, members in bodies.items()}
    starts: dict[int, int] = {index: unit for unit, index in first.items()}
    rebuilt: list[ast.stmt] = []
    rebuilt_reads: list[list[str]] = []
    available = set(arguments)
    pending: list[int] = []

    def call(unit: int) -> None:
        needed, sends = signatures[unit]
        target = f"({', '.join(sends)}{',' if len(sends) == 1 else ''}) = " if sends else ""
        rebuilt.append(ast.parse(f"{target}{name_of[unit]}({', '.join(needed)})").body[0])
        rebuilt_reads.append(needed)
        available.update(sends)

    def flush() -> None:
        while pending and all(name in available for name in signatures[pending[0]][0]):
            call(pending.pop(0))

    for i, node in enumerate(statements):
        if i in starts:
            pending.append(starts[i])
            flush()
        if _OUTER in runs[i]:
            if any(name not in available for name in reads[i]):
                flush()
                if any(name not in available for name in reads[i]):
                    return None
            rebuilt.append(node)
            rebuilt_reads.append(reads[i])
            available.update(writes[i])
            flush()
    flush()
    if pending or any(name not in available for name in returned):
        return None

    main.body = _released(rebuilt, rebuilt_reads, arguments, returned)
    module.body = [n for n in module.body if n is not main] + [*kinds, main]
    ast.fix_missing_locations(module)
    return Regions(
        ast.unparse(module) + "\n",
        tuple(f.name for f in kinds),
        len(functions),
    )


def _function(body: list[ast.stmt], needed: Sequence[str], sends: Sequence[str]) -> ast.FunctionDef:
    """An element's function: its values released after their last use."""
    reads = [list(dict.fromkeys(_names(node, ast.Load))) for node in body]
    last: dict[str, int] = {}
    for i, read in enumerate(reads):
        for name in read:
            last[name] = i
    kept = set(sends)
    released: dict[int, list[str]] = {}
    for name, i in last.items():
        if name not in kept and (name in needed or any(name in _stores(n) for n in body)):
            released.setdefault(i, []).append(name)
    out: list[ast.stmt] = []
    for i, node in enumerate(body):
        out.append(node)
        if i in released:
            out.append(_delete(released[i]))
    out.append(
        ast.Return(
            value=ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in sends], ctx=ast.Load())
        )
    )
    function = ast.parse(f"def _element({', '.join(needed)}):\n    pass\n").body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = out
    return function


def _reads(node: ast.AST, known: Container[str]) -> list[str]:
    """The known names `node` reads, once each, in order."""
    found: dict[str, None] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in known:
            found[child.id] = None
    return list(found)


def _names(node: ast.AST, context: type) -> list[str]:
    return [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, context)]


def _local_names(function: ast.FunctionDef) -> list[str]:
    """The function's arguments, then its assigned names, in order."""
    found: dict[str, None] = {a.arg: None for a in function.args.args}
    for node in function.body:
        if isinstance(node, ast.Assign):
            for name in _stores(node):
                found[name] = None
    return list(found)


def _renamed(function: ast.FunctionDef) -> ast.FunctionDef:
    """`function` with its own names numbered in order of appearance, so
    that elements the same but for their names print the same."""
    names = {name: f"x{i}" for i, name in enumerate(_local_names(function))}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id in names:
                return ast.copy_location(ast.Name(id=names[node.id], ctx=node.ctx), node)
            return node

        def visit_arg(self, node: ast.arg) -> ast.arg:
            if node.arg in names:
                node.arg = names[node.arg]
            return node

    copy = ast.parse(ast.unparse(function)).body[0]
    assert isinstance(copy, ast.FunctionDef)
    renamed = Rename().visit(copy)
    assert isinstance(renamed, ast.FunctionDef)
    renamed.name = function.name
    return renamed


def _normalized(function: ast.FunctionDef) -> str:
    renamed = _renamed(function)
    renamed.name = "_"
    return ast.dump(renamed, annotate_fields=False)


def _released(
    body: list[ast.stmt], reads: list[list[str]], arguments: set[str], returned: Sequence[str]
) -> list[ast.stmt]:
    """`main`'s statements, each value it made released after its last use."""
    last: dict[str, int] = {}
    for i, read in enumerate(reads):
        for name in read:
            last[name] = i
    released: dict[int, list[str]] = {}
    for name, i in last.items():
        if name not in arguments and name not in returned:
            released.setdefault(i, []).append(name)
    out: list[ast.stmt] = []
    for i, node in enumerate(body):
        out.append(node)
        if i in released:
            out.append(_delete(released[i]))
    names = ", ".join(returned)
    out.append(ast.parse(f"return ({names}{',' if len(returned) == 1 else ''})").body[0])
    return out


__all__ = ["Regions", "regional"]

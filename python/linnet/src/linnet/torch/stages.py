"""An entry's generated source split into pipeline stages.

`linnet torch` prints an entry as one straight-line function, `main`. A
pipeline runs it as one function per stage, each process its own, the
values one stage computes and a later one reads passing between them.
`split` makes those functions from `main` by its data flow:

- a value computed from a parameter is computed on the stage that holds
  the parameter, or the latest stage among its operands' (as early as it
  can be, so a layer's output crosses once rather than its operands);
- a value computed from the entry's inputs alone (a mask, the positions)
  is computed again on every stage that reads it: every stage receives
  the inputs, and computing a mask is cheaper than sending one;
- constants (`constants(device)`: rotary tables) are computed on every
  stage that reads them.

A stage takes the values crossing into it positionally, then its own
parameters and constants, then the inputs it reads and `_device` by
keyword; it returns the values crossing out of it, and the last stage the
entry's results.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..plan import PlanError


@dataclass(frozen=True, slots=True)
class Stage:
    """One stage function's signature."""

    name: str
    receives: tuple[str, ...]  # values crossing in, in order
    parameters: tuple[int, ...]  # indices into PARAMETERS
    constants: tuple[str, ...]  # names from CONSTANTS
    inputs: tuple[str, ...]  # the entry inputs it reads (`in_tokens`, ...)
    sends: tuple[str, ...]  # values crossing out; the results on the last stage


@dataclass(frozen=True, slots=True)
class Split:
    """The split module's source, the entry's inputs in order (`main`'s
    `in_` arguments), its parameters (`PARAMETERS`), constants
    (`CONSTANTS`) and number of results (`RESULTS`), and the stages."""

    source: str
    inputs: tuple[str, ...]
    parameters: tuple[str, ...]
    constants: tuple[str, ...]
    results: int
    stages: tuple[Stage, ...]


def split(source: str, stage_of: Callable[[str], int], stages: int) -> Split:
    """Splits generated `source` into `stages` functions `stage_0`, ...;
    `stage_of(path)` is the stage that holds the parameter at `path`."""
    module = ast.parse(source)
    main = next(
        (n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None
    )
    if main is None:
        raise PlanError("the generated source has no `main`")
    parameter_paths = _list_constant(module, "PARAMETERS")
    stage_of_parameter = [stage_of(path) for path in parameter_paths]
    constant_list = _list_constant(module, "CONSTANTS")
    constant_names = set(constant_list)
    if _list_constant(module, "STATES") or _list_constant(module, "NEXT_STATES"):
        raise PlanError("a pipeline runs entries that keep no state between calls")
    if _list_constant(module, "PREPARED"):
        raise PlanError("a pipeline's entry is generated without `--prepare`")

    # name -> (stage or None for "decide later", anchored to a parameter)
    made_on: dict[str, int | None] = {}
    anchored: dict[str, bool] = {}
    inputs: list[str] = []
    for arg in main.args.args:
        name = arg.arg
        if name.startswith("in_"):
            inputs.append(name)
            made_on[name], anchored[name] = None, False
        elif name.startswith("p") and name[1:].isdigit():
            index = int(name[1:])
            if index >= len(stage_of_parameter):
                raise PlanError(f"no stage given for parameter {index}")
            made_on[name], anchored[name] = stage_of_parameter[index], True
        elif name in constant_names:
            made_on[name], anchored[name] = None, False
        else:
            raise PlanError(f"unexpected argument `{name}` of the generated `main`")

    statements: list[ast.stmt] = []
    results: list[str] = []
    for node in main.body:
        if isinstance(node, ast.Delete):
            continue  # released again per stage
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_device"
        ):
            continue  # each stage is given its device
        if isinstance(node, ast.Return):
            if node.value is None:
                break
            values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
            for value in values:
                if not isinstance(value, ast.Name):
                    raise PlanError("the generated `main` returns something other than values")
                results.append(value.id)
            break
        if not isinstance(node, (ast.Assign, ast.Expr)):
            raise PlanError(
                "a pipeline splits straight-line entries; this one has "
                f"`{type(node).__name__.lower()}` control flow"
            )
        statements.append(node)

    uses: list[list[str]] = []
    defs: list[list[str]] = []
    bound: list[bool] = []
    placed: list[set[int]] = []
    for node in statements:
        read = _loads(node, made_on)
        written = _stores(node)
        uses.append(read)
        defs.append(written)
        anchors = [made_on[name] or 0 for name in read if anchored[name]]
        bound.append(bool(anchors))
        placed.append({max(anchors)} if anchors else set())
        for name in written:
            made_on[name], anchored[name] = (max(anchors) if anchors else None), bool(anchors)

    # Values of the inputs alone are computed on every stage that reads
    # them, latest statement first so that its readers are placed.
    last = stages - 1
    readers: dict[str, list[int]] = {}
    for index, read in enumerate(uses):
        for name in read:
            readers.setdefault(name, []).append(index)
    for index in range(len(statements) - 1, -1, -1):
        if bound[index]:
            continue
        wanted: set[int] = set()
        for name in defs[index]:
            for reader in readers.get(name, []):
                wanted |= placed[reader]
            if name in results:
                wanted.add(last)
        placed[index] = wanted

    # Values made from parameters cross from the stage that makes them to
    # every later stage that reads them.
    made: dict[str, int] = {}
    order: dict[str, int] = {}
    for index, written in enumerate(defs):
        if bound[index]:
            for name in written:
                made[name] = min(placed[index])
                order.setdefault(name, index)
    read_on: dict[str, set[int]] = {}
    for index, read in enumerate(uses):
        for name in read:
            read_on.setdefault(name, set()).update(placed[index])
    for name in results:
        read_on.setdefault(name, set()).add(last)

    crossing: list[list[str]] = []
    for boundary in range(last):
        out = [
            name
            for name in sorted(made, key=lambda n: order[n])
            if made[name] <= boundary and any(s > boundary for s in read_on.get(name, ()))
        ]
        crossing.append(out)

    functions: list[ast.FunctionDef] = []
    signatures: list[Stage] = []
    for k in range(stages):
        body = [node for node, where in zip(statements, placed, strict=True) if k in where]
        local_uses = [read for read, where in zip(uses, placed, strict=True) if k in where]
        local_defs = {
            name
            for written, where in zip(defs, placed, strict=True)
            if k in where
            for name in written
        }
        receives = crossing[k - 1] if k > 0 else []
        sends = results if k == last else crossing[k]
        read = {name for names in local_uses for name in names} | set(sends)
        parameters = sorted(int(name[1:]) for name in read if _parameter(name) and name not in made)
        constants = [a.arg for a in main.args.args if a.arg in constant_names and a.arg in read]
        stage_inputs = [name for name in inputs if name in read]
        have = {*receives, *local_defs, *stage_inputs}
        missing = [name for name in sends if name not in have and not _parameter(name)]
        if missing:
            raise PlanError(f"stage {k} sends values it does not have: {', '.join(missing)}")
        functions.append(
            _function(
                f"stage_{k}",
                receives,
                [f"p{i}" for i in parameters],
                constants,
                stage_inputs,
                body,
                local_uses,
                sends,
            )
        )
        signatures.append(
            Stage(
                f"stage_{k}",
                tuple(receives),
                tuple(parameters),
                tuple(constants),
                tuple(stage_inputs),
                tuple(sends),
            )
        )

    kept = [n for n in module.body if n is not main]
    module.body = [*kept, *functions]
    ast.fix_missing_locations(module)
    return Split(
        ast.unparse(module) + "\n",
        tuple(inputs),
        tuple(parameter_paths),
        tuple(constant_list),
        len(results),
        tuple(signatures),
    )


def _parameter(name: str) -> bool:
    return name.startswith("p") and name[1:].isdigit()


def _list_constant(module: ast.Module, name: str) -> list[str]:
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, list):
                return [str(v) for v in value]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return []


def _loads(node: ast.AST, known: dict[str, int | None]) -> list[str]:
    """The local values `node` reads, once each, in order."""
    found: dict[str, None] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in known:
            found[child.id] = None
    return list(found)


def _stores(node: ast.stmt) -> list[str]:
    if not isinstance(node, ast.Assign):
        return []
    found: dict[str, None] = {}
    for target in node.targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                found[child.id] = None
    return list(found)


def _function(
    name: str,
    receives: Sequence[str],
    parameters: Sequence[str],
    constants: Sequence[str],
    inputs: Sequence[str],
    body: Sequence[ast.stmt],
    uses: Sequence[Sequence[str]],
    sends: Sequence[str],
) -> ast.FunctionDef:
    """A stage function whose values are released after their last use, as
    `main`'s are."""
    positional = [*receives, *parameters, *constants]
    last: dict[str, int] = {}
    for index, read in enumerate(uses):
        for value in read:
            last[value] = index
    for index, node in enumerate(body):
        for value in _stores(node):
            last.setdefault(value, index)
    kept = set(sends)
    released: dict[int, list[str]] = {}
    for value in [*positional, *inputs]:
        if value not in kept and value not in last:
            released.setdefault(-1, []).append(value)
    for value, index in last.items():
        if value not in kept:
            released.setdefault(index, []).append(value)
    statements: list[ast.stmt] = []
    if -1 in released:
        statements.append(_delete(released[-1]))
    for index, node in enumerate(body):
        statements.append(node)
        if index in released:
            statements.append(_delete(released[index]))
    returned: ast.expr = ast.Tuple(
        elts=[ast.Name(id=value, ctx=ast.Load()) for value in sends], ctx=ast.Load()
    )
    statements.append(ast.Return(value=returned))
    # Parsed rather than built, so the node has every field this Python's
    # `ast` expects.
    signature = ", ".join([*positional, "*", *inputs, "_device", "**_unused"])
    function = ast.parse(f"def {name}({signature}):\n    pass\n").body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = statements
    return function


def _delete(names: Sequence[str]) -> ast.Delete:
    return ast.Delete(targets=[ast.Name(id=name, ctx=ast.Del()) for name in names])


__all__ = ["Split", "Stage", "split"]

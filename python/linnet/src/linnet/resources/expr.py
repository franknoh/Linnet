"""Symbolic resource expressions: byte counts that stay functions of the
dimensions left free (a batch size, a sequence length) until a solver or a
report binds them.

An `Expr` is a small immutable tree over non-negative integers: constants,
named symbols, sums, products, floor division, `min` and `max`. The smart
constructors fold constants and flatten nested sums and products, so the
same quantity built two ways usually prints the same. Every operator is
non-decreasing in its symbols except division's divisor, which is only ever
a constant or a dimension the model fixes; that monotonicity is what lets
`linnet.resources.planner` search for the largest configuration that fits
by bisection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Literal


class ExprError(ValueError):
    """An expression needs a symbol that is not bound, or divides by zero."""


@dataclass(frozen=True, slots=True)
class Const:
    value: int


@dataclass(frozen=True, slots=True)
class Sym:
    name: str


@dataclass(frozen=True, slots=True)
class Nary:
    op: Literal["add", "mul", "min", "max"]
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Binary:
    op: Literal["floordiv", "mod", "ceildiv"]
    lhs: Expr
    rhs: Expr


Expr = Const | Sym | Nary | Binary

ZERO = Const(0)
ONE = Const(1)


def const(value: int) -> Expr:
    return Const(value)


def sym(name: str) -> Expr:
    return Sym(name)


def _nary(op: Literal["add", "mul", "min", "max"], args: Iterable[Expr]) -> Expr:
    flat: list[Expr] = []
    for arg in args:
        if isinstance(arg, Nary) and arg.op == op:
            flat.extend(arg.args)
        else:
            flat.append(arg)
    constants = [a.value for a in flat if isinstance(a, Const)]
    rest = [a for a in flat if not isinstance(a, Const)]
    if op == "add":
        total = sum(constants)
        terms = _like_terms(rest) + ([Const(total)] if total or not rest else [])
    elif op == "mul":
        product = reduce(lambda x, y: x * y, constants, 1)
        if product == 0:
            return ZERO
        terms = ([Const(product)] if product != 1 or not rest else []) + rest
    else:
        if constants:
            folded = min(constants) if op == "min" else max(constants)
            terms = [*dict.fromkeys(rest), Const(folded)]
        else:
            terms = list(dict.fromkeys(rest))
    if len(terms) == 1:
        return terms[0]
    return Nary(op, tuple(terms))


def _like_terms(terms: Sequence[Expr]) -> list[Expr]:
    """`2 * B + 3 * B` as `5 * B`: terms that differ only in their constant
    factor, combined."""
    coefficients: dict[Expr, int] = {}
    for term in terms:
        coefficient, base = 1, term
        if isinstance(term, Nary) and term.op == "mul" and isinstance(term.args[0], Const):
            coefficient = term.args[0].value
            base = term.args[1] if len(term.args) == 2 else Nary("mul", term.args[1:])
        coefficients[base] = coefficients.get(base, 0) + coefficient
    out: list[Expr] = []
    for base, coefficient in coefficients.items():
        if coefficient == 0:
            continue
        out.append(base if coefficient == 1 else _nary("mul", (Const(coefficient), base)))
    return out


def add(*args: Expr) -> Expr:
    return _nary("add", args)


def mul(*args: Expr) -> Expr:
    return _nary("mul", args)


def minimum(*args: Expr) -> Expr:
    return _nary("min", args)


def maximum(*args: Expr) -> Expr:
    return _nary("max", args)


def total(args: Iterable[Expr]) -> Expr:
    return _nary("add", args)


def product(args: Iterable[Expr]) -> Expr:
    return _nary("mul", args)


def floordiv(lhs: Expr, rhs: Expr) -> Expr:
    if isinstance(rhs, Const) and rhs.value == 1:
        return lhs
    if isinstance(lhs, Const) and isinstance(rhs, Const) and rhs.value:
        return Const(lhs.value // rhs.value)
    return Binary("floordiv", lhs, rhs)


def ceildiv(lhs: Expr, rhs: Expr) -> Expr:
    if isinstance(rhs, Const) and rhs.value == 1:
        return lhs
    if isinstance(lhs, Const) and isinstance(rhs, Const) and rhs.value:
        return Const(-(-lhs.value // rhs.value))
    return Binary("ceildiv", lhs, rhs)


def mod(lhs: Expr, rhs: Expr) -> Expr:
    if isinstance(lhs, Const) and isinstance(rhs, Const) and rhs.value:
        return Const(lhs.value % rhs.value)
    return Binary("mod", lhs, rhs)


class _Expressions:
    """Dimension arithmetic over expressions (`ir.fold_dim`)."""

    def const(self, value: int) -> Expr:
        return const(value)

    def total(self, args: Sequence[Expr]) -> Expr:
        return total(args)

    def product(self, args: Sequence[Expr]) -> Expr:
        return product(args)

    def floordiv(self, a: Expr, b: Expr) -> Expr:
        return floordiv(a, b)

    def mod(self, a: Expr, b: Expr) -> Expr:
        return mod(a, b)

    def minimum(self, args: Sequence[Expr]) -> Expr:
        return minimum(*args)

    def maximum(self, args: Sequence[Expr]) -> Expr:
        return maximum(*args)


EXPRESSIONS = _Expressions()


def evaluate(expr: Expr, env: Mapping[str, int]) -> int:
    """The expression's value with every symbol bound by `env`."""
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Sym):
        if expr.name not in env:
            raise ExprError(f"`{expr.name}` is not bound")
        return env[expr.name]
    if isinstance(expr, Nary):
        values = [evaluate(a, env) for a in expr.args]
        if expr.op == "add":
            return sum(values)
        if expr.op == "mul":
            return reduce(lambda x, y: x * y, values, 1)
        return min(values) if expr.op == "min" else max(values)
    lhs, rhs = evaluate(expr.lhs, env), evaluate(expr.rhs, env)
    if rhs == 0:
        raise ExprError("division by zero")
    if expr.op == "floordiv":
        return lhs // rhs
    if expr.op == "ceildiv":
        return -(-lhs // rhs)
    return lhs % rhs


def substitute(expr: Expr, env: Mapping[str, Expr]) -> Expr:
    """The expression with the symbols `env` names replaced, folded again."""
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Sym):
        return env.get(expr.name, expr)
    if isinstance(expr, Nary):
        return _nary(expr.op, (substitute(a, env) for a in expr.args))
    lhs, rhs = substitute(expr.lhs, env), substitute(expr.rhs, env)
    if expr.op == "floordiv":
        return floordiv(lhs, rhs)
    if expr.op == "ceildiv":
        return ceildiv(lhs, rhs)
    return mod(lhs, rhs)


def symbols(expr: Expr) -> frozenset[str]:
    if isinstance(expr, Const):
        return frozenset()
    if isinstance(expr, Sym):
        return frozenset({expr.name})
    if isinstance(expr, Nary):
        found: frozenset[str] = frozenset()
        return found.union(*(symbols(a) for a in expr.args))
    return symbols(expr.lhs) | symbols(expr.rhs)


def is_constant(expr: Expr) -> bool:
    return isinstance(expr, Const)


_PRECEDENCE = {"add": 1, "mul": 2}


def format_expr(expr: Expr) -> str:
    """The expression as the language spells dimensions: `2 * B * S * 4096`."""
    return _format(expr, 0)


def _format(expr: Expr, outer: int) -> str:
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Sym):
        return expr.name
    if isinstance(expr, Nary):
        if expr.op in ("min", "max"):
            return f"{expr.op}({', '.join(_format(a, 0) for a in expr.args)})"
        precedence = _PRECEDENCE[expr.op]
        text = (" + " if expr.op == "add" else " * ").join(
            _format(a, precedence) for a in expr.args
        )
        return f"({text})" if precedence < outer else text
    if expr.op == "ceildiv":
        return f"ceil({_format(expr.lhs, 0)} / {_format(expr.rhs, 0)})"
    symbol = " / " if expr.op == "floordiv" else " % "
    text = f"{_format(expr.lhs, 3)}{symbol}{_format(expr.rhs, 3)}"
    return f"({text})" if outer >= 2 else text

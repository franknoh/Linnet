"""The type of what running a compiled entry returns.

An entry returns one tensor (or array), a tuple of them when it has several
results, and, under JAX, the state beside them when it touches `state`.
Which a call gets is settled when the entry compiles, so no static type can
follow it: a call that runs an entry returns `Result`, unchecked, and the
caller types it by what its own entry returns, as it would any value its
program computes. Everything else in `linnet` is typed exactly."""

from typing import Any, TypeAlias

Result: TypeAlias = Any

__all__ = ["Result"]

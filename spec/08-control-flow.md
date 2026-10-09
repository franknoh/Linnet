# 8. Control flow

Linnet has runtime `if`, `while`, and `for`, compile-time `static for`, and no recursion.

## 8.1 Runtime `if`

`if` is an expression over a scalar `bool`. Both branches must have identical result types.

## 8.2 `static for`

`static for` expands its body at compile time:

```text
var x = embedding.forward(tokens)

static for layer in layers {
    x = layer.forward(x)
}
```

The iterable MUST be a statically known structural collection, such as a sub-block array or a compile-time integer range. The loop MUST NOT depend on runtime tensor values.

## 8.3 Static integer ranges

```text
static for i in 0..Steps {
    ...
}
```

`start..stop` iterates over the integers `start <= i < stop`. Both bounds MUST be compile-time integers: literals, generic dimensions, or arithmetic on them. The loop variable is an `i64` scalar holding the iteration's position, not a compile-time integer, so it cannot index a sub-block array or appear in a shape.

## 8.4 Runtime loops

```text
var count = 0
var running = true
while running && count < MaxNew {
    ...
    count = count + 1
    running = !done
}
```

`while` repeats its body while its condition, a scalar `bool` checked before each iteration, holds. The `var` locals assigned in the body are the carried values; their types are fixed (§5.3), so shapes are invariant across iterations. The body may assign `state` members (§9.3), and those writes are visible after the loop. `return` is not allowed inside a loop. Termination is the program's responsibility.

## 8.5 Counted loops and scans

```text
var h = h0
let hs = for t in 0..T {
    h = cell(h, xs[t])
    yield h
}
```

`for i in start..stop` runs its body at runtime once for each integer `start <= i < stop`. Both bounds MUST be compile-time integers, as in §8.3, and `stop >= start` MUST be provable. The loop variable is an `i64` scalar. As in `while`, the `var` locals assigned in the body are carried, the body may assign `state`, and `return` is not allowed inside it.

A `for` statement has no value. A `for` that is the whole value of a `let`, `var`, assignment or `return` is a scan: its body MUST end with `yield value`, and the loop's value is every iteration's `value`, stacked in order along a new leading axis of size `stop - start`. A yielded `Tensor[S; T]` becomes `Tensor[stop - start, S; T]`, a scalar `T` becomes `Tensor[stop - start; T]`, and a tuple stacks each element. `yield` appears nowhere else.

## 8.6 Recursion

Function, op, and block-method recursion is forbidden in the initial language version. Future recursion support MUST be explicit in this specification, not inherited from backend behavior.

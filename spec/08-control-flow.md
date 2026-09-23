# 8. Control Flow and Structural Iteration

## 8.1 Runtime `if`

`if` is an expression over a scalar `bool`. Both branches must have identical result types.

It may lower to backend control flow or selection depending on the backend and context.

## 8.2 `static for`

`static for` performs compile-time structural expansion:

```text
var x = embedding.forward(tokens)

static for layer in layers {
    x = layer.forward(x)
}
```

The iterable MUST be a statically known structural collection such as:

- a sub-block array;
- a compile-time integer range;
- another future compile-time collection.

The loop MUST NOT depend on runtime tensor values.

## 8.3 Static integer ranges

```text
static for i in 0..Steps {
    ...
}
```

`start..stop` iterates over the compile-time integers `start <= i < stop`; both bounds MUST be compile-time integers (literals, generic dimensions, or arithmetic on them). The body is expanded at compile time exactly as the array form is. The loop variable is an `i64` scalar value holding the position of each iteration: it may be cast, compared, or used in arithmetic like any scalar, but it is not a compile-time integer, so it cannot index a sub array or appear in a shape — iterate the array itself for that.

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

`while` repeats its body while the condition, a scalar `bool` evaluated before each iteration, holds. The `var` locals assigned in the body are the loop's carried values: their types are fixed (§5.3), so every shape is invariant across iterations, and the body may assign `state` members (§9.3). `return` is not allowed inside a loop. Termination is the program's responsibility; a bound on a counter is the usual form.

In Core IR a `while` is one operation carrying those values, with a condition region and a body region, so effects inside it are ordered like any other. Backends execute it as a loop; a graph export that cannot express a data-dependent loop reports it rather than unrolling. `for` remains reserved.

## 8.5 Recursion

Function, op, and block-method recursion is forbidden in the initial language version.

Future recursion support, if any, MUST be explicit in the language specification rather than accidentally inherited from backend behavior.

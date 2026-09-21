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

Reserved syntax:

```text
static for i in 0..Layers {
    ...
}
```

An implementation MAY support this in the initial version. If unsupported, it MUST be rejected rather than parsed with different semantics.

## 8.4 Runtime loops

`while` is reserved for future runtime control flow and MUST NOT be accepted as a user identifier.

Runtime loops require explicit rules for state threading, shape invariance, termination-independent graph semantics, and backend export. They are intentionally excluded from the first implementation.

## 8.5 Recursion

Function, op, and block-method recursion is forbidden in the initial language version.

Future recursion support, if any, MUST be explicit in the language specification rather than accidentally inherited from backend behavior.

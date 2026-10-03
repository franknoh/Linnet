# 5. Expressions and bindings

Bindings, operators, calls, `if`, `match`, and indexing.

## 5.1 Purity

Ordinary `fn` and `op` bodies are pure. They cannot mutate parameters, buffers, files, environment variables, external state, or hidden global values.

Local `var` assignment is not observable outside the function body.

## 5.2 `let`

```text
let q = linear(x, q_weight)
```

A `let` binding is immutable. A type annotation also gives contextual literals their type:

```text
let scale: f32 = 0.5
```

Tuple destructuring:

```text
let (q, k) = rope(q, k, positions)
```

## 5.3 `var`

```text
var x = embedding(tokens)
x = layer.forward(x)
```

Only locals declared with `var` may be reassigned. Reassignment MUST preserve the variable's declared or inferred type.

A block's `state` members (§9.3) are also assigned with this statement; that assignment is the one observable effect in the language.

## 5.4 Arithmetic

Supported operators:

```text
+ - * / %
== != < <= > >=
&& || !
& | ^
```

Arithmetic operators apply to compatible scalars and lift elementwise to same-dtype tensors under valid broadcasting. Integer arithmetic wraps on overflow (two's complement) at the dtype's width. Integer `/` and `%` truncate toward zero.

`&`, `|`, and `^` are bitwise on integers and elementwise logical on booleans, scalar or tensor, broadcast like arithmetic; `mask ^ true` negates a mask. `shl(x, bits)` and `shr(x, bits)` shift integers; `shr` is arithmetic for signed dtypes and logical for unsigned ones. On compile-time integers these five fold when both operands are constants.

A contextual numeric literal lifts across a tensor: `x * 0.5` is valid when `x` has a floating dtype that can represent `0.5`. A scalar variable is not implicitly converted to a tensor of a different dtype.

Comparing tensors gives a boolean tensor of the broadcast shape.

`&&`, `||`, and `!` take scalar `bool` operands only.

Binary operators bind, loosest first: `||`; `&&`; `==` `!=`; `<` `<=` `>` `>=`; `|`; `^`; `&`; `+` `-`; `*` `/` `%`. All are left-associative. Unary `!`, `-`, and `+` bind tighter than every binary operator. Two comparisons combined with `&` therefore need parentheses: `(a == b) & (i < n)`.

## 5.5 Calls

```text
linear(x, weight)
layer.forward(x)
cast<f32>(x)
```

Generic arguments may be inferred where unambiguous. Explicit generic arguments bind generic parameters in declaration order; the rest are inferred.

Arguments are positional or named (`name = value`), positional first. A parameter with a default may be omitted. An optional parameter accepts `none` or an optional value; a plain value MUST be wrapped as `some(value)`.

Every `where` constraint of the callee MUST be provable at the call site from the caller's own constraints.

`name<` begins a generic call only when the matching `>` is immediately followed by `(`; otherwise `<` is a comparison. In a generic argument list, a non-type argument is an arithmetic expression, and comparison and logical operators there MUST be parenthesized.

## 5.6 `if` expression

```text
let y =
    if causal {
        causal_mask(score)
    } else {
        score
    }
```

The condition MUST be a scalar `bool`; for a tensor condition, use `select` or an equivalent elementwise operation.

Both branches MUST have the same type, including tensor shape and dtype.

## 5.7 `match`

The initial version requires `match` on optional values and simple enums:

```text
let y = match bias {
    some(b) => x + b
    none    => x
}
```

Patterns MUST be exhaustive. An arm pattern is a variant or binding name, `some(pattern)`, or `none`. A tuple pattern may appear only inside `some(...)`.

## 5.8 Slicing and indexing

Value-level indexing forms include:

```text
x[i]
x[b, h, s, d]
x[..., 0::2]
x[:, start:end]
x[:, start:end:step]
```

An integer index removes its axis. A compile-time integer index MUST be provably within the axis. A runtime integer scalar index reads the position it holds and has no defined result outside the axis. Negative indices are rejected: the last element of an axis of size `N` is `x[N - 1]`.

A slice `start:stop:step` on an axis of size `D` keeps the axis, with extent `max(0, (min(stop, D) - start + step - 1) / step)`. `start` defaults to `0`, `stop` to `D`, and `step` to `1`. Slice bounds MUST be non-negative compile-time integers and the step a positive integer constant.

`...` stands for all axes not addressed explicitly and may appear once. Axes of a shape pack cannot be indexed individually.

Inside index notation (section 6), every tensor access MUST index all axes, with index variables, a pack index, or integers. An integer there may come from runtime data; such an access is a gather (§6.4).

Outside index notation, an access with a runtime index means the comprehension over the remaining axes: `x[i, :]` is `let r[j] = x[i, j]`, and `x[i, 1:3]` slices that result.

## 5.9 Shape literals

A bracketed list of dimension expressions is a compile-time shape literal:

```text
[B, S, H]
[H / Heads, Heads]
```

Shape literals are not runtime values. They are accepted only in compile-time shape contexts and by primitives that consume a shape, such as `reshape`:

```text
reshape(x, [B, Heads, S, H / Heads])
```

## 5.10 No arbitrary expression statements

The initial language has no general expression statement, so an unused call result has no meaning in pure code.

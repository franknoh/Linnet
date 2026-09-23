# 5. Expressions and Local Bindings

## 5.1 Purity

Ordinary `fn` and `op` bodies are pure in the initial language version. They cannot mutate parameters, buffers, files, environment variables, external state, or hidden global values.

Local `var` assignment is syntactic convenience and lowers to SSA values; it is not observable mutation outside the local function body.

## 5.2 `let`

```text
let q = linear(x, q_weight)
```

A `let` binding is immutable. It may carry a type annotation, which also gives contextual literals their type:

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

Only locals declared with `var` may be reassigned. Reassignment MUST preserve the statically inferred or declared type of the variable.

Parameters, buffers, sub-blocks, function arguments, and `let` bindings cannot be reassigned. A block's `state` members (§9.3) are assigned with the same statement; that assignment is the one observable effect in the language.

## 5.4 Arithmetic

Supported operators:

```text
+ - * / %
== != < <= > >=
&& || !
& | ^
```

Arithmetic operators are defined for compatible scalar types and are elementwise-lifted to tensors of the same dtype with valid broadcasting. Integer arithmetic wraps on overflow (two's complement) at the dtype's width; `/` and `%` on integers truncate toward zero.

`&`, `|`, and `^` are bitwise on integer operands and elementwise logical on boolean operands (scalars or tensors, broadcast like arithmetic); they bind looser than `+` and tighter than comparisons. The prelude functions `shl(x, bits)` and `shr(x, bits)` shift integers; `shr` is arithmetic for signed dtypes and logical for unsigned ones. On compile-time integers all five fold when both operands are constants. A contextual scalar numeric literal may be lifted across a tensor, for example `x * 0.5` when `x` has floating dtype and `0.5` is representable in that dtype. General scalar variables are not implicitly converted to a tensor of a different dtype.

Comparison of tensors produces a boolean tensor with broadcasted shape.

Logical `&&`, `||`, and `!` apply to scalar booleans. Elementwise boolean tensor operations are provided by library functions or explicitly specified primitive operators.

## 5.5 Calls

```text
linear(x, weight)
layer.forward(x)
cast<f32>(x)
```

Generic arguments may be inferred where unambiguous. Explicit generic arguments bind generic parameters in declaration order, and the remainder are inferred.

Arguments are positional, or named with `name = value`; positional arguments come first. A parameter with a default may be omitted. An optional parameter accepts `none` or an optional value; a plain value MUST be wrapped as `some(value)`.

Every `where` constraint of the callee MUST be provable at the call site from the caller's own constraints.

`name<` begins a generic call only when the matching `>` is immediately followed by `(`; otherwise `<` is the comparison operator. Inside a generic argument list, an argument that is not a type is an arithmetic expression; comparison and logical operators there MUST be parenthesized.

## 5.6 `if` expression

```text
let y =
    if causal {
        causal_mask(score)
    } else {
        score
    }
```

The condition MUST be a scalar `bool`.

Both branches MUST have the same type, including tensor shape and dtype.

Tensor-valued conditions require `select` or an equivalent elementwise operation; they are not accepted by `if`.

## 5.7 `match`

Initial `match` support is required for optional values and simple enums.

```text
let y = match bias {
    some(b) => x + b
    none    => x
}
```

Patterns MUST be exhaustive.

An arm pattern is a variant or binding name, `some(pattern)`, or `none`; a tuple pattern may appear only inside `some(...)`. Arms have no separator, so a tuple pattern at the start of an arm would be read as a call on the previous arm's value.

## 5.8 Slicing and indexing

Value-level tensor indexing syntax includes:

```text
x[i]
x[b, h, s, d]
x[..., 0::2]
x[:, start:end]
x[:, start:end:step]
```

A slice creates a logical tensor view in HIR. Backends determine whether materialization is required.

An integer index removes its axis. A slice `start:stop:step` keeps it with extent `max(0, (min(stop, D) - start + step - 1) / step)` for an axis of size `D`; `start` defaults to `0`, `stop` to `D`, and `step` to `1`. Slice bounds MUST be non-negative compile-time integers and the step a positive integer constant, so that the result shape never depends on runtime data. `...` stands for all axes not addressed explicitly and may appear once. Axes that belong to a shape pack cannot be indexed individually.

Inside index notation (section 6) every tensor access MUST index all axes, using index variables, a pack index, or integers.

Negative literal indices MAY be supported when the dimension is statically known or when the backend plan preserves well-defined indexing semantics. The initial implementation MAY reject negative indices conservatively.

## 5.9 Shape literals

A bracketed list of dimension expressions is a compile-time shape literal:

```text
[B, S, H]
[H / Heads, Heads]
```

Shape literals are accepted only in compile-time shape contexts and by primitives that explicitly consume a shape, such as `reshape` or `broadcast_to`. They are not runtime tensor/list values.

Example:

```text
reshape(x, [B, Heads, S, H / Heads])
```

## 5.10 No arbitrary expression statements

The initial language has no generic expression statement. Calls whose result is intentionally unused are therefore not meaningful in pure code.

This restriction keeps semicolon-free syntax unambiguous and reinforces effect-free semantics.

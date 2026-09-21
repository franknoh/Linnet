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

Parameters, buffers, sub-blocks, function arguments, and `let` bindings cannot be reassigned.

## 5.4 Arithmetic

Supported operators:

```text
+ - * / %
== != < <= > >=
&& || !
```

Arithmetic operators are defined for compatible scalar types and are elementwise-lifted to tensors of the same dtype with valid broadcasting. A contextual scalar numeric literal may be lifted across a tensor, for example `x * 0.5` when `x` has floating dtype and `0.5` is representable in that dtype. General scalar variables are not implicitly converted to a tensor of a different dtype.

Comparison of tensors produces a boolean tensor with broadcasted shape.

Logical `&&`, `||`, and `!` apply to scalar booleans. Elementwise boolean tensor operations are provided by library functions or explicitly specified primitive operators.

## 5.5 Calls

```text
linear(x, weight)
layer.forward(x)
cast<f32>(x)
```

Generic arguments may be inferred where unambiguous.

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

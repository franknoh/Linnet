# 4. Shapes and constraints

The compiler proves every shape relation at compile time.

## 4.1 Dimension expressions

A tensor dimension is a non-negative compile-time integer expression:

```text
constant
symbol
A + B
A - B
A * B
A / B
A % B
min(A, B)
max(A, B)
```

Division is integer division and is valid only when its semantics are statically well-defined for the operation being checked. Library code SHOULD state divisibility constraints when it intends exact division.

## 4.2 Compile-time constants

```text
const Hidden = 4096
const Heads = 32
const HeadDim = Hidden / Heads
```

Compile-time constants MUST be pure and evaluable without runtime tensor data.

An unannotated constant is contextual (§1.7): an integer constant is a compile-time integer usable in shapes, and a floating constant adopts a `Float` dtype.

An annotated constant may have a scalar or enum type:

```text
pub const THETA: f32 = 500000.0
pub const POOLING: Pooling = Pooling.ClassToken
```

Only integer constants participate in dimension expressions.

## 4.3 `where` constraints

```text
fn split_heads<H: Dim, N: Dim>(...)
    -> ...
where
    H % N == 0,
    N > 0
{
    ...
}
```

Supported relations:

```text
== != < <= > >=
```

Either side may be an arithmetic expression.

## 4.4 Proof requirement

When an operation requires two dimensions to match, the compiler MUST prove the equality from:

- syntactic identity;
- constant evaluation;
- generic substitutions;
- declared constraints;
- solver deductions that are sound under the specification.

If the compiler cannot prove a required relation, strict static checking fails.

## 4.5 Solver completeness

A conforming compiler MUST be sound and may be incomplete.

For a true relation beyond the solver, the compiler SHOULD emit a diagnostic suggesting an explicit `where` constraint or a simpler equivalent shape expression.

## 4.6 Broadcasting

Elementwise tensor operations use statically provable, right-aligned broadcasting. Two aligned dimensions are compatible when the compiler can prove them equal or one of them exactly `1`:

```text
Tensor[B, S, H; T] + Tensor[H; T]
    -> Tensor[B, S, H; T]
```

If compatibility cannot be proven, the operation is an error.

## 4.7 Shape packs

A shape pack can appear in generic prefix or suffix patterns:

```text
Tensor[*S, In; T]
```

Matched against `Tensor[B, S, H; T]` with `In = H`, `*S` becomes `[B, S]`. Within one signature, a pack symbol denotes the same sequence of dimensions.

## 4.8 Negative and zero dimensions

Concrete tensor dimensions MUST be non-negative. Operations may impose stronger constraints such as `N > 0`. A compile-time expression proven negative is an error.

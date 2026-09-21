# 4. Symbolic Shapes and Constraints

## 4.1 Dimension expressions

A tensor dimension is a non-negative compile-time integer expression.

Supported expression forms:

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

Division in a dimension expression is integer division and is valid only when its semantics are statically well-defined for the operation being checked. Library code SHOULD state divisibility constraints when exact division is intended.

## 4.2 Compile-time constants

```text
const Hidden = 4096
const Heads = 32
const HeadDim = Hidden / Heads
```

Compile-time constants MUST be pure and evaluable without runtime tensor data.

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

Supported relational forms in the initial specification:

```text
== != < <= > >=
```

Arithmetic expressions may appear on either side.

## 4.4 Proof requirement

When an operation requires two dimensions to match, the compiler MUST prove the equality from:

- syntactic identity;
- constant evaluation;
- generic substitutions;
- declared constraints;
- solver deductions that are sound under the specification.

If the compiler cannot prove a required relation, strict static checking fails. A backend's ability to guard or dynamically handle the case does not make the Linnet program valid.

## 4.5 Solver completeness

The language does not require an implementation to decide every true nonlinear integer proposition. A conforming compiler MUST be sound; it may be incomplete.

When a required relationship is true but beyond the solver's supported reasoning, the compiler SHOULD emit a diagnostic suggesting an explicit `where` constraint or a simpler equivalent shape expression.

## 4.6 Broadcasting

Elementwise tensor operations use statically provable right-aligned broadcasting.

For each aligned axis, dimensions are compatible when the compiler can prove either:

- the dimensions are equal; or
- one dimension is exactly `1`.

Example:

```text
Tensor[B, S, H; T] + Tensor[H; T]
    -> Tensor[B, S, H; T]
```

If symbolic compatibility cannot be proven, the operation is an error rather than a runtime guess.

Explicit broadcast operations in the tensor standard library may be used to state intent.

## 4.7 Shape packs

A shape pack can participate in generic suffix or prefix patterns:

```text
Tensor[*S, In; T]
```

When matching against `Tensor[B, S, H; T]` with `In = H`, `*S` becomes `[B, S]`.

The same pack symbol within a signature denotes the same sequence of dimensions.

## 4.8 Negative and zero dimensions

Concrete tensor dimensions MUST be non-negative. Operations may impose stronger constraints such as `N > 0`.

A compile-time expression proven negative is an error.

# 3. Type System

## 3.1 Scalar types

Initial scalar types:

```text
bool

i8 i16 i32 i64
u8 u16 u32 u64

f16 bf16 f32 f64
```

Low-bit and floating-point extensions such as `i4`, `u4`, `f8e4m3`, and `f8e5m2` are reserved for future versions.

## 3.2 Tensor type syntax

Tensor types separate shape from element dtype with `;`:

```text
Tensor[B, S, H; bf16]
Tensor[M, N; f32]
Tensor[H; i32]
```

A tensor rank is the number of dimensions before `;` after shape-pack expansion.

Zero-rank tensor syntax is reserved. Scalar values SHOULD use scalar types instead of rank-zero tensors in the initial language version.

## 3.3 Shape packs

A generic shape pack is declared:

```text
*S: Shape
```

and used:

```text
Tensor[*S, H; T]
```

A shape pack may be empty.

## 3.4 Generic kinds

The initial language provides built-in generic kinds and constraints:

```text
Dim
Shape
DType
Numeric
Integer
Float
```

Examples:

```text
fn identity<N: Dim, T: Numeric>(...)
fn relu<*S: Shape, T: Float>(...)
```

User-defined traits are reserved for a future version.

## 3.5 Type aliases

```text
type Hidden<B: Dim, S: Dim, H: Dim, T: Float = bf16> =
    Tensor[B, S, H; T]
```

Aliases are transparent; they do not create nominal runtime types.

## 3.6 Tuples

```text
(f32, i32)
(Tensor[B, H; T], Tensor[B, H; T])
```

Tuple destructuring:

```text
let (q, k) = rope(q, k, positions)
```

## 3.7 Structs

Structs provide nominal aggregate types:

```text
struct KVPair<K, V> {
    key: K
    value: V
}
```

Struct fields are immutable values in the initial language version.

## 3.8 Enums

Enums are closed tagged unions:

```text
enum MaskKind {
    None,
    Causal,
}
```

Payload-carrying enum variants are reserved for a future version. `Option<T>` behavior is provided directly through `T?`.

## 3.9 Optional values

```text
Tensor[H; f32]?
```

Values:

```text
none
some(value)
```

Pattern matching:

```text
match bias {
    some(b) => y + b
    none    => y
}
```

`none` requires contextual type information.

## 3.10 Structural arrays

Compile-time structural arrays are written:

```text
[DecoderLayer<Hidden, T>; Layers]
```

They are not runtime tensors. They are intended for repeated sub-block declarations and `static for` traversal.

## 3.11 Dtype conversion

There is no implicit tensor-to-tensor dtype promotion in the strict language core.

This is an error:

```text
Tensor[B, H; bf16] + Tensor[B, H; f32]
```

The author must explicitly convert:

```text
cast<f32>(x) + y
```

Contextual numeric literals remain allowed as described in the lexical specification.

This rule is deliberately stricter than NumPy, PyTorch, and many host frameworks. Backends MUST preserve Linnet semantics rather than inheriting framework-specific promotion behavior.

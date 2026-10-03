# 3. Type system

Linnet's types are scalars, tensors, tuples, structs, enums, optionals, and structural arrays.

## 3.1 Scalar types

Scalar types:

```text
bool

i8 i16 i32 i64
u8 u16 u32 u64

f16 bf16 f32 f64
```

Types such as `i4`, `u4`, `f8e4m3`, and `f8e5m2` are reserved for future versions.

## 3.2 Tensor type syntax

`;` separates a tensor type's shape from its element dtype:

```text
Tensor[B, S, H; bf16]
Tensor[M, N; f32]
Tensor[H; i32]
```

A tensor's rank is the number of dimensions before `;`, after shape-pack expansion.

Zero-rank tensor syntax is reserved: `Tensor[; f32]` is an error; scalar values use scalar types. Where a backend has no scalars, an entry's scalar inputs and results cross its boundary as rank-zero tensors.

## 3.3 Shape packs

A generic shape pack is declared as:

```text
*S: Shape
```

and used as:

```text
Tensor[*S, H; T]
```

A shape pack may be empty.

## 3.4 Generic kinds

Built-in generic kinds and constraints:

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

Aliases are transparent: they create no nominal type.

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

Structs are nominal aggregate types:

```text
struct KVPair<K: DType, V: DType> {
    key: K
    value: V
}
```

Struct fields are immutable and read with `value.field`.

No expression constructs a struct value, and a struct name is not callable. Construction syntax is reserved for a future version.

## 3.8 Enums

Enums are closed tagged unions:

```text
enum MaskKind {
    None,
    Causal,
}
```

A variant is written `MaskKind.Causal`; `match` arms name the variants:

```text
const MASK: MaskKind = MaskKind.Causal

let scores = match MASK {
    Causal => causal_mask(scores)
    None => scores
}
```

Every arm is checked. When the scrutinee is a constant, the arm is chosen at compile time and a backend evaluates only that arm.

Payload-carrying variants are reserved for a future version.

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

Compile-time structural arrays hold repeated sub-blocks for `static for` (§8.2):

```text
[DecoderLayer<Hidden, T>; Layers]
```

A structural array is not a runtime tensor. It is indexed with exactly one compile-time integer, as in `layers[0]`.

## 3.11 Dtype conversion

Tensor dtypes are never promoted implicitly. This is an error:

```text
Tensor[B, H; bf16] + Tensor[B, H; f32]
```

Convert explicitly:

```text
cast<f32>(x) + y
```

Backends MUST preserve Linnet semantics, not a framework's promotion behavior.

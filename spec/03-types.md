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

`bool` is built in. The others are `dtype` declarations in the prelude, `std.prelude` (`stdlib/prelude.linnet`), which every module sees without a `use` and which no module may shadow. Each is a member of a family:

| Family | Members |
| --- | --- |
| `bool` | `bool` |
| `int(bits: N, signed: B)` | `i8` to `i64` (`signed: true`, the default), `u8` to `u64` |
| `float(exponent: E, mantissa: M, inf: B, nan: B)` | `f16` (5, 10), `bf16` (8, 7), `f32` (8, 23), `f64` (11, 52); `inf` and `nan` default to `true` |

A `dtype` declaration names a member of a family:

```text
pub dtype f8e4m3 = float(exponent: 4, mantissa: 3, inf: false)
dtype word = int(bits: 32)
```

`dtype` is a keyword only at the start of an item followed by a name. Each parameter of `int` and `float` is a non-negative integer literal (a width) or `true`/`false` (a flag); a `float` is signed unless `signed: false`. A format no target stores is an error (E2128): implementations store the built-in formats, the FP8 formats E4M3 without infinities (`float(exponent: 4, mantissa: 3, inf: false)`) and E5M2 (`float(exponent: 5, mantissa: 2)`), which `std.dtypes` declares as `f8e4m3` and `f8e5m2`, E8M0 below, and the 4-bit formats below.

A format the prelude does not declare is a storage format: a tensor of it is cast to and from, reshaped, permuted, sliced, concatenated and indexed, but no operator or function computes in it (E2104), no literal has its type, and it belongs only to the class `DType` (§3.4). `cast` rounds to the nearest representable value, ties to even. A value beyond the format's range becomes infinity where the format has one, else NaN, else the largest value of its sign; NaN becomes zero in a format without it. A cast to an integer format wraps at its width, rounding a float toward zero first.

E8M0 (`float(exponent: 8, mantissa: 0, signed: false, inf: false)`, `std.dtypes::f8e8m0`) is the scale of an MX block: byte `e` is `2 ** (e - 127)` for 0 to 254 and NaN for 255; it has no sign, zero or infinity. `cast` to it takes a value's magnitude to a power of two, the next one up from 1.5 times a power (and at it); a magnitude below `2 ** -127`, zero included, to `2 ** -127`; NaN, infinities and magnitudes from `1.5 * 2 ** 127` up, to NaN. A checkpoint's `F8_E8M0` or `U8` tensor binds to it as its bytes.

A format of fewer than 8 bits is packed: a tensor of it stores two elements a byte along its last axis, element `2k` in the low four bits and `2k + 1` in the high four, so the last axis has an even length. An operation keeps that axis whole (E2129): `reshape` and `broadcast_to` keep its length, `permute` keeps it last, `concat` does not join along it, slicing takes all of it, and index notation does not read its elements. `std.dtypes` declares `i4 = int(bits: 4)`, `u4 = int(bits: 4, signed: false)` and `f4e2m1 = float(exponent: 2, mantissa: 1, inf: false, nan: false)` (FP4 E2M1: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives). A checkpoint stores such a tensor as its bytes, `U8` with the last axis halved.

A codebook format gives each code of an unsigned integer format a value:

```text
pub dtype nf4 = codebook(carrier: u4, values: [-1.0, -0.6961928009986877, ..., 0.7229568362236023, 1.0])
```

(`...` stands for the twelve values between.) The carrier is an unsigned `int` format of at most 8 bits, `values` one finite number per code in strictly increasing order; another carrier, count or order is an error (E2128). A tensor of a codebook format stores its codes as the carrier does, packed when the carrier is. Each declaration is a storage format of its own, apart from its carrier and from other codebooks. `cast` from one takes each code to its value as `f32`, then casts that; `cast` to one takes each value to the code of the nearest, a tie to the lower code and NaN to code 0. `std.dtypes` declares `nf4`, QLoRA's 4-bit NormalFloat over `u4`.

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

Calling a struct's name builds a value. Every field is given once, by name, in any order:

```text
let kv = KVPair(key = k, value = v)
```

Generic arguments are inferred from the fields or written out: `KVPair<f32, i32>(key = k, value = v)`. Struct fields are immutable and read with `value.field`.

An entry's struct result crosses the backend boundary as a tuple of its fields, in declaration order.

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

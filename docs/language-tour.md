# Language tour

Linnet is a small, strict language for describing tensor computation and model
structure. This tour covers what a model author needs; `spec/` is normative.

## Modules

Every file declares its module and imports by logical path:

```text
module models.mini

use std.nn.linear::{linear, Linear}
use std.nn.norm::{rms_norm}
use crate.layers::{Block}
```

Items are private unless `pub`. See [modules-and-packages.md](modules-and-packages.md).

## Tensor types

A tensor type names its shape and its element type, separated by `;`:

```text
Tensor[B, S, H; bf16]
Tensor[H; f32]
```

Dimensions are compile-time integers: literals, `Dim` generic parameters, or
arithmetic over them (`H / Heads`, `2 * S`). Scalars use the scalar types
directly: `bool`, `i8`…`i64`, `u8`…`u64`, `f16`, `bf16`, `f32`, `f64`.

## Functions and generics

```text
fn scale<*S: Shape, T: Float>(x: Tensor[*S; T], k: T) -> Tensor[*S; T] {
    return x * k
}
```

- `T: Float` is an element type constrained to floating types (`DType`,
  `Numeric`, `Integer`, `Float`).
- `*S: Shape` is a shape pack: any number of leading dimensions, so `scale`
  applies to tensors of every rank.
- Generic arguments are inferred at calls: `scale(x, 0.5)`. They can also be
  written: `cast<f32>(x)`.

## Strict dtypes

Tensors combine only with tensors of the same dtype; there is no implicit
promotion:

```text
let y = cast<f32>(x_bf16) + z_f32   // explicit
let bad = x_bf16 + z_f32            // error E2103
```

Literals adopt the dtype of their context: `x * 0.5` works for any floating
`x`; `x * 2` works for any numeric `x`.

## Broadcasting

Elementwise operations broadcast right-aligned, and only when the compiler can
prove each pair of axes equal or one of them `1`:

```text
Tensor[B, S, H; T] + Tensor[H; T]     // fine
Tensor[A; f32] + Tensor[B; f32]       // error E2207 unless A == B is provable
```

## Index notation

Contractions are written with explicit indices. Output indices go on the left;
reduction indices are introduced by `sum`, `prod`, `max`, `min`, `any`, `all`:

```text
let c[m, n] = sum[k] a[m, k] * b[k, n]          // matrix multiplication
let y[*s, o] = sum[i] x[*s, i] * w[o, i]        // linear layer over any rank
let t = sum[i] a[i, i]                          // trace
let score[b, h, q, k] =
    sum<f32>[d] cast<f32>(q[b, h, q, d]) * cast<f32>(k[b, h, k, d])
```

An index that is neither an output nor reduced is an error: Linnet never sums
implicitly. `sum<f32>[...]` states the accumulation dtype.

## Constraints

Functions and blocks state what their shapes require in a `where` clause; the
compiler proves the clause at every call:

```text
fn split_heads<B: Dim, S: Dim, H: Dim, N: Dim, T: Float>(
    x: Tensor[B, S, H; T],
) -> Tensor[B, N, S, H / N; T]
where H % N == 0, N > 0 {
    return permute(reshape(x, [B, S, N, H / N]), [0, 2, 1, 3])
}
```

Without `H % N == 0`, `reshape` could not be shown to keep the element count.

## Semantic operations

`op` is a function with a stable identity. Its body is the normative
definition; a backend may replace it with an optimized implementation that
must agree with the body:

```text
pub op linear<*S: Shape, In: Dim, Out: Dim, T: Float>(
    x: Tensor[*S, In; T],
    weight: Tensor[Out, In; T],
    bias: Tensor[Out; T]? = none,
) -> Tensor[*S, Out; T] {
    let y[*s, o] = sum[i] x[*s, i] * weight[o, i]
    return match bias {
        some(b) => y + b
        none => y
    }
}
```

`Tensor[Out; T]?` is an optional; `none` and `some(value)` are its values and
`match` must cover both.

## Blocks and parameters

A block owns parameters and sub-blocks; its methods compute with them:

```text
pub block Linear<In: Dim, Out: Dim, T: Float = bf16> {
    param weight: Tensor[Out, In; T]
    param bias: Tensor[Out; T]? = none

    pub fn forward<*S: Shape>(x: Tensor[*S, In; T]) -> Tensor[*S, Out; T] {
        return linear(x, weight, bias)
    }
}

pub block Model<H: Dim, Layers: Dim> {
    sub embedding: Embedding<Vocab, H>
    sub layers: [DecoderLayer<H>; Layers]
    sub head: Linear<H, Vocab>

    pub entry forward<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, S, Vocab; bf16] {
        var x = embedding.forward(tokens)
        static for layer in layers {
            x = layer.forward(x)
        }
        return head.forward(x)
    }
}
```

- A `param` never has data in source; `= none` only marks it optional.
- A `state` member is execution state the block owns, such as a KV cache:
  `state keys: Tensor[B, Max, D; T]`. Its name reads the current value and
  `keys = updated` replaces it; backends keep the value between calls
  (zeros at first) and graph exports thread it as an extra input and output.
- `sub` arrays give repeated layers; `static for` iterates them at compile time.
  `static for step in 0..Steps { ... }` iterates a compile-time integer range;
  `step` is an `i64` scalar of each iteration (cast it, compare it, add it to
  a position), and `var` locals carry values from one iteration to the next.
- `while running && count < MaxNew { ... }` is a runtime loop over a scalar
  `bool`; the `var`s it assigns are its carried values (shapes fixed), so a
  decode loop can stop at an end-of-sequence token. It exports as
  `stablehlo.while`, an ONNX `Loop`, or a Python loop in generated PyTorch.
- `entry` marks the callables a backend exposes.
- Parameter paths follow the structure: `layers.0.attention.q_proj.weight`.
  `linnet inspect --parameters` lists them.

## Slicing, shapes, and prelude functions

```text
x[..., 0::2]                 // every other element of the last axis
x[:, 1:3]                    // rows 1 and 2
reshape(x, [B * S, H])       // element count must be provably equal
permute(x, [0, 2, 1])
concat(a, b, axis = -1)
select(mask, a, b)           // elementwise choice
iota(N)                      // 0, 1, ..., N - 1
fill<f32>([B, S], 0.0)
```

`exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `abs`, `min`, `max`,
`shl`/`shr` (shifts), and `cast<T>` are also built in; `&`, `|`, `^` are
bitwise on integers and elementwise logical on booleans. Everything else — `softmax`, `attention`,
`rms_norm`, `rope`, random numbers (`std.random`, a Threefry PRNG over
integer tensors), and quantized weights (`std.quant`) — is library code in
`stdlib/`.

## What is not in the language

No expression statements, no recursion, no data-dependent shapes, no
mutation of anything but local `var`s and a block's own `state`, no
host-language escape hatches, and no tensor payloads in source. That is what makes a `.linnet` file safe to inspect and
load.

# Language tour

Everything a model author needs, on one page. `spec/` is the normative
specification; this is the short version.

## Modules

Each file declares one module and imports by logical path. Items are private
unless marked `pub`.

```linnet
module models.mini

use std.nn.linear::{linear, Linear}
use std.nn.norm::{rms_norm}
use crate.layers::{Block}
```

`std.` is the standard library, `crate.` the current package. See
[Modules and packages](modules-and-packages.md).

## Tensor types

A tensor type is a shape and an element type:

```linnet
Tensor[B, S, H; bf16]
Tensor[H; f32]
```

Dimensions are compile-time integers: literals, `Dim` generics, or arithmetic
on them (`H / Heads`, `2 * S`). Scalar types are `bool`, `i8` to `i64`,
`u8` to `u64`, `f16`, `bf16`, `f32`, `f64`.

## Functions and generics

```linnet
fn scale<*S: Shape, T: Float>(x: Tensor[*S; T], k: T) -> Tensor[*S; T] {
    return x * k
}
```

| Generic | Meaning |
| --- | --- |
| `N: Dim` | one dimension |
| `*S: Shape` | a shape pack: any number of leading dimensions |
| `T: Float` | an element type; constraints are `DType`, `Numeric`, `Integer`, `Float` |

Generic arguments are inferred at calls (`scale(x, 0.5)`) or written out
(`cast<f32>(x)`).

## Dtypes and broadcasting

Tensors combine only with the same dtype; there is no implicit promotion.
Literals take the dtype of their context.

```linnet
let y = cast<f32>(x_bf16) + z_f32   // explicit cast
let bad = x_bf16 + z_f32            // E2103
let half = x * 0.5                  // 0.5 takes x's dtype
```

Elementwise operations broadcast right-aligned, and only when each pair of
axes is provably equal or one of them is `1`:

```linnet
Tensor[B, S, H; T] + Tensor[H; T]   // fine
Tensor[A; f32] + Tensor[B; f32]     // E2207 unless A == B is known
```

## Index notation

Contractions name every axis. Output indices go on the left; reduction
indices are bound by `sum`, `prod`, `max`, `min`, `any`, or `all`.

```linnet
let c[m, n] = sum[k] a[m, k] * b[k, n]          // matrix product
let y[*s, o] = sum[i] x[*s, i] * w[o, i]        // linear layer, any rank
let t = sum[i] a[i, i]                          // trace
let score[b, h, q, k] = sum<f32>[d] cast<f32>(q[b, h, q, d]) * cast<f32>(k[b, h, k, d])
```

An index that is neither an output nor reduced is an error: nothing is summed
implicitly. `sum<f32>[...]` sets the accumulation dtype.

## Constraints

A `where` clause states what the shapes require. The compiler proves it at
every call and uses it inside the body.

```linnet
fn split_heads<B: Dim, S: Dim, H: Dim, N: Dim, T: Float>(
    x: Tensor[B, S, H; T],
) -> Tensor[B, N, S, H / N; T]
where H % N == 0, N > 0 {
    return permute(reshape(x, [B, S, N, H / N]), [0, 2, 1, 3])
}
```

Without `H % N == 0` the `reshape` cannot be shown to keep the element count
and is rejected.

## Operations with an identity

`op` is a function whose body is the reference definition. A backend may
replace it with a kernel that agrees with the body; `linnet explain` shows
when it does.

```linnet
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

`Tensor[Out; T]?` is an optional. Its values are `none` and `some(v)`, and a
`match` must cover both.

## Blocks

A block owns parameters, state, and sub-blocks; its functions compute with
them. `entry` marks the functions a backend exposes.

```linnet
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

| Member | Holds |
| --- | --- |
| `param` | a weight, bound from a checkpoint; `= none` marks it optional |
| `buffer` | non-trainable data, bound like a parameter |
| `state` | execution state such as a KV cache: read by name, replaced by assignment, kept between calls |
| `sub` | a child block or an array of them |

Parameter paths follow the structure (`layers.0.attention.q_proj.weight`);
`linnet inspect --parameters` lists them.

## Loops

```linnet
static for layer in layers { ... }          // over a sub array, expanded at compile time
static for i in 0..Steps { ... }            // over a compile-time range; i is an i64 scalar
while running && count < MaxNew { ... }     // runtime loop over a scalar bool
```

`var` locals assigned in a loop body carry to the next iteration; their types
are fixed, so shapes are invariant. `static for` is unrolled by every
backend; `while` becomes `stablehlo.while`, an ONNX `Loop`, or a Python
loop in generated code.

## Slicing and built-ins

```linnet
x[..., 0::2]                 // every other element of the last axis
x[:, 1:3]                    // rows 1 and 2
reshape(x, [B * S, H])       // element count must be provable
permute(x, [0, 2, 1])
concat(a, b, axis = -1)
select(mask, a, b)
iota(N)                      // 0, 1, ..., N - 1
fill<f32>([B, S], 0.0)
```

Also built in: `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `abs`,
`min`, `max`, `shl`, `shr`, `cast<T>`, and the operators `& | ^` (bitwise on
integers, logical on booleans). Everything else is library code in `stdlib/`:
`softmax`, `attention`, `rms_norm`, `rope`, random numbers (`std.random`),
quantized weights (`std.quant`).

## Not in the language

Expression statements, recursion, data-dependent shapes, mutation other than
`var` locals and a block's own `state`, host-language escapes, and tensor
data in source. These omissions are what make a `.linnet` file safe to check
and load.

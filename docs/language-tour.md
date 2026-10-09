# Language tour

Linnet by example. `spec/` has the normative rules.

## Modules

Each file is one module. Imports use logical paths: `std.` is the standard
library, `crate.` the current package. Items are private unless
`pub`. See [Modules and packages](modules-and-packages.md).

```linnet
module models.mini

use std.nn.linear::{linear, Linear}
use std.nn.norm::{rms_norm}
use crate.layers::{Block}
```

## Tensor types

A tensor type is a shape and an element type. Dimensions are compile-time
integers: literals, `Dim` generics, or arithmetic on them (`H / Heads`).
Element types are `bool`, `i8` to `i64`, `u8` to `u64`, `f16`, `bf16`,
`f32`, and `f64`.

```linnet
Tensor[B, S, H; bf16]
Tensor[H; f32]
```

## Functions and generics

`N: Dim` is one dimension, `*S: Shape` any number of leading dimensions, and
`T: Float` an element type (also `DType`, `Numeric`, `Integer`). Generic
arguments are inferred, or written out as in `cast<f32>(x)`.

```linnet
fn scale<*S: Shape, T: Float>(x: Tensor[*S; T], k: T) -> Tensor[*S; T] {
    return x * k
}
```

## Dtypes and broadcasting

Dtypes never promote implicitly; use `cast`. Literals take the context's
dtype. Broadcasting is right-aligned and needs each pair of axes provably
equal or `1`.

```linnet
let y = cast<f32>(x_bf16) + z_f32   // explicit cast
let bad = x_bf16 + z_f32            // E2103
let half = x * 0.5                  // 0.5 takes x's dtype
```

```linnet
Tensor[B, S, H; T] + Tensor[H; T]   // fine
Tensor[A; f32] + Tensor[B; f32]     // E2207 unless A == B is known
```

## Index notation

Contractions name every axis: outputs on the left, reductions bound by
`sum`, `prod`, `max`, `min`, `any`, or `all`. Nothing is summed implicitly.
`sum<f32>` sets the accumulation dtype.

```linnet
let c[m, n] = sum[k] a[m, k] * b[k, n]          // matrix product
let y[*s, o] = sum[i] x[*s, i] * w[o, i]        // linear layer, any rank
let t = sum[i] a[i, i]                          // trace
let score[b, h, q, k] = sum<f32>[d] cast<f32>(q[b, h, q, d]) * cast<f32>(k[b, h, k, d])
```

## Constraints

A `where` clause states what the shapes require. The compiler proves it at
every call; here, without `H % N == 0`, it would reject the `reshape`.

```linnet
fn split_heads<B: Dim, S: Dim, H: Dim, N: Dim, T: Float>(
    x: Tensor[B, S, H; T],
) -> Tensor[B, N, S, H / N; T]
where H % N == 0, N > 0 {
    return permute(reshape(x, [B, S, N, H / N]), [0, 2, 1, 3])
}
```

## Ops

An `op`'s body is its reference definition. A backend may substitute a
kernel that agrees with it; `linnet explain` shows where. `T?` is an
optional, `none` or `some(v)`, and a `match` must cover both.

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

## Blocks

A block holds `param` weights bound from a checkpoint, `buffer` data,
`state` such as a KV cache (kept between calls, replaced by assignment), and
`sub` child blocks or arrays of them. `= none` makes a `param` or `sub`
optional. `entry` marks what a backend exposes. `linnet inspect --parameters`
lists the parameter paths, such as `layers.0.attention.q_proj.weight`.

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

An `entry` outside any block, such as a loss, has no parameters or state
and exports on its own.

```linnet
pub entry mse<B: Dim, N: Dim>(predicted: Tensor[B, N; f32], target: Tensor[B, N; f32]) -> f32 {
    let error[b, n] = predicted[b, n] - target[b, n]
    return sum[b, n] error[b, n] * error[b, n] / cast<f32>(B * N)
}
```

## Structs

A struct groups values under field names. Calling its name builds one, every
field given by name.

```linnet
struct Moments<N: Dim, T: Float> {
    mean: Tensor[N; T]
    scale: Tensor[N; T]
}

fn moments<N: Dim, T: Float>(x: Tensor[N; T]) -> Moments<N, T> {
    return Moments(mean = x - x, scale = x * x)   // generics inferred from the fields
}

let m = moments(x)
let y = (x - m.mean) * m.scale
```

An entry that returns a struct hands the backend its fields as a tuple.

## Loops

`static for` is unrolled at compile time; `while` loops at runtime. `var`
locals carry across iterations with fixed types.

```linnet
static for layer in layers { ... }          // over a sub array, expanded at compile time
static for i in 0..Steps { ... }            // over a compile-time range; i is an i64 scalar
while running && count < MaxNew { ... }     // runtime loop over a scalar bool
```

## Slicing and built-ins

Slicing and these calls are built in, as are `exp`, `log`, `sqrt`, `rsqrt`,
`sin`, `cos`, `tanh`, `abs`, `min`, `max`, `shl`, `shr`, `cast<T>`, and
`& | ^` (logical on `bool`). The rest (`softmax`, `attention`, `rms_norm`,
`rope`, `std.random`, `std.quant`) is library code in `stdlib/`.

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

## Not in the language

No expression statements, recursion, data-dependent shapes, host-language
escapes, tensor data in source, or mutation beyond `var` locals and a
block's `state`.

# The Linnet language

Everything needed to write Linnet in one file: the workflow, the syntax, the
rules the checker enforces, and the standard library. It is meant for
people and coding agents alike. [`spec/`](spec/) has the normative rules;
where they differ, the spec wins. The package installs this file at
`share/linnet/LANGUAGE.md` under its environment (`.venv/share/linnet/` in a
uv project), and the site serves it at
`https://linnet.franknoh.dev/LANGUAGE.md`.

Linnet describes a neural network architecture as checked source. Shapes are
types, every dimension is a compile-time expression, and the compiler proves
each shape before anything runs. Weights stay in SafeTensors, bound by
parameter path. The same file runs in PyTorch, JAX and ONNX Runtime.

## Workflow

```bash
uv add "linnet-lang[torch]"                 # the compiler, its standard library, linnet.torch
linnet init my-model                        # or write one .linnet file
linnet check --json src/                    # repeat until there are no errors
linnet fmt src/                             # one canonical style
linnet inspect --parameters src/lib.linnet  # the tensors a checkpoint must supply
```

```python
from linnet.torch import load

model = load("src/lib.linnet", generics={"Vocab": 32000, "H": 512, "Heads": 8,
             "Inner": 1376, "Layers": 4}, weights="model.safetensors")
logits = model(tokens)                      # the root block's `forward` entry
```

`linnet check` prints each error with a code, the span, and a `help` line
that usually names the fix. `--json` gives the same as one document. The
[codes and their fixes](#diagnostics) are below.

## A complete model

```linnet
module tiny_lm

use std.nn.attention::{attention, causal_mask}
use std.nn.embedding::{Embedding}
use std.nn.linear::{Linear}
use std.nn.mlp::{SwiGlu}
use std.nn.norm::{RmsNorm}

// One pre-norm decoder layer.
pub block Layer<H: Dim, Heads: Dim, Inner: Dim, T: Float = bf16>
where
    Heads > 0,
    H % Heads == 0
{
    sub attn_norm: RmsNorm<H, T>
    sub qkv: Linear<H, 3 * H, T>
    sub out: Linear<H, H, T>
    sub mlp_norm: RmsNorm<H, T>
    sub mlp: SwiGlu<H, Inner, T>

    pub fn forward<B: Dim, S: Dim>(x: Tensor[B, S, H; T]) -> Tensor[B, S, H; T] {
        let projected = qkv.forward(attn_norm.forward(x))
        let q = heads<B, S, Heads, H / Heads, T>(projected[:, :, 0:H])
        let k = heads<B, S, Heads, H / Heads, T>(projected[:, :, H:2 * H])
        let v = heads<B, S, Heads, H / Heads, T>(projected[:, :, 2 * H:3 * H])
        let mixed = attention(q, k, v, rsqrt(cast<f32>(H / Heads)), some(causal_mask<S, S>()))
        let y = x + out.forward(reshape(permute(mixed, [0, 2, 1, 3]), [B, S, H]))
        return y + mlp.forward(mlp_norm.forward(y))
    }
}

pub block Model<Vocab: Dim, H: Dim, Heads: Dim, Inner: Dim, Layers: Dim, T: Float = bf16>
where
    Heads > 0,
    H % Heads == 0
{
    sub embedding: Embedding<Vocab, H, T>
    sub layers: [Layer<H, Heads, Inner, T>; Layers]
    sub norm: RmsNorm<H, T>
    sub lm_head: Linear<H, Vocab, T>

    pub entry forward<B: Dim, S: Dim>(tokens: Tensor[B, S; i32]) -> Tensor[B, S, Vocab; T] {
        var x = embedding.forward(tokens)
        static for layer in layers {
            x = layer.forward(x)
        }
        return lm_head.forward(norm.forward(x))
    }
}

// `[B, S, N * D]` to `[B, N, S, D]`.
fn heads<B: Dim, S: Dim, N: Dim, D: Dim, T: Float>(
    x: Tensor[B, S, N * D; T],
) -> Tensor[B, N, S, D; T] {
    return permute(reshape(x, [B, S, N, D]), [0, 2, 1, 3])
}
```

Its parameter paths are the PyTorch `state_dict` names a matching
`nn.Module` would have: `embedding.weight`, `layers.0.qkv.weight`,
`layers.0.mlp.gate.weight`, `lm_head.weight`, and so on.

## What differs from Python and PyTorch

Most mistakes come from these.

- **Shapes are compile-time.** Every dimension is a literal, a `Dim`
  generic, or arithmetic on them (`H / Heads`, `3 * H`). There is no
  `x.shape`, no `-1` in `reshape`, and no shape that depends on data.
- **No implicit dtype promotion.** `bf16 + f32` is an error; write
  `cast<f32>(x)`. A literal takes its context's dtype: `x * 0.5` is fine for
  any float `x`. A scalar variable is not converted: for `k: f32` and a
  tensor of dtype `T`, write `x * cast<T>(k)`.
- **Constraints are stated.** `reshape` must prove the element counts equal,
  and a division by a dimension must prove the divisor positive. State what
  the shapes rely on in `where` (`H % Heads == 0`, `Heads > 0`). A caller
  must prove its callee's `where` clauses, so restate them up the chain.
- **Broadcasting is right-aligned and must be provable**: each aligned pair
  of axes equal, or one of them `1`.
- **Index notation never sums implicitly.** Every index is an output index
  on the left or bound by a reduction (`sum[k]`).
- **No negative indices**, except the `axis` keyword of `concat`. The last
  element of an axis of size `N` is `x[N - 1]`.
- **No `self`.** Inside a block, members and methods are bare names:
  `weight`, `q_proj.forward(x)`, `step(x)`.
- **Statements.** No semicolons, one statement per line, and no expression
  statements: a call's result must be bound or returned. Every function ends
  with `return`; `return` is not allowed inside a loop.
- **Mutation.** `let` is immutable; only `var` locals and the block's own
  `state` members are assigned.
- **Booleans.** `&&`, `||` and `!` take scalar `bool`s. For boolean tensors
  use `&`, `|`, `^` (`mask ^ true` negates) and `select(mask, a, b)`. `if`
  needs a scalar `bool`, an `else`, and branches of one type. Comparisons
  combined with `&` need parentheses: `(a == b) & (i < n)`.
- **Optional arguments** take `none` or `some(value)`, never a bare value.
- **Generic arguments** are inferred, or written in declaration order:
  `heads<B, S, Heads, H / Heads, T>(x)`. Write them when an output dimension
  appears in no argument, as in `causal_mask<S, S>()`.
- **Names.** Prelude names (`max`, `min`, `exp`, `select`, `iota`, `fill`,
  `cast`, ...) cannot be redeclared, and keywords and reserved words cannot
  be names. Watch for `type`, `state`, `param`, `sub`, `buffer`, `rng`,
  `device`, `kernel`, `ref`, `mut`, `macro` and `trait`.
- **Not in the language:** recursion, struct construction, `for` (use
  `static for` or `while`), host-language escapes (`extern` is reserved),
  and tensor data in source.

## Modules and packages

One file is one module and starts with `module`. Imports name logical
paths: `std.` is the standard library, `crate.` the current package, and a
dependency's key its package.

```text
module models.mini

use std.nn.linear::{linear, Linear}       // items
use std.nn.norm::{rms_norm as norm}        // renamed
use std.random                             // the module: random.normal<N, f32>(key)
use crate.layers::{Block}                  // <package>/src/layers.linnet
```

Items are private unless `pub`. A package is a directory with
`linnet.toml`; `crate` is its `src/lib.linnet`, and `crate.a.b` is
`src/a/b.linnet`.

```toml
[package]
name = "my-model"
version = "0.1.0"
language = "0.1"

[dependencies]
shared = { path = "../shared" }
```

## Types

| Type | Written |
| --- | --- |
| scalars | `bool`, `i8` `i16` `i32` `i64`, `u8` `u16` `u32` `u64`, `f16` `bf16` `f32` `f64` |
| tensor | `Tensor[B, S, H; bf16]`: dimensions, `;`, element type. A rank-0 value is a scalar (`f32`) |
| optional | `Tensor[H; f32]?`, with values `none` and `some(x)` |
| tuple | `(Tensor[B, H; T], i32)`, destructured by `let (a, b) = f(x)` |
| block array | `[Layer<H, T>; Layers]`, only as a `sub` |
| alias | `type Hidden<B: Dim, H: Dim> = Tensor[B, H; bf16]` |
| enum | `enum Pooling { Mean, ClassToken }`, variants `Pooling.Mean` |

Generic kinds: `N: Dim` (one dimension), `*S: Shape` (any number of
dimensions, possibly none), `T: DType` (any element type), `T: Numeric`,
`T: Integer`, `T: Float`. A generic may have a default: `T: Float = bf16`.
Inside a body a `Dim` is also a value: `cast<f32>(H)`, `count < MaxNew`.

Literals: `42`, `1_000`, `0xff`, `0b1010`, `1.5`, `1e-5`, `true`, `"text"`.
Without context an integer is `i64` and a float `f64`.

## Constants and enums

```linnet
module pooling

pub enum Pooling {
    Mean,
    ClassToken,
}

pub const POOLING: Pooling = Pooling.ClassToken
pub const THETA: f32 = 10000.0
const PATCH = 16

pub fn pool<B: Dim, N: Dim, H: Dim, T: Float>(x: Tensor[B, N, H; T]) -> Tensor[B, H; T]
where N > 0 {
    return match POOLING {
        Mean => mean(x)
        ClassToken => x[:, 0, :]
    }
}

fn mean<B: Dim, N: Dim, H: Dim, T: Float>(x: Tensor[B, N, H; T]) -> Tensor[B, H; T]
where N > 0 {
    let total[b, h] = sum<f32>[n] cast<f32>(x[b, n, h])
    return cast<T>(total / cast<f32>(N))
}
```

An unannotated integer constant (`PATCH`) is a compile-time integer usable
in shapes. A `match` on a constant picks its arm at compile time.

## Functions, ops and entries

```text
fn helper<...>(...) -> R { ... }            // pure helper, may be inlined anywhere
pub op linear<...>(...) -> R { ... }        // named operation; the body defines it,
                                            // and a backend may substitute a kernel
pub entry forward<...>(...) -> R { ... }    // what a backend exposes
```

Parameters may have defaults (`eps: f32 = 1e-5`, `bias: Tensor[N; T]? =
none`). Calls take positional arguments, then named ones (`axis = -1`). A
function returns one value or a tuple. An `entry` outside any block has no
parameters and exports alone, like a loss:

```linnet
module losses

pub entry mse<B: Dim, N: Dim>(predicted: Tensor[B, N; f32], target: Tensor[B, N; f32]) -> f32 {
    let error[b, n] = predicted[b, n] - target[b, n]
    return sum[b, n] error[b, n] * error[b, n] / cast<f32>(B * N)
}
```

## Blocks

A block is a component with members and methods; it is not a value.

| Member | Meaning |
| --- | --- |
| `param weight: Tensor[Out, In; T]` | a weight bound from the checkpoint |
| `param bias: Tensor[Out; T]? = none` | an optional weight |
| `buffer mean: Tensor[C; f32]` | non-trained data bound from the checkpoint |
| `state cache: Tensor[B, N, S, D; T]` | runtime state, zeros at first, kept between entry calls |
| `sub proj: Linear<H, H, T>` | a child block |
| `sub layers: [Layer<H, T>; Layers]` | an array of child blocks |
| `sub pooler: Linear<H, H, T>? = none` | an optional child, read with `match` |

A block's `where` holds in every method and must be proven where the block
is used. `pub fn` methods are callable by the parent; `pub entry` methods
are what a backend runs. Parameter paths join member names with dots and
array positions: `layers.3.attention.q_proj.weight`.

Assigning a `state` member is the one effect in the language. A block
assigns only its own state:

```linnet
module running

pub block RunningMean<N: Dim> {
    state total: Tensor[N; f32]
    state count: Tensor[1; f32]

    pub entry update(x: Tensor[N; f32]) -> Tensor[N; f32] {
        total = total + x
        count = count + 1.0
        return total / count
    }
}
```

An optional child:

```linnet
module heads

use std.nn.linear::{Linear}

pub block Head<H: Dim, Labels: Dim, T: Float = f32> {
    sub pooler: Linear<H, H, T>? = none
    sub classifier: Linear<H, Labels, T>

    pub entry forward<B: Dim>(x: Tensor[B, H; T]) -> Tensor[B, Labels; T] {
        let pooled = match pooler {
            some(p) => p.forward(x)
            none => x
        }
        return classifier.forward(pooled)
    }
}
```

The weights decide: when they lack a parameter the child requires, the
child is absent.

## Expressions

Operators, loosest first: `||`; `&&`; `==` `!=`; `<` `<=` `>` `>=`; `|`;
`^`; `&`; `+` `-`; `*` `/` `%`. All are left-associative; unary `!` `-` `+`
bind tightest. Integer arithmetic wraps; integer `/` and `%` truncate toward
zero. Tensor comparisons give `bool` tensors.

```text
let y = if causal { masked } else { scores }      // scalar bool, same types
let z = match bias { some(b) => y + b  none => y }
```

### Index notation

An indexed `let` defines a tensor element by element. Left-hand indices are
outputs; `sum`, `prod`, `max`, `min`, `any` and `all` reduce. `sum<f32>[k]`
accumulates in `f32` (the summand must already be `f32`).

```text
let c[m, n] = sum[k] a[m, k] * b[k, n]                 // matmul
let y[*s, o] = sum[i] x[*s, i] * w[o, i]               // linear over any leading axes
let t[j, i] = x[i, j]                                  // transpose
let d[i] = a[i, i]                                     // diagonal
let picked[b] = logits[b, labels[b]]                   // gather by a runtime index
let seen[s] = positions[s] <= pos                      // a mask
let score[b, h, q, k] = sum<f32>[d] cast<f32>(qs[b, h, q, d]) * cast<f32>(ks[b, h, k, d])
```

A reduction reaches as far right as it can: `x + sum[i] a[i] * b[i]`
reduces the product. An index is only a position inside a tensor access;
use `iota(N)[i]` for its value. Every access in index notation indexes
every axis.

### Slicing

```text
x[0]                  // drops the axis; a compile-time index must be in range
x[:, 1:3]             // keeps the axis: positions 1 and 2
x[..., 0::2]          // every other element of the last axis
x[:, S - 1, :]        // the last position
x[b, pos]             // a runtime integer index reads that position
```

Slice bounds are non-negative compile-time integers; the step is a positive
constant. `...` stands for the axes not written and appears at most once.

### Prelude

These need no import:

| Call | Does |
| --- | --- |
| `cast<T>(x)` | convert the element type |
| `exp` `log` `sqrt` `rsqrt` `sin` `cos` `tanh` `(x)` | float math, elementwise |
| `abs(x)`, `min(a, b)`, `max(a, b)` | elementwise; on two compile-time integers, a dimension |
| `shl(x, bits)`, `shr(x, bits)` | integer shifts |
| `select(cond, a, b)` | elementwise choice |
| `reshape(x, [A, B])` | same element count, provably |
| `permute(x, [0, 2, 1, 3])` | reorder axes |
| `broadcast_to(x, [B, S, H])` | trailing axes equal or `1` |
| `concat(a, b, axis = -1)` | join along an axis |
| `iota<T = i64>(N)` | `0, 1, ..., N - 1` |
| `fill<T>([B, S], value)` | a constant tensor |

## Control flow

```text
static for layer in layers { x = layer.forward(x) }   // unrolled over a sub array
static for i in 0..Steps { ... }                      // unrolled; i is an i64 scalar
while running && count < MaxNew { ... }               // runtime loop over a scalar bool
```

`static for` bounds are compile-time. Its loop variable is a runtime `i64`
value: it cannot index a block array or appear in a shape. A `while` body
carries its `var`s with fixed types and may assign `state`.

Decoding one token at a time keeps the cache in `state`:

```linnet
module cached

use std.nn.attention::{attention}
use std.nn.cache::{write_at}

pub block Cache<Batch: Dim, Heads: Dim, MaxSeq: Dim, D: Dim, T: Float = bf16>
where MaxSeq > 0 {
    state keys: Tensor[Batch, Heads, MaxSeq, D; T]
    state values: Tensor[Batch, Heads, MaxSeq, D; T]

    pub entry step(
        q: Tensor[Batch, Heads, 1, D; T],
        k: Tensor[Batch, Heads, 1, D; T],
        v: Tensor[Batch, Heads, 1, D; T],
        pos: i32,
    ) -> Tensor[Batch, Heads, 1, D; T] {
        keys = write_at(keys, k, pos)
        values = write_at(values, v, pos)
        let positions = iota<i32>(MaxSeq)
        let seen[s] = positions[s] <= pos
        return attention(q, keys, values, rsqrt(cast<f32>(D)), some(reshape(seen, [1, MaxSeq])))
    }
}
```

## Standard library

Import with `use std.<module>::{...}`. Blocks take their weights as
`param`s with the PyTorch names (`weight`, `bias`), and `T` defaults to
`bf16` (`f32` for convolutions). Each block has a `forward` method.

| Module | Items |
| --- | --- |
| `std.nn.linear` | `linear(x, weight, bias?)`, `Linear<In, Out, T>` |
| `std.nn.embedding` | `embedding(ids, table)`, `Embedding<Vocab, H, T>` |
| `std.nn.norm` | `rms_norm(x, weight, eps = 1e-5)`, `layer_norm(x, weight, bias?, eps)`, `batch_norm`, `group_norm`, `RmsNorm<H, T>` |
| `std.nn.activations` | `relu`, `sigmoid`, `silu`, `gelu` (tanh), `gelu_erf` |
| `std.nn.softmax` | `softmax(x)` over the last axis |
| `std.nn.attention` | `attention(q, k, v, scale, mask?)` over `[B, H, Q, D]`, `grouped_attention` (fewer key/value heads), `causal_mask<Q, K>()`, per-row masks `attention_rows`, `grouped_attention_rows`, and `sink_attention` |
| `std.nn.rope` | `rope(x, cos_table, sin_table)`, `rope_rows`, `rotate_half` |
| `std.nn.mlp` | `swiglu(x, gate, up, down)`, `SwiGlu<H, Inner, T>` (`gate`, `up`, `down`) |
| `std.nn.cache` | `write_at(cache, value, at)` and the batched `write_span`, `write_rows`, `write_slot`, `write_slots`, `write_tokens` |
| `std.nn.conv` | `conv1d`, `conv2d`, `conv2d_rect`, `Conv1d`, `Conv2d`, `Conv2dRect`, `zero_pad*` |
| `std.nn.pool` | `max_pool2d`, `global_average_pool2d` |
| `std.nn.resize` | `upsample_nearest2d` |
| `std.nn.loss` | `log_softmax`, `cross_entropy(logits, targets, weights)`, `linear_cross_entropy(hidden, weight, targets, weights)` (no `[N, V]` logits), `token_log_probs`, `linear_token_log_probs`, `entropy` |
| `std.nn.decoding` | `argmax(x)` over the last axis |
| `std.nn.moe` | `linear_experts`, `linear_experts_shared`, `combine_experts` |
| `std.nn.parallel` | `all_reduce`, `all_gather` across tensor-parallel shards |
| `std.linalg` | `dot`, `outer`, `matmul`, `batched_matmul`, `transpose` |
| `std.random` | keys are `Tensor[2; i64]`: `split<N>`, `fold_in`, `uniform<N, T>`, `normal<N, T>`, `categorical(key, logits)` |
| `std.quant` | `Int8Linear`, `Int4Linear`, `Int4GroupLinear`, `dequantize_*`, MXFP4 experts |

The sources in `stdlib/` are short and readable; read one before guessing
its signature.

## Porting a PyTorch or Hugging Face model

- **Name members after the checkpoint.** `sub` and `param` names join into
  the paths the weights must have. Match the `state_dict` keys
  (`model.layers.0.self_attn.q_proj.weight` means a `sub model` with
  `sub layers: [...]`, and so on), or map Linnet paths to tensor names in a
  `bindings.json`. `linnet inspect --parameters` lists the paths.
- **Keep the checkpoint's layout.** Declare a `param` in the shape the file
  stores and write the contraction to match: GPT-2's `Conv1D` stores
  `[In, Out]`, so `sum[i] x[*s, i] * weight[i, o]`.
- **Tied weights.** Read the one parameter where both uses need it (an
  output head can use `embedding.weight` with `linear`), or bind both paths
  to one tensor.
- **Fixed sizes are generics.** Hidden size, heads, layers and vocabulary
  are `Dim` generics bound at load; the batch and sequence lengths of an
  entry are its own generics.
- **Decoding** keeps caches in `state` (sized by generics such as `Batch`
  and `MaxSeq`) with an entry taking one token and a position.
- **Or let it be converted.** `linnet.nest.load("org/name")` converts a
  Hub checkpoint of the Llama, Mistral, Qwen2, Qwen3, Phi-3 and GPT-2
  families from its `config.json`, and `python -m linnet.nest convert
  org/name -o dir` writes the result to edit. For anything else,
  `linnet.torch.export_linnet(module, (example,), output="model.linnet")`
  traces with `torch.export` and writes checked Linnet with the same
  parameter names, or names the operation it cannot express.

[Nest](https://nest.franknoh.dev) has 24 checked models (Llama, Qwen, Phi,
Mistral, gpt-oss, BERT, ViT, CLIP, Whisper, Stable Diffusion) to copy from.

## Diagnostics

| Code | Cause | Fix |
| --- | --- | --- |
| E1101 | unexpected token | check the syntax above; no semicolons |
| E1004 | a reserved word as a name | rename it |
| E1201 | undeclared name | import it with `use`; members are bare names |
| E1207 | a prelude name redeclared | rename it (`max`, `exp`, `select`, ...) |
| E2102 | wrong arguments | match the callee's parameters; optionals take `some(x)` |
| E2103 | dtypes differ | `cast<T>(x)` one side |
| E2110 | a generic cannot be inferred | write it: `f<A, B>(x)` |
| E2113 | `none` without an optional type | give the context a `T?` type |
| E2114 | condition not a scalar `bool` | use `select` for tensors |
| E2118 | `match` misses a case | cover every variant, or `some` and `none` |
| E2120 | no final `return`, or one inside a loop | end the function with `return` |
| E2202 | wrong shape | compare the shapes in the notes |
| E2203 | `reshape` count unprovable | add `where H % N == 0` |
| E2205 | divisor not provably positive | add `where N > 0` |
| E2206 | a callee's `where` unprovable | restate it in the caller's `where` |
| E2207 | broadcast unprovable | make the axes equal or `1`, or state their equality |
| E2209 | invalid index or slice | no negative indices; bounds within the axis |
| E3101 | index neither output nor reduced | add it to the left or reduce it with `sum[i]` |
| E3102, E3103 | an index unused | remove it |
| E4102 | a `param` with a value | parameters come from the checkpoint only |

Warnings (W1001 unused import, W1002 unused local, W1003 unused member)
fail only `linnet lint`. Every code is in
[docs/diagnostics](docs/diagnostics/README.md).

## Commands

| Command | Does |
| --- | --- |
| `linnet check [--json] [--strict] <paths>` | check files and what they import |
| `linnet lint <paths>` | `check --strict` |
| `linnet fmt [--check] <paths>` | format in place |
| `linnet inspect --parameters [--json] <file>` | the parameter manifest |
| `linnet explain <file>` | which kernel each library operation gets |
| `linnet torch\|jax\|onnx\|stablehlo --entry <name> --bind G=v ... <file>` | export one entry |
| `linnet init <name>` | a new package |
| `linnet serve <model>` | serve a model with OpenAI's API (the `serve` extra) |
| `linnet memory <model>`, `linnet fit <model>` | the memory a configuration needs, before running it; the largest batch or context that fits |

`--std <dir>` or `LINNET_STD` points at another standard library.

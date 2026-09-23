# Attention

Scaled dot-product attention and a softmax written from scratch with index
notation, reductions, and an explicit accumulation dtype. Nothing here is a
primitive: `softmax_last` is `max`, `exp`, and `sum` over the last axis, and
`attention` is two contractions around it.

## Reductions with a dtype

```linnet
let score[b, h, q, k] = sum<f32>[d] cast<f32>(query[b, h, q, d]) * cast<f32>(key[b, h, k, d])
```

`sum<f32>[d]` accumulates in `f32` whatever `T` is, so `bf16` inputs get
`f32` scores — the choice is in the source, not in a backend flag.

## Softmax over the last axis

`softmax_last<*S, N>` subtracts the row maximum, exponentiates, and divides
by the row sum; `*S` keeps it shape-generic. The standard library's
`std.nn.softmax::softmax` is the same code, and a backend may replace it with
a fused kernel (`linnet explain` shows when).

## Run it

```bash
linnet check examples/02-attention/attention.linnet
linnet explain examples/02-attention/attention.linnet
```

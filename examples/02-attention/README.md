# Attention

Softmax and scaled dot-product attention in index notation, with no
primitives: `attention` is two contractions around `softmax_last`.

## Commands

```bash
linnet check examples/02-attention/attention.linnet
linnet explain examples/02-attention/attention.linnet
```

## Key ideas

### Accumulation dtype

```linnet
let score[b, h, q, k] = sum<f32>[d] cast<f32>(query[b, h, q, d]) * cast<f32>(key[b, h, k, d])
```

In `attention`, `sum<f32>[d]` accumulates in `f32` whatever `T` is, so
`bf16` inputs get `f32` scores. The source sets this, not a backend flag.

### Softmax from reductions

`softmax_last<*S, N>` subtracts the row `max`, applies `exp`, and divides by
the row `sum`, for any leading shape `*S`. `std.nn.softmax::softmax` is the
same code, and a backend may swap it for a fused kernel. `linnet explain`
shows when.

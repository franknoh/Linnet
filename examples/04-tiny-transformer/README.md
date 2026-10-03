# Tiny transformer

A decoder-only transformer built from the standard library's `Embedding`,
`RmsNorm`, `Linear`, `rope`, `attention` and `swiglu`. These are ordinary
Linnet in `stdlib/`, so the whole model is inspectable source down to the
arithmetic.

## Commands

```bash
linnet check --std stdlib examples/04-tiny-transformer
linnet stablehlo --std stdlib --bind Vocab=11 --bind H=8 --bind Heads=2 --bind Inner=16 \
                 --bind Layers=2 --bind T=f32 --bind B=2 --bind S=5 \
                 examples/04-tiny-transformer/src/lib.linnet
```

The round-trip tests run this model in PyTorch, XLA, JAX and ONNX Runtime
with matching outputs.

## Key ideas

### Package and imports

The package is `linnet.toml` plus `src/lib.linnet`.
`use std.nn.attention::{attention, causal_mask}` imports from the standard
library directory passed as `--std`. A `crate.` import would reach sibling
modules.

### Divisibility constraints

Every block and helper in `src/lib.linnet` declares:

```linnet
where
    H % Heads == 0,
    Heads > 0,
    (H / Heads) % 2 == 0
```

The head split `reshape(x, [B, S, Heads, H / Heads])` and the rotary
embedding over half the head dimension check only because of these facts.
Remove one, and `linnet check` names the reshape that no longer follows.

### Rotary tables as inputs

`Model.forward` takes `cos_table` and `sin_table` of shape
`[S, H / Heads]` from the caller. `05-llama` computes them from `iota`
instead.

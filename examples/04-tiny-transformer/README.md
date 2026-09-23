# Tiny transformer

A decoder-only transformer assembled from the standard library: `Embedding`,
`RmsNorm`, `Linear`, `rope`, `attention`, `swiglu`. Every one of those is
ordinary Linnet in `stdlib/`, so the whole model is inspectable source down
to the arithmetic. This is the model the round-trip tests run in PyTorch,
XLA, JAX, and ONNX Runtime with matching outputs.

## A package

The example is a package (`linnet.toml` plus `src/lib.linnet`): `use
std.nn.attention::{attention, causal_mask}` imports from the standard
library directory (`--std`), and a `crate.` import would reach sibling
modules.

## Divisibility as a constraint

```linnet
where
    H % Heads == 0,
    Heads > 0,
    (H / Heads) % 2 == 0
```

The head split `reshape(x, [B, S, Heads, H / Heads])` and rotary embedding
over half the head dimension only check because these facts are declared;
remove one and `linnet check` names the reshape that no longer follows.

## Rotary tables as inputs

`forward` takes `cos_table` and `sin_table` of shape `[S, H / Heads]` — this
version leaves them to the caller. The Llama example computes them from
`iota` inside the model instead.

## Commands

```bash
linnet check --std stdlib examples/04-tiny-transformer
linnet stablehlo --std stdlib --bind Vocab=11 --bind H=8 --bind Heads=2 --bind Inner=16 \
                 --bind Layers=2 --bind T=f32 --bind B=2 --bind S=5 \
                 examples/04-tiny-transformer/src/lib.linnet
```

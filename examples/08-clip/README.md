# CLIP

A dual encoder as a package of four modules: one `Encoder` shared by a
vision tower and a text tower, and a root block exposing three entries over
the same parameters — `embed_image`, `embed_text`, and `similarity`.

## One encoder, two towers

`encoder.linnet` defines `Encoder<D, Heads, Inner, Layers, T>`; `vision.linnet`
runs it without a mask (`none`) over patch tokens, `text.linnet` with
`some(causal_mask(...))` over token embeddings. `lib.linnet` imports both with
`crate.` paths.

## Three entries, one parameter set

`embed_image` and `embed_text` return L2-normalized embeddings;
`similarity<B, C, S>` computes `[images, texts]` cosine similarities scaled by
a learned temperature stored as `Tensor[1; f32]`. A backend exposes each
entry as a separate function over the same weights.

## Commands

```bash
linnet lint --std stdlib examples/08-clip
linnet inspect --parameters --std stdlib examples/08-clip/src/lib.linnet
linnet stablehlo --std stdlib --entry similarity --bind Height=32 --bind Width=32 --bind Channels=3 \
                 --bind Patch=8 --bind Vocab=100 --bind Context=16 --bind D=64 --bind Heads=4 \
                 --bind Inner=128 --bind Layers=2 --bind Embed=32 --bind T=f32 \
                 --bind B=1 --bind C=3 --bind S=8 examples/08-clip/src/lib.linnet
```

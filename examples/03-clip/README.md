# CLIP

A CLIP-style dual encoder: a package of four modules whose two towers share
one encoder, with three entries over one parameter set.

## Commands

```bash
linnet lint --std stdlib examples/03-clip
linnet inspect --parameters --std stdlib examples/03-clip/src/lib.linnet
linnet stablehlo --std stdlib --entry similarity --bind Height=32 --bind Width=32 --bind Channels=3 \
                 --bind Patch=8 --bind Vocab=100 --bind Context=16 --bind D=64 --bind Heads=4 \
                 --bind Inner=128 --bind Layers=2 --bind Embed=32 --bind T=f32 \
                 --bind B=1 --bind C=3 --bind S=8 examples/03-clip/src/lib.linnet
```

## Key ideas

### One encoder, two towers

`src/encoder.linnet` defines `Encoder<D, Heads, Inner, Layers, T>`.
`src/vision.linnet` runs it over patch tokens without a mask (`none`), and
`src/text.linnet` over token embeddings with `some(causal_mask(...))`.
`src/lib.linnet` imports both towers with `crate.` paths.

### Three entries, one parameter set

The root block `Clip` has three entries. `embed_image` and `embed_text`
return L2-normalized embeddings. `similarity<B, C, S>` contracts them into
`[images, texts]` cosine similarities, scaled by a learned temperature
stored as `Tensor[1; f32]`. A backend exposes each entry as a separate
function over the same weights.

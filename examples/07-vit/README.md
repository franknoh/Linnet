# Vision Transformer

Images in, classes out: patches cut with `reshape` and `permute` whose
element counts the checker proves from `Height % Patch == 0`, a class token
concatenated after `broadcast_to`, a position table sized by an expression,
unmasked attention, and a pooling strategy chosen by matching a compile-time
`enum` constant.

## Patchify is a reshape the checker trusts

`patchify<B, C, Height, Width, P, T>` reshapes `[B, C, Height, Width]` to
`[B, (Height / P) * (Width / P), C * P * P]` through a `permute`. With
`Height % Patch == 0` and `Width % Patch == 0` declared, the solver proves
the element counts match; without them it refuses.

## Sizes as expressions

```linnet
param positions: Tensor[(Height / Patch) * (Width / Patch) + 1, D; T]
```

The position table has one row per patch plus the class token — a shape
expression, checked against the concatenation that uses it.

## A choice made at compile time

```linnet
pub enum Pooling { ClassToken, Mean }
pub const POOLING: Pooling = Pooling.ClassToken
```

`match POOLING { ... }` picks the pooled representation; because the value is
a constant, the emitter and the exporters keep only the chosen arm.

## Run it

```bash
linnet check --std stdlib examples/07-vit/vit.linnet
linnet stablehlo --std stdlib --bind Height=32 --bind Width=32 --bind Channels=3 --bind Patch=8 \
                 --bind D=64 --bind Heads=4 --bind Inner=128 --bind Layers=2 --bind Classes=10 \
                 --bind T=f32 --bind B=1 examples/07-vit/vit.linnet
```

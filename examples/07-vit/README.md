# Vision Transformer

A Vision Transformer that maps images to class logits, with unmasked
attention (`none`). It shows reshapes the checker proves, shapes sized by
expressions, and pooling chosen at compile time.

## Commands

```bash
linnet check --std stdlib examples/07-vit/vit.linnet
linnet stablehlo --std stdlib --bind Height=32 --bind Width=32 --bind Channels=3 --bind Patch=8 \
                 --bind D=64 --bind Heads=4 --bind Inner=128 --bind Layers=2 --bind Classes=10 \
                 --bind T=f32 --bind B=1 examples/07-vit/vit.linnet
```

## Key ideas

### Patchify by reshape

`patchify<B, C, Height, Width, P, T>` reshapes `[B, C, Height, Width]` to
`[B, (Height / P) * (Width / P), C * P * P]` through a `permute`. With
`Height % Patch == 0` and `Width % Patch == 0` declared, the solver proves
the element counts match. Without them, it refuses.

### Sizes as expressions

```linnet
param positions: Tensor[(Height / Patch) * (Width / Patch) + 1, D; T]
```

`Model.forward` expands the class token to `[B, 1, D]` with `broadcast_to`
and concatenates it in front of the patches. The position table has one row
per patch plus one for the class token, and the checker matches it against
that concatenation.

### Compile-time choice

```linnet
pub enum Pooling { ClassToken, Mean }
pub const POOLING: Pooling = Pooling.ClassToken
```

`match POOLING { ... }` picks the pooled representation. `POOLING` is a
constant, so the emitter and the exporters keep only the chosen arm.

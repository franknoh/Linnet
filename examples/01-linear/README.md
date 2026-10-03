# Linear

A linear layer as an `op`, and a `block` that owns its weight and calls the
op. Later examples build on both.

## Commands

```bash
linnet check examples/01-linear/linear.linnet
linnet inspect --parameters examples/01-linear/linear.linnet
linnet inspect --core-ir examples/01-linear/linear.linnet
```

`--parameters` lists `weight: Tensor[Out, In; T]` and the optional `bias`.
`--core-ir` shows the contraction as a `comprehension` with a `sum`
reduction. Exporting needs a block with an `entry`, which later examples add.

## Key ideas

### Shape packs

In `linear<*S, In, Out, T>`, the shape pack `*S` is any number of leading
axes. One op serves `[B, In]` and `[B, S, In]`, and the checker infers the
result `Tensor[*S, Out; T]` without running anything.

### Index notation

```linnet
let y[*s, o] = sum[i] x[*s, i] * weight[o, i]
```

`sum[i]` sums over `i`. The left side names the result's axes, `*s` and
`o`. A backend with a `matmul` maps this to it.

### Optional parameters

`bias: Tensor[Out; T]? = none` is a parameter a checkpoint may leave out.
`linear` `match`es on it, and the checker requires both arms.

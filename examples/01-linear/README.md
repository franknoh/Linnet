# Linear

The smallest useful Linnet file: a linear layer as an `op` and as a `block`
that owns its weight. It shows what every other example is built from —
generic dimensions, index notation, an optional parameter, and a block whose
method calls the op.

## Shapes are names

`linear<*S, In, Out, T>` takes `x: Tensor[*S, In; T]` and a weight
`Tensor[Out, In; T]`. `*S` is a shape pack — any number of leading axes —
so the same op serves `[B, In]` and `[B, S, In]` inputs, and the checker
knows the result is `Tensor[*S, Out; T]` without running anything.

## The contraction is written out

```linnet
let y[*s, o] = sum[i] x[*s, i] * weight[o, i]
```

Index notation names every axis; `i` is summed because a `sum[i]` binds it,
and `o` and `*s` are the result's axes because they appear on the left. A
backend maps this to `matmul` when it has one.

## An optional bias

`bias: Tensor[Out; T]? = none` is a parameter a checkpoint may leave out;
the body `match`es on it, and the checker insists both arms exist.

## Commands

```bash
linnet check examples/01-linear/linear.linnet
linnet inspect --parameters examples/01-linear/linear.linnet
linnet inspect --core-ir examples/01-linear/linear.linnet
```

`inspect --parameters` lists `weight: Tensor[Out, In; T]` and the optional
`bias`; `--core-ir` shows the contraction as a `comprehension` with a `sum`
reduction. The exporters need a block with an `entry`, which the next
examples add.

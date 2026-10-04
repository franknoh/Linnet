# Training a model from scratch

The ViT example (`02-vit`) trained from scratch on a task it learns in about
a minute on a CPU: which quadrant of a 32x32 image holds a bright square.

## Run

```bash
python examples/05-train-vit/train.py            # on a GPU when there is one
python examples/05-train-vit/train.py --device cpu
```

It prints the loss and the held-out accuracy every 50 steps, and reaches
over 0.95 in 300 steps.

## What it shows

- **A Linnet model is a `torch.nn.Module`.** `linnet.torch.load` compiles
  the `.linnet` source; `trainable=True` makes its parameters require
  gradients. A plain PyTorch loop trains it: `model(images)`, a loss,
  `backward()` and an optimizer.
- **The source fixes the structure; the script fixes the sizes.** The
  generics (`D`, `Heads`, `Layers`, `Patch`, ...) bind at load. The checker
  proves the patch reshapes and the head split for those values before
  anything runs.
- **Weights are data.** Loaded without weights, the parameters are zeros:
  the script initializes them by path. `model.save_weights(path)` writes them
  under the same paths for `load(..., weights=path)` to read back.

`compile=True` runs the generated PyTorch source; `compile="inductor"`
passes it through `torch.compile` as well. The same source trains in JAX
with `linnet.jax.load_source` and `jax.grad`
([JAX](https://linnet.franknoh.dev/docs/jax#training)).

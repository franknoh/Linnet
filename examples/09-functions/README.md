# Functions

Entries declared at module level, outside any block, for the training or
evaluation loop around a model. With no parameters or state, each is a
function of its inputs alone.

## Commands

```bash
linnet stablehlo --std stdlib --entry cross_entropy --bind B=8 --bind V=10 --bind T=bf16 \
    examples/09-functions/functions.linnet
linnet plan --functions --std stdlib examples/09-functions/functions.linnet
```

```python
from linnet.torch import load, load_function

model = load("functions.linnet", generics={"In": 64, "Classes": 10}, trainable=True)
cross_entropy = load_function("functions.linnet", "cross_entropy")
cross_entropy(model(x), labels).backward()
```

`linnet.jax.load_function` returns the same functions as `jax.numpy` code
that composes with `jax.grad`, `jax.jit` and `jax.vmap`.
`linnet.onnx.export_function` writes one as a self-contained ONNX model.

## Key ideas

### Module-level entries

| Entry | Computes |
| --- | --- |
| `cross_entropy<B, V, T>` | the mean cross-entropy of logits in any float dtype against class labels, in f32 |
| `normalize_images<B, H, W, C>` | `u8` pixels in height-width-channel order to normalized f32 in channel-height-width order |
| `token_log_probs<B, S, V, T>` | the log-probability of each produced token, for a policy-gradient update |
| `match_reward<B, S>` | the fraction of each reference reproduced, less `penalty` (a scalar `f32` input) for the fraction missed |

### Calling a function

A block's entries can call these functions. `Classifier.loss` returns
`cross_entropy(head.forward(x), labels)`, so the loss runs in the same
program as the logits.

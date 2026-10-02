# Functions

Entries declared at module level, outside any block. They have no
parameters or state, so each is a function of its inputs alone -- the parts
of a training or evaluation loop around a model:

| Entry | Computes |
| --- | --- |
| `cross_entropy<B, V, T>` | the mean cross-entropy of logits in any float dtype against class labels, in f32 |
| `normalize_images<B, H, W, C>` | `u8` pixels in height-width-channel order to normalized f32 in channel-height-width order |
| `token_log_probs<B, S, V, T>` | the log-probability of each produced token, for a policy-gradient update |
| `match_reward<B, S>` | the fraction of each reference reproduced, less `penalty` (a scalar `f32` input) for the fraction missed |

`Classifier` shows a model's entry calling one: its `loss` entry is the
cross-entropy of its own logits.

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
that `jax.grad`, `jax.jit`, and `jax.vmap` compose with, and
`linnet.onnx.export_function` writes one as a self-contained ONNX model.

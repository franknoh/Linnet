# Randomness

Linnet has no random primitive. `std.random` is a counter-based PRNG written
in Linnet — Threefry-2x32, the algorithm behind `jax.random` — so a model
that samples is still pure source, every backend draws exactly the same
numbers, and a key is ordinary data: `Tensor[2; i64]` holding two 32-bit
words.

```linnet
use std.random::{categorical, normal, split, uniform}

let keys = split<Steps>(key)          // Steps derived keys, one per step
let noise = normal<N, f32>(keys[0, :])
let u = uniform<N, bf16>(fold_in(key, 3))
let token = categorical(key, logits)  // one draw per row, ∝ exp(logits)
```

| Function | Result |
| --- | --- |
| `threefry2x32<N>(key, counter: Tensor[N, 2; i64])` | the raw block function: `N` output pairs |
| `bits<N>(key)` | `N` random 32-bit words (as `i64`) |
| `split<N>(key)` | `N` new keys |
| `fold_in(key, data: i32)` | a key derived from `key` and an integer |
| `uniform<N, T>(key)` | `N` numbers in `[0, 1)` |
| `normal<N, T>(key)` | `N` standard normal numbers (Box–Muller) |
| `categorical<B, V, T>(key, logits)` | one category per row, Gumbel-max |

Keys made with `jax.random.key(seed)` (`key_data` gives the two words) give
the same `split`, `bits`, `uniform`, and `fold_in` results as JAX itself, bit
for bit, under JAX's default partitionable Threefry layout; the test suite
checks this through the PyTorch materializer. `normal` and `categorical` use
the same bits but their own arithmetic, so they agree across Linnet backends,
not with JAX's versions.

The arithmetic is 64-bit integers masked to 32 bits with the operators `&`,
`|`, `^` and the builtins `shl`/`shr`, so it needs nothing beyond integer
tensors: it runs in the PyTorch interpreter and generated code, under XLA,
and in ONNX Runtime alike. Threading keys explicitly is the whole design:
there is no hidden generator, a `static for` loop over `split<Steps>(key)`
gives each step its own key, and reproducing a run means passing the same
key. The Llama example's `sample<Steps>(token, pos, key, temperature)` entry
does exactly that.

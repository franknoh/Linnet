# Randomness

`std.random` is a counter-based PRNG written in Linnet. There is no random
primitive: a key is data (`Tensor[2; i64]`, two 32-bit words), Threefry-2x32
turns keys and counters into bits, and every backend draws the same numbers.

```linnet
use std.random::{categorical, fold_in, normal, split, uniform}

let keys = split<Steps>(key)          // one derived key per step
let noise = normal<N, f32>(keys[0, :])
let u = uniform<N, bf16>(fold_in(key, 3))
let token = categorical(key, logits)  // one draw per row, proportional to exp(logits)
```

| Function | Result |
| --- | --- |
| `threefry2x32<N>(key, counter: Tensor[N, 2; i64])` | `N` output pairs |
| `bits<N>(key)` | `N` random 32-bit words as `i64` |
| `split<N>(key)` | `N` new keys |
| `fold_in(key, data: i32)` | a key derived from `key` and an integer |
| `uniform<N, T>(key)` | `N` numbers in `[0, 1)` |
| `normal<N, T>(key)` | `N` standard normal numbers (Box-Muller) |
| `categorical<B, V, T>(key, logits)` | one category per row (Gumbel-max) |

## Agreement with JAX

For a key from `jax.random.key(seed)` (`key_data` gives the two words),
`split`, `bits`, `uniform`, and `fold_in` return exactly what `jax.random`
returns under its default partitionable Threefry layout. The test suite
checks this through the PyTorch materializer. `normal` and `categorical` use
the same bits with their own arithmetic, so they agree across Linnet backends
but not with JAX's versions.

## Why a library

The arithmetic is 64-bit integers masked to 32 bits with `&`, `|`, `^`,
`shl`, and `shr`, so it runs wherever integer tensors do: the PyTorch
interpreter and generated code, XLA, ONNX Runtime. Keys are threaded
explicitly, so there is no hidden generator, and reproducing a run means
passing the same key. The Llama example's `sample<Steps>(token, pos, key,
temperature)` entry splits one key per step.

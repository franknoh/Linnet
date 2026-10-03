# Randomness

`std.random` draws the same random numbers on every backend from keys you
pass; there is no hidden generator, so the same key reproduces a run. A key
is a `Tensor[2; i64]` of two 32-bit words, expanded with Threefry-2x32.

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
`split`, `bits`, `uniform`, and `fold_in` match `jax.random` exactly under
its default partitionable Threefry layout. `normal` and `categorical` agree
across Linnet backends but not with JAX.

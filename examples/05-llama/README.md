# Llama

A Llama-style decoder with grouped-query attention, rotary positions,
SwiGLU, a KV cache, and six entries from a full forward pass to generation
loops. Its dtype generic `T` defaults to `bf16`. The benchmarks measure this
model.

## Commands

```bash
linnet lint --std stdlib examples/05-llama
linnet inspect --parameters --std stdlib examples/05-llama/src/lib.linnet
linnet stablehlo --std stdlib --entry decode --bind Vocab=32000 --bind H=512 --bind Heads=8 \
                 --bind KvHeads=8 --bind Inner=1376 --bind Layers=8 --bind Batch=1 \
                 --bind MaxSeq=512 --bind T=bf16 examples/05-llama/src/lib.linnet
```

```python
from linnet.torch import load
model = load("examples/05-llama/src/lib.linnet", generics={...}, weights="weights/",
             numerics="equivalent", compile="inductor")
tokens = model.run_entry("generate", [prompt, torch.tensor(0, dtype=torch.int32)],
                         generics={"Steps": 32})
```

## Key ideas

### Modules and rotary tables

`src/lib.linnet` imports `GroupedQueryAttention` from `src/attention.linnet`,
which imports `THETA` and `tables` from `src/rope.linnet`, both through
`crate.` imports. `THETA` is the base frequency, a `pub const`.
`tables<S, D, T>` computes the cos and sin tables at compile time from
`iota`, so the model needs no table inputs.

### Grouped-query attention

`GroupedQueryAttention<H, Heads, KvHeads, Batch, MaxSeq, T>` projects
`KvHeads` key/value heads. `std.nn.attention::grouped_attention` repeats
each for `Heads / KvHeads` query heads with `broadcast_to` and a reshape.
The block's `where` clause states the divisibility the reshapes rely on.

### KV cache in `state`

```linnet
state cache_k: Tensor[Batch, KvHeads, MaxSeq, H / Heads; T]
state cache_v: Tensor[Batch, KvHeads, MaxSeq, H / Heads; T]
```

`decode(x, pos)` writes the new key and value at `pos` with
`std.nn.cache::write_at`, a masked `select`, then attends over positions
`<= pos`. The runtime keeps the caches between calls; PyTorch holds them in
buffers, reset with `reset_state()`. Graph exports thread them in and out as
extra arguments and results.

### Entries

| Entry | Returns |
| --- | --- |
| `forward` | logits for every position |
| `next_token` | last-position logits, sliced with `S - 1` |
| `decode` | logits for one token at `pos`, through the per-layer KV caches |
| `generate<Steps>` | `Steps` greedy tokens from `std.nn.decoding::argmax` |
| `sample<Steps>` | `Steps` tokens drawn with a key |
| `generate_until<MaxNew>` | up to `MaxNew` greedy tokens and their count |

`generate` loops with `static for i in 0..Steps`, expanded at compile time,
so the whole generation is one graph. `generate_until` loops
`while running && count < MaxNew` at runtime over scalar carried values,
and stops once every row has produced the end token. It exports as
`stablehlo.while`, an ONNX `Loop`, or a Python loop.

`sample` splits `key` into one key per step and draws with
`std.random::categorical`. The same key gives the same tokens in PyTorch
and under XLA.

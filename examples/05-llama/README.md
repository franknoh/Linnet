# Llama

A Llama-style decoder with grouped-query attention, rotary positions computed
from `iota`, SwiGLU, and six entries that show most of the language: a full
forward pass, last-position logits, a KV-cache `decode` step with `state`
members, greedy `generate` with a compile-time range loop, keyed `sample`
with `std.random`, and `generate_until`, a runtime `while` loop that stops at
an end token. It is the model the benchmarks measure.

## Grouped-query attention

`GroupedQueryAttention<H, Heads, KvHeads, Batch, MaxSeq, T>` projects
`KvHeads` key/value heads and repeats each for `Heads / KvHeads` query heads
with `broadcast_to` and a reshape — a `where` clause states the divisibility
the reshapes rely on.

## State: the KV cache

```linnet
state cache_k: Tensor[Batch, KvHeads, MaxSeq, H / Heads; T]
state cache_v: Tensor[Batch, KvHeads, MaxSeq, H / Heads; T]
```

`decode(x, pos)` writes the new key and value at `pos` with a masked
`select` and attends over positions `<= pos`. The runtime keeps the caches
between calls (PyTorch: buffers with `reset_state()`); graph exports thread
them in and out as extra arguments and results.

## Loops

`generate<Steps>` is `static for i in 0..Steps` — expanded at compile time,
so the whole generation is one graph. `generate_until<MaxNew>` is `while
running && count < MaxNew`, a runtime loop over scalar carried values, and
exports as `stablehlo.while`, an ONNX `Loop`, or a Python loop.

## Sampling with a key

`sample<Steps>(token, pos, key, temperature)` splits the key once per step
and draws with `std.random::categorical`; the same key gives the same tokens
in PyTorch and under XLA.

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

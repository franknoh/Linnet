# Backend plugins

A backend runs part of a Linnet model on a library of its own
([Backends](https://linnet.franknoh.dev/docs/backends)). Each directory here
is a package of its own that plugs one library in through the
`linnet.backends` entry point, without changes to Linnet:

| Plugin | Library | Takes |
| --- | --- | --- |
| `deepgemm` | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | block-scaled FP8 `linear` calls (`Fp8BlockLinear`, blocks of 128) under `numerics="fast"`, on Hopper |
| `bitsandbytes` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | NF4 `linear` calls (`Nf4Linear`), on CUDA |
| `flash-attn` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | decode attention over a KV cache (one query a row, a prefix mask), on Ampere and later |

Install one beside Linnet and its library:

```bash
pip install -e examples/09-backends/deepgemm
```

`linnet.backends.available()` then lists it, and `model.plan_for(entry,
inputs)` offers it every region it matches. `model.conform(entry, inputs,
backend)` checks it against the generated source.

## DeepGEMM

DeepGEMM is built from source against the CUDA toolkit PyTorch was built
with:

```bash
git clone --recursive --branch v2.1.1.post3 https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && ./install.sh
```

The plugin takes a `std.quant::linear_fp8_blocks` call whose input the
export may round to FP8 (`numerics="fast"`). It rounds each 128 inputs of a
row as Linnet's own FP8 path does and multiplies with
`deep_gemm.fp8_gemm_nt`. On one H100, Qwen3 8B FP8 prefills 2,048 tokens in
35 ms with the measured plan, against 46 ms for the generated source alone.

For a model loaded with `numerics="fast"` and `compile=True`, such as
Nest's `qwen3-8b-fp8`:

```python
from linnet_deepgemm import DeepGemm

plan = model.plan_for("prefill", [tokens, pos], backends=[DeepGemm()])
```

## bitsandbytes

Compiled PyTorch widens an `Nf4Linear` weight once and keeps the 16-bit
copy, four times the bytes of the codes. The plugin takes the layer's
`std.quant::linear_nf4` call to `bitsandbytes.matmul_4bit` instead, which
multiplies a row straight from the 4-bit codes. bitsandbytes reads the
codes in another order (the first of each pair in the high four bits), so
the plugin keeps them once more in that order: half a byte a weight, which
the plan's memory budget counts. On an RTX A5000, an 8192 by 8192 layer
holds 68 MiB in the plan against 172 MiB in the entry, at the same speed
for one row.

```bash
pip install -e examples/09-backends/bitsandbytes
```

## FlashAttention

The plugin takes a decode step's attention, one query a row over the KV
cache, to `flash_attn_with_kvcache`. It reads Linnet's cache as it is and
each row's length from the mask on the device, so a plan captured as one
CUDA graph replays it at any position.

```bash
pip install flash-attn --no-build-isolation
pip install -e examples/09-backends/flash-attn
```

## Checking them

`python/linnet/tests/test_backend_examples.py` checks each plugin here: what
it matches on any machine, and on a GPU with its library installed, its
results against the entry's.

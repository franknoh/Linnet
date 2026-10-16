# Backend plugins

A backend runs part of a Linnet model on a library of its own
([Backends](https://linnet.franknoh.dev/docs/backends)). Each directory here
is a package of its own that plugs one library in through the
`linnet.backends` entry point, without changes to Linnet:

| Plugin | Library | Takes |
| --- | --- | --- |
| `deepgemm` | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | block-scaled FP8 `linear` calls (`Fp8BlockLinear`, blocks of 128) under `numerics="fast"`, on Hopper |
| `bitsandbytes` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | NF4 `linear` calls (`Nf4Linear`), on CUDA |
| `marlin` | [vLLM](https://github.com/vllm-project/vllm)'s Marlin | group-wise int4 `linear` calls (`Int4GroupLinear`: GPTQ and AWQ), on Ampere and later |
| `flash-attn` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) 2 and 3 | decode attention over a KV cache, packed prompts, and paged decode and prompts, on Ampere and later (3 on Hopper) |

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

## Marlin

GPTQ and AWQ checkpoints import into `Int4GroupLinear`
([Quantization](https://linnet.franknoh.dev/docs/quantization)), which
compiled PyTorch multiplies with its own int4 product on CUDA. The plugin
takes the layer's `std.quant::linear_int4_groups` call to vLLM's
`gptq_marlin_gemm` instead. It repacks the weight into Marlin's tiles once
(half a byte a weight, which the plan's memory budget counts) and takes
groups of 32, 64 or 128 inputs, or one a row, with an output count a
multiple of 64 and an input count a multiple of 128, in `bf16` or `f16`.
A GPTQ layer's zero of 8 and an AWQ layer's zero points take the same
path. On one H100, a measured plan of a 14336 by 4096 layer takes Marlin
from 1 to 256 rows (3.5 to 5 times the generated source's speed at 16 to
64) and leaves 1,024 rows to the generated source's dequantized product.

The plugin is written against vLLM 0.11.0 (PyTorch 2.8), whose Marlin
functions it calls; later releases renamed them.

```bash
pip install vllm==0.11.0
pip install -e examples/09-backends/marlin
```

## FlashAttention

The plugin takes four kinds of attention, each reading Linnet's cache as
it is and every length on the device, so a plan captured as one CUDA graph
replays it at any position:

- a decode step's attention, one query a row over the KV cache, to
  `flash_attn_with_kvcache`, each row's length from the mask;
- a packed pass of prompts (`prefill_packed`, the prompts of
  `step_packed`), each token seeing its own prompt up to itself, to
  `flash_attn_varlen_func`, the prompts' bounds from the segments; the
  pass's mask is then never built;
- `paged_attention` and `paged_prefill_attention` over a pool of pages, to
  the same two with the rows' page table. FlashAttention 2 reads pages of
  a multiple of 256 positions: load the model with `PageSize=256`.

On Hopper with FlashAttention 3 installed, the plugin runs every kind on
it instead. FlashAttention 3 reads pages of any size, the default 64
included, and takes a paged pass of prompts as `flash_attn_with_kvcache`
with the prompts' bounds. `FlashAttention(three=False)` keeps
FlashAttention 2.

```bash
pip install flash-attn --no-build-isolation
pip install -e examples/09-backends/flash-attn
```

FlashAttention 3 is built from the repository's `hopper` directory:

```bash
git clone --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && python setup.py install
```

## Checking them

`python/linnet/tests/test_backend_examples.py` checks each plugin here: what
it matches on any machine, and on a GPU with its library installed, its
results against the entry's.

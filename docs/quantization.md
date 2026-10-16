# Quantization

Run 8-bit (int8 and FP8, scaled per row, per group or per block), int4,
group-wise 4-bit (GPTQ, AWQ), NF4 (bitsandbytes) and MXFP4 weights on every
backend. A quantized weight is an ordinary parameter, a tensor of a narrow
format (`i8`, or `std.dtypes`' `f8e4m3`, `i4`, `u4`, `f4e2m1`, `nf4`) plus
scales, that `std.quant` casts and scales. The arithmetic dtype `T` is
separate from the storage.

```linnet
use std.quant::{Int4Linear, Int8Linear}

pub block Mlp<H: Dim, Inner: Dim, T: Float = bf16>
where H % 2 == 0 {
    sub up: Int4Linear<H, Inner, T>      // weight: Tensor[Inner, H; i4], scale: Tensor[Inner, 1; T]
    sub down: Int8Linear<Inner, H, T>    // weight: Tensor[H, Inner; i8], scale: Tensor[H, 1; T]
    ...
}
```

The 4-bit formats hold two values a byte along the last axis, the even
element in the low nibble; a checkpoint stores them as those bytes (`U8`,
the last axis halved).

| `std.quant` | Format |
| --- | --- |
| `Int8Linear<In, Out, T>` | `weight: Tensor[Out, In; i8]`, `scale: Tensor[Out, 1; T]`, optional bias |
| `Int8GroupLinear<In, Out, Group = 128, T>` | `weight: Tensor[Out, In; i8]`, `scale: Tensor[Out, In / Group; T]`, optional bias |
| `Int4Linear<In, Out, T>` | `weight: Tensor[Out, In; i4]`, `scale: Tensor[Out, 1; T]`, optional bias |
| `Int4GroupLinear<In, Out, Group = 128, T>` | `weight: Tensor[Out, In / Group, Group; u4]`, `scale` and `zero: [Out, In / Group]`, optional `order: [In; i32]` and bias |
| `Fp8Linear<In, Out, T>` | `weight: Tensor[Out, In; f8e4m3]`, `scale: Tensor[Out, 1; T]`, optional bias |
| `Fp8BlockLinear<In, Out, Block = 128, T>` | `weight: Tensor[Out, In; f8e4m3]`, `scale: Tensor[Out / Block, In / Block; T]`, optional bias |
| `Nf4Linear<In, Out, Block = 64, T>` | `weight: Tensor[Out, In; nf4]`, `absmax: Tensor[Out, In / Block; f32]`, optional bias |
| `dequantize_int8<*S, N, T>(q, scale)` | symmetric per-row int8: `q * scale` in `f32`, cast to `T` |
| `dequantize_int8_groups<Out, Groups, Group, T>(q, scale)` | symmetric int8, a scale per `Group` inputs |
| `linear_int8(x, weight, scale, bias)`, `linear_int8_groups<Group>(...)` | linear layers over int8 weights |
| `dequantize_int4<R, N, T>(q, scale)` | symmetric per-row int4: `cast<f32>(q) * scale`, cast to `T` |
| `dequantize_int4_groups<Out, Groups, Group, T>(q, scale, zero)` | asymmetric 4-bit: `(q - zero) * scale` per group |
| `linear_int4_groups(x, q, scale, zero, bias)` | a linear layer over group-wise 4-bit weights |
| `decode_fp8(bits)` | FP8 E4M3 bytes held as `u8`, as `f32` values |
| `dequantize_fp8<*S, N, T>(q, scale)` | per-row FP8: `cast<f32>(q) * scale`, cast to `T` |
| `dequantize_fp8_blocks<RowBlocks, ColBlocks, Block, T>(q, scale)` | FP8, a scale per `Block` by `Block` block |
| `linear_fp8(x, weight, scale, bias)`, `linear_fp8_blocks<Block>(...)` | linear layers over FP8 weights |
| `dequantize_nf4<Out, Blocks, Block, T>(q, absmax)` | NF4 codes, a scale per `Block` inputs: `cast<f32>(q) * absmax` |
| `linear_nf4<Block>(x, q, absmax, bias)` | a linear layer over NF4 weights |
| `dequantize_mxfp4<E, Out, G, T>(blocks, scales)` | MXFP4: `f4e2m1` values in blocks of 32 (`[E, Out, G, 32]`), one E8M0 scale byte a block |
| `mxfp4_experts(x, blocks, scales, experts)` | a mixture's chosen experts multiplied from MXFP4, each slot its own input |
| `mxfp4_experts_shared(x, blocks, scales, experts)` | the same, each row's slots sharing one input (`x: [R, 1, In]`) |
| `mxfp4_linear_experts_shared(x, blocks, scales, experts)` | `std.nn.moe::linear_experts_shared` with MXFP4 experts, many rows (`x: [R, In]`) |
| `mxfp4_combine_experts(x, blocks, scales, experts, weights)` | `std.nn.moe::combine_experts` with MXFP4 experts |

## Checkpoints

A checkpoint must contain the parameter paths (`up.weight`, `up.scale`,
...). Quantize with any tool, pack as above, and bind the result with
`bind_weights`. `linnet explain` shows where a backend uses a fused kernel.

## Group-wise 4-bit weights

`Int4GroupLinear` is the GPTQ and AWQ scheme: 16 levels per group of `Group`
consecutive inputs, each group with a scale and zero point. Its layout is
ONNX Runtime's `MatMulNBits`. Under `numerics="fast"`:

| Backend | `linear_int4_groups` runs as |
| --- | --- |
| PyTorch, CUDA, `bf16`, groups of 32 to 256 | fused int4 kernels up to 128 rows, weights repacked at load (`--prepare`); longer prompts dequantize per call |
| PyTorch plan with the `marlin` plugin | vLLM's Marlin, where a measured plan finds it faster ([Backends](backends.md)) |
| ONNX Runtime | `com.microsoft.MatMulNBits` (`f32`, `f16`; `bf16` goes through `f32`) |
| JAX | weights dequantized once at load (`--prepare`), then run as bf16 |
| everywhere else | dequantize, then `linear` |

On Llama 3.1 8B Instruct, imported GPTQ and AWQ checkpoints reach a
perplexity of 9.98 (bf16: 9.55; rounded to nearest: 10.28) and decode at
192 and 214 tokens per second, against 134 for bf16 in the same run (an
earlier one than the 8-bit table below).

### Quantizing a checkpoint

`linnet.quant.quantize_checkpoint` rounds each group of a float checkpoint
to the nearest level:

```python
from linnet.quant import quantize_checkpoint

quantize_checkpoint("model.safetensors", "model-int4.safetensors",
                    patterns=["*_proj.weight"], group=128, dtype="bf16",
                    bindings="bindings.json")
```

`bindings` maps checkpoint names to Linnet paths. Each matched `proj.weight`
becomes `proj.weight`, `proj.scale`, and `proj.zero`; other tensors are
copied. Declare `Int4GroupLinear` where the source had `Linear`. With
`bits=8`, each becomes `proj.weight` and `proj.scale` for `Int8GroupLinear`.

### Importing GPTQ and AWQ

`linnet.quant.import_quantized` repacks a 4-bit GPTQ or AWQ checkpoint into
the same layout; the `quantization_config` in its `config.json` says which:

```python
from linnet.quant import import_quantized

import_quantized("Meta-Llama-3.1-8B-Instruct-GPTQ-INT4/", "model-int4.safetensors",
                 bindings="bindings.json")
```

GPTQ checkpoints in activation order (`desc_act`) import without rounding
again: the layer gets an `order` and gathers its inputs first, at some cost
in speed.

### Importing bitsandbytes NF4

The same call imports a bitsandbytes 4-bit checkpoint (QLoRA's, and the
`bnb-4bit` models on the Hub) whose `bnb_4bit_quant_type` is `nf4`. Each
layer becomes `.weight` (NF4 codes, `std.dtypes::nf4`) and `.absmax` (`f32`,
one per block of 64 inputs) for `Nf4Linear`. Nested absmax codes
(`bnb_4bit_use_double_quant`) are decoded at import. bitsandbytes' `fp4`
type and its 8-bit layers do not import. `Nf4Linear` widens the weight and
multiplies in `T` on every backend; compiled PyTorch widens it once and
keeps the 16-bit copy. The bitsandbytes backend plugin in
`examples/09-backends` multiplies from the 4-bit codes instead, without
that copy.

## 8-bit weights

Four blocks cover the 8-bit checkpoints in use. Bind `scale` to the
checkpoint's scale tensor. The FP8 blocks take `f8e4m3` weights
(`std.dtypes`): an `F8_E4M3` tensor binds as itself, and a `U8` one as its
bytes. An `F8_E4M3` tensor also binds to a `u8` parameter as its bytes,
which `decode_fp8` decodes.

| Block | Checkpoints | `scale` binds to |
| --- | --- | --- |
| `Int8Linear` | `compressed-tensors` int8 per channel (W8A8, W8A16) | `weight_scale`, `[Out, 1]` |
| `Int8GroupLinear<..., Group>` | `compressed-tensors` int8 per group, `quantize_checkpoint(bits=8)` | `weight_scale`, `[Out, In / Group]` |
| `Fp8Linear` | `compressed-tensors` FP8 (FP8-dynamic) | `weight_scale`, `[Out, 1]` |
| `Fp8BlockLinear` | DeepSeek-V3, Qwen3 FP8 | `weight_scale_inv`, `[Out / 128, In / 128]` |

`Int8GroupLinear` and `Fp8BlockLinear` need the group or block to divide
the widths: state `where In % 128 == 0` (and `Out % 128 == 0` for blocks)
up to the root block.

PyTorch on CUDA, in bf16 or f16, multiplies the 8-bit weight as it is:

| Rows | `numerics="equivalent"` | `numerics="fast"` (the default) |
| --- | --- | --- |
| 1 | a CUDA-core kernel: the weight read once, widened in registers | the same |
| 2 to 32 | a tensor-core kernel: the weight widened to the input's dtype | the same |
| 33 to 64 | the same | the input rounded to 8 bits too, multiplied in 8 bits |
| more | the weight dequantized for the call, then cuBLAS | the same as 33 to 64 |

Rounded inputs multiply by `torch._scaled_mm` for FP8 (a scale per row, or
per 128 inputs against blocks) and by Linnet's kernel for int8. Int8 groups
never round the input.

FP8 needs compute capability 8.9 or later (9.0 for the FP8 products), and
In a multiple of 128. Elsewhere, and on JAX, ONNX and StableHLO, the weight
is dequantized and multiplied; JAX with `--prepare` dequantizes it once.
Sibling projections (query, key and value; gate and up) multiply as one
product, as bf16's do.

On one H100 (same host, CUDA graphs for decoding, a 512-token prompt run
eagerly):

| Model | Weights | Perplexity | Time to first token | Decoding | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama 3.1 8B Instruct | bf16 | 9.55 | 18.9 ms | 163 tokens/s | 16.3 GiB |
| | FP8, a scale per row | 9.62 | 18.7 ms | 275 tokens/s | 9.9 GiB |
| | int8 W8A8 | 9.70 | 21.0 ms | 271 tokens/s | 9.9 GiB |
| Qwen3 8B | bf16 | 12.71 | 21.6 ms | 157 tokens/s | 16.9 GiB |
| | FP8, 128 x 128 blocks | 12.84 | 23.2 ms | 238 tokens/s | 10.4 GiB |

Rounding the input costs some accuracy: under `numerics="equivalent"` the
same checkpoints reach 9.59 (FP8) and 9.55 (int8) on Llama, and 12.77 on
Qwen3. Block-scaled prompts run at bf16's speed: cuBLAS's block-scaled
FP8 product is no faster at these sizes.

## MXFP4 experts

MXFP4 is a 4-bit microscaling float format, 4.25 bits a weight; gpt-oss
publishes its experts in it.

| Backend | MXFP4 experts run as |
| --- | --- |
| PyTorch, CUDA, decoding | a Triton kernel that reads the MXFP4 bytes directly |
| PyTorch, Hopper GPU, `bf16`, prompts and serving | grouped by expert; multiplied from MXFP4 with `triton_kernels` installed, otherwise dequantized to bf16 once |
| JAX | dequantized to bf16 once at load (`--prepare`); on a GPU, prompts and serving grouped by expert in a Pallas kernel. Reading four times the bytes, prompts run slower than in PyTorch |
| others with `--prepare` | dequantized once at load, kept in 16 bits |
| everywhere else | the bodies: dequantize every expert, gather the chosen ones, multiply in `f32` |

OpenAI's
[`triton_kernels`](https://github.com/triton-lang/triton/tree/main/python/triton_kernels)
is not on PyPI; install the one matching your Triton:

```bash
pip install "triton_kernels @ git+https://github.com/triton-lang/triton.git@v$(python -c 'import triton; print(triton.__version__)')#subdirectory=python/triton_kernels"
```

With `triton_kernels` and CUDA graphs, gpt-oss-20b on one H100 decodes at
368 tokens per second and serves 256 requests at 6031 tokens per second
(vLLM: 299 and 4313). Without CUDA graphs, `triton_kernels` adds host
overhead to every routed product. More measurements:
[Benchmarks](https://linnet.franknoh.dev/benchmarks).

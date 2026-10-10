# Quantization

Run int8, int4, group-wise 4-bit (GPTQ, AWQ), FP8, and MXFP4 weights on
every backend. A quantized weight is an ordinary parameter, an integer tensor plus
scales, that `std.quant` dequantizes; there is no quantized dtype, and the
arithmetic dtype `T` is separate from the storage.

```linnet
use std.quant::{Int4Linear, Int8Linear}

pub block Mlp<H: Dim, Inner: Dim, T: Float = bf16>
where Inner % 2 == 0 {
    sub up: Int4Linear<H, Inner, T>      // weight: Tensor[Inner, H / 2; i8], scale: Tensor[Inner; f32]
    sub down: Int8Linear<Inner, H, T>    // weight: Tensor[H, Inner; i8], scale: Tensor[H; f32]
    ...
}
```

Packed 4-bit formats hold two values a byte, the even element in the low
nibble.

| `std.quant` | Format |
| --- | --- |
| `Int8Linear<In, Out, T>` | `weight: Tensor[Out, In; i8]`, `scale: Tensor[Out; f32]`, optional bias |
| `Int4Linear<In, Out, T>` | `weight: Tensor[Out, In / 2; i8]`, `scale`, optional bias |
| `Int4GroupLinear<In, Out, Group = 128, T>` | `weight: Tensor[Out, In / Group, Group / 2; u8]`, `scale` and `zero: [Out, In / Group]`, optional `order: [In; i32]` and bias |
| `Fp8Linear<In, Out, T>` | `weight: Tensor[Out, In; u8]` (FP8 E4M3 bytes), `scale: Tensor[Out, 1; T]`, optional bias |
| `dequantize_int8<*S, N, T>(q, scale)` | symmetric per-row int8: `q * scale` in `f32`, cast to `T` |
| `unpack_int4<R, H>(packed)`, `unpack_uint4<R, H>(packed)` | signed or unsigned nibbles |
| `dequantize_int4<R, H, T>(packed, scale)` | unpack, then dequantize |
| `dequantize_int4_groups<Out, Groups, Half, T>(packed, scale, zero)` | asymmetric 4-bit: `(q - zero) * scale` per group |
| `linear_int4_groups(x, packed, scale, zero, bias)` | a linear layer over group-wise 4-bit weights |
| `decode_fp8(bits)` | FP8 E4M3 bytes as `f32` values |
| `dequantize_fp8<*S, N, T>(q, scale)` | per-row FP8: `decode_fp8(q) * scale`, cast to `T` |
| `linear_fp8(x, weight, scale, bias)` | a linear layer over per-row FP8 weights |
| `dequantize_mxfp4<E, Out, G, T>(blocks, scales)` | MXFP4: E2M1 values in blocks of 32, one E8M0 scale byte a block |
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
| ONNX Runtime | `com.microsoft.MatMulNBits` (`f32`, `f16`; `bf16` goes through `f32`) |
| JAX | weights dequantized once at load (`--prepare`), then run as bf16 |
| everywhere else | dequantize, then `linear` |

On Llama 3.1 8B Instruct, imported GPTQ and AWQ checkpoints reach a
perplexity of 9.98 (bf16: 9.55; rounded to nearest: 10.28) and decode at
192 and 214 tokens per second (bf16: 134).

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
copied. Declare `Int4GroupLinear` where the source had `Linear`.

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

## FP8 weights

`Fp8Linear` reads FP8 E4M3 weights with one scale a row, the layout of
`compressed-tensors` checkpoints (`FP8-dynamic`): bind `scale` to the
checkpoint's `weight_scale`. An `F8_E4M3` tensor binds to a `u8`
parameter as its bytes, and `decode_fp8` turns them into values; PyTorch,
JAX and StableHLO decode them as their own FP8 type.

| Backend | `linear_fp8` runs as |
| --- | --- |
| PyTorch, CUDA (compute capability 9 or later), `bf16`, `numerics="fast"`, one row | a Triton kernel that widens the FP8 bytes in registers; the input stays in `bf16` |
| the same, more rows | each input row rounded to FP8 with its own scale, then `torch._scaled_mm` |
| everywhere else | the weight decoded and scaled, then `linear` |

RedHatAI's Llama 3.1 8B Instruct FP8-dynamic checkpoint against the bf16
one, on one H100. Same host, CUDA graphs for decoding, a 512-token prompt
run eagerly:

| | Perplexity | Time to first token | Decoding | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 9.55 | 19.1 ms | 162 tokens/s | 16.3 GiB |
| FP8 | 9.62 | 20.3 ms | 197 tokens/s | 9.9 GiB |

Projections that read one input (a layer's query, key and value, its gate
and up) multiply as one FP8 product, as bf16's do. The prompt's FP8
products take less device time than bf16's, but run eagerly its calls
cost the host about as much as they save.

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

Not supported yet: activation quantization other than FP8's, and per-group
scales for 8-bit
weights.

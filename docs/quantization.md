# Quantization

Quantized weights are ordinary parameters: an integer tensor plus its scales,
declared like any `param` and dequantized by library code. The language has
no quantized dtype. The arithmetic dtype `T` and the storage of the weights
(`i8`, packed `i4`) stay separate, and a backend that has a fused kernel can
select it for the dequantize-and-multiply it sees.

```linnet
use std.quant::{Int4Linear, Int8Linear}

pub block Mlp<H: Dim, Inner: Dim, T: Float = bf16>
where Inner % 2 == 0 {
    sub up: Int4Linear<H, Inner, T>      // weight: Tensor[Inner, H / 2; i8], scale: Tensor[Inner; f32]
    sub down: Int8Linear<Inner, H, T>    // weight: Tensor[H, Inner; i8], scale: Tensor[H; f32]
    ...
}
```

| `std.quant` | |
| --- | --- |
| `dequantize_int8<*S, N, T>(q, scale)` | symmetric per-row int8: `q * scale` in `f32`, cast to `T` |
| `unpack_int4<R, H>(packed)` | two signed nibbles per byte (low = even element, high = odd) |
| `dequantize_int4<R, H, T>(packed, scale)` | unpack, then dequantize |
| `Int8Linear<In, Out, T>` | `weight: Tensor[Out, In; i8]`, `scale: Tensor[Out; f32]`, optional bias |
| `Int4Linear<In, Out, T>` | `weight: Tensor[Out, In / 2; i8]`, `scale`, optional bias |
| `unpack_uint4<R, H>(packed)` | two unsigned nibbles per byte (low = even element, high = odd) |
| `dequantize_int4_groups<Out, Groups, Half, T>(packed, scale, zero)` | asymmetric 4-bit in groups: `(q - zero) * scale` per group |
| `linear_int4_groups(x, packed, scale, zero, bias)` | a linear layer over group-wise 4-bit weights |
| `Int4GroupLinear<In, Out, Group = 128, T>` | `weight: Tensor[Out, In / Group, Group / 2; u8]`, `scale` and `zero: [Out, In / Group]`, optional `order: [In; i32]` and bias |

## Group-wise 4-bit weights

`Int4GroupLinear` stores a weight as 16 levels per group of `Group`
consecutive inputs, each group with its own scale and zero point, the scheme
GPTQ and AWQ checkpoints use. Its weight is `[Out, In / Group, Group / 2]`
bytes, two values a byte with the even one in the low nibble. That layout is
ONNX Runtime's `MatMulNBits` as it is. Under `numerics="fast"` a backend runs a
fused kernel in place of dequantize-then-multiply:

| Backend | `linear_int4_groups` |
| --- | --- |
| PyTorch, CUDA, `bf16`, groups of 32 to 256 | tinygemm (`_weight_int4pack_mm`); the weights are repacked for it once, at load (`--prepare`) |
| ONNX Runtime | `com.microsoft.MatMulNBits` (`f32` and `f16`; `bf16` goes through `f32`) |
| everywhere else | the body: unpack, dequantize, `linear` |

`linnet.quant.quantize_checkpoint` writes such a checkpoint from a float one,
rounding each group to the nearest level:

```python
from linnet.quant import quantize_checkpoint

quantize_checkpoint("model.safetensors", "model-int4.safetensors",
                    patterns=["*_proj.weight"], group=128, dtype="bf16",
                    bindings="bindings.json")
```

The tensors land under Linnet paths (`bindings` maps them from the
checkpoint's names), `proj.weight` becoming `proj.weight`, `proj.scale`, and
`proj.zero`, and everything the patterns do not match is copied as it is.
Declaring `Int4GroupLinear` where the source had `Linear` then loads it.

Rounding to nearest is the plainest scheme. A checkpoint someone quantized
with calibration repacks into the same layout: `linnet.quant.import_quantized`
reads a 4-bit GPTQ or AWQ checkpoint (the `quantization_config` in its
`config.json` says which), unpacks its int32 words (GPTQ's zero points stored
less one, AWQ's interleaved outputs), and writes `Int4GroupLinear`'s tensors
under Linnet paths, the rest copied as it is:

```python
from linnet.quant import import_quantized

import_quantized("Meta-Llama-3.1-8B-Instruct-GPTQ-INT4/", "model-int4.safetensors",
                 bindings="bindings.json")
```

A GPTQ checkpoint in activation order (`desc_act`) formed its groups over
the inputs in another order than their own, given by `g_idx`. Its inputs are
sorted by group, which makes each group contiguous, and the permutation is
written as the layer's `order`: `Int4GroupLinear` takes its inputs in that
order first (`std.quant::take_inputs`, a gather: `index_select` in PyTorch,
`Gather` in ONNX, `jnp.take` in JAX), so no weight is rounded again.

On Llama 3.1 8B Instruct (WikiText-2, eight windows of 1024 tokens), bf16 has
a perplexity of 9.55 and rounding to nearest 10.28. The hugging-quants GPTQ
checkpoint (activation order) imports to 9.98, and the AWQ one to 9.98. Both
decode on the tinygemm path in 10.3 GiB, at 192 and 214 tokens per second
(the gather costs GPTQ's order), against 134 in bf16.

## Checkpoints

The parameter paths (`up.weight`, `up.scale`, ...) are what a checkpoint must
contain. Quantize a float checkpoint with any tool, pack int4 pairs with the
even element in the low nibble, and bind the result with `bind_weights`.
Unpacking is `shr` and `&`, dequantization a broadcast multiply, so the same
source runs in PyTorch, XLA, and ONNX Runtime. `linnet explain` shows where a
backend replaced the arithmetic with a kernel.

Not written yet: activation quantization, and per-group scales for 8-bit
weights.

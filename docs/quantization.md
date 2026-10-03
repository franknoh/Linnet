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
| `dequantize_mxfp4<E, Out, G, T>(blocks, scales)` | MXFP4: two E2M1 values a byte (low = even element) in blocks of 32, an E8M0 scale byte a block |
| `mxfp4_experts(x, blocks, scales, experts)` | a mixture's chosen experts multiplied straight from MXFP4, each slot its own input |
| `mxfp4_experts_shared(x, blocks, scales, experts)` | the same, every slot of a row reading the row's one input (`x: [R, 1, In]`) |
| `mxfp4_linear_experts_shared(x, blocks, scales, experts)` | `std.nn.moe::linear_experts_shared` with MXFP4 experts, for many rows at once (`x: [R, In]`) |
| `mxfp4_combine_experts(x, blocks, scales, experts, weights)` | `std.nn.moe::combine_experts` with MXFP4 experts: each row's products weighed and summed |

## Group-wise 4-bit weights

`Int4GroupLinear` stores a weight as 16 levels per group of `Group`
consecutive inputs, each group with its own scale and zero point, the scheme
GPTQ and AWQ checkpoints use. Its weight is `[Out, In / Group, Group / 2]`
bytes, two values a byte with the even one in the low nibble. That layout is
ONNX Runtime's `MatMulNBits` as it is. Under `numerics="fast"` a backend runs a
fused kernel in place of dequantize-then-multiply:

| Backend | `linear_int4_groups` |
| --- | --- |
| PyTorch, CUDA, `bf16`, groups of 32 to 256 | tinygemm (`_weight_int4pack_mm`) for up to 7 rows; the weights are repacked for it once, at load (`--prepare`). 8 to 128 rows (a batch of decoding requests) go through Linnet's own kernel, `linnet::int4_linear`, which unpacks each group inside the matrix product. More rows (a prompt) multiply the weight dequantized by one Triton kernel |
| ONNX Runtime | `com.microsoft.MatMulNBits` (`f32` and `f16`; `bf16` goes through `f32`) |
| JAX | the body, dequantized once at load (`--prepare`) |
| everywhere else | the body: unpack, dequantize, `linear` |

tinygemm reads the packed weights, so a decoding step reads a quarter of
bf16's bytes, but its time grows with the rows. Linnet's kernel reads them
packed too and takes about as long for 8 rows as for 16, multiplying on
tensor cores; over the projections of a Llama 3.1 8B layer on an H100 it
takes 123 us for 8 rows where tinygemm takes 148, and 202 us for 64 where
dequantizing first takes 359. Past 128, a prompt's rows go faster through
cuBLAS over the weight dequantized for that call, a layer at a time. For
Llama 3.1 8B the first token of a 512-token prompt takes 24.8 ms this way,
against 17.5 ms in bf16. XLA has no fused kernel for a 4-bit weight: left in
the step, the dequantization runs as its own pass every call (44 tokens per
second for 8B), so JAX dequantizes once when the model loads and then
computes, and holds its weights, as in bf16.

## MXFP4 experts

gpt-oss publishes its experts in MXFP4, 4.25 bits a weight. The bodies of
`mxfp4_experts` and `mxfp4_experts_shared` dequantize every expert, gather
the chosen ones and multiply, accumulating in `f32`; since the
dequantization reads nothing but weights, a backend that prepares
weight-only work (`--prepare`) does it once at load and keeps the experts in
16 bits. PyTorch on CUDA instead runs a Triton kernel (`linnet.torch.kernels`)
for a decoding step's few rows: each chosen expert's bytes are read as they
are, a quarter of a 16-bit weight's, and unpacked in registers eight at a
time -- each 32-bit word of a block shifted and masked into four pairs of
`f16`s, every E2M1 nibble's bits placed in one -- with the block's scale
applied once per 32 weights. On an H100 it reads gpt-oss-20b's experts at
about 2 TB/s for one to four rows, where unpacking each nibble on its own
read 1.4.

Many rows at once -- a prompt, a serving step -- take
`mxfp4_linear_experts_shared` and `mxfp4_combine_experts`, whose bodies
dequantize every expert and take `std.nn.moe`'s dense form (prepared once at
load where a backend prepares). On a Hopper GPU in bf16, PyTorch instead
sorts the (row, choice) pairs by expert and multiplies each expert by its
pairs' inputs (`linnet.torch.moe`): with OpenAI's
[`triton_kernels`](https://github.com/triton-lang/triton/tree/main/python/triton_kernels)
installed, its MXFP4 product reads the four-bit weights in a layout swizzled
for the GPU once; without it, the experts are dequantized to bf16 once and
`torch._grouped_mm` multiplies them. `triton_kernels` is not on PyPI; install
the one matching your Triton:

```bash
pip install "triton_kernels @ git+https://github.com/triton-lang/triton.git@v$(python -c 'import triton; print(triton.__version__)')#subdirectory=python/triton_kernels"
```

With it and CUDA graphs, gpt-oss-20b on one H100 takes a 512-token prompt in
14.1 ms rather than 25.0, decodes at 368 tokens per second, and serves 256
requests at 5349 tokens per second (vLLM 17.8 ms, 299 and 4313). Without CUDA graphs
each routed product also costs `triton_kernels`' Python on the host, about
0.4 ms a call, which a captured step does not pay: the same prompt takes
34.9 ms under `torch.compile` alone.

In JAX, generated code dequantizes the experts to bf16 once, when the model
loads (`--prepare`), and `linnet.jax.moe` multiplies the pairs by that copy.
A decoding step's few pairs read their experts' weights inside one reduction.
A prompt's many go through a Pallas kernel on the GPU that multiplies each
expert by the tiles of rows that chose it. XLA's own `ragged_dot` multiplies
every row by every expert and masks the rest, as the bodies do, so it is
what runs on the CPU. The kernel reads the bf16 copy, four times the bytes
of MXFP4, so a prompt takes longer than PyTorch's.

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

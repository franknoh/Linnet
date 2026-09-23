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

## Checkpoints

The parameter paths (`up.weight`, `up.scale`, ...) are what a checkpoint must
contain. Quantize a float checkpoint with any tool, pack int4 pairs with the
even element in the low nibble, and bind the result with `bind_weights`.
Unpacking is `shr` and `&`, dequantization a broadcast multiply, so the same
source runs in PyTorch, XLA, and ONNX Runtime. `linnet explain` shows where a
backend replaced the arithmetic with a kernel.

Not written yet: activation quantization, per-group scales, asymmetric
(zero-point) schemes. Each is a few more lines of the same library.

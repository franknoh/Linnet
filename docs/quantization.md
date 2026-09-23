# Quantization

Quantized weights in Linnet are ordinary parameters: an integer tensor and
its scales, declared like any other `param` and dequantized by library code
the checker verifies. There is no quantized dtype in the language — the
logical dtype of a layer's arithmetic (`T`) and the storage of its weights
(`i8`, packed `i4`) are separate, exactly as the specification asks, and a
backend that has a fused kernel can select it as a candidate for the
dequantize-and-multiply it sees.

```linnet
use std.quant::{Int4Linear, Int8Linear}

pub block Mlp<H: Dim, Inner: Dim, T: Float = bf16>
where Inner % 2 == 0 {
    sub up: Int4Linear<H, Inner, T>      // param weight: Tensor[Inner, H / 2; i8]
    sub down: Int8Linear<Inner, H, T>    // param weight: Tensor[H, Inner; i8]
    ...
}
```

`std.quant` provides:

| | |
| --- | --- |
| `dequantize_int8<*S, N, T>(q: Tensor[*S, N; i8], scale: Tensor[*S; f32])` | symmetric per-row int8: `q * scale` in `f32`, cast to `T` |
| `unpack_int4<R, H>(packed: Tensor[R, H; i8]) -> Tensor[R, 2 * H; i8]` | two signed nibbles per byte (low = even element, high = odd) |
| `dequantize_int4<R, H, T>(packed, scale)` | unpack, then dequantize |
| `Int8Linear<In, Out, T>` | `weight: Tensor[Out, In; i8]`, `scale: Tensor[Out; f32]`, optional `bias` |
| `Int4Linear<In, Out, T>` | `weight: Tensor[Out, In / 2; i8]`, `scale`, optional `bias` |

The parameter paths are what a checkpoint must contain (`up.weight`,
`up.scale`, ...): quantize a float checkpoint with any tool, pack int4 pairs
low-nibble-first, and bind the result with `bind_weights`. Because
unpacking is `shr`/`&` arithmetic and dequantization a broadcast multiply,
the same source runs in PyTorch, XLA, and ONNX Runtime; `linnet explain`
shows where a backend replaced the dequantize-multiply with a kernel of its
own.

Not covered yet: activation quantization, per-group scales, and asymmetric
(zero-point) schemes — each is a few lines of the same kind of library code.

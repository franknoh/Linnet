# 15. Reserved future extensions

Planned extensions and the constraints each must keep. This chapter is informative except where it reserves keywords or explicitly constrains future compatibility.

## 15.1 Runtime state and KV caches

Structured state (a `struct` of tensors), state carried by runtime loops, and state passed explicitly through entry signatures are open. Each must keep state flow explicit in the IR.

## 15.2 Explicit RNG

[`std.random`](../docs/random.md) implements a counter-based PRNG (Threefry-2x32) in Linnet, with `Tensor[2; i64]` keys that programs split and pass. `rng` stays reserved for a possible language-level construct.

## 15.3 Runtime loops and scan

A first-class `scan` that collects per-iteration outputs, beyond `while` (§8.4).

## 15.4 Custom gradients and training

Training already works by differentiating exported entries, with objectives written as module-level entries (§7.6). Possible additions: autodiff as a compiler transform, so StableHLO and ONNX exports can carry gradients; `@custom_vjp` or equivalent annotations; gradient-specific semantic ops; and optimizer-state structures. Training support must not require model-specific Python classes.

## 15.5 Quantization

[`std.quant`](../docs/quantization.md) already stores integer weights beside their scales and dequantizes in source. Quantized dtypes or storage annotations in the type system are future work and must keep dequantization explicit.

## 15.6 Sparse and structured tensors

Type- or value-level metadata for sparse, diagonal/triangular, low-rank, or block-sparse tensors. It should not bake one backend's storage layout into semantic source.

## 15.7 User-defined traits

`Float`, `Numeric`, and related constraints may generalize to a small, safe trait system. It must not become arbitrary compile-time execution.

## 15.8 `extern op`

A trusted escape hatch for operations with externally supplied backend implementations (§13.5). They should be clearly non-portable and must never be required merely to add a model built from expressible tensor primitives.

## 15.9 Kernel language

A low-level kernel DSL for Triton-like tile programs, targeting the same or a lower IR and kept separate from model-level `.linnet` semantics.

## 15.10 Distributed execution

Today a `Shards` generic sizes each process's weights, and `std.nn.parallel::all_reduce` and `all_gather` mark where partial results are summed or joined; the program with `Shards = 1` is the reference. Further collectives and placement or sharding annotations in the type system are future work.

## 15.11 Importers

Importers for formats beyond `torch.export`, JAX through StableHLO, and ONNX. Importers are frontends and do not redefine Linnet source semantics.

# 15. Reserved Future Extensions

This chapter is informative except where it reserves keywords or explicitly constrains future compatibility.

## 15.1 Runtime state and KV caches

`state` members (§9.3) cover block-owned tensors updated by assignment. Structured state (a `struct` of tensors), state carried by runtime loops, and state passed explicitly through entry signatures are open; any of them must keep state flow explicit in the IR rather than hidden mutation.

## 15.2 Explicit RNG

Randomness is explicit data: `std.random` implements a counter-based PRNG (Threefry-2x32) in Linnet over integer tensors, with keys as `Tensor[2; i64]` values that programs split and pass. No primitive draws random numbers and no hidden generator exists, so graph transformations preserve reproducibility by construction. `rng` stays reserved should a language-level construct ever be needed.

## 15.3 Runtime loops and scan

`while` (§8.4) covers data-dependent loops with invariant shapes on every backend; a first-class `scan` that collects per-iteration outputs remains future work.

## 15.4 Custom gradients and training

Training works today without language support for gradients. Every entry exports to arithmetic a framework differentiates: PyTorch autograd runs through interpreted or generated entries, and `jax.grad` through generated JAX. Losses, rewards, and other objectives are written as module-level entries (§7.6), which a model's own entries can call. No model-specific Python class is involved.

Potential extensions include:

- autodiff as a compiler transform, so that graph formats without one (StableHLO, ONNX) can carry gradients;
- `@custom_vjp` or equivalent annotations;
- gradient-specific semantic ops;
- optimizer-state structures.

Training support must not require model-specific Python classes.

## 15.5 Quantization

Logical dtype and storage are separate today by construction: `std.quant` stores weights as integer parameters beside their scales and dequantizes in ordinary source, so a backend sees the dequantize-and-multiply explicitly and may select a fused kernel for it. The library covers symmetric per-row 8-bit weights (`Int8Linear`), per-row 4-bit weights packed two to a byte (`Int4Linear`, nibbles unpacked by `shr` and `&`), group-wise 4-bit weights with a scale and zero point per group as GPTQ and AWQ store them (`Int4GroupLinear`, `linear_int4_groups`), and MXFP4 expert weights with a shared exponent per block of 32 (`dequantize_mxfp4`, `mxfp4_experts`). Quantized dtypes or storage annotations in the type system remain future work and must keep that explicitness.

## 15.6 Sparse and structured tensors

Potential type-level or value-level structure metadata:

- sparse formats;
- diagonal/triangular structure;
- low-rank representations;
- block sparsity.

These should avoid baking one backend's storage layout into semantic source.

## 15.7 User-defined traits

The initial built-in `Float`, `Numeric`, and related constraints may eventually generalize to a small safe trait system. This must not become arbitrary compile-time execution.

## 15.8 `extern op`

A trusted escape hatch may allow operations with externally supplied backend implementations. Such operations should be clearly non-portable and must never be required merely to add a new model composed of expressible tensor primitives.

## 15.9 Kernel language

A future low-level kernel DSL may target the same or a lower IR and allow Triton-like tile programs. This is intentionally separate from model-level `.linnet` semantics.

## 15.10 Distributed execution

*Informative.* A model that splits itself across processes states it in ordinary source: a `Shards` generic sizes each process's part of the split weights, and `std.nn.parallel::all_reduce` marks where partial results combine. The operation's canonical body is the identity, which is exact for one shard, so the program with `Shards = 1` is the reference; a backend running several processes sums across them there. Collective operations beyond this, and placement or sharding annotations in the type system, are future work.

## 15.11 Importers

Importers already translate `torch.export` graphs, JAX programs through StableHLO, and ONNX graphs into Linnet source. Future importers may translate other semantic tensor formats.

Importers are frontends. They do not redefine Linnet source semantics.

# 15. Reserved future extensions

Planned extensions and the constraints each must keep. This chapter is informative except where it reserves keywords or explicitly constrains future compatibility.

## 15.1 Runtime state and KV caches

Structured state (a `struct` of tensors), state carried by runtime loops, and state passed explicitly through entry signatures are open. Each must keep state flow explicit in the IR.

## 15.2 Explicit RNG

[`std.random`](../docs/random.md) implements a counter-based PRNG (Threefry-2x32) in Linnet, with `Tensor[2; i64]` keys that programs split and pass. `rng` stays reserved for a possible language-level construct.

## 15.3 Runtime loops and scan

A `for` with a value (§8.5) collects per-iteration outputs. Iterating a tensor's leading axis directly (`for x in xs`) and runtime trip counts are open.

## 15.4 Custom gradients and training

Training works by differentiating exported entries in the target framework. Objectives are entries: module-level ones (§7.6), or a block's entries over packed sequences that return each position's loss or log-probability. `std.nn.loss` defines the losses as semantic ops (§7.3). A backend MAY compute `linear_cross_entropy` and `linear_token_log_probs` a block of rows at a time, with its own backward pass, so the `[N, Vocab]` logits are never whole. Low-rank adapters, gathering sharded parameters, and recomputing blocks in the backward pass are code-generation options and leave source semantics unchanged.

The compiler differentiates exports too: `--grad` (docs/tooling.md) emits an entry's loss and its gradient with respect to the parameters, the backward pass written from the exported operations. A `for` loop's backward pass keeps each iteration's starting values and runs the iterations in reverse; `while` loops, whose iteration counts are not known, are not differentiated yet. An op's `grad` clause (§7.2) replaces the backward pass of its body wherever an entry is differentiated.

Possible additions are:

- gradient-specific semantic ops;
- optimizer-state structures.

Training support must not require model-specific Python classes.

## 15.5 Quantization

[`std.quant`](../docs/quantization.md) already stores integer weights beside their scales and dequantizes in source. Quantized dtypes or storage annotations in the type system are future work and must keep dequantization explicit.

## 15.6 Sparse and structured tensors

Type- or value-level metadata for sparse, diagonal/triangular, low-rank, or block-sparse tensors. It should not bake one backend's storage layout into semantic source.

## 15.7 User-defined traits

`Float`, `Numeric`, and related constraints may generalize to a small, safe trait system. It must not become arbitrary compile-time execution.

## 15.8 `extern op`

A trusted escape hatch for operations with externally supplied backend implementations (§13.5). They should be clearly non-portable and must never be required merely to add a model built from expressible tensor primitives.

## 15.9 Kernel language

Kernels (§7.9) are tile programs an op may run in place of its body. Scans within a tile, explicit shared memory, atomics that return the old value, and kernels outside an op are open.

## 15.10 Distributed execution

Today a `Shards` generic sizes each process's weights for tensor parallelism, and `std.nn.parallel::all_reduce` and `all_gather` mark where partial results are summed or joined; the program with `Shards = 1` is the reference. Fully sharded data parallelism needs nothing in source. A code generator splits a block's parameters across processes or devices by block path and gathers each where the block runs; the program computes what it computes unsplit. Further collectives and placement or sharding annotations in the type system are future work.

## 15.11 Importers

Importers for formats beyond `torch.export`, JAX through StableHLO, and ONNX. Importers are frontends and do not redefine Linnet source semantics.

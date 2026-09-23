# 15. Reserved Future Extensions

This chapter is informative except where it reserves keywords or explicitly constrains future compatibility.

## 15.1 Runtime state and KV caches

`state` members (§9.3) cover block-owned tensors updated by assignment. Structured state (a `struct` of tensors), state carried by runtime loops, and state passed explicitly through entry signatures are open; any of them must keep state flow explicit in the IR rather than hidden mutation.

## 15.2 Explicit RNG

Randomness should use explicit RNG values or state tokens so graph transformations preserve reproducibility.

## 15.3 Runtime loops and scan

`while` and a first-class `scan` primitive are expected for recurrent/state-space models and decoding loops. Shape invariants and state effects must be statically checked.

## 15.4 Custom gradients and training

Potential extensions include:

- autodiff as a compiler transform;
- `@custom_vjp` or equivalent annotations;
- gradient-specific semantic ops;
- optimizer-state structures.

Training support must not require model-specific Python classes.

## 15.5 Quantization

Future types/metadata should represent logical dtype separately from storage/quantization encoding. Quantization must remain semantically explicit enough for backends to choose dequantization/fused-kernel placement.

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

## 15.10 Importers

Future importers may translate:

- `torch.export` graphs;
- JAX/Jaxpr or StableHLO;
- ONNX;
- other semantic tensor formats.

Importers are frontends. They do not redefine Linnet source semantics.

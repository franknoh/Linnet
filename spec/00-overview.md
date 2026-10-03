# 0. Overview and design contract

## 0.1 Purpose

Linnet is a statically typed, declarative tensor language. A `.linnet` file describes tensor computation and model structure, with no tensor payloads or host-language code.

## 0.2 Architectural invariants

### A. Model independence

The compiler core MUST NOT contain model-specific names such as `Llama`, `Qwen`, or `Gemma`.

### B. Library-defined ops

High-level operations such as `linear`, `rope`, `rms_norm`, and `attention` SHOULD be `op` declarations in `.linnet` libraries. They MUST NOT require compiler changes unless the primitive tensor algebra cannot express them.

### C. Primitive fallback

Every non-extern semantic `op` MUST have a valid body that defines its canonical semantics in lower-level Linnet operations. A backend MAY replace an `op` or a recognized decomposition with a native implementation, but correctness MUST NOT depend on it.

### D. Weight separation

A `.linnet` source file MUST NOT contain raw parameter tensor payloads. Weight data is bound separately, through formats such as SafeTensors and a non-executable binding manifest.

### E. No hidden host execution

Parsing, type checking, graph inspection, and parameter-manifest generation MUST NOT execute Python, C++, shell commands, dynamic libraries, network requests, package build scripts, or arbitrary user code.

### F. Strict static semantics

Shape and dtype compatibility MUST be proven statically whenever the language claims an operation is valid. The compiler MUST NOT silently insert dtype conversions or accept unresolved broadcasting.

### G. Single semantic frontend

The CLI, language server, formatter, graph visualizer, and backend pipeline MUST use the same parser and semantic-analysis implementation. Editor tooling MUST NOT reimplement Linnet's type or shape system in another language.

### H. Source stability

`.linnet` source and this specification are the durable interface. HIR, Core IR, optimizer representations, and plan formats are implementation details until explicitly versioned as public interchange formats.

## 0.3 Compilation model

A conforming implementation conceptually runs:

```text
.linnet source
    -> syntax tree
    -> name/module resolution
    -> type + shape analysis
    -> typed HIR
    -> Core Tensor IR
    -> optional optimization
    -> backend plan/export/materialization
```

An executor is optional: a conforming frontend can stop after semantic validation or IR generation.

## 0.4 Terminology

- **scalar**: a non-tensor value such as `f32`, `i32`, or `bool`.
- **tensor**: a value with a static rank and symbolic or concrete dimensions.
- **dimension expression**: a compile-time integer expression in a tensor shape.
- **shape pack**: a variadic dimension sequence: `*S` in generic declarations, `*s` in index notation.
- **semantic op**: an `op` whose identity may survive optimization; its body gives the canonical semantics.
- **block**: a structural component with parameters, buffers, state, sub-blocks, and methods.
- **entry**: a public execution entry point. A block's entry uses the block's parameters and state; a module-level entry is a **function** of its inputs alone (§7.6).
- **state member**: mutable per-instance block state, such as a KV cache (§9.3).
- **primitive**: an operation Core Tensor IR understands directly, not a user or library `op`.

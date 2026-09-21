# 0. Overview and Design Contract

## 0.1 Purpose

Linnet is a statically typed declarative tensor language. A `.linnet` file describes tensor computation and model structure without embedding tensor payloads or arbitrary host-language code.

The language is intended to be:

- directly authored by humans;
- suitable as a durable, reviewable model artifact;
- statically checked before weights are loaded;
- portable across execution frameworks and compiler IRs;
- optimizable without requiring model-specific compiler changes.

## 0.2 Architectural invariants

The following rules are normative design constraints for the implementation.

### A. Model independence

The compiler core MUST NOT contain model-specific names such as `Llama`, `Qwen`, `Gemma`, or equivalent architecture identifiers.

### B. Library-defined semantic operations

High-level operations such as `linear`, `rope`, `rms_norm`, `attention`, `conv`, and `moe` SHOULD be implemented as Linnet `op` declarations in `.linnet` libraries. They MUST NOT require compiler modifications unless the operation fundamentally cannot be expressed with the existing primitive tensor algebra.

### C. Primitive fallback

Every non-extern semantic `op` MUST have a valid body defining its canonical semantics in lower-level Linnet operations. A backend MAY replace an `op` or a recognized decomposition with a native implementation, but correctness MUST NOT depend on such a replacement.

### D. Weight separation

A `.linnet` source file MUST NOT contain raw parameter tensor payloads. Parameters are symbolic declarations. Weight data is bound separately through formats such as SafeTensors and a non-executable binding manifest.

### E. No hidden host execution

Parsing, type checking, graph inspection, and parameter-manifest generation MUST NOT execute Python, C++, shell commands, dynamic libraries, network requests, package build scripts, or arbitrary user code.

### F. Strict static semantics

Shape and dtype compatibility MUST be proven statically whenever the language claims an operation is valid. The compiler MUST NOT silently insert dtype conversions or accept unresolved broadcasting relationships merely because a runtime framework might tolerate them.

### G. Single semantic frontend

The CLI, language server, formatter, graph visualizer, and backend pipeline MUST consume the same parser and semantic-analysis implementation. Editor tooling MUST NOT reimplement Linnet's type or shape system in TypeScript, Vimscript, Python, or another language.

### H. Source stability over IR stability

`.linnet` source and the language specification are the durable interface. Internal HIR, Core IR, optimizer representations, and plan formats are implementation details until explicitly versioned as public interchange formats.

## 0.3 Compilation model

A conforming implementation conceptually performs:

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

A compiler is not required to provide an executor. A conforming frontend can stop after semantic validation or IR generation.

## 0.4 Terminology

- **scalar**: a non-tensor value such as `f32`, `i32`, or `bool`.
- **tensor**: a value with a statically described rank and symbolic or concrete dimensions.
- **dimension expression**: a compile-time integer expression used in tensor shapes.
- **shape pack**: a variadic sequence of dimensions, written `*S` in generic declarations and `*s` in index notation.
- **semantic op**: an `op` declaration whose identity may be preserved through optimization while its body provides canonical semantics.
- **block**: a structural component containing parameters, buffers, sub-blocks, and callable methods.
- **entry**: a public execution entry point.
- **primitive**: an operation understood directly by Core Tensor IR rather than implemented as a user/library `op`.

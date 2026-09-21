# 13. Safety, Determinism, and Reviewability

## 13.1 Threat model

A user should be able to download an unfamiliar `.linnet` model definition and statically inspect/check it without granting arbitrary code execution.

## 13.2 Forbidden implicit capabilities

Core Linnet source has no facilities for:

- filesystem access;
- sockets/network access;
- subprocess creation;
- dynamic library loading;
- environment-variable access;
- host-language evaluation;
- arbitrary native callbacks;
- package installation scripts;
- compile-time code execution from user packages.

## 13.3 Imports

Imports resolve only through the package/module resolver and fixed package metadata. Source code cannot construct dynamic import paths from runtime values.

## 13.4 Macros

Procedural macros and arbitrary compile-time execution are intentionally absent. Future syntax-level metaprogramming, if introduced, must preserve the safety model and deterministic expansion.

## 13.5 Extern operations

`extern` is reserved but disabled in the initial language version.

A future `extern op` mechanism must be opt-in at compile/materialization time and must clearly separate safe semantic checking from trusted backend plugin execution.

## 13.6 Determinism

Pure Linnet code is deterministic for a fixed set of inputs under the language's numeric semantics. Randomness, state, clocks, and external effects require explicit future constructs.

## 13.7 Numeric nondeterminism

Backend parallel reductions may differ in floating-point bit patterns when the selected numeric-equivalence policy allows reassociation or other non-IEEE-exact transformations. Such optimizer policies are outside the source-language semantics and must be explicit compiler options.

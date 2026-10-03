# 13. Safety and determinism

Source cannot reach the host.

## 13.1 Threat model

A user should be able to statically check an unfamiliar `.linnet` model without granting it arbitrary code execution.

## 13.2 Forbidden implicit capabilities

Core Linnet source has no facilities for filesystem, network, or environment-variable access, subprocess creation, dynamic library loading, host-language evaluation, arbitrary native callbacks, package installation scripts, or compile-time code execution from user packages.

## 13.3 Imports

Imports resolve only through the package/module resolver and fixed package metadata; no import path is built from runtime values.

## 13.4 Macros

Linnet has no procedural macros or arbitrary compile-time execution. Future syntax-level metaprogramming must preserve the safety model and deterministic expansion.

## 13.5 Extern operations

`extern` is reserved but disabled in the initial language version. A future `extern op` (§15.8) must be opt-in at compile or materialization time and separate safe semantic checking from trusted backend plugin execution.

## 13.6 Determinism

For fixed inputs, parameters, and initial state, Linnet code is deterministic under the language's numeric semantics, with no clocks or external effects. Randomness comes only from keys the caller passes (§15.2); no primitive draws random numbers.

## 13.7 Numeric nondeterminism

Backend parallel reductions may differ in floating-point bits when the selected numeric-equivalence policy allows reassociation or other non-IEEE-exact transformations; such policies are outside source-language semantics and must be explicit compiler options.

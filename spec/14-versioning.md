# 14. Versioning and Compatibility

## 14.1 Language version

A package manifest declares the language version it targets:

```toml
[package]
language = "0.1"
```

## 14.2 Pre-1.0 policy

Before language 1.0, syntax and semantics may evolve between minor language versions. Implementations SHOULD provide clear migration diagnostics where feasible.

## 14.3 Stable artifact boundary

The long-term stable artifact is human-readable Linnet source plus declared external parameter artifacts/bindings.

Internal HIR, Core IR, e-graph representation, optimizer caches, and backend Plan IR are not stable interchange formats unless separately versioned.

## 14.4 Reserved syntax

Reserved keywords and syntax forms exist so future capabilities can be introduced without reinterpreting previously valid user identifiers.

## 14.5 Standard library versioning

Semantic `op` identity depends on the library/package version. A backend optimization registered for one semantic version range MUST NOT silently apply to incompatible semantics.

## 14.6 Backend compatibility

A backend may reject a valid Linnet program when it cannot represent or execute required semantics. Such rejection is a backend capability error, not a source-language type error.

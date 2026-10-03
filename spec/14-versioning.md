# 14. Versioning and compatibility

Version compatibility rules.

## 14.1 Language version

A package manifest declares its language version:

```toml
[package]
language = "0.1"
```

## 14.2 Pre-1.0 policy

Before 1.0, minor versions may change syntax and semantics. Implementations SHOULD provide migration diagnostics where feasible.

## 14.3 Stable artifacts

Stable long-term: source plus declared parameter artifacts and bindings. Unstable unless separately versioned: compiler IRs (HIR, Core IR, Plan IR), e-graphs, optimizer caches.

## 14.4 Reserved syntax

Reserved syntax (§1.4) lets future features avoid reinterpreting previously valid identifiers.

## 14.5 Library versions

Semantic `op` identity depends on the library/package version; backend optimizations registered for one semantic version range MUST NOT silently apply to incompatible semantics.

## 14.6 Backends

A backend may reject a valid program it cannot represent or execute: a capability error, not a type error.

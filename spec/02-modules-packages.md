# 2. Modules and Packages

## 2.1 Module declaration

A source file declares one module:

```text
module models.llama.attention
```

A package compiler SHOULD verify that module paths correspond consistently to source paths, but standalone checking MAY relax that relationship.

## 2.2 Imports

Imports use logical module paths, never filesystem path strings:

```text
use std.nn::{linear, rms_norm}
use std.nn.attention::{attention}
use crate.layers.decoder::{DecoderLayer}
use my_dependency.ops::{custom_op}
```

This is intentionally safer than executable or relative path imports. Imports MUST NOT contain `..`, absolute filesystem paths, URLs, or shell expansions.

### Namespaces

- `std`: toolchain-provided standard library.
- `crate`: current package root.
- dependency key: a dependency declared in `linnet.toml`.

`self` and `super` are reserved; support MAY be added later.

## 2.3 Core prelude

Every module has an implicit, non-shadowable language prelude containing only compiler-defined foundational names. The initial prelude includes:

- scalar dtype names and `Tensor`;
- generic kinds/constraints such as `Dim`, `Shape`, `DType`, `Numeric`, `Integer`, and `Float`;
- primitive casts and tensor-shape functions such as `cast`, `reshape`, `permute`, `broadcast_to`, `concat`, and `pad`;
- primitive data functions such as `iota`, `fill`, `gather`, and `scatter`;
- primitive elementwise math functions such as `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, and `abs`;
- `select`.

The prelude MUST remain small and model-independent. `linear`, `softmax`, `attention`, `rope`, normalization layers, convolutions, and similar semantic operations are library code and are not implicit prelude names.

Prelude names may be referenced without a `use` declaration. No declaration — item, parameter, generic parameter, or local — may use a prelude name.

### Prelude functions

In the signatures below, `x` stands for a scalar or a tensor, and the result has the shape of its tensor operands after broadcasting.

```text
cast<T>(x)                     same shape, dtype T; T is required
exp log sqrt rsqrt sin cos tanh (x)    x must have a Float dtype
abs(x)                         x must have a Numeric dtype
min(a, b)  max(a, b)           elementwise on Numeric operands of one dtype;
                               on two compile-time integers, a dimension
select(condition, a, b)        condition is bool; a and b share one dtype
reshape(x, shape)              element counts must be provably equal
broadcast_to(x, shape)         each trailing axis of x must equal the target or be 1
permute(x, axes)               axes is a permutation of 0..rank-1; rank must be known
concat(a, b, ..., axis = k)    equal shapes except along axis k; `axis` is a keyword.
                               k >= 0 counts axes from the front, k < 0 from the back;
                               no shape pack may lie on the side being counted
iota<T = i64>(n)               Tensor[n; T] holding 0, 1, ..., n - 1
fill<T>(shape, value)          T may be omitted when `value` is a typed scalar
```

`shape` and `axes` arguments are shape literals. `pad`, `gather`, and `scatter` are reserved prelude names whose signatures are not yet specified; an implementation MUST reject calls to them rather than guess.

## 2.4 Visibility

Items are private to their module unless declared `pub`.

```text
pub op linear(...) { ... }
fn helper(...) { ... }
```

## 2.5 Package manifest

The canonical project manifest is `linnet.toml`.

Minimum form:

```toml
[package]
name = "example-model"
version = "0.1.0"
language = "0.1"

[dependencies]
foo = { path = "../foo" }
```

Future dependency forms MAY include pinned Git revisions and a registry. Dependency resolution MUST be reproducible when a lockfile exists.

## 2.6 Lockfile

The canonical lockfile is `linnet.lock`.

A lockfile is generated metadata and MUST record exact dependency identities required for deterministic package resolution. The precise serialization is implementation-defined until separately standardized.

## 2.7 Package safety

Fetching a package MUST NOT execute package-provided scripts.

A Linnet package MAY contain source, metadata, tests, and non-executable assets. Merely resolving or checking a package MUST NOT execute arbitrary code.

## 2.8 Cycles

Value-level initialization cycles are forbidden.

Module import cycles MAY be accepted only if the implementation can resolve declarations without order-dependent initialization. The initial implementation SHOULD reject import cycles with a clear diagnostic; a later version may permit safe declaration-only cycles.

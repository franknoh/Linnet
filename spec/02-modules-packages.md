# 2. Modules and packages

One file is one module; `linnet.toml` defines a package.

## 2.1 Module declaration

A source file declares one module:

```text
module models.llama.attention
```

A package compiler SHOULD verify that module paths correspond to source paths; standalone checking MAY relax this.

## 2.2 Imports

Imports use logical module paths, never filesystem paths:

```text
use std.nn::{linear, rms_norm}
use std.nn.attention::{attention}
use crate.layers.decoder::{DecoderLayer}
use my_dependency.ops::{custom_op}
```

Imports MUST NOT contain `..`, absolute filesystem paths, URLs, or shell expansions.

### Namespaces

- `std`: toolchain-provided standard library.
- `crate`: current package root.
- dependency key: a dependency declared in `linnet.toml`.

`self` and `super` are reserved; a later version MAY support them.

### Locating modules

A logical path maps to exactly one source file:

```text
crate            <package>/src/lib.linnet
crate.a.b        <package>/src/a/b.linnet
std.a.b          <standard library>/a/b.linnet
dep.a.b          <dependency dep>/src/a/b.linnet
```

`<package>` is the nearest directory above the importing file that contains `linnet.toml`. `dep` is a key of that manifest's `[dependencies]` table: an identifier other than `std` or `crate`.

`use a.b::{x, y as z}` imports items from module `a.b`. `use a.b` imports the module as `b`, so its public items are written `b.x`.

The current implementation rejects import cycles between modules (§2.8).

## 2.3 Core prelude

Every module has an implicit prelude of compiler-defined names, including:

- scalar dtype names and `Tensor`;
- generic kinds and constraints: `Dim`, `Shape`, `DType`, `Numeric`, `Integer`, `Float`;
- casts and shape functions: `cast`, `reshape`, `permute`, `broadcast_to`, `concat`, `pad`;
- data functions: `iota`, `fill`, `cumsum`, `gather`, `scatter`;
- elementwise math: `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `abs`;
- `min`, `max`, `shl`, `shr`, and `select`.

The prelude MUST remain small and model-independent.

Prelude names need no `use` declaration. No item, parameter, generic parameter, or local may be declared with a prelude name.

### Prelude functions

`x` is a scalar or a tensor; the result has the broadcast shape of the tensor operands.

```text
cast<T>(x)                     same shape, dtype T; T is required
exp log sqrt rsqrt sin cos tanh (x)    x must have a Float dtype
abs(x)                         x must have a Numeric dtype
min(a, b)  max(a, b)           elementwise on Numeric operands of one dtype;
                               on two compile-time integers, a dimension
shl(x, bits)  shr(x, bits)     integer shifts; `shr` is arithmetic for signed
                               dtypes and logical for unsigned ones (§5.4)
select(condition, a, b)        condition is bool; a and b share one dtype
reshape(x, shape)              element counts must be provably equal
broadcast_to(x, shape)         each trailing axis of x must equal the target or be 1
permute(x, axes)               axes is a permutation of 0..rank-1; rank must be known
concat(a, b, ..., axis = k)    equal shapes except along axis k; `axis` is a keyword.
                               k >= 0 counts axes from the front, k < 0 from the back;
                               no shape pack may lie on the side being counted
iota<T = i64>(n)               Tensor[n; T] holding 0, 1, ..., n - 1
fill<T>(shape, value)          T may be omitted when `value` is a typed scalar
cumsum(x, axis = k)            running sums of a Numeric tensor along axis k, same type;
                               k counts axes as in `concat`
```

`shape` and `axes` arguments are shape literals. `pad`, `gather`, and `scatter` have no specified signature; an implementation MUST reject calls to them.

## 2.4 Visibility

Items are private to their module unless declared `pub`.

```text
pub op linear(...) { ... }
fn helper(...) { ... }
```

## 2.5 Package manifest

The canonical project manifest is `linnet.toml`. Minimum form:

```toml
[package]
name = "example-model"
version = "0.1.0"
language = "0.1"

[dependencies]
foo = { path = "../foo" }
```

`language` is the targeted language version; a toolchain MUST reject a version it does not implement.

Future dependency forms MAY include pinned Git revisions and a registry. Dependency resolution MUST be reproducible when a lockfile exists.

## 2.6 Lockfile

The canonical lockfile is `linnet.lock`. It MUST record the exact dependency identities that deterministic resolution requires; its serialization is implementation-defined.

## 2.7 Package safety

A package MAY contain source, metadata, tests, and non-executable assets.

- Fetching a package MUST NOT execute package-provided scripts.
- Resolving or checking a package MUST NOT execute arbitrary code.

## 2.8 Cycles

Value-level initialization cycles are forbidden.

An implementation MAY accept module import cycles only if it can resolve declarations without order-dependent initialization. The initial implementation SHOULD reject import cycles with a clear diagnostic.

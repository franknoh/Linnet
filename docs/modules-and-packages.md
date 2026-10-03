# Modules and packages

A module is a file; a package is a directory with a `linnet.toml`. Imports
are logical paths inside their root.

## Imports

```linnet
module models.llama.attention

use std.nn::{linear, rms_norm as norm}     // items, optionally renamed
use crate.layers.decoder::{DecoderLayer}   // from this package
use shared_layers.ops                      // a dependency's module, used as `ops.x`
```

| Path | File |
| --- | --- |
| `crate` | `<package>/src/lib.linnet` |
| `crate.a.b` | `<package>/src/a/b.linnet` |
| `std.a.b` | `<standard library>/a/b.linnet` |
| `dep.a.b` | `<dependency dep>/src/a/b.linnet` |

`<package>` is the nearest directory above with a `linnet.toml`;
`<standard library>` is `--std <dir>` or `LINNET_STD`. Items are private
unless `pub`; import cycles are errors.

## Packages

```bash
linnet init my-model
```

```text
my-model/
  linnet.toml
  src/lib.linnet
```

```toml
[package]
name = "my_model"
version = "0.1.0"
language = "0.1"

[dependencies]
shared_layers = { path = "../shared-layers" }
```

- `language` is the targeted language version; `0.1` is accepted.
- A dependency's key prefixes its imports; `std` and `crate` are reserved.
- Dependencies are only paths, relative to the manifest, each with its own
  `linnet.toml`. Git, registries, and lock files are not supported.

Checking a package only reads its files.

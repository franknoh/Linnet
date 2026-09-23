# Modules and packages

How files find each other. A module is a file; a package is a directory with
a `linnet.toml`; imports are logical paths that map to files without ever
leaving their root.

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

`<package>` is the nearest directory above the importing file that contains
`linnet.toml`. The standard library directory is `--std <dir>` or
`LINNET_STD`. Items are private unless `pub`, and modules may not import each
other in a cycle.

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

- `language` is the language version the package targets; this toolchain
  accepts `0.1`.
- A dependency key is an identifier and becomes the first segment of imports
  from that package. `std` and `crate` are reserved.
- Only path dependencies exist. They are relative to the manifest and must
  point at a directory with its own `linnet.toml`; their own dependencies
  resolve relative to that manifest. Git and registry sources, and a lock
  file, are future work.

Resolving a package only reads files. `linnet check` never executes anything
in it.

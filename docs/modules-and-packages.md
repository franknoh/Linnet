# Modules and packages

## Modules

Every source file declares one module and may import others by logical path:

```text
module models.llama.attention

use std.nn::{linear, rms_norm as norm}
use crate.layers.decoder::{DecoderLayer}
use my_dependency.ops
```

`use a.b::{x, y as z}` imports items; `use a.b` imports the module itself, so
that its items are written `b.x`. Items are private unless declared `pub`.
Modules must not import each other in a cycle.

A logical path maps to one file:

| Path | File |
| --- | --- |
| `crate` | `<package>/src/lib.linnet` |
| `crate.a.b` | `<package>/src/a/b.linnet` |
| `std.a.b` | `<standard library>/a/b.linnet` |
| `dep.a.b` | `<dependency dep>/src/a/b.linnet` |

`<package>` is the nearest directory above the importing file that contains
`linnet.toml`. The standard library directory comes from `--std <dir>` or the
`LINNET_STD` environment variable. Because path segments are identifiers, an
import can never leave its root directory.

## Packages

`linnet init [dir]` creates a package:

```text
my-model/
├── linnet.toml
└── src/
    └── lib.linnet
```

`linnet.toml`:

```toml
[package]
name = "my_model"
version = "0.1.0"
language = "0.1"

[dependencies]
shared_layers = { path = "../shared-layers" }
```

- `language` is the language version the package is written for; this
  toolchain accepts `0.1`.
- A dependency key is an identifier (letters, digits, `_`), not `std` or
  `crate`; it is the first segment of imports from that package.
- Path dependencies are relative to the manifest's directory and must point at
  a directory with its own `linnet.toml`. Their own dependencies resolve
  relative to their own manifest.
- Only path dependencies exist. Git and registry sources, and the `linnet.lock`
  file that pins them, come later.

Reading a manifest or resolving a package only reads files. Nothing in a
package is ever executed by `linnet check`.

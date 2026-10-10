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
- A path dependency is relative to the manifest. Every dependency is a
  directory with its own `linnet.toml`.

## Git dependencies

```toml
[dependencies]
layers = { github = "owner/repo", tag = "v0.2" }
llama = { github = "franknoh/nest", subdir = "models/llama-3.1-8b-instruct" }
blocks = { hf = "owner/repo", branch = "main" }
other = { git = "https://gitlab.com/owner/repo.git", rev = "4f2c1e0" }
```

| Key | Meaning |
| --- | --- |
| `github`, `hf`, `git` | a GitHub repository, a Hugging Face Hub repository (`datasets/...` and `spaces/...` too), or any git URL (https, ssh, file) |
| `tag`, `branch`, `rev` | which commit; at most one, the default branch without any |
| `subdir` | the package's directory inside the repository |

```bash
linnet fetch      # check out what linnet.toml names; record commits in linnet.lock
linnet update     # move every git dependency to its newest commit
```

- `linnet.lock` records the commit each dependency resolved to. Commit it:
  every build then reads the same files, whatever the tag or branch names
  later.
- Repositories go to `$LINNET_HOME/git` (`~/.linnet/git`): a bare clone per
  repository and the files of each commit used, named as Hugging Face's
  cache names repositories (`github.com--owner--repo`).
- `check`, the exporters and the other commands fetch what the cache lacks.
  `LINNET_OFFLINE=1` (or `linnet fetch --offline`) uses the cache alone; the
  language server never fetches.
- A dependency not on GitHub or the Hub is fetched with a warning (W1004).
- Linnet reads a dependency's `.linnet` files and `linnet.toml`. It runs
  nothing from it, and leaves Git LFS files (checkpoints) unfetched.

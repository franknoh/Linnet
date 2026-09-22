#!/usr/bin/env bash
# `linnet init` creates a package that checks cleanly and refuses to overwrite.
set -euo pipefail
scratch="$1"
rm -rf "$scratch"
"$LINNET_BIN" init "$scratch"
"$LINNET_BIN" check "$scratch"
if "$LINNET_BIN" init "$scratch" 2>/dev/null; then
    echo "second init should fail" >&2
    exit 1
fi
test -f "$scratch/linnet.toml"
test -f "$scratch/src/lib.linnet"

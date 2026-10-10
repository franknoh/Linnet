#!/usr/bin/env bash
# A git dependency: `linnet fetch` checks it out under $LINNET_HOME and
# records its commit in linnet.lock, `check` imports through it (offline
# too), the lock holds its commit until `linnet update`, and an empty cache
# without the network is reported rather than fetched.
set -euo pipefail
scratch="$1"
rm -rf "$scratch"
mkdir -p "$scratch"
export LINNET_HOME="$scratch/home"
export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.com
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.com

upstream="$scratch/upstream"
mkdir -p "$upstream/packages/layers/src"
cat > "$upstream/packages/layers/linnet.toml" <<'EOF'
[package]
name = "layers"
version = "0.1.0"
language = "0.1"
EOF
cat > "$upstream/packages/layers/src/lib.linnet" <<'EOF'
module layers

pub fn double(x: f32) -> f32 {
    return x * 2.0
}
EOF
git -C "$upstream" init -q
git -C "$upstream" add .
git -C "$upstream" commit -q -m first
git -C "$upstream" tag v1
first=$(git -C "$upstream" rev-parse HEAD)

app="$scratch/app"
"$LINNET_BIN" init "$app"
echo "layers = { git = \"file://$upstream\", tag = \"v1\", subdir = \"packages/layers\" }" \
    >> "$app/linnet.toml"
cat > "$app/src/lib.linnet" <<'EOF'
module app

use layers::{double}

pub fn quadruple(x: f32) -> f32 {
    return double(double(x))
}
EOF

"$LINNET_BIN" fetch "$app" 2> "$scratch/fetch.log"
grep -q "not on GitHub or the Hugging Face Hub" "$scratch/fetch.log"
grep -q "$first" "$app/linnet.lock"
test -f "$LINNET_HOME"/git/file--*upstream/snapshots/"$first"/packages/layers/src/lib.linnet
"$LINNET_BIN" check "$app"
LINNET_OFFLINE=1 "$LINNET_BIN" check "$app"

# A new commit under the same tag: the lock keeps the first until `update`.
echo "// second" >> "$upstream/packages/layers/src/lib.linnet"
git -C "$upstream" commit -q -am second
git -C "$upstream" tag -f v1 > /dev/null
second=$(git -C "$upstream" rev-parse HEAD)
"$LINNET_BIN" check "$app"
grep -q "$first" "$app/linnet.lock"
"$LINNET_BIN" update "$app"
grep -q "$second" "$app/linnet.lock"
"$LINNET_BIN" check "$app"

# Nothing cached and no network: reported, with what to run.
if LINNET_HOME="$scratch/empty" LINNET_OFFLINE=1 "$LINNET_BIN" check "$app" 2> "$scratch/offline.log"; then
    echo "check should fail without the dependency" >&2
    exit 1
fi
grep -q "linnet fetch" "$scratch/offline.log"

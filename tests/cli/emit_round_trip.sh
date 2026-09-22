#!/usr/bin/env bash
# Every library and example file emits source from its Core IR (plain and
# optimized) that checks and lints clean and that emits again unchanged. A
# file inside a package is checked in a copy of its package, so that its
# `crate` imports resolve.
set -euo pipefail
scratch="$1"
std="$2"
shift 2
rm -rf "$scratch"
mkdir -p "$scratch"

# The package directory holding a file, or empty for a standalone file.
package_of() {
    local dir
    dir=$(dirname "$1")
    while [ "$dir" != "/" ] && [ "$dir" != "." ]; do
        if [ -f "$dir/linnet.toml" ]; then
            echo "$dir"
            return
        fi
        dir=$(dirname "$dir")
    done
}

for file in "$@"; do
    name=$(echo "$file" | tr '/:' '__')
    package=$(package_of "$file")
    for flag in "" "-O"; do
        out="$scratch/$name$flag.linnet"
        # shellcheck disable=SC2086
        "$LINNET_BIN" inspect --emit $flag --std "$std" "$file" > "$out"
        if [ -n "$package" ]; then
            copy="$scratch/pkg_$name$flag"
            rm -rf "$copy"
            cp -r "$package" "$copy"
            target="$copy/${file#"$package"/}"
            cp "$out" "$target"
            "$LINNET_BIN" lint --std "$std" "$copy"
            "$LINNET_BIN" inspect --emit --std "$std" "$target" > "$out.again"
        else
            "$LINNET_BIN" lint --std "$std" "$out"
            "$LINNET_BIN" inspect --emit --std "$std" "$out" > "$out.again"
        fi
        if ! cmp -s "$out" "$out.again"; then
            echo "emitting $file$flag twice differs:" >&2
            diff "$out" "$out.again" >&2 || true
            exit 1
        fi
    done
done

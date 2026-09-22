#!/usr/bin/env bash
# Every library and example file emits source from its Core IR (plain and
# optimized) that checks and lints clean and that emits again unchanged.
set -euo pipefail
scratch="$1"
std="$2"
shift 2
rm -rf "$scratch"
mkdir -p "$scratch"
for file in "$@"; do
    name=$(echo "$file" | tr '/:' '__')
    for flag in "" "-O"; do
        out="$scratch/$name$flag.linnet"
        # shellcheck disable=SC2086
        "$LINNET_BIN" inspect --emit $flag --std "$std" "$file" > "$out"
        "$LINNET_BIN" lint --std "$std" "$out"
        "$LINNET_BIN" inspect --emit --std "$std" "$out" > "$out.again"
        if ! cmp -s "$out" "$out.again"; then
            echo "emitting $file$flag twice differs:" >&2
            diff "$out" "$out.again" >&2 || true
            exit 1
        fi
    done
done

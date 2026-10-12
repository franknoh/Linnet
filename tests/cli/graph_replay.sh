#!/usr/bin/env bash
# `linnet graph --replay` prints what the target prints, so the recorded
# graph holds every call the export made. Each line of the cases file is
# `<target> <args...>`, paths relative to the repository.
set -euo pipefail
scratch="$1"
repo="$2"
cases="$3"
rm -rf "$scratch"
mkdir -p "$scratch"
cd "$repo"
count=0
while read -r target args; do
    case "$target" in "" | "#"*) continue ;; esac
    count=$((count + 1))
    # shellcheck disable=SC2086
    "$LINNET_BIN" "$target" --std stdlib $args > "$scratch/$count.direct"
    # shellcheck disable=SC2086
    "$LINNET_BIN" graph --target "$target" --replay --std stdlib $args > "$scratch/$count.replay"
    if ! cmp -s "$scratch/$count.direct" "$scratch/$count.replay"; then
        echo "replay differs: $target $args"
        diff "$scratch/$count.direct" "$scratch/$count.replay" | head -20
        exit 1
    fi
    # shellcheck disable=SC2086
    "$LINNET_BIN" graph --target "$target" --std stdlib $args > "$scratch/$count.json"
    grep -q '^{"version":1,"target":' "$scratch/$count.json"
done < "$cases"
echo "$count cases replay unchanged"

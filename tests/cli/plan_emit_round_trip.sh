#!/usr/bin/env bash
# `linnet emit` of a plan prints the same source as emitting the file
# directly, and a plan reduced to the file's own declarations (what a
# framework adapter produces) still emits a module that lints clean.
set -euo pipefail
scratch="$1"
std="$2"
shift 2
rm -rf "$scratch"
mkdir -p "$scratch"
for file in "$@"; do
    name=$(basename "$file" .linnet)
    "$LINNET_BIN" plan --std "$std" "$file" > "$scratch/$name.json"
    "$LINNET_BIN" emit "$scratch/$name.json" > "$scratch/$name.from_plan.linnet"
    "$LINNET_BIN" inspect --emit --std "$std" "$file" > "$scratch/$name.direct.linnet"
    "$LINNET_BIN" fmt "$scratch/$name.direct.linnet"
    if ! cmp -s "$scratch/$name.direct.linnet" "$scratch/$name.from_plan.linnet"; then
        echo "plan and direct emission of $file differ:" >&2
        diff "$scratch/$name.direct.linnet" "$scratch/$name.from_plan.linnet" >&2 || true
        exit 1
    fi
    module=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["module"])' "$scratch/$name.json")
    python3 - "$scratch/$name.json" "$module" "$scratch/$name.own.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
module = sys.argv[2]
plan["functions"] = [f for f in plan["functions"] if f["name"].startswith(module + "::")]
plan["blocks"] = {k: v for k, v in plan["blocks"].items() if v["module"] == module}
json.dump(plan, open(sys.argv[3], "w"))
PY
    "$LINNET_BIN" emit "$scratch/$name.own.json" > "$scratch/$name.own.linnet"
    "$LINNET_BIN" lint --std "$std" "$scratch/$name.own.linnet"
done

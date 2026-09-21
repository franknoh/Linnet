#!/usr/bin/env bash
# Verify that all project C++ sources are clang-format clean.
# Pass --fix to rewrite files in place instead.
set -euo pipefail

cd "$(dirname "$0")/.."

mode=(--dry-run --Werror)
if [[ "${1:-}" == "--fix" ]]; then
    mode=(-i)
fi

git ls-files -z -c -o --exclude-standard -- '*.cpp' '*.hpp' |
    xargs -0 -r "${CLANG_FORMAT:-clang-format}" "${mode[@]}"

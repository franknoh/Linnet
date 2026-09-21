#!/usr/bin/env bash
# Run the local quality gates: format check, then configure, build, and test
# each requested CMake preset (default: debug).
set -euo pipefail

cd "$(dirname "$0")/.."

presets=("$@")
if [[ ${#presets[@]} -eq 0 ]]; then
    presets=(debug)
fi

scripts/check-format.sh

for preset in "${presets[@]}"; do
    cmake --preset "$preset"
    cmake --build --preset "$preset"
    if [[ "$preset" != "tidy" ]]; then
        ctest --preset "$preset"
    fi
done

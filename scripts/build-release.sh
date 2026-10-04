#!/usr/bin/env bash
# Builds and tests the compiler for a release, and installs it into a prefix:
# bin/linnet and share/linnet/stdlib. The release workflow packs the prefix
# into an archive and into the Python wheels.
set -euo pipefail

prefix=${1:?usage: scripts/build-release.sh <prefix>}
cd "$(dirname "$0")/.."

# A newer compiler than CI's may warn where CI does not; that is no reason to
# fail a release.
cmake --preset release -DLINNET_WARNINGS_AS_ERRORS=OFF
cmake --build --preset release
ctest --preset release
cmake --install build/release --prefix "$prefix"

#!/bin/sh
# Installs the `linnet` command (linnet-cli: the compiler, the language
# server and `linnet serve`) as a uv tool, and puts uv's tool directory on
# PATH.
#
#     curl -LsSf https://linnet.franknoh.dev/install.sh | sh
#     curl -LsSf https://linnet.franknoh.dev/install.sh | sh -s -- --no-modify-path
#
# The latest release's wheel for this platform carries a built compiler; where
# there is none (or with --from-source), uv builds the compiler from the
# repository, which needs a C++23 compiler (GCC 13, Clang 19). uv is
# installed first if it is missing.
#
# Options (or environment variables):
#   --no-modify-path   LINNET_NO_MODIFY_PATH=1   leave the shell's PATH alone
#   --from-source      LINNET_FROM_SOURCE=1      build from the repository
#   --ref <ref>        LINNET_REF=<ref>          the branch, tag or commit to build
set -eu

repo="franknoh/Linnet"
modify_path="${LINNET_NO_MODIFY_PATH:-0}"
from_source="${LINNET_FROM_SOURCE:-0}"
ref="${LINNET_REF:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --no-modify-path) modify_path=1 ;;
        --from-source) from_source=1 ;;
        --ref) ref="$2"; from_source=1; shift ;;
        -h | --help) sed -n '2,19p' "$0" 2>/dev/null || true; exit 0 ;;
        *) echo "install.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

say() { printf 'linnet: %s\n' "$*"; }

fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$1"
    else
        echo "install.sh: needs curl or wget" >&2
        exit 1
    fi
}

if ! command -v uv >/dev/null 2>&1; then
    say "installing uv (https://docs.astral.sh/uv/)"
    if [ "$modify_path" = 1 ]; then
        fetch https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    else
        fetch https://astral.sh/uv/install.sh | sh
    fi
    PATH="$HOME/.local/bin:$PATH"
    if [ -n "${XDG_BIN_HOME:-}" ]; then PATH="$XDG_BIN_HOME:$PATH"; fi
    export PATH
fi

# The wheel for this platform from the latest release, if there is one.
wheel=""
if [ "$from_source" != 1 ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) pattern="manylinux.*_x86_64" ;;
        Linux-aarch64 | Linux-arm64) pattern="manylinux.*_aarch64" ;;
        Darwin-arm64) pattern="macosx_.*_arm64" ;;
        *) pattern="" ;;
    esac
    if [ -n "$pattern" ]; then
        wheel=$(fetch "https://api.github.com/repos/$repo/releases/latest" 2>/dev/null |
            grep -o '"browser_download_url": *"[^"]*linnet_cli-[^"]*\.whl"' |
            grep -o 'https://[^"]*' | grep -E "$pattern" | head -n 1) || wheel=""
    fi
fi

if [ -n "$wheel" ]; then
    say "installing ${wheel##*/}"
    uv tool install --force "linnet-cli @ $wheel"
else
    source="git+https://github.com/$repo"
    if [ -n "$ref" ]; then source="$source@$ref"; fi
    say "building from $source (needs a C++23 compiler)"
    uv tool install --force "linnet-cli @ $source#subdirectory=python/cli"
fi

if [ "$modify_path" = 1 ]; then
    say "PATH left alone; run \`uv tool update-shell\` to add $(uv tool dir --bin)"
else
    uv tool update-shell || true
fi

if command -v linnet >/dev/null 2>&1; then
    say "$(linnet --version) at $(command -v linnet)"
else
    say "installed in $(uv tool dir --bin); open a new shell to use \`linnet\`"
fi

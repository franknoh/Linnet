#!/bin/sh
# Installs the `linnet` command (linnet-cli: the compiler, the language
# server and `linnet serve`) as a uv tool, and puts uv's tool directory on
# PATH.
#
#     curl -LsSf https://linnet.franknoh.dev/install.sh | sh
#     curl -LsSf https://linnet.franknoh.dev/install.sh | sh -s -- --with jax,onnx
#
# The latest release's wheel for this platform carries a built compiler; where
# there is none (or with --from-source), uv builds the compiler from the
# repository, which needs a C++23 compiler (GCC 13, Clang 19). uv is
# installed first if it is missing. PyTorch is the build for this machine's
# GPU (uv's --torch-backend auto: CUDA by the driver's version, ROCm, Intel
# XPU, else the CPU); a Mac gets MLX too. `linnet self update` runs this
# again with the same options.
#
# Options (or environment variables):
#   --with <backends>      LINNET_WITH=<backends>       more backends, comma-separated:
#                                                        jax, onnx, mlx
#   --torch-backend <b>    LINNET_TORCH_BACKEND=<b>     PyTorch's build: auto (default),
#                                                        cpu, cu130, cu126, rocm7.2, xpu, ...
#   --no-modify-path       LINNET_NO_MODIFY_PATH=1      leave the shell's PATH alone
#   --from-source          LINNET_FROM_SOURCE=1         build from the repository
#   --ref <ref>            LINNET_REF=<ref>             the branch, tag or commit to build
set -eu

repo="franknoh/Linnet"
modify_path="${LINNET_NO_MODIFY_PATH:-0}"
from_source="${LINNET_FROM_SOURCE:-0}"
ref="${LINNET_REF:-}"
with="${LINNET_WITH:-}"
torch_backend="${LINNET_TORCH_BACKEND:-auto}"
# The options as given, for `linnet self update` to install with again.
options=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-modify-path) modify_path=1 ;;
        --from-source) from_source=1 ;;
        --ref) ref="$2"; from_source=1; options="$options --ref $2"; shift ;;
        --with) with="$2"; shift ;;
        --torch-backend) torch_backend="$2"; shift ;;
        -h | --help) sed -n '2,25p' "$0" 2>/dev/null || true; exit 0 ;;
        *) echo "install.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
if [ "$modify_path" = 1 ]; then options="$options --no-modify-path"; fi
if [ "$from_source" = 1 ] && [ -z "$ref" ]; then options="$options --from-source"; fi
if [ -n "$with" ]; then options="$options --with $with"; fi
if [ "$torch_backend" != auto ]; then options="$options --torch-backend $torch_backend"; fi

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

platform="$(uname -s)-$(uname -m)"

# The wheel for this platform from the latest release, if there is one, and
# the release's tag.
wheel=""
tag=""
if [ "$from_source" != 1 ]; then
    case "$platform" in
        Linux-x86_64) pattern="manylinux.*_x86_64" ;;
        Linux-aarch64 | Linux-arm64) pattern="manylinux.*_aarch64" ;;
        Darwin-arm64) pattern="macosx_.*_arm64" ;;
        *) pattern="" ;;
    esac
    if [ -n "$pattern" ]; then
        release=$(fetch "https://api.github.com/repos/$repo/releases/latest" 2>/dev/null) || release=""
        wheel=$(printf '%s\n' "$release" |
            grep -o '"browser_download_url": *"[^"]*linnet_cli-[^"]*\.whl"' |
            grep -o 'https://[^"]*' | grep -E "$pattern" | head -n 1) || wheel=""
        tag=$(printf '%s\n' "$release" | grep -o '"tag_name": *"[^"]*"' |
            head -n 1 | sed 's/.*"\([^"]*\)"$/\1/') || tag=""
    fi
fi

# The CUDA of an NVIDIA driver: 13 from driver 580, else 12.
cuda=""
if command -v nvidia-smi >/dev/null 2>&1; then
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null |
        head -n 1 | cut -d. -f1) || driver=""
    case "$driver" in
        '' | *[!0-9]*) ;;
        *) if [ "$driver" -ge 580 ]; then cuda=13; else cuda=12; fi ;;
    esac
fi

# The packages each backend adds, for this machine.
extras=""
add() { extras="$extras --with $1"; }
case "$platform" in Darwin-arm64) with="mlx,$with" ;; esac
for backend in $(printf '%s' "$with" | tr ',' ' ' | tr ' ' '\n' | sort -u); do
    case "$backend" in
        jax)
            if [ -n "$cuda" ]; then add "jax[cuda$cuda]"; else add jax; fi ;;
        onnx)
            add onnx
            # onnxruntime-gpu's CUDA libraries from PyPI are CUDA 13's.
            if [ "$cuda" = 13 ]; then add "onnxruntime-gpu[cuda,cudnn]"; else add onnxruntime; fi ;;
        mlx)
            case "$platform" in
                Darwin-*) add mlx ;;
                *) if [ -n "$cuda" ]; then add "mlx[cuda$cuda]"; else add "mlx[cpu]"; fi ;;
            esac ;;
        *) echo "install.sh: unknown backend $backend (jax, onnx, mlx)" >&2; exit 2 ;;
    esac
done

# uv picks PyTorch's build itself from 0.8; an older uv takes PyPI's.
torch=""
if uv tool install --help 2>/dev/null | grep -q -- '--torch-backend'; then
    torch="--torch-backend $torch_backend"
else
    say "this uv cannot pick PyTorch's build for the GPU; \`uv self update\` updates it"
fi

if [ -n "$wheel" ]; then
    say "installing ${wheel##*/}$extras"
    # shellcheck disable=SC2086 # options, split as words
    uv tool install --force $torch $extras "linnet-cli @ $wheel"
else
    source="git+https://github.com/$repo"
    if [ -n "$ref" ]; then source="$source@$ref"; fi
    say "building from $source (needs a C++23 compiler)"
    # shellcheck disable=SC2086
    uv tool install --force $torch $extras "linnet-cli @ $source#subdirectory=python/cli"
fi

# What `linnet self update` reads: the release installed and the options to
# install with again.
record="${LINNET_HOME:-$HOME/.linnet}/install"
mkdir -p "$(dirname "$record")"
{
    echo "tag=$tag"
    echo "options=${options# }"
} > "$record"

if [ "$modify_path" = 1 ]; then
    say "PATH left alone; run \`uv tool update-shell\` to add $(uv tool dir --bin)"
else
    uv tool update-shell || true
fi

if command -v linnet >/dev/null 2>&1; then
    say "$(linnet --version) at $(command -v linnet)${tag:+ ($tag)}"
else
    say "installed in $(uv tool dir --bin); open a new shell to use \`linnet\`"
fi

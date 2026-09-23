#!/usr/bin/env bash
# Prepares a fresh GPU pod (RunPod's `runpod/pytorch:*-cu128*-ubuntu2404`
# image or any Ubuntu 24.04 with CUDA PyTorch) to run `bench/run.py`:
# builds the compiler, installs the Python adapters next to the image's
# PyTorch, and adds a CUDA JAX for the XLA rows. Run it in the checkout.
#
#     git clone --branch "$BRANCH" https://github.com/franknoh/Linnet.git && cd Linnet
#     bash bench/setup-pod.sh
#     source /workspace/venv/bin/activate
#     LINNET_BIN=build/release/linnet python bench/run.py --device cuda --configs small,medium,large
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends cmake ninja-build g++-13 gcc-13 git >/dev/null

# The compiler, release build.
CC=gcc-13 CXX=g++-13 cmake --preset release >/dev/null
cmake --build --preset release
build/release/linnet --version

# Adapters in a venv that sees the image's CUDA PyTorch (the system Python
# is externally managed, so nothing is installed into it directly).
python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e python/linnet_torch -e python/linnet_jax
python -m pip install --quiet "jax[cuda12]" safetensors
python - <<'EOF'
import jax, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("jax", jax.__version__, jax.default_backend(), jax.devices())
EOF

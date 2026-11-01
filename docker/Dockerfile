# syntax=docker/dockerfile:1
#
# Linnet's serving images, one target each, built from the repository root.
# docker-bake.hcl names every combination with its versions pinned
# (`docker buildx bake -f docker/docker-bake.hcl --list targets`); one alone:
#
#     docker build -f docker/Dockerfile --target serve-torch -t linnet-serve:torch-cu130 .
#     docker run --gpus all -p 8000:8000 -v linnet-cache:/cache linnet-serve:torch-cu130 llama-3.1-8b-instruct
#
# serve-torch, serve-jax, serve-onnx,  `linnet serve` with that backend;
# serve-mlx                            arguments after the image are its
# triton, vllm                         that server, serving MODEL exported
#                                      on first start (export-and-serve.sh)

ARG UBUNTU=24.04
ARG UV=0.11.15
ARG TRITON=26.09
ARG VLLM=v0.32.0

FROM ghcr.io/astral-sh/uv:${UV} AS uv

# ---- the compiler, and the wheels that carry it

FROM ubuntu:${UBUNTU} AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake g++ ninja-build python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /src
COPY CMakeLists.txt CMakePresets.json LICENSE README.md LANGUAGE.md ./
COPY cmake cmake
COPY include include
COPY src src
COPY stdlib stdlib
COPY editors/vim editors/vim
# CI tests the compiler; this only builds it. Its C++ library goes inside,
# so it runs on the older system of another project's image.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLINNET_BUILD_TESTS=OFF \
        "-DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++ -static-libgcc" \
        "-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc" \
    && cmake --build build \
    && cmake --install build --prefix /opt/linnet

# linnet-cli for `linnet serve`; linnet-lang for an image that brings its
# own PyTorch (Triton's, vLLM's).
COPY python/cli python/cli
COPY python/linnet python/linnet
RUN LINNET_PREFIX=/opt/linnet uv build --wheel --out-dir /wheels python/cli \
    && LINNET_PREFIX=/opt/linnet uv build --wheel --out-dir /wheels python/linnet

# ---- `linnet serve`

FROM ubuntu:${UBUNTU} AS serve
# Triton and Inductor compile C against Python's headers while warming up.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy
RUN uv venv --python /usr/bin/python3 /opt/venv

# Checkpoints, Nest cards, compiled entries and kernels, kept across runs
# when /cache is a volume.
ENV HF_HOME=/cache/huggingface \
    LINNET_HOME=/cache/linnet \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    PYTHONUNBUFFERED=1
# Ubuntu's own user, uid 1000, under Linnet's name.
RUN usermod --login linnet --home /home/linnet --move-home ubuntu \
    && groupmod --new-name linnet ubuntu \
    && mkdir /cache \
    && chown linnet:linnet /cache
VOLUME /cache
EXPOSE 8000
# Run directly, `linnet serve` is the process `docker stop` signals, and
# SIGINT is its ^C.
STOPSIGNAL SIGINT

# PyTorch from TORCH_INDEX: a CUDA build (cu130; cu126 for drivers before
# 580), ROCm, Intel XPU or CPU.
FROM serve AS serve-torch
ARG TORCH=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUN --mount=from=build,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch==${TORCH}" --index-url "$TORCH_INDEX" \
    && uv pip install /wheels/linnet_cli-*.whl
USER linnet
ENTRYPOINT ["python", "-m", "linnet.serve", "--host", "0.0.0.0"]

# JAX: JAX_EXTRA `cuda13`, `cuda12`, or empty for the CPU. PyTorch, which
# `linnet-cli` takes, is its CPU build: nothing here runs on it.
FROM serve AS serve-jax
ARG TORCH=2.14.1
ARG JAX=0.12.0.1
ARG JAX_EXTRA=cuda13
RUN --mount=from=build,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch==${TORCH}" --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install /wheels/linnet_cli-*.whl "jax${JAX_EXTRA:+[$JAX_EXTRA]}==${JAX}"
USER linnet
ENTRYPOINT ["python", "-m", "linnet.serve", "--host", "0.0.0.0", "--backend", "jax"]

# ONNX Runtime: ORT `onnxruntime-gpu[cuda,cudnn]` (CUDA 13, its libraries
# from PyPI) or `onnxruntime` for the CPU.
FROM serve AS serve-onnx
ARG TORCH=2.14.1
ARG ONNX=1.23.2
ARG ORT=onnxruntime-gpu[cuda,cudnn]
ARG ORT_VERSION=1.31.1
RUN --mount=from=build,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch==${TORCH}" --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install /wheels/linnet_cli-*.whl "onnx==${ONNX}" "${ORT}==${ORT_VERSION}"
USER linnet
ENTRYPOINT ["python", "-m", "linnet.serve", "--host", "0.0.0.0", "--backend", "onnx"]

# MLX: MLX_EXTRA `cuda13`, `cuda12`, or `cpu`. (Apple silicon's Metal has
# no container: Docker on macOS reaches no GPU.)
FROM serve AS serve-mlx
ARG TORCH=2.14.1
ARG MLX=0.32.4
ARG MLX_EXTRA=cuda13
RUN --mount=from=build,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch==${TORCH}" --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install /wheels/linnet_cli-*.whl "mlx[${MLX_EXTRA}]==${MLX}"
USER linnet
ENTRYPOINT ["python", "-m", "linnet.serve", "--host", "0.0.0.0", "--backend", "mlx"]

# ---- Triton Inference Server and vLLM, at the versions given

FROM nvcr.io/nvidia/tritonserver:${TRITON}-py3 AS triton
COPY --from=uv /uv /usr/local/bin/uv
# The Python backend runs `linnet.torch` in the server's own Python.
ARG TORCH=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUN --mount=from=build,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv pip install --system --break-system-packages \
        "torch==${TORCH}" --index-url "$TORCH_INDEX" \
    && UV_LINK_MODE=copy uv pip install --system --break-system-packages \
        "$(ls /wheels/linnet_lang-*.whl)[onnx,nest]" transformers sentencepiece
ENV HF_HOME=/cache/huggingface \
    LINNET_HOME=/cache/linnet \
    LINNET_SERVER=triton
COPY docker/export-and-serve.sh /usr/local/bin/export-and-serve
VOLUME /cache
EXPOSE 8000 8001 8002
ENTRYPOINT ["export-and-serve"]

FROM vllm/vllm-openai:${VLLM}-ubuntu2404 AS vllm
COPY --from=uv /uv /usr/local/bin/uv
# vLLM's PyTorch and Transformers export; Linnet adds the compiler.
RUN --mount=from=build,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv pip install --system --break-system-packages \
        "$(ls /wheels/linnet_lang-*.whl)[nest]"
ENV HF_HOME=/cache/huggingface \
    LINNET_HOME=/cache/linnet \
    LINNET_SERVER=vllm
COPY docker/export-and-serve.sh /usr/local/bin/export-and-serve
VOLUME /cache
EXPOSE 8000
ENTRYPOINT ["export-and-serve"]

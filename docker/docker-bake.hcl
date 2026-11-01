# Every image, its versions pinned here. From the repository root:
#
#     docker buildx bake -f docker/docker-bake.hcl                  # the default group
#     docker buildx bake -f docker/docker-bake.hcl torch-cpu jax-cuda13
#     docker buildx bake -f docker/docker-bake.hcl --list targets
#
# Images are tagged linnet-serve:<target> and loaded into the local daemon;
# nothing is pushed. Arm (aarch64) builds where its wheels exist: CUDA and
# CPU PyTorch, JAX, ONNX Runtime, MLX, Triton and vLLM. ROCm and XPU are
# x86-64 only.

variable "UBUNTU" { default = "24.04" }
variable "TORCH" { default = "2.14.1" }
variable "JAX" { default = "0.12.0.1" }
variable "ONNX" { default = "1.23.2" }
variable "ORT" { default = "1.31.1" }
variable "MLX" { default = "0.32.4" }
variable "TRITON" { default = "26.09" }
variable "VLLM" { default = "v0.32.0" }
variable "PYTORCH_WHL" { default = "https://download.pytorch.org/whl" }

group "default" {
  targets = ["torch-cu130", "torch-cpu"]
}

group "all" {
  targets = [
    "torch-cu130", "torch-cu126", "torch-rocm", "torch-xpu", "torch-cpu",
    "jax-cuda13", "jax-cuda12", "jax-cpu",
    "onnx-cuda13", "onnx-cpu",
    "mlx-cuda13", "mlx-cpu",
    "triton", "vllm",
  ]
}

target "_common" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  output     = ["type=docker"]
  args = {
    UBUNTU = UBUNTU
    TORCH  = TORCH
  }
}

# `linnet serve`, PyTorch

target "torch-cu130" {
  inherits = ["_common"]
  target   = "serve-torch"
  args     = { TORCH_INDEX = "${PYTORCH_WHL}/cu130" }
  tags     = ["linnet-serve:torch-cu130"]
}

# NVIDIA drivers before 580, which CUDA 13 needs.
target "torch-cu126" {
  inherits = ["_common"]
  target   = "serve-torch"
  args     = { TORCH_INDEX = "${PYTORCH_WHL}/cu126" }
  tags     = ["linnet-serve:torch-cu126"]
}

target "torch-rocm" {
  inherits  = ["_common"]
  target    = "serve-torch"
  args      = { TORCH_INDEX = "${PYTORCH_WHL}/rocm7.2" }
  tags      = ["linnet-serve:torch-rocm7.2"]
  platforms = ["linux/amd64"]
}

target "torch-xpu" {
  inherits  = ["_common"]
  target    = "serve-torch"
  args      = { TORCH_INDEX = "${PYTORCH_WHL}/xpu" }
  tags      = ["linnet-serve:torch-xpu"]
  platforms = ["linux/amd64"]
}

target "torch-cpu" {
  inherits = ["_common"]
  target   = "serve-torch"
  args     = { TORCH_INDEX = "${PYTORCH_WHL}/cpu" }
  tags     = ["linnet-serve:torch-cpu"]
}

# `linnet serve --backend jax`

target "jax-cuda13" {
  inherits = ["_common"]
  target   = "serve-jax"
  args     = { JAX = JAX, JAX_EXTRA = "cuda13" }
  tags     = ["linnet-serve:jax-cuda13"]
}

target "jax-cuda12" {
  inherits = ["_common"]
  target   = "serve-jax"
  args     = { JAX = JAX, JAX_EXTRA = "cuda12" }
  tags     = ["linnet-serve:jax-cuda12"]
}

target "jax-cpu" {
  inherits = ["_common"]
  target   = "serve-jax"
  args     = { JAX = JAX, JAX_EXTRA = "" }
  tags     = ["linnet-serve:jax-cpu"]
}

# `linnet serve --backend onnx`

target "onnx-cuda13" {
  inherits = ["_common"]
  target   = "serve-onnx"
  args     = { ONNX = ONNX, ORT = "onnxruntime-gpu[cuda,cudnn]", ORT_VERSION = ORT }
  tags     = ["linnet-serve:onnx-cuda13"]
}

target "onnx-cpu" {
  inherits = ["_common"]
  target   = "serve-onnx"
  args     = { ONNX = ONNX, ORT = "onnxruntime", ORT_VERSION = ORT }
  tags     = ["linnet-serve:onnx-cpu"]
}

# `linnet serve --backend mlx`

target "mlx-cuda13" {
  inherits = ["_common"]
  target   = "serve-mlx"
  args     = { MLX = MLX, MLX_EXTRA = "cuda13" }
  tags     = ["linnet-serve:mlx-cuda13"]
}

target "mlx-cpu" {
  inherits = ["_common"]
  target   = "serve-mlx"
  args     = { MLX = MLX, MLX_EXTRA = "cpu" }
  tags     = ["linnet-serve:mlx-cpu"]
}

# Exports, served by their own servers

target "triton" {
  inherits = ["_common"]
  target   = "triton"
  args     = { TRITON = TRITON, TORCH_INDEX = "${PYTORCH_WHL}/cu130" }
  tags     = ["linnet-serve:triton-${TRITON}"]
}

target "vllm" {
  inherits = ["_common"]
  target   = "vllm"
  args     = { VLLM = VLLM }
  tags     = ["linnet-serve:vllm-${VLLM}"]
}

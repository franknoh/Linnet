# Compatibility

What runs where, what converts in which direction, and what is missing, all
tested under `python/` and `tests/`.

## Backends

| Target | Call | Notes |
| --- | --- | --- |
| [PyTorch](/docs/torch) | `linnet.torch.load` | interpreted, or generated source (`compile=True`); `trainable=True` for autograd |
| [JAX](/docs/jax) | `linnet.jax.load`, `load_source`, `load_nnx` | XLA (no VJP), differentiable generated `jnp` code, or either as Flax NNX |
| [StableHLO](/docs/tooling) | `linnet stablehlo --bind ...` | static shapes |
| [ONNX](/docs/onnx) | `linnet onnx --bind ...` | opset 20 text format |
| ONNX, packaged | `linnet.onnx.export_model` | checkpoint embedded; runs in any ONNX runtime |
| vLLM, SGLang, TGI, transformers | [`linnet.hf.export`](/docs/integrations#vllm-sglang-tgi) | `llama`, `qwen2`, `qwen3`, `phi3`, and `gpt2` families |
| llama.cpp, Ollama | [`linnet.gguf.export`](/docs/integrations#llama-cpp-and-ollama) | the same families |
| Triton Inference Server | [`linnet.triton.export`](/docs/integrations#triton-inference-server) | packaged ONNX, or a Python backend for stateful entries |
| ComfyUI | [linnet-comfyui](https://github.com/franknoh/linnet-comfyui) | node pack |

## Importers

| Source | Call | Recovers |
| --- | --- | --- |
| PyTorch modules | `linnet.torch.export_linnet` | module tree as blocks; common layers as library calls; `matmul` and `einsum` as index notation |
| JAX functions | `linnet.jax.export_linnet` | parameter pytree as blocks; softmax, sigmoid, silu, gelu, and RMS norm |
| StableHLO text | `linnet.jax.import_stablehlo` | the same, for modules from elsewhere |
| ONNX models | `linnet.onnx.import_onnx` | initializer names as the hierarchy, `dim_param`s as generics, the common operators, PyTorch's RMS norm, SiLU, and softmax |

Importers never guess: an unmapped operation stops the import by name.

## Real checkpoints

The Llama and GPT-2 examples with TinyLlama-1.1B's and GPT-2's weights match
`transformers` logits within 2e-2 in f32, with identical argmax, through
KV-cache decoding. `LINNET_HF_TESTS=1` runs
`python/linnet/tests/torch/test_hf_checkpoints.py`.

## Language features by backend

| Feature | PyTorch | JAX and StableHLO | ONNX |
| --- | --- | --- | --- |
| Symbolic shapes | bound at load and from inputs | bound per compile | bound per export |
| `static for` | yes | unrolled | unrolled |
| `while` | yes, also in generated source | `stablehlo.while`, `lax.while_loop` | `Loop` |
| Index notation and reductions | yes | yes | yes |
| Optional parameters | yes | `--optionals` | yes |
| Enum constants and `match` | yes | folded | folded |
| Tuple results | yes | yes | one output per element |
| `state` members | buffers, `reset_state()` | threaded in and out | threaded in and out |
| `std.random` | yes | yes | yes (right shifts are logical) |
| Training | `trainable=True` | `load_source` with `jax.grad` | no |
| dtypes | f16, bf16, f32, f64, i8 to i64, u8, bool | as the backend supports | as opset 20 supports |

## Not yet

- A `scan` that collects per-iteration outputs; data-dependent shapes.
- Importers: `erf`-based GELU, softmax over an axis other than the last,
  general gather and scatter, custom calls.
- Activation quantization. [Weights](/docs/quantization) can be int8, int4,
  group-wise 4-bit (GPTQ, AWQ), or MXFP4.

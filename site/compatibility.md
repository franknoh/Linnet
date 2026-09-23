# Compatibility

What runs where, what converts in which direction, and what is missing. All
of it is exercised by the test suites under `python/` and `tests/`.

## Backends

| Target | Call | Notes |
| --- | --- | --- |
| PyTorch | `linnet.torch.load` | interpreted Core IR, or generated source with `compile=True`; `numerics="equivalent"` selects native kernels, `"fast"` the input-dtype variants; `trainable=True` for autograd |
| JAX | `linnet.jax.load`, `load_source`, `load_nnx` | `load` compiles StableHLO with XLA (no VJP); `load_source` runs generated `jnp` code and differentiates; `load_nnx` wraps either as a Flax NNX module |
| StableHLO | `linnet stablehlo --bind ...` | static shapes; parameters by `linnet.path`, state by `linnet.state` and `linnet.states` |
| ONNX | `linnet onnx --bind ...` | opset 20 text format; parameters as `param<N>` with metadata, state as `state<N>` and `next_state<N>` |

## Importers

| Source | Call | Recovers |
| --- | --- | --- |
| PyTorch modules | `linnet.torch.export_linnet` | module tree as blocks and sub arrays; `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, softmax, GELU, SiLU as library calls; `matmul` and `einsum` as index notation |
| JAX functions | `linnet.jax.export_linnet` | the parameter pytree as blocks; `dot_general`, `reduce`, row `gather`; softmax, sigmoid, silu, gelu, and RMS norm decompositions |
| StableHLO text | `linnet.jax.import_stablehlo` | the same translation for modules produced elsewhere |
| ONNX models | `linnet.onnx.import_onnx` | initializer names as the hierarchy, `dim_param`s as generics, shape arithmetic folded, the common operators, PyTorch's RMS norm, SiLU, and softmax decompositions |

Every importer refuses rather than guesses. Unmapped operations stop the
import by name; anything dropped is listed in `notes`.

## Real checkpoints

The Llama example loaded with TinyLlama-1.1B's weights and the GPT-2
example loaded with GPT-2's weights give the logits `transformers` gives
(within 2e-2 in f32, identical argmax), including decoding through the KV
caches. `python/linnet_torch/tests/test_hf_checkpoints.py` runs the
comparison with `LINNET_HF_TESTS=1`.

## Language features by backend

| Feature | PyTorch | JAX and StableHLO | ONNX |
| --- | --- | --- | --- |
| Symbolic shapes | bound at load and from inputs | bound per compile | bound per export |
| `static for` (sub arrays, integer ranges) | yes | unrolled | unrolled |
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
- Quantization schemes beyond per-row int8 and packed int4 (per-group,
  asymmetric); activation quantization.

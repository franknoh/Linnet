# Compatibility

What runs where, what converts in which direction, and what does not (yet).
Everything here is exercised by the test suites under `python/` and `tests/`.

## Backends

| Target | How | Runs the model on | Notes |
| --- | --- | --- | --- |
| PyTorch | `linnet_torch.load` → `torch.nn.Module` | CPU, CUDA (any device PyTorch has) | Core IR interpreted on index grids, or with `compile=True` run as straight-line PyTorch source from `linnet torch` (`compile="inductor"` adds `torch.compile`); `numerics="equivalent"` selects native kernels for library ops (`matmul`, `softmax`, `layer_norm`, `rms_norm`, `attention`, activations). `state` members are non-persistent buffers. |
| JAX | `linnet_jax.load` → callable, `load_source` → generated `jnp` code, `load_nnx` → Flax NNX module | Any XLA backend (CPU, GPU, TPU) | `load` compiles through `linnet stablehlo` (no VJP); `load_source` runs `linnet jax` output under `jax.jit` and is differentiable with `jax.grad`. `state` is threaded: `f(*inputs, state=...) -> (out, new_state)`. |
| XLA / StableHLO | `linnet stablehlo --bind ...` | Anything that consumes StableHLO | Static shapes per binding; `@main` with parameters named by `linnet.path`, state by `linnet.state` / `linnet.states`. |
| ONNX | `linnet onnx --bind ...` → ONNX text format | ONNX Runtime and other ONNX consumers | opset 20; parameters as `param<N>` inputs with `linnet.path.*` metadata; state as `state<N>` / `next_state<N>`. |

## Importers

| Source | How | Recovers |
| --- | --- | --- |
| PyTorch modules | `linnet_torch.export_linnet(module, inputs)` via `torch.export` | Module hierarchy as blocks and sub arrays; `nn.Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, softmax, GELU, SiLU as library calls; index notation for `matmul`/`einsum`; SafeTensors weights by path. |
| JAX functions | `linnet_jax.export_linnet(fn, params, inputs)` via `jax.export` | The parameter pytree as blocks (lists and `0..n-1` dicts as sub arrays); `dot_general`, `reduce`, `gather` rows, `broadcast_in_dim`; decompositions of `jax.nn.softmax`, `sigmoid`, `silu`, tanh-`gelu`, and `rsqrt(mean(x*x)+eps)` RMS norms as library calls. |
| StableHLO text | `linnet_jax.import_stablehlo(text, params)` | Same translator as above, for modules produced elsewhere; parameters may be `jax.ShapeDtypeStruct`s. |
| ONNX models | `linnet_onnx.import_onnx(model)` | Initializer names as the block hierarchy; `dim_param`s as generics; shape arithmetic folded at import; `Softmax`, `LayerNormalization`, `Gelu`, `Sigmoid`, `Relu`, `MatMul`, `Gemm`, `Reduce*`, `Gather`, `Slice`, ...; PyTorch's decomposed RMS norm, SiLU, and softmax recovered. |

Every importer refuses rather than guesses: an operation without a Linnet
mapping stops the import with a message naming it, and anything dropped on
the way (a NaN guard Linnet does not need) is reported in `notes`.

## Round trips

The same `.linnet` file produces matching outputs (within 1e-4 in f32) in
PyTorch eager, XLA, JAX, and ONNX Runtime, and a model exported from PyTorch
or JAX, written as Linnet, and loaded back agrees with the original. These
are the `test_round_trip` / `test_export` / `test_import` suites.

## Language features by backend

| Feature | PyTorch | JAX / StableHLO | ONNX |
| --- | --- | --- | --- |
| Symbolic shapes | bound at load, entry generics from inputs | bound per compile (`--bind`, or from inputs at call) | bound per export |
| `static for` over sub arrays and integer ranges | yes | unrolled | unrolled |
| `while` (runtime loop, scalar `bool` condition) | yes (also generated source) | `stablehlo.while` | `Loop` |
| Index notation, reductions | yes | yes | yes |
| Optional parameters (`= none`) | yes | `--optionals` | yes |
| Enum-typed constants and `match` | yes | folded | folded |
| Tuple results | yes | yes | one output per element |
| `state` members | buffers, `reset_state()` | threaded in and out | threaded in and out |
| `std.random` (keyed Threefry PRNG) | yes | yes | yes (right shifts are logical) |
| dtypes | f16, bf16, f32, f64, i8–i64, u8, bool | as the backend supports | as opset 20 supports |

## Not supported (yet)

- A `scan` that collects per-iteration outputs, and data-dependent shapes.
  `while` covers data-dependent loops with invariant shapes (the Llama
  example's `generate_until` stops at an end token) on every backend.
- Randomness is a library, not a primitive: `std.random` (Threefry-2x32,
  matching `jax.random` bit for bit) runs on every backend; there is no
  hidden generator state.
- Training: PyTorch via `load(..., trainable=True)` (autograd through the
  interpreted or generated code); JAX via `load_source` (generated `jnp`
  code, `jax.grad` over `f.apply(params, ...)`). The StableHLO-compiled
  `load` function itself has no VJP.
- Quantized *dtypes*: quantization is library code (`std.quant`: per-row
  int8 and packed int4 weights with scales, `Int8Linear`/`Int4Linear`), not a
  storage type; per-group and asymmetric schemes are not written yet.
- Importers: `erf`-based GELU (no `erf` primitive), softmax over an axis other
  than the last, general `gather`/scatter, custom calls.

The [roadmap](https://github.com/franknoh/Linnet#readme) tracks these; state
threading is the first of the milestone-16 features (state, RNG, loops,
training, quantization) and landed with the `decode` entry of the
[Llama example](/examples/05-llama).

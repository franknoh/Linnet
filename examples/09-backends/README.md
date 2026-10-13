# Backend plugins

A backend runs part of a Linnet model on a library of its own
([Backends](../../docs/backends.md)). Each directory here is a package of
its own that plugs one library in through the `linnet.backends` entry
point, without changes to Linnet:

| Plugin | Library | Takes |
| --- | --- | --- |
| `deepgemm` | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | block-scaled FP8 `linear` calls (`Fp8BlockLinear`, blocks of 128) under `numerics="fast"`, on Hopper |

Install one beside Linnet and its library:

```bash
pip install -e examples/09-backends/deepgemm
```

`linnet.backends.available()` then lists it, and `model.plan_for(entry,
inputs)` offers it every region it matches. `model.conform(entry, inputs,
backend)` checks it against the generated source.

## DeepGEMM

DeepGEMM is built from source against the CUDA toolkit PyTorch was built
with:

```bash
git clone --recursive --branch v2.1.1.post3 https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && ./install.sh
```

The plugin takes a `std.quant::linear_fp8_blocks` call whose input the
export may round to FP8 (`numerics="fast"`). It rounds each 128 inputs of a
row as Linnet's own FP8 path does and multiplies with
`deep_gemm.fp8_gemm_nt`. On one H100, Qwen3 8B FP8 prefills 2,048 tokens in
35 ms with the measured plan, against 46 ms for the generated source alone.

For a model loaded with `numerics="fast"` and `compile=True`, such as
Nest's `qwen3-8b-fp8`:

```python
from linnet_deepgemm import DeepGemm

plan = model.plan_for("prefill", [tokens, pos], backends=[DeepGemm()])
```

`python/linnet/tests/test_backend_examples.py` checks each plugin here: what
it matches on any machine, and on a Hopper GPU with the library installed,
its results against the entry's.

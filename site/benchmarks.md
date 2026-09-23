# Benchmarks

Linnet adds a compiler in front of the frameworks; these numbers show what
that costs and what it does not. Every row is produced by `bench/run.py` in
the repository, and the JSON it writes (`bench/results/latest.json`) is what
this page renders — nothing here is typed in by hand.

<BenchTable />

## What is measured

The models are the repository's examples with random weights at realistic
widths: the Llama-style decoder (`examples/05-llama`) and GPT-2
(`examples/06-gpt2`). For each configuration the harness times:

- **PyTorch reference** — the hand-written `torch` implementation from the
  test suite, eager, the baseline everything is compared against.
- **PyTorch reference, compiled** — the same code under `torch.compile`.
- **Linnet → PyTorch** — `linnet_torch.load(..., numerics="equivalent")`: the
  checked source materialized as an `nn.Module`, library operations dispatched
  to native PyTorch kernels.
- **Linnet → XLA** — `linnet_jax.load` under `jax.jit`: the source compiled
  through `linnet stablehlo` and run by XLA.

Latency is the median of timed iterations after warm-up, with CUDA
synchronized around each call; throughput is tokens per second for the
`forward` entry over a full sequence and steps per second for the Llama
`decode` entry (one token per step through the KV caches). Outputs of every
variant are compared against the reference (`max |Δ|`) so a fast wrong
number cannot appear in the table.

Compile and load times are reported separately: `linnet check` on the
package, `load` (which runs `linnet plan` and builds the module), and the
first XLA compile.

## Reading the numbers

- Where Linnet → PyTorch matches the reference, the interpreter's overhead
  is paid once per call in Python and the arithmetic is the same kernels; the
  gap that remains is the dispatch cost of the interpreted Core IR, which
  shrinks as the model grows.
- Linnet → XLA is a whole-program compile: no Python in the loop, fused
  elementwise chains, and static shapes per binding. It is the number to
  compare with `torch.compile`.
- `linnet check` is milliseconds; it never runs the model.

## Reproducing

```bash
cd python/linnet_torch && uv sync --all-extras
cd ../linnet_jax && uv sync --group dev
cd ../..
LINNET_BIN=build/release/linnet python bench/run.py --device cuda --out bench/results/latest.json
```

`bench/README.md` describes the configurations and the environment the
published numbers came from.

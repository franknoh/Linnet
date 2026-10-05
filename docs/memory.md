# Memory planning

`linnet memory` says how much memory a model needs under a configuration,
before anything runs. `linnet fit` finds the largest batch, context or
cache that fits a device. Both read the compiled program; neither loads
weights or touches a GPU.

```bash
linnet memory llama-3.1-8b-instruct --batch 1 --seq-len 8192
```

```text
Weights                  14.96 GiB  [exact]
KV cache                  1.00 GiB  [exact]
Peak activations          2.02 GiB  [exact]
CUDA context            512.00 MiB  [estimated]
cuBLAS workspace         32.00 MiB  [estimated]
------------------------------------
Graph peak               17.98 GiB
Expected peak            18.51 GiB  [estimated]

Peak at step 1603: linear in `lm_head`.
Transient storage: 62.89 GiB with a buffer per tensor, 2.02 GiB planned, 2.02 GiB live at most.

Not included (unknown):
  - Allocator fragmentation: the caching allocator's unused reserve depends on the allocation order
```

The model is anything [`nest.load`](nest.md) takes, or a `.linnet` file with
`--bind` for its generics.

## Graph memory and runtime memory

Every number carries how it was obtained:

| Confidence | What it covers |
| --- | --- |
| `exact` | the program's own tensors: weights, buffers, state, KV caches, the peak of live activations, and in training gradients, master weights and optimizer states |
| `backend-modeled` | what a backend implementation is known to allocate: the chunked loss's logits blocks, autograd's saved tensors, a paged cache's blocks |
| `estimated` | a typical value for something the runtime decides: the CUDA context, the cuBLAS workspace |
| `unknown` | nothing models it, so it is listed and counted as nothing: allocator fragmentation, a kernel no backend model describes |

The graph peak is the exact part. The expected peak adds the modeled and
estimated parts. Unknown items are never folded in as zero: they are
listed under "Not included".

## What the analysis does

- **Traces the entry** as the backend runs it: calls followed into their
  callees, layer loops unrolled, and each library call either running the
  native implementation the plan selected (`--numerics`, `fast` by default
  as in `linnet.torch.load`) or its canonical body.
- **Tracks storage, not tensors.** A view (`permute`, `slice`, `broadcast`,
  a `reshape` of contiguous storage) shares its operand's bytes; a cache
  write the generated code makes in place updates the cache; tied
  parameters are counted once.
- **Computes liveness.** Each tensor lives from the step that makes it to
  its last reader; the peak is the largest sum of live tensors, not the sum
  of all. A static planner that reuses dead tensors' storage packs them
  into the arena shown as "planned".
- **Keeps sizes symbolic.** The batch, sequence and cache lengths stay
  free symbols, so analyzing another size is one sweep, not a new trace.

## Inference

```bash
linnet memory llama-3.1-8b-instruct --batch 8 --seq-len 32768
linnet memory llama-3.1-8b-instruct --entry decode_rows --batch 64 --seq-len 8192
linnet memory model.linnet --bind H=1024 --bind Layers=12 --dtype bf16 --batch 4 --seq-len 2048
```

`--batch` binds the generics `B` and `Batch`, `--seq-len` binds `S` and
`P`, and `--cache-len` binds `MaxSeq` (by default the sequence length).
KV caches are the model's `state` members written through `std.nn.cache`,
so their size follows the attention the model declares: multi-head,
grouped-query or multi-query, per layer. `--kv-block-size 16` lays them out
in pages as a paging engine would; `--kv-dtype` stores them in another
dtype.

## Training

```bash
linnet memory model.linnet --entry loss --training --optimizer adamw --master-dtype f32 \
    --batch 1 --seq-len 8192 --checkpoint none --checkpoint block
```

A training step is the entry's forward pass, its backward pass and the
optimizer step on one timeline:

- autograd keeps what each operation's backward reads, by PyTorch's rules
  for native kernels and by the operation's arithmetic otherwise, until
  that backward runs;
- a parameter's gradient lives from its first backward to the optimizer
  step, an activation's gradient from its last reader's backward to its
  producer's;
- the optimizer keeps nothing (`sgd`), one tensor per parameter
  (`sgd-momentum`) or two (`adam`, `adamw`), in the dtype of what it
  updates: the master weights with `--master-dtype`, the parameters
  otherwise;
- `--checkpoint block` recomputes each layer before its backward,
  `--checkpoint 'layers.*.mlp'` the blocks a pattern names; the report adds
  the recomputation's cost;
- `--trainable 'layers.*'` limits gradients and optimizer states to those
  parameters, and `--shards N` splits parameters, gradients, master weights
  and optimizer states across N devices (FSDP), adding the layer being
  gathered.

Repeat `--checkpoint` to compare policies in one report. Tensor and
pipeline parallelism are not analyzed yet, and say so.

## The largest configuration that fits

```bash
linnet fit llama-3.1-8b-instruct --entry decode_rows --seq-len 8192 \
    --device-memory 80GiB --reserve 2GiB --maximize batch
```

```text
Maximum batch size: 62

Estimated peak memory: 77.50 GiB
Headroom: 507.84 MiB of a 78.00 GiB budget
```

`--maximize` is `batch`, `context` or `kv-cache`. The budget is the
device memory less `--reserve` and `--reserve-percent`. Memory never
shrinks as these grow, so the search doubles until a value does not fit
and then bisects: about two dozen sweeps. The model's own `where` clauses
bound it too.

## JSON and Python

`--json` prints one document: the configuration, every component with its
bytes, confidence and formula, both peaks, the unknown items, warnings and
assumptions.

```python
from linnet.resources import ExecutionConfig, ExecutionPlanner, MemoryModel, ResourceConstraint, load_source

source = load_source("llama-3.1-8b-instruct")
model = MemoryModel(source, ExecutionConfig(batch=8, context=8192))
result = model.analyze()                   # MemoryAnalysisResult
result.expected_peak, result.component("KV cache").nbytes
model.analyze(batch=16).expected_peak      # the same trace, another size

found = ExecutionPlanner(source).maximize(
    ExecutionConfig(entry="decode_rows", context=8192), ResourceConstraint(80 << 30), "batch"
)
```

The layers are separate modules: `trace` (the program as memory objects
and steps), `graph` (liveness, peaks and buffer planning), `backends`
(`BackendResourceModel`: workspaces, saved tensors and runtime memory per
backend), `kvcache`, `training`, `analysis` and `planner`.

## Checking a prediction

```bash
python -m linnet.resources.validate llama-3.1-8b-instruct --batch 4 --seq-len 2048
```

runs the configuration on a CUDA device and compares the prediction with
PyTorch's allocator peak, and the expected peak with the device memory in
use. The errors are reported as they are; nothing is tuned to make one
benchmark match.

On an H100 with PyTorch 2.14 (generated source, `compile=True`, fast
numerics), the graph and workspaces against the allocator peak:

| Model | Configuration | Predicted | Measured | Error |
| --- | --- | ---: | ---: | ---: |
| TinyLlama 1.1B | batch 1, 2,048 tokens | 2.25 GiB | 2.26 GiB | -0.2% |
| TinyLlama 1.1B | batch 8, 2,048 tokens | 3.46 GiB | 3.47 GiB | -0.1% |
| TinyLlama 1.1B | decode, batch 32, cache 4,096 | 4.83 GiB | 4.87 GiB | -0.8% |
| Llama 3.1 8B | batch 1, 8,192 tokens | 18.01 GiB | 18.02 GiB | -0.0% |
| Llama 3.1 8B | decode, batch 16, cache 8,192 | 31.00 GiB | 31.03 GiB | -0.1% |
| Qwen2.5 0.5B | batch 4, 4,096 tokens | 6.06 GiB | 6.07 GiB | -0.2% |
| GPT-2 | batch 8, 1,024 tokens | 2.61 GiB | 2.62 GiB | -0.1% |
| Qwen2.5 0.5B | training, 2,048 tokens, SGD | 6.75 GiB | 6.80 GiB | -0.7% |
| Qwen2.5 0.5B | training, 4,096 tokens, SGD | 11.14 GiB | 11.01 GiB | +1.2% |
| Qwen2.5 0.5B | training, 8,192 tokens, AdamW | 20.92 GiB | 20.61 GiB | +1.5% |
| TinyLlama 1.1B | training, 4,096 tokens, SGD | 11.24 GiB | 10.92 GiB | +2.9% |
| TinyLlama 1.1B | training, 4,096 tokens, AdamW | 15.34 GiB | 15.02 GiB | +2.1% |
| TinyLlama 1.1B | training, 8,192 tokens, SGD | 21.28 GiB | 20.62 GiB | +3.2% |

Training is a whole step (forward, backward and a fused optimizer step)
on packed sequences. The process adds more than the graph:

- **CUDA context:** 690 to 770 MiB measured outside the allocator; the
  analysis estimates 512 MiB. Pass the size you see with `--context-bytes`.
- **Allocator reserve:** what the caching allocator holds beyond its peak.
  It measured from 0.14 to 7.9 GiB and depends on the order of
  allocations, so the analysis lists it as unknown. Leave room for it with
  `--reserve` or `--reserve-percent` when fitting.

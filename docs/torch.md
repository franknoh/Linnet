# PyTorch

`linnet.torch` loads a Linnet root block as a `torch.nn.Module` and exports
a `torch.nn.Module` as Linnet source.

## Install

```bash
uv add "linnet-lang[torch]"
```

## Load a model

```python
from linnet.torch import load

model = load(
    "examples/01-llama/src/lib.linnet",
    generics={"Vocab": 32000, "H": 512, "Heads": 8, "KvHeads": 4, "Inner": 1376,
              "Layers": 2, "Batch": 1, "MaxSeq": 128, "T": "bf16"},
    std_root="stdlib",
    weights="weights/",          # a .safetensors file or a directory of them
)
logits = model(tokens)
```

Printing the module shows the block hierarchy with Linnet names, shapes
and dtypes:

```text
LinnetModule(
  (root): Model(
    (embedding): Embedding(weight=bfloat16[32000, 512])
    (layers): ModuleList(
      (0-1): 2 x DecoderLayer(
        (attention_norm): RmsNorm(weight=bfloat16[512])
        (attention): GroupedQueryAttention(
          cache_k=bfloat16[1, 4, 128, 64] state, cache_v=bfloat16[1, 4, 128, 64] state
          (q_proj): Linear(weight=bfloat16[512, 512], bias=bfloat16[512]?)
          (k_proj): Linear(weight=bfloat16[256, 512], bias=bfloat16[256]?)
          (v_proj): Linear(weight=bfloat16[256, 512], bias=bfloat16[256]?)
          (o_proj): Linear(weight=bfloat16[512, 512], bias=bfloat16[512]?)
        )
        (mlp_norm): RmsNorm(weight=bfloat16[512])
        (mlp): SwiGlu(
          (gate): Linear(weight=bfloat16[1376, 512], bias=bfloat16[1376]?)
          (up): Linear(weight=bfloat16[1376, 512], bias=bfloat16[1376]?)
          (down): Linear(weight=bfloat16[512, 1376], bias=bfloat16[512]?)
        )
      )
    )
    (norm): RmsNorm(weight=bfloat16[512])
    (lm_head): Linear(weight=bfloat16[32000, 512], bias=bfloat16[32000]?)
  )
)
```

`?` marks an optional parameter, `state` a KV-cache member. `state_dict()`
names each tensor `root.` and its Linnet path
(`root.layers.0.attention.q_proj.weight`).

`load` checks the program with `linnet plan` without running it. With
`weights`, it checks every tensor's name, shape and dtype against the
manifest before copying; optional parameters can be absent. `state`
members start at zero. Entries become methods: `forward` is the entry of
that name or the only one, and entry generics such as `B` and `S` bind
from the inputs.

## Load options

| Option | Effect |
| --- | --- |
| `numerics=` | see [Numerics policy](#numerics-policy) |
| `compile=True` | entries run as generated PyTorch source, once per entry and input shape. Default on CUDA |
| `compile=False` | the interpreter. Default elsewhere |
| `compile="inductor"` | generated source through `torch.compile` |
| `compile="reduce-overhead"` | CUDA graphs; see [CUDA graphs](#cuda-graphs) |
| `trainable=True` | parameters require gradients; glob patterns (`["layers.*.mlp.*"]`) choose a subset. See [Training](#training) |
| `bindings="bindings.json"` | maps Linnet paths to checkpoint tensor names |
| `cast_dtype=True` | converts floating-point weights to the model's dtype as they are read |
| `amp="bf16"` or `"f16"` | mixed precision under `torch.autocast`; weights keep the model's dtype (`T=f32` keeps f32 masters) and gradients arrive in f32. With `"f16"`, scale the loss with `torch.amp.GradScaler` |
| `device_map`, `max_memory`, `offload` | see [Placement and offload](#placement-and-offload) |
| `tensor_parallel`, `tp_rules` | see [Tensor parallelism](#tensor-parallelism) |

### Compiling a layer once

Under `torch.compile` (`compile="inductor"`, or the capture of
`"reduce-overhead"`), each kind of repeated block compiles once. The
generated step's block-array elements (`layers.0`, `layers.1`, ...) become
one function, which the step calls once per layer, and only that function
is compiled. The first call compiles one layer rather than all of them:

| On an H100 | Layer compiled once | Whole step compiled |
| --- | ---: | ---: |
| Llama 3.1 8B, 2,048 tokens, first call | 4.8 s | 18.9 s |
| Llama 3.1 8B decoding, CUDA graphs, first call | 7.2 s | 50.4 s |
| TinyLlama training step, 4,096 tokens, first step | 10.9 s | 69.8 s |
| TinyLlama serving warmup (`Engine.warmup`) | 20.1 s | 113.4 s |

Steps run at the same speed (Llama decoding 6.55 ms both ways, a training
step 95.8 against 93.8 ms).

`generated_source()` still shows the step as `linnet torch` printed it. Set
`model.regional = False` to compile the whole step. Entries whose blocks
differ, or that branch or loop, compile whole.

### CUDA graphs

`compile="reduce-overhead"` is for decoding. The first call runs through
`torch.compile`, the second captures the step as one CUDA graph, and later
calls replay it. KV caches the entry writes are updated in place.
Replacing a state or weight (`reset_state`) triggers a new capture. It
works under `tensor_parallel` and requires `offload=False`. With blocks on
several devices, or with gradients, `torch.compile`'s own CUDA graphs run
instead.

## Numerics policy

| `numerics=` | Library operations run as |
| --- | --- |
| `"fast"` (default) | PyTorch kernels, with layer normalization and attention in the input dtype, as PyTorch's reference implementations do |
| `"equivalent"` | the same kernels, accumulating in f32 as the canonical bodies specify. Slower on `bf16`, which loses fused attention ([benchmarks](https://linnet.franknoh.dev/benchmarks)) |
| `"exact"` | their canonical `.linnet` bodies: the slowest path, for comparison |

Under `"fast"` on CUDA, attention under an explicit mask (serving batches,
packed prompts, sliding windows, gpt-oss's sink attention) computes only
the key blocks the mask reaches. Mixture-of-experts layers run as one
grouped product over the chosen experts on compute capability 9 or later
with bf16.

## Entries and state

| Call | Effect |
| --- | --- |
| `model.run_entry("generate", [prompt, pos], generics={"Steps": 16})` | runs an entry, naming generics the inputs do not determine |
| `model.run_entry(..., compile=...)` | overrides `compile` for one call, such as CUDA graphs for decoding steps only |
| `model.reset_state()` | zeroes every `state` member |
| `model.state_paths()` | lists the `state` members |
| `model.generated_source(entry)` | shows the code `compile` ran |

## Multiple devices

### Placement and offload

```python
model = linnet.torch.load("model.linnet", generics=generics, weights="weights/",
                          device_map="auto")
print(model.placement.describe())
```

```text
cuda:0: embedding, layers.0-15
cuda:1: layers.16-31, norm, head
```

`device_map="auto"` places each sub-block of the root, and each element of
a sub-block array, on the GPUs in declaration order, keeping a reserve for
activations. Placement is decided from the manifest before any weight is
read. What does not fit stays in host memory and is *offloaded*: copied to
the last GPU when it runs, at the speed of the copies.

| Option | Effect |
| --- | --- |
| `max_memory={0: "20GiB"}` | caps one GPU |
| `offload=False` | raises an error instead of offloading |
| `device_map={"layers.30": "cpu", "layers.31": "cpu"}` | places units by hand |

`linnet torch --place layers.16=1 --offload layers.31` compiles a placement
into the generated source, with a `.to()` where a value crosses devices:

```python
    v212 = v209.to(_dev[1], non_blocking=True)   # the residual stream, once
    ...
    v480 = p291.to(_dev[1], non_blocking=True)   # an offloaded layer's weights
    v481 = F.linear(v479, v480, None)
    ...
    del v480                                     # gone when the layer returns
```

### Tensor parallelism

```python
# torchrun --nproc-per-node 2 serve.py
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
model = linnet.torch.load("model.linnet", generics=generics, weights="weights/",
                          device=f"cuda:{dist.get_rank()}", tensor_parallel=mesh)
```

Tensor parallelism splits every large weight across the GPUs, so all work
on each layer at once. `device_map` instead gives each GPU whole blocks,
which take turns.

A model with a `Shards` generic (1 by default), such as the zoo's
Llama-shaped decoders, declares its own split. A shard holds
`Heads / Shards` query heads, `KvHeads / Shards` key and value heads and
`Inner / Shards` hidden units; projections back out end in
`std.nn.parallel::all_reduce`.
An output projection not tied to the embedding splits into
`Vocab / Shards` rows, joined by `std.nn.parallel::all_gather`. On one
device both return their input. `tensor_parallel=mesh` binds `Shards` to
the mesh size, and each process reads only its part of the checkpoint.
Such a model trains split as well: see [Split training](training.md#split-training).

Other models are split as DTensors by `linnet.parallel.DEFAULT_RULES`:

| Weights | Split |
| --- | --- |
| projections into the heads, feed-forward width | by output |
| projections back | by input |
| KV caches | by heads |
| everything else | copied to each process |

`tp_rules={"*.experts.*": 0}` adds or overrides rules by path pattern. An
axis the device count does not divide is copied. Any split computes the
same numbers, and results come back whole on every process.

With `compile="reduce-overhead"`, run `del model` before
`dist.destroy_process_group()`, which otherwise waits on the CUDA graphs'
communicators.

### Pipeline parallelism

```python
# torchrun --nproc-per-node 4 train.py
import torch, torch.distributed as dist
from linnet.torch import pipeline

dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())
pipe = pipeline("model.linnet", generics=generics, weights="weights/",
                entry="loss_packed", microbatches=8)
optimizer = torch.optim.AdamW(pipe.parameters(), lr=1e-5)
loss = pipe.step(tokens, positions, segments, targets, weights)  # on the last stage
optimizer.step()
optimizer.zero_grad()
```

Each process runs one stage: a run of the root's blocks, balanced by
parameter bytes (`stages=["layers.8", "layers.16", "layers.24"]` sets the
splits). A process holds only its own blocks' weights and reads only
those from the checkpoint.

The entry's generated source is split into one function per stage. Only
values computed from weights cross between processes, usually the residual
stream. Masks and positions are computed again on each stage that reads
them.

| Call | Effect |
| --- | --- |
| `pipe.step(*inputs)` | one training step over `microbatches` parts of the inputs' first axis; the last stage returns the summed result |
| `pipe.run(*inputs)` | the forward pass alone; the last stage returns the results joined along the first axis |
| `schedule="1f1b"` (default) | a stage keeps at most as many micro-batches in flight as there are stages after it |
| `schedule="gpipe"` | every micro-batch's forward, then every backward |
| `compile="inductor"` | each stage function through `torch.compile` |

Every process calls with the same inputs. For a packed entry, pack each
micro-batch's part on its own so no sequence spans two. Parameters tied
across stages (an embedding and its output head) have their gradients
summed after each step. Entries that write `state`, or loop with `while`,
do not split.

A stage can span several processes, from a two-dimensional mesh whose
`"pp"` dimension is the pipeline:

```python
# torchrun --nproc-per-node 4: 2 stages, each over 2 GPUs
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("pp", "split"))
pipe = pipeline("model.linnet", ..., group=mesh["pp"].get_group(),
                tensor_parallel=mesh["split"])   # or data_parallel=
```

| Option | Each stage's processes |
| --- | --- |
| `tensor_parallel=` | split its weights, for a model with a `Shards` generic; they take the same micro-batches |
| `data_parallel=` | shard its weights (`linnet.torch.fsdp`) and train on batches of their own; gradients are summed |

A sharded stage gathers its weights once a step and keeps them whole for
every micro-batch. As each micro-batch's backward finishes a gradient, it
is summed into the parts while the rest of the backward runs.

A weight two stages share (an embedding and its output head) is sharded the
same way on both, and its gradient parts are summed between them.

## Training

```python
model = load("src/model.linnet", generics={...}, weights="init/", trainable=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = criterion(model(tokens), labels)
loss.backward()
optimizer.step()
model.save_weights("trained.safetensors")
```

Entries are differentiable, interpreted or generated. `state` members are
detached between calls.

| Call | Effect |
| --- | --- |
| `trainable=True`, or `model.set_trainable(True)` | every floating-point parameter trains |
| `trainable=["layers.*.mlp.*"]` | only parameters whose path matches a glob pattern; returns the paths |
| `model.save_weights(path)` | one SafeTensors file under the checkpoint's own names, so it replaces that checkpoint (`linnet.hf.export(card, weights=path)` exports it) |
| `model.save_weights(path, names="linnet")` | the same under Linnet paths |
| `model.save_weights(path, dtype=torch.bfloat16)` | converts floating-point tensors as it writes |

Parameters the checkpoint ties (two paths bound to one tensor, such as an
embedding and its output head) load as one parameter and are written once.
Optional parameters the checkpoint lacks never train and are not written.
A model trains without weight-only work done ahead or hand-captured CUDA
graphs; a model that ran without gradients compiles its entries again once
any parameter requires them.

Train with `compile="inductor"`. For long packed batches, have it recompute
activations instead of keeping them: set
`torch._functorch.config.activation_memory_budget = 0.3` before the first
step. On one H100, Llama 3.1 8B trains its layers (7 B parameters, SGD) over
4096 packed tokens at 10.9K tokens per second in 29 GiB this way, against
9.2K and 42.5 GiB as generated source without `torch.compile`.

[Training](training.md) covers the rest: supervised fine-tuning with
packed batches, checkpoints, sharding a model across GPUs, transformers
and TRL trainers, reinforcement learning (GRPO), preferences (DPO),
low-rank adapters, and the losses.

## Functions

An `entry` declared at module level, outside any block, is a function of
its inputs alone, such as a loss.
`load_function` returns one as a PyTorch function.

```python
from linnet.torch import load, load_function

model = load("functions.linnet", generics={"In": 64, "Classes": 10}, trainable=True)
cross_entropy = load_function("functions.linnet", "cross_entropy")
loss = cross_entropy(model(x), labels)
loss.backward()
```

Generics bind from each call's inputs, so one function serves every batch
size and dtype. Name any the inputs leave open: `positions(offset, N=8)`.
A Python number passes for a scalar input. `compile=` works as for `load`;
unset, it generates source for inputs on CUDA and interprets otherwise.
Autograd differentiates the function, and a model's entries can call it
(the example's `Classifier.loss`).

## PyTorch to Linnet

```python
from torch.export import Dim
from linnet.torch import export_linnet

export_linnet(
    model,
    example_args=(tokens,),
    dynamic_shapes={"tokens": {0: Dim("batch"), 1: Dim("seq")}},
    output="src/model.linnet",
    weights="weights/",          # optional: SafeTensors under the PyTorch names
)
```

`export_linnet` captures the model with `torch.export` and writes checked
Linnet source:

| PyTorch | Linnet |
| --- | --- |
| module tree | blocks; a `ModuleList` or `Sequential` of alike children becomes a sub array |
| forward graph | the root `entry` |
| dynamic dimensions | generics named after their `Dim`s |
| ATen operations | the primitive or library operation with the same meaning (`aten.mm` is `std.linalg::matmul`) |

An unmapped operation stops the export with its name. A parameter path
Linnet cannot spell is renamed, and `weights/bindings.json` records the
mapping.

## Tests

```bash
uv run pytest                       # references, round trips, state, training
LINNET_HF_TESTS=1 uv run pytest tests/torch/test_hf_checkpoints.py   # TinyLlama and GPT-2 vs transformers
uv run pyright
uv run ruff check src tests && uv run ruff format --check src tests
```

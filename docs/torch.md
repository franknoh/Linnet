# PyTorch

`linnet.torch` loads a Linnet root block as a `torch.nn.Module` and exports
a `torch.nn.Module` as Linnet source.

## Install

```bash
cd python/linnet && uv sync --extra torch    # or: pip install ".[torch]"
export LINNET_BIN=/path/to/build/release/linnet     # or put `linnet` on PATH
```

## Load a model

```python
from linnet.torch import load

model = load(
    "examples/05-llama/src/lib.linnet",
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

### Losses

`std.nn.loss` holds the losses, over flattened tokens (`[N, V]` logits),
computed in f32:

| Op | Result |
| --- | --- |
| `cross_entropy(logits, targets, weights)` | `sum_n weights[n] * -log p(targets[n])`; pass `mask / count` for a mean over `count` tokens |
| `token_log_probs(logits, targets)` | each row's log-probability of its target, `[N]` |
| `log_softmax(logits)`, `entropy(logits)` | per row |
| `linear_cross_entropy(hidden, weight, targets, weights)` | `cross_entropy` of the output head `hidden @ weight.T` |
| `linear_token_log_probs(hidden, weight, targets)` | `token_log_probs` of the output head |

The `linear_` forms never hold the `[N, V]` logits in PyTorch: they run a
block of about 1 GiB of f32 logits at a time (2 GB in full for 4096 tokens
of Llama 3's vocabulary), and accumulate the weight's gradient in f32.

## Functions

An `entry` declared at module level, outside any block, is a function of
its inputs alone, such as a loss (`examples/09-functions`).
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

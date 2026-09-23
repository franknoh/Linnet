# PyTorch

`python/linnet_torch` turns a Linnet root block into a `torch.nn.Module` and
a `torch.nn.Module` into Linnet source. The adapter knows nothing about any
model; everything comes from the compiler.

```bash
cd python/linnet_torch && uv sync
export LINNET_BIN=/path/to/build/release/linnet     # or put `linnet` on PATH
```

## Loading a model

```python
from linnet_torch import load

model = load(
    "examples/05-llama/src/lib.linnet",
    generics={"Vocab": 32000, "H": 512, "Heads": 8, "KvHeads": 4, "Inner": 1376,
              "Layers": 2, "Batch": 1, "MaxSeq": 128, "T": "bf16"},
    std_root="stdlib",
    weights="weights/",          # a .safetensors file or a directory of them
)
logits = model(tokens)
```

The result is an ordinary module. Printing it shows the block hierarchy with
the Linnet names, shapes, and dtypes:

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
uses the Linnet parameter paths (`layers.0.attention.q_proj.weight`), so a
checkpoint written from PyTorch loads here and back.

What `load` does:

1. Runs `linnet plan`, which checks the program and prints its structure,
   parameter manifest, and Core IR. Nothing in the package is executed.
2. Builds the module tree: `sub` becomes a child module (`ModuleList` for
   arrays), `param` an `nn.Parameter`, `buffer` a buffer, `state` a
   non-persistent buffer starting at zero.
3. With `weights`, checks every tensor's name, shape, and dtype against the
   manifest, then copies the data. Optional parameters may be absent.
4. Exposes entries as methods. `forward` is the entry of that name or the
   only one; entry generics such as `B` and `S` are bound from the inputs.

| Option | |
| --- | --- |
| `numerics="exact"` (default) | library operations run as their canonical `.linnet` bodies |
| `numerics="equivalent"` | library operations dispatch to PyTorch kernels (`F.linear`, `scaled_dot_product_attention`, `torch.softmax`, `torch.rms_norm`, `F.layer_norm`, activations, `torch.matmul`) that agree up to rounding |
| `compile=True` | entries run as PyTorch source from `linnet torch`, generated once per entry and input shape |
| `compile="inductor"` | the same, passed through `torch.compile` |
| `trainable=True` | parameters require gradients |
| `bindings="bindings.json"` | maps Linnet paths to checkpoint tensor names |

Entries with generics the inputs do not determine take them by name:
`model.run_entry("generate", [prompt, pos], generics={"Steps": 16})`.
`model.reset_state()` zeroes every `state` member; `model.state_paths()`
lists them; `model.generated_source(entry)` shows the code `compile` ran.

## Training

```python
model = load("src/model.linnet", generics={...}, weights="init/", trainable=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = criterion(model(tokens), labels)
loss.backward()
optimizer.step()
```

Entries are ordinary differentiable PyTorch arithmetic, interpreted or
generated, so autograd needs nothing else. `state` members are detached
between calls: they carry values, not gradients.

## Exporting a PyTorch model

```python
from torch.export import Dim
from linnet_torch import export_linnet

export_linnet(
    model,
    example_args=(tokens,),
    dynamic_shapes={"tokens": {0: Dim("batch"), 1: Dim("seq")}},
    output="src/model.linnet",
    weights="weights/",          # optional: SafeTensors under the PyTorch names
)
```

`export_linnet` captures the model with `torch.export`, decomposes it to core
ATen, and translates it into a plan that `linnet emit` prints as source. The
module tree becomes blocks (`ModuleList` and `Sequential` of alike children
become sub arrays), the forward graph becomes the root `entry`, dynamic
dimensions become generics named after their `Dim`s, and each ATen operation
becomes the primitive or standard-library operation with the same meaning
(`aten.mm` is `std.linalg::matmul`, `_softmax` is `softmax`,
`native_layer_norm` is `layer_norm`). The result is checked before it is
written.

The translation never guesses: an unmapped operation stops the export with
its name. Parameter paths follow PyTorch's; when one cannot be spelled in
Linnet, the member is renamed and `weights/bindings.json` records the
mapping. Exporting runs Python with the model loaded; checking the result
never does.

## Tests

```bash
uv run pytest                       # references, round trips, state, training
LINNET_HF_TESTS=1 uv run pytest tests/test_hf_checkpoints.py   # TinyLlama and GPT-2 vs transformers
uv run pyright
uv run ruff check src tests && uv run ruff format --check src tests
```

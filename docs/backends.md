# Backends

A backend runs part of a Linnet model on a library of its own: a kernel
library, a fusing compiler, another framework. Linnet plans an entry over the
backends it has, gives each the regions it can run, and runs the rest as
generated PyTorch source. A backend from another package plugs in through an
entry point, without changes to Linnet.

```python
from linnet.torch import load
from linnet.torch.cublaslt import CublasLt
from linnet.torch.planned import Captured

model = load("llama.linnet", weights="model.safetensors", device="cuda", compile=True)
plan = model.plan_for("decode", [token, pos], backends=[CublasLt()], compile="inductor")
step = Captured(plan, model.context_for(plan))   # one CUDA graph from the second call on
logits, = step([token, pos])
```

This interface is new and changes between releases.

## Plans

`plan_for(entry, inputs, backends=..., compile=...)` reads the entry's
graph for inputs like `inputs` and asks each backend which regions it can
run. It times each region on its backend and on the generated source, with
`inputs`, and gives a backend the regions it runs faster. The generated
source runs every step left over, a slice of the entry at a time; `compile`
is the `torch.compile` backend for those slices, each kind compiled once.
Without `backends`, every backend `linnet.backends.available()` lists takes
part.

- Regions of one kind (each layer's q, k and v, say) are timed once. The
  times are kept by device, backend version and kind in
  `$LINNET_HOME/benchmarks.json`, so the next plan reads them instead.
- A backend takes a region only if it saves 2% or more of the generated
  source's time for the same steps (`Options.margin`), and still does in
  place: each kind taken is timed again against the generated source's
  code around it, which cutting the region out splits in two.
- The regions that save the most go first. Two regions that overlap
  (one `linear` call, or it joined with its siblings) never both run.
- `memory=bytes` caps what the plan keeps beyond the model's own weights,
  such as weights a backend converted to another layout.
- `against=` (with the timing, on CUDA) times the whole plan, replayed as a
  CUDA graph, against the entry as `run_entry(compile=against)` runs it.
  Where the plan is not faster, it runs the entry as it is: regions priced
  one by one miss what they do to each other, such as a cast at a boundary
  or a fusion two backends split. `model.plan_beats(plan, entry, inputs,
  against)` is that comparison on its own.
- `measure=False` skips the timing: each backend takes what it matches,
  the largest regions first.
- Under `amp`, or a model split into DTensors, no backend takes part: the
  plan runs the entry as it is. A card that splits itself (`Shards`) plans
  alike on every process: each times every region, and the slowest
  process's time stands, so none runs a collective the others leave out.

`plan.run(inputs, model.context_for(plan))` runs the plan step by step.
`Captured(plan, context)` runs it once, then captures it whole as a CUDA
graph and replays it.

| Backend | Takes |
| --- | --- |
| `linnet.torch.cublaslt.CublasLt` | `linear` calls without a bias, by cuBLASLt; sibling calls that read the same input (q, k and v) as one product |
| `linnet.torch.flashinfer.FlashInfer` | decode attention (one query a row, a prefix mask) by FlashInfer, on Hopper and later |

Backends for other libraries are packages of their own. The examples in
[`examples/09-backends`](https://github.com/franknoh/Linnet/tree/main/examples/09-backends)
are such packages, each installed with `pip install -e` and found through
its entry point:

| Example | Takes |
| --- | --- |
| `deepgemm` | block-scaled FP8 `linear` calls (`Fp8BlockLinear`, blocks of 128) under `numerics="fast"`, by DeepGEMM, on Hopper |
| `bitsandbytes` | NF4 `linear` calls (`Nf4Linear`), by bitsandbytes' 4-bit product |
| `flash-attn` | decode attention (one query a row, a prefix mask) by FlashAttention, on Ampere and later |

The serving engine plans its own steps: `Engine(model, plans=True)`
([Serving](integrations.md#plans)).

## Writing one

A backend is an object with a name, a version and four methods:

```python
from linnet.backends import Capabilities, ExecutableMeta, Region

class MyLinear:
    name = "my-linear"
    version = "0.1"
    contract = 1  # the contract version it was written for

    def probe(self, target):
        return Capabilities() if target.kind == "cuda" else None

    def match(self, graph, target):
        return [
            Region.of(graph, call.start, call.stop, call.index)
            for call in graph.semantic_calls("std.nn.linear::linear")
        ]

    def compile(self, region, target, options):
        return MyLinearCall(region.parameters[0])

    def estimate_cost(self, region, target):
        return None  # or Cost(seconds)

class MyLinearCall:
    def __init__(self, weight):
        self.weight = weight
        self.meta = ExecutableMeta((), ())

    def prepare(self, context):
        pass

    def execute(self, inputs, outputs, context):
        return [my_library.linear(inputs[0], context.parameters[self.weight])]
```

- `probe` declines a target it cannot run on by returning `None`.
- `match` returns regions: a call by its callee (`graph.semantic_calls`), a
  block's call (`graph.calls`, with its `path`), any range of steps
  (`Region.of`), or steps apart that run together where the first stands
  (`Region.of_steps`).
- `graph.scalar(value)` is the number a value computed from constants
  holds (an attention's scale), and `graph.reaches_prefix(mask)` whether an
  attention mask reaches a prefix of the keys, as a decode step's does.
  `graph.segment_causal(mask)` finds a packed pass's mask (each token
  seeing its own prompt up to itself): its segments, its positions and
  the first step that builds it.
- A region's `inputs` are the values it reads that earlier steps made.
  Parameters and states are not inputs: an executable reads them from
  `context.parameters` by path.
- `execute` returns the region's `outputs`, in order, and does not wait for
  the device. On CUDA it launches on PyTorch's current stream, so that a
  captured graph records it.
- A backend that compiles many regions better together may also define
  `compile_regions(regions, target, options)`.
- `context.join(paths)` lays parameters out one after another in one
  buffer and returns them as one tensor, so that an executable reads
  sibling weights (q, k and v) joined without a copy. It returns `None`
  where they cannot be laid out so.
- `prepare` runs again after the model's weights change
  (`model.copy_weights`): what the executable keeps (weights joined or
  converted) it writes into the same tensors, since a captured plan reads
  them where they are.
- An executable may also define `backward(inputs, grad_outputs, context)`,
  returning its inputs' gradients and its weights' by path. With
  gradients on, a backend's step is part of autograd: by that backward,
  or else by the region run again on the generated source and
  differentiated there, which keeps less memory but computes it twice.
  `train(plans=...)` and `linnet.torch.planned.Planned` run training
  steps through plans this way.
- `context.adopt(path, convert, restore)` converts a weight in place into
  the layout a kernel reads (its shape and dtype kept) and returns it, so
  the executable keeps no copy. A model run only through its plans offers
  it (`model.context_for(plan, adopt=True)`). It returns `None` while
  anything else may read the weight as it is: training, a tied path,
  another step of the plan, or an entry or slice already compiled. An adopted weight saves as
  it was (`restore`), and compiling an entry that reads it raises.
- `ExecutableMeta(holds=bytes)` declares memory the executable keeps
  beyond the model's weights, which the plan's `memory` budget counts.
- `estimate_cost` returns `Cost(seconds, bytes)` when the backend knows
  them without running. A measured plan times the region instead.

`graph.steps` and `graph.calls` follow the graph format
([Graph format](graph-format.md)): each step is one call the export made,
with its operands, results and attributes.

## Installing one

A package names its backends in the `linnet.backends` entry point group:

```toml
[project.entry-points."linnet.backends"]
my-linear = "my_package:MyLinear"
```

Each entry point names a function or class that takes nothing and returns a
backend. `linnet.backends.available()` lists the backends found, and
`linnet.backends.skipped()` the ones left out and why: written for another
contract version (`linnet.backends.CONTRACT`), or failing to load.

## Checking one

`model.conform(entry, inputs, backend)` runs each region the backend takes
both ways, on the same inputs: by the backend and by the generated source.
Each finding has the largest difference between the two, and `passed` when
it is within `tolerance` (0.01 by default).

```python
for finding in model.conform("decode", [token, pos], MyLinear()):
    print(finding.region.start, finding.difference, finding.passed)
```

A region that writes state runs once and is not compared.

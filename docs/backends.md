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
graph for inputs like `inputs`, asks each backend which regions it can run,
and keeps the largest ones that do not overlap. The generated source runs
every step left over, a slice of the entry at a time; `compile` is the
`torch.compile` backend for those slices, each kind compiled once.
Without `backends`, every backend `linnet.backends.available()` lists takes
part.

`plan.run(inputs, model.context_for(plan))` runs the plan step by step.
`Captured(plan, context)` runs it once, then captures it whole as a CUDA
graph and replays it.

| Backend | Takes |
| --- | --- |
| `linnet.torch.cublaslt.CublasLt` | `linear` calls without a bias, by cuBLASLt; sibling calls that read the same input (q, k and v) as one product |
| `linnet.torch.flashinfer.FlashInfer` | decode attention (one query a row, a prefix mask) by FlashInfer, on Hopper and later |

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
- A region's `inputs` are the values it reads that earlier steps made.
  Parameters and states are not inputs: an executable reads them from
  `context.parameters` by path.
- `execute` returns the region's `outputs`, in order, and does not wait for
  the device. On CUDA it launches on PyTorch's current stream, so that a
  captured graph records it.
- A backend that compiles many regions better together may also define
  `compile_regions(regions, target, options)`.

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

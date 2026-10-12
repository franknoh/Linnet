# Execution architecture

Linnet decides what to compute and where each part runs; backends do the
computing. This document fixes the contracts between the parts: the graph
the planner reads, the backend and executable interfaces, the execution
plan, and the runtime that runs it. Until the planner matches today's
speed, models keep running as generated PyTorch or JAX source.

## Layers

| Layer | Holds | Today |
| --- | --- | --- |
| Frontends | Linnet source; `torch.export`, StableHLO and ONNX importers | done |
| Core IR | the checked program: functions, generics, blocks, state; shapes symbolic | done (`include/linnet/ir`) |
| Graph | one entry under one binding: values with shape, dtype, layout and device; primitive nodes; semantic calls with their bodies | partly: `linnet graph` records it as a target sees it (M0), and `linnet.resources` traces its own |
| Planner | partition, backend selection, memory and stream plan | partly: `linnet.backends` (greedy partition by measured savings, fallback, memory budget) |
| Execution plan | executables, buffers, streams, capture segments | partly: `Plan`, run in order or captured whole (`linnet.torch.planned`) |
| Runtime | runs a plan: allocation, launches, CUDA graphs | partly: `liblinnet_runtime` (native kernels, cuBLASLt, capture); PyTorch's allocator and streams |

Five rules hold at every layer:

- No backend API (CUDA, PyTorch, XLA) appears in the Core IR or the graph.
- A backend takes a region of any size, from one node to the whole graph,
  through one interface.
- The planner chooses backends by shape, dtype, device and measured speed.
  Users may pin a backend but need not.
- Buffers, layouts, streams and transfers are explicit, so data crosses
  backends without copies.
- Every region has a fallback: a semantic call's body, and a fallback
  backend that runs every primitive.

The rules hold for CPUs, TPUs and NPUs as they do for GPUs.

## Graph

The planner reads one entry under one binding of its generics, as
`graph_export` evaluates it today: calls inlined, `static for` unrolled,
index notation lowered.

### Values

| Field | Meaning |
| --- | --- |
| `shape` | integers; symbolic dimensions with ranges come later |
| `dtype` | a format (see [Dtypes](#dtypes)) |
| `layout` | `contiguous`, explicit strides, or a layout a backend names, such as DeepGEMM's TMA-aligned transposed scales |
| `device` | a slot of the placement |
| `storage` | the value whose bytes it shares: a view, or an in-place write |

### Nodes

| Kind | What |
| --- | --- |
| primitive | the evaluator's operations: elementwise, compare, select, convert, reshape, transpose, broadcast, slice, concat, iota, gather, reduce, contract, cumsum, scatter-add |
| semantic call | a call of an `op`: its identity (`std.nn.attention::attention`), generics, operands and results, and its body as a nested graph |
| scope | a block method's call, such as `layers.3.mlp.forward`: a named group a backend may take whole |
| state | a state member read before the call and written after it, aliased to its buffer |
| collective | `all_reduce` or `all_gather` over a process group |
| kernel | a launch of an op's `kernel` |

Semantic calls and scopes nest. A backend matches at any level, and an
unmatched semantic call is replaced by its body. A megakernel backend takes
a whole scope, such as a decoder layer, and nothing else changes.

`RecordingTarget` builds the graph in C++: it records every call the
exporter makes while passing it on to a target, so the graph holds that
target's choices. `linnet graph` prints it as JSON
([Graph format](../graph-format.md)), and `--replay` rebuilds the target's
source from it unchanged. The Python package reads it in process later. `linnet.resources` moves to this graph, so the
analysis and the runtime see the same steps.

## Backends

A backend is a Python object with four methods. `estimate_cost` is
optional.

```python
class Backend(Protocol):
    name: str
    version: str

    def probe(self, target: Target) -> Capabilities | None: ...
    def match(self, graph: Graph, target: Target) -> Iterable[Region]: ...
    def compile(self, region: Region, target: Target, options: Options) -> Executable: ...
    def estimate_cost(self, region: Region, target: Target) -> Cost | None: ...
```

- `probe` says whether the backend runs on the target (device kind,
  compute capability, library versions) and what it supports.
- `match` proposes the regions it can run: a semantic call by identity, a
  scope, or any set of nodes.
- `compile` returns an executable for one region.
- `estimate_cost` returns seconds and the bytes the executable keeps beyond
  the model's own. Without it, the planner benchmarks the region.

Operator libraries (cuBLASLt, FlashInfer, DeepGEMM), fusing compilers
(Triton, Inductor) and whole-graph compilers (XLA, TensorRT) implement the
same four methods. Only the regions `match` returns differ.

### Executables

```python
class Executable(Protocol):
    meta: ExecutableMeta

    def prepare(self, context: Context) -> None: ...
    def execute(self, inputs: Sequence[Tensor], outputs: Sequence[Tensor], context: Context) -> None: ...
```

- `prepare` runs once per set of weights. It converts them into the layout
  the kernels read, such as DeepGEMM's scales or packed int4, and keeps
  the result.
- `execute` enqueues the work on the context's stream and returns. It never
  waits for the device.

`ExecutableMeta` declares what the planner and the runtime need:

| Field | Meaning |
| --- | --- |
| `inputs`, `outputs` | each tensor's dtype, shape, layout and device |
| `dynamic` | the dimensions that may change between calls without a new compile |
| `aliases` | outputs that reuse an input's buffer: in-place state, views |
| `workspace` | scratch bytes, taken from the runtime's arena |
| `capturable` | whether a CUDA graph may record it |
| `stream` | `caller` (runs on the given stream) or `own` (synchronizes through events) |
| `host` | `native` (callable from C) or `python` |
| `effects` | `pure`, `collective`, `random`, or `state` |

### Native execution

Compiling may happen in Python; executing should not need it. DeepGEMM,
TileKernels and Triton compile in Python to a cubin and its launch
parameters, and the runtime launches it. An executable that runs only
through Python declares `host: python`. The planner counts its host cost,
and inside a captured CUDA graph the cost disappears.

Native executables follow a C ABI (`include/linnet/runtime/runtime.h`):

```c
typedef struct LnExecutable {
    uint32_t abi_version;
    void* state;
    int (*prepare)(void* state, LnContext* context);
    int (*execute)(void* state, const LnTensor* inputs, size_t input_count,
                   LnTensor* outputs, size_t output_count, LnContext* context);
    const char* (*error)(void* state);
    void (*destroy)(void* state);
} LnExecutable;
```

`LnContext` holds the device, the stream, the call's workspace and whether a
CUDA graph records the call. `LnTensor` is laid out as DLPack's `DLTensor`,
so PyTorch, JAX and CuPy share buffers without copies.

## Planner

The planner turns a graph into an execution plan in four steps:

1. **Probe.** Keep the backends whose `probe` accepts the target.
2. **Partition.** Walk the graph from the outermost scope inward. At each
   scope and semantic call, collect the regions backends propose and take
   the cheapest. Replace an unclaimed semantic call by its body and
   continue inside it. Primitives no backend claimed go to the fallback
   backend, in the largest regions that stay convex.
3. **Connect.** Insert layout, dtype and device conversions where
   neighbouring regions disagree. Step 2 counts their cost.
4. **Schedule.** Place buffers by lifetime, assign streams, and group
   consecutive capturable executables on one stream into capture segments.

A region is convex: no path leaves it and comes back. Partitioning and
costing sit behind separate interfaces (`Partitioner`, `CostModel`). The
first partitioner is greedy with a benchmark cache; a global solver can
replace it without touching backends.

A region's cost comes from the benchmark cache (keyed by device, backend
version and the region's kind, the same for each layer), else from timing
it (`Measured`: the graph run once by the fallback, cut at every region's
boundary, gives each region real inputs), else from the backend's
`estimate_cost`. The fallback is priced on the same steps (steps apart, as
their ranges). A region is taken when it saves at least `Options.margin`
of the fallback's time, the largest saving first, within `Options.memory`
bytes kept beyond the model's own. Conversion costs between regions are
not counted yet.

## Execution plan

A plan holds:

- the executables in order, each with its input and output buffers;
- the buffers: activations in one arena placed by lifetime, persistent
  weights and state, and workspaces;
- the streams and the events between them;
- the capture segments.

Plans serialize. A compile cache keyed by the graph's hash, the target and
the backends' versions skips planning and compiling on the next load. (The
execution plan is unrelated to `linnet plan`, which prints the Core IR.)

## Runtime

The runtime is a C++ library in this repository with Python bindings. It
links no CUDA library: it loads the driver at run time, as Triton does, so
it builds and tests on every CI platform.

- **Memory.** The runtime owns the allocator: one arena per plan for
  activations and a pool for persistent buffers. PyTorch draws from it
  through a `MemPool`, so two allocators never compete. XLA keeps its own
  pool and runs only whole graphs at first.
- **Launch.** A plan runs as a flat list of resolved calls. Dispatch costs
  tens of nanoseconds a call against 2-4 µs a kernel launch, so generated
  host code (LLVM) waits until a profile shows dispatch.
- **CUDA graphs.** Each capture segment is captured once per binding and
  replayed. A non-capturable executable, such as XLA with its own command
  buffers or a host callback, ends a segment.
- **Interop.** Inputs and results are DLPack tensors. `linnet.torch.load`
  keeps returning an `nn.Module` whose `forward` runs the plan.

## Dtypes

The compiler keeps three dtype families, and the prelude names every
dtype:

```linnet
pub dtype f32    = float(exponent: 8, mantissa: 23)
pub dtype bf16   = float(exponent: 8, mantissa: 7)
pub dtype f8e4m3 = float(exponent: 4, mantissa: 3, inf: false)
pub dtype e2m1   = float(exponent: 2, mantissa: 1, inf: false, nan: false)
pub dtype i4     = int(bits: 4)
pub dtype u8     = int(bits: 8, signed: false)
```

- `bool`, `int(bits, signed)` and `float(exponent, mantissa, specials)`
  give the compiler all it checks: size, constant folding, which operators
  apply, literal types, cast rounding.
- A format outside the families, such as NF4 or a codebook, declares a
  carrier dtype and the ops that widen to it and narrow from it.
- Formats whose elements share a scale (MXFP4, block-scaled FP8) stay
  structs with ops. A dtype describes one element.
- Backends match dtypes by format. A backend without a format stores it in
  a container dtype and widens it where it computes.

The change updates spec chapter 3, the grammar and the spec tests with it.
The spec fixes cast rounding and what overflow does in a format without
infinities.

## Training and distribution

The first planner runs inference. Training stays on the current PyTorch
path until the planner has a cost model (P3); its backward pass will be the
Linnet autodiff graph, planned like any other. Collectives are graph nodes,
and NCCL and the one-shot all-reduce become communication backends. A
tensor-parallel model is one plan per process, as `Shards` builds it today.

## Milestones

| | Work | Done when |
| --- | --- | --- |
| M0 | `linnet graph`: the graph as JSON, built by a recording `GraphTarget` | the existing targets, replayed from the graph, print the same source for every Nest card |
| P0 | Backend and Executable contract; runtime core; the generated PyTorch source as one full-graph backend | every Nest model runs through a plan at today's speed and passes today's tests |
| P1 | partition and fallback; cuBLASLt and FlashInfer backends; the text passes over generated Python become graph passes | Llama 3.1 8B decode on an H100 runs attention on FlashInfer, the MLP on cuBLASLt and the rest on the fallback in one CUDA graph, at least as fast as `linnet-cudagraphs` |
| P2 | plugin SDK: entry points, versions, capability checks, conformance against the generated source (docs/backends.md) | an external package adds a backend without changing Linnet |
| P3 | cost-based selection: benchmark and compile caches, memory budget, conversion costs | the planner picks backends with no user choice |
| P4 | DeepGEMM, TileKernels, CUTLASS, Triton, XLA, TensorRT, external megakernels | each passes its conformance tests |

Dtype families land during P0 and P1, before the plugin SDK, because
backends declare dtypes by format.

P1 on an H100, Llama 3.1 8B bf16, each plan replayed as one CUDA graph,
against the entry as the bench runs it (Inductor, one CUDA graph):

| | entry | cuBLASLt + Inductor slices | + FlashInfer attention |
| --- | ---: | ---: | ---: |
| `decode`, 1 row, positions 0-128 | 160 tok/s | 159 tok/s | 151 tok/s |
| `decode_rows`, 16 rows, positions 1,000-3,377 | 1,772 tok/s | 1,825 tok/s | 1,904 tok/s |

FlashInfer pays at long caches and costs at short ones. P3 leaves the choice
to measured costs. On another H100 host, with both backends offered:

| | entry | cuBLASLt everywhere | + FlashInfer everywhere | measured |
| --- | ---: | ---: | ---: | ---: |
| `decode`, 1 row | 155 tok/s | 154 tok/s | | 154 tok/s: cuBLASLt, no FlashInfer |
| `decode_rows`, 16 rows | 1,720 tok/s | 1,780 tok/s | 1,842 tok/s | 1,881 tok/s: FlashInfer, no cuBLASLt |

Measuring takes about 10 s on the first plan; with the times kept, planning
takes about 1 s. Timed alone, a region misses what its neighbours do: at
16 rows FlashInfer's attention saves 5 us a layer timed alone, while the
plan that uses it runs about 25 us a layer faster than the entry. Counting
those effects is part of the conversion costs still to do.

## Not planned

- A Linnet megakernel compiler or runtime.
- Complete support of every framework at once.
- A global partition solver from the start.
- Replacing backends with Linnet's own kernels.
- Requiring users to pick one backend.

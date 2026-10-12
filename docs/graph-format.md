# Graph format

`linnet graph` prints one entry, under one binding of its generics, as the
calls the exporter makes to a target: each tensor operation, each library
call the target took, and every function, op and block method call around
them. The execution planner reads this graph.

```bash
linnet graph --std stdlib --entry decode --bind Batch=1 --bind MaxSeq=4096 model.linnet
linnet graph --target jax --replay --entry decode ... model.linnet
```

`--target` picks the target whose calls are recorded, `torch` (the default)
or `jax`. The graph holds that target's choices, such as which library calls
it takes. Every other option is the target's own (`linnet torch`,
`linnet jax`). `--replay` makes the same calls on a fresh target and prints
its source, which is what `linnet torch` or `linnet jax` prints.

`--slices <file>` prints the source of ranges of the graph's steps, each
an entry of its own, as `{"sources": [source]}`. The file lists them:

```text
[ { "name", "start", "stop", "inputs": [id], "outputs": [id], "states": [[path, id]] } ]
```

A range's steps are the graph's nodes but `call`, numbered in order. Its
`inputs` become the entry's inputs, and the parameters and states its steps
read become its own. A constant it reads is made again. `outputs` and
`states` (each state's value after the call, by path) are its results. The
planner gives the generated PyTorch source the ranges no other backend
takes this way.

Version 1 follows the exporter's target interface and changes with it.

## Document

```text
{ "version": 1, "target": "torch",
  "values": [ { "id", "name", "shape": [int], "dtype" } ],
  "nodes":  [ node ] }
```

| Field | Meaning |
| --- | --- |
| `values` | every tensor a call made, in order; `name` is the target's |
| `nodes` | the calls, in order |

## Nodes

A node is `{ "op", ...attributes, "operands": [operand], "results": [id] }`.
An operand is a value's id, `null` for an absent optional, or
`{ "value", "shape", "dtype" }` where the call viewed the value with another
shape or dtype.

A call of a function, op or block method is
`{ "op": "call", "callee", "semantic", "path", "generics", "arguments", "body", "results" }`:
`semantic` is 1 for an `op`, `path` is a method's block (`layers.3.mlp`), and
`body` holds its nodes. An op whose library call the target took holds one
`native_call`.

| `op` | Attributes |
| --- | --- |
| `input`, `parameter`, `state` | `name` or `path`, `shape`, `dtype` |
| `constant` | `literal` (`kind`, `value`), `dtype` |
| `elementwise` | `kind` (`add`, `exp`, ...), `shape`, `dtype` |
| `compare` | `kind` (`lt`, `eq`, ...), `shape` |
| `select`, `convert` | `shape`, `dtype` |
| `reshape`, `transpose`, `broadcast` | `shape`; `permutation`; `dims` |
| `slice` | `starts`, `limits`, `strides`, `shape` |
| `concat`, `cumsum` | `axis` |
| `iota` | `length` |
| `gather`, `reduce` | `shape`; `kind` and `dims` for `reduce` |
| `contract` | `lhs_axes`, `rhs_axes`, `out_axes`, `shape`, `dtype` |
| `native_call` | `implementation`, `generics`, `shape`, `dtype` |
| `update_row`, `scatter_add` | `shape` and `dtype` for `scatter_add` |
| `enable_placement`, `set_slot`, `transfer`, `release` | `slots` or `slot`: placement across devices |
| `gather_parts` | a fully sharded parameter gathered whole |
| `begin_remat`, `end_remat` | a block the backward pass computes again |
| `begin_while`, `while_condition`, `end_while` | a runtime loop; `while_initial_condition`, `while_trailing_condition` where the target checks there |
| `begin_counted`, `end_counted` | a loop of known length: `start`, `stop` |
| `begin_custom_gradient`, `end_custom_gradient` | an op with a `grad`; `arguments` counts its tensor arguments |
| `launch_kernel` | `callee` (the op), `name`, `program`, `grid`, `result_shapes`, `result_dtypes`, `warps`, `stages` |
| `finish` | `module`, `block`, `entry`; `returns` counts the entry's results, `states` names the rest |

`"declined": true` marks a call the target declined (`contract`,
`native_call`, `update_row`, `scatter_add`, `launch_kernel`); the exporter's
fallback follows it. A callback the target ran, an op's backward pass or a
kernel's fallback body, is a region of its node:
`{ "role": "pullback" | "body", "arguments", "nodes", "results" }`.

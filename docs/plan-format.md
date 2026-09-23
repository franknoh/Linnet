# Plan format

`linnet plan` prints the root block as one JSON document. A materializer
reads it to build a module, bind weights, and run entries. The plan contains
no tensor data. Version 1:

```text
{ "version": 1,
  "module":    "a.b.c",
  "root":      { "name", "generics": [generic], "constraints": [constraint] },
  "manifest":  [ { "path", "kind": "param" | "buffer" | "state", "dtype", "shape": [unit],
                   "repeat": [dim], "optional" } ],
  "blocks":    { <name>: { "module", "pub", "generics": [generic], "constraints": [constraint],
                           "members": [ { "name", "kind": "param" | "buffer" | "state" | "sub",
                                          "type": type } ] } },
  "functions": [ { "name", "kind": "fn" | "op" | "entry", "block": name | null, "pub",
                   "generics": [generic], "constraints": [constraint],
                   "results": [type], "states": [path], "body": region } ],
  "constants": [ { "name", "pub", "type": type, "contextual", "body": region } ] }
```

| Field | Meaning |
| --- | --- |
| `module` | the root file's module path; function names are `module::name`, methods `module::Block.method` |
| `manifest` | every parameter, buffer, and state member of the instantiated hierarchy; `[*]` in a path stands for each element of a sub array and `repeat` lists those lengths |
| `states` | the state members a function reads or writes, directly or through calls, relative to its block |
| `constants` | module-level `const` items as regions yielding one value; `contextual` marks one declared without a type |

## Dimensions and types

| | |
| --- | --- |
| `generic` | `{ "name", "kind": "dim" \| "shape" \| "dtype", "sym" \| "var", "default"? }`; a dtype generic also has `"class"` |
| `dim` | an integer, `{ "sym", "name" }`, `{ "packsize", "name" }`, or `{ "op": "add" \| "mul" \| "floordiv" \| "mod" \| "min" \| "max", "args": [dim] }` |
| `unit` | a `dim`, or `{ "pack", "name" }` for every axis of a shape pack |
| `type` | `{ "kind": "scalar", "dtype" }`, `{ "kind": "tensor", "shape": [unit], "dtype" }`, `tuple`, `optional`, `array`, or `block` / `struct` / `enum` with `name`, `module`, `args` |
| dtype | a name such as `"bf16"`, or `{ "var", "name" }` |

Dimensions stay symbolic. A materializer binds the root block's generics,
then each block instance's, then an entry's from its inputs, and evaluates
the expressions.

## Regions and operations

| | |
| --- | --- |
| `region` | `{ "args": [value], "ops": [op] }`; `value` is `{ "id", "name", "type" }` |
| `op` | `{ "kind", "operands": [id], "results": [value], "attrs": {...}, "regions": [region] }` |

Kinds and attributes follow the Core IR operations in
`include/linnet/ir/ir.hpp`. `comprehension` and `reduce` list their `indices`
with domains; `slice` lists per-axis `start`, `stop`, `step`, `squeeze` (or
`whole` for a shape pack); calls carry `callee`, `substitution`, and
`generics` in the callee's declaration order; `state.read` and `state.write`
name their member; `while` has a condition region and a body region.

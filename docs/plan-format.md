# Plan format

`linnet plan` prints the root block as a JSON plan, version 1, from which
a materializer builds a module. It holds no tensor data.

## Document

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
| `module` | the root file's module path; functions are `module::name`, methods `module::Block.method` |
| `manifest` | every parameter, buffer and state of the instantiated hierarchy; `[*]` in a path is each element of a sub array, whose lengths `repeat` lists |
| `states` | the state members a function reads or writes, directly or through calls, relative to its block |
| `constants` | module-level `const` items, each a region yielding one value; `contextual` marks an untyped declaration |

`--functions` prints the module-level entries, with `root` `null` and an
empty `manifest`.

## Dimensions and types

| Name | Form |
| --- | --- |
| `generic` | `{ "name", "kind": "dim" \| "shape" \| "dtype", "sym" \| "var", "default"? }`; a dtype generic also has `"class"` |
| `dim` | an integer, `{ "sym", "name" }`, `{ "packsize", "name" }`, or `{ "op": "add" \| "mul" \| "floordiv" \| "mod" \| "min" \| "max", "args": [dim] }` |
| `unit` | a `dim`, or `{ "pack", "name" }` for every axis of a shape pack |
| `type` | `{ "kind": "scalar", "dtype" }`, `{ "kind": "tensor", "shape": [unit], "dtype" }`, `tuple`, `optional`, `array`, or `block` / `struct` / `enum` with `name`, `module`, `args` |
| dtype | a name such as `"bf16"`, or `{ "var", "name" }` |

Bind the root block's generics, then each block instance's, then an
entry's from its inputs, and evaluate the expressions.

## Regions and operations

| Name | Form |
| --- | --- |
| `region` | `{ "args": [value], "ops": [op] }`; `value` is `{ "id", "name", "type" }` |
| `op` | `{ "kind", "operands": [id], "results": [value], "attrs": {...}, "regions": [region] }` |

Kinds and attributes follow the Core IR operations in
`include/linnet/ir/ir.hpp`:

- `comprehension`, `reduce`: `indices` with domains
- `slice`: per-axis `start`, `stop`, `step`, `squeeze` (or `whole` for a
  shape pack)
- calls: `callee`, `substitution`, and `generics` in the callee's
  declaration order
- `state.read`, `state.write`: the member
- `while`: a condition region and a body region

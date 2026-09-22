# The plan format

`linnet plan [--root <Block>] <file>` prints one JSON document that a
materializer consumes. Version 1:

```text
{ "version": 1,
  "module":    "a.b.c",
  "root":      { "name", "generics": [generic], "constraints": [constraint] },
  "manifest":  [ { "path", "kind": "param" | "buffer", "dtype", "shape": [unit],
                   "repeat": [dim], "optional" } ],
  "blocks":    { <name>: { "module", "pub", "generics": [generic], "constraints": [constraint],
                           "members": [ { "name", "kind": "param" | "buffer" | "sub",
                                          "type": type } ] } },
  "functions": [ { "name", "kind": "fn" | "op" | "entry", "block": name | null, "pub",
                   "generics": [generic], "constraints": [constraint],
                   "results": [type], "body": region } ] }
```

- `module` is the path of the root file's module; function names are
  `module.path::name` (`module.path::Block.method` for methods).
- `generic`: `{ "name", "kind": "dim" | "shape" | "dtype", "sym" | "var", "default"? }`;
  a dtype generic also has `"class": "any" | "numeric" | "integer" | "float"`.
  Symbols and dtype variables are numbered by the compiler; a call's
  `substitution` maps the callee's numbers to expressions in the caller.
- `dim`: an integer, `{ "sym", "name" }`, `{ "packsize", "name" }`, or
  `{ "op": "add" | "mul" | "floordiv" | "mod" | "min" | "max", "args": [dim] }`.
- `unit` (one element of a shape): a `dim`, or `{ "pack", "name" }` standing
  for every axis of a shape pack.
- `type`: `{ "kind": "scalar", "dtype" }`, `{ "kind": "tensor", "shape": [unit], "dtype" }`,
  `tuple`, `optional`, `array`, `block` / `struct` / `enum` with `name`,
  `module`, and `args`, `shape`, `unit`. A dtype is a name such as `"bf16"` or
  `{ "var", "name" }`.
- `region`: `{ "args": [value], "ops": [op] }` with one block per region;
  `value` is `{ "id", "name", "type" }`.
- `op`: `{ "kind", "operands": [id], "results": [value], "attrs": {...}, "regions": [region] }`.
  Kinds and attributes follow the Core IR operations
  (`include/linnet/ir/ir.hpp`); `comprehension` and `reduce` list their
  `indices` with domains, `slice` lists per-axis `start`/`stop`/`step`/`squeeze`
  or `whole` (a shape pack), and calls carry `callee`, `substitution`, and
  `generics` (the same bindings as generic values, in the callee's declaration
  order, so that a reader without the callee can still spell the call).

Dimensions stay symbolic. The materializer binds the root block's generics,
then each block instance's, then an entry's from its inputs, and evaluates the
expressions. The plan contains no tensor data.

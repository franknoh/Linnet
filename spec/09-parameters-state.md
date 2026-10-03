# 9. Block members

Members declare a block's parameters, buffers, state, and child blocks.

## 9.1 `param`

`param` declares an externally supplied parameter tensor, with no payload:

```text
param weight: Tensor[Out, In; T]
```

An optional parameter defaults to `none` and may be absent:

```text
param bias: Tensor[Out; T]? = none
```

A `param` initializer other than `none` is forbidden.

## 9.2 `buffer`

`buffer` declares persistent non-parameter data, such as statistics or fixed tables:

```text
buffer running_mean: Tensor[H; f32]
```

Its payload is bound externally, like a `param`'s.

## 9.3 `state`

`state` declares mutable execution state owned by a block instance, such as a KV cache:

```text
state cache: Tensor[Batch, KvHeads, MaxSeq, Head; T]
```

A state member has a tensor type whose shape the block's generics fix, and cannot be optional. It carries no payload: the runtime supplies the initial value (zeros by default) and keeps the latest value between entry calls.

Inside the block's functions and entries, the member name reads the current value, and an assignment replaces it:

```text
cache = updated
```

- The assigned value MUST have the declared type.
- Reads after an assignment observe it, in program order.
- A function may assign only its own block's state members. A parent reads a child's state through its path (`layers[0].attention.cache`) but cannot assign it.
- Hidden global mutation is not permitted.
- The optimizer MUST NOT merge, remove, or reorder state reads and writes relative to one another.

State members appear in the parameter manifest with kind `state`. Weight files never contain them.

## 9.4 `sub`

`sub` declares child blocks:

```text
sub norm: RMSNorm<Hidden, T>
sub layers: [DecoderLayer<Hidden, T>; Layers]
```

A child block may be optional, present or absent as a whole:

```text
sub pooler: Linear<Hidden, Hidden, T>? = none
```

Its value is `some(block)` or `none`, read with `match`. Every parameter inside it is optional in the manifest. A materializer treats the block as absent when the weights lack any parameter it requires (one declared without `?`), and as present otherwise. An array of blocks cannot be optional.

## 9.5 Parameter paths

A compiler instantiating a root block MUST derive deterministic hierarchical paths from sub-block and parameter names:

```text
layers.0.attention.q_proj.weight
```

These are the canonical parameter-manifest names unless a binding manifest (§10.2) maps them to external tensor names.

## 9.6 Parameter manifest

Semantic analysis of a fully instantiated root block can produce a parameter manifest. It holds no tensor bytes and records at least each tensor's canonical path, kind (`param`, `buffer`, or `state`), shape expression after known substitutions, dtype, and optional or required status.

# 9. Parameters, Buffers, State, and Structural Members

## 9.1 `param`

`param` declares an externally supplied parameter tensor:

```text
param weight: Tensor[Out, In; T]
```

A parameter declaration does not contain tensor payload data.

Optional parameters are permitted:

```text
param bias: Tensor[Out; T]? = none
```

The default `none` means the parameter may be absent. A non-`none` tensor initializer for `param` is forbidden.

## 9.2 `buffer`

`buffer` declares persistent non-parameter data such as statistics or fixed tables:

```text
buffer running_mean: Tensor[H; f32]
```

Payload binding is external just like `param` unless a future specification explicitly defines generated constants.

Backends may map buffers to framework-specific non-trainable state.

## 9.3 `state`

`state` declares mutable execution state owned by a block instance, such as a KV cache:

```text
state cache: Tensor[Batch, KvHeads, MaxSeq, Head; T]
```

A state member has a tensor type whose shape is fixed by the block's generics, like a `param`. It carries no payload: the runtime supplies the initial value (backends default to zeros) and keeps the latest value between entry calls.

Inside the block's functions and entries, the member name reads the current value, and an assignment statement replaces it:

```text
cache = updated
```

The assigned value MUST have the declared type. Reads after an assignment observe it, in program order. A function may assign only state members of its own block; a parent block reads a child's state through its path (`layers[0].attention.cache`) but cannot assign it.

State flow is explicit. In Core IR every read and write is an operation on the block instance, ordered within its region, and each function records the state members it touches directly or through the functions it calls. The optimizer MUST NOT merge, remove, or reorder these operations relative to one another. Backends thread the values: a graph export takes the initial values as extra inputs and returns the final values as extra results, and a framework materializer keeps them as the module's non-trainable state, reset on request. Hidden global mutation is not permitted.

State members appear in the parameter manifest with kind `state`; weight files never contain them.

## 9.4 `sub`

`sub` declares structural child blocks:

```text
sub norm: RMSNorm<Hidden, T>
sub layers: [DecoderLayer<Hidden, T>; Layers]
```

## 9.5 Parameter paths

A compiler instantiating a root block MUST derive deterministic hierarchical paths from sub-block and parameter names.

Example:

```text
layers.0.attention.q_proj.weight
```

These paths form the canonical parameter-manifest names unless a binding manifest maps them to external tensor names.

## 9.6 Parameter manifest

Semantic analysis of a fully instantiated root block can produce a parameter manifest containing at least:

- canonical path;
- kind (`param`, `buffer`, or `state`);
- tensor shape expression after known substitutions;
- dtype;
- optional/required status.

This manifest is compiler data and contains no tensor bytes.

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

`state` is reserved for mutable execution state such as KV caches or RNG streams.

The initial language version reserves the keyword but does not define state mutation semantics.

A future design MUST make state flow explicit in IR. Hidden global mutation is not permitted.

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
- kind (`param` or `buffer`);
- tensor shape expression after known substitutions;
- dtype;
- optional/required status.

This manifest is compiler data and contains no tensor bytes.

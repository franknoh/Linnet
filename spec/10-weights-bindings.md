# 10. External Weights and Binding Manifests

## 10.1 Separation

Linnet source describes structure and computation. Tensor payloads live in external artifacts.

SafeTensors is a recommended first-class weight format because it naturally separates named tensor metadata and raw tensor data, but the Linnet language is not semantically tied to one storage format.

## 10.2 Binding manifest

A binding manifest maps canonical Linnet parameter paths to external tensor keys.

Example human-facing YAML representation:

```yaml
sources:
  model:
    type: safetensors
    files: "model-*.safetensors"

bindings:
  "embedding.weight":
    source: model
    key: "model.embed_tokens.weight"

  "layers.{i}.attention.q_proj.weight":
    source: model
    key: "model.layers.{i}.self_attn.q_proj.weight"
```

## 10.3 Binding manifests are not programs

Binding manifests MUST NOT contain executable expressions, arbitrary transformations, imports, shell interpolation, Python tags, custom YAML object tags, callbacks, or conditions.

Permitted behavior is limited to declarative source selection and key mapping.

Tensor transformations such as transpose, reshape, dequantization, or concatenation belong in Linnet computation semantics or a separately specified artifact-conversion tool, not in the binding manifest.

## 10.4 Safe YAML subset

If YAML is accepted, implementations MUST parse it in a safe mode that permits only ordinary scalar, mapping, and sequence nodes. Arbitrary tagged object construction is forbidden.

A JSON representation MAY be supported as an equivalent serialization of the same abstract binding model.

## 10.5 Validation

When weight metadata is available, a binder SHOULD validate:

- required key existence;
- duplicate bindings;
- tensor rank;
- concrete dimensions that are known;
- dtype compatibility;
- optional parameter absence;
- unexpected external tensors according to configured strictness.

Shape or dtype mismatch is an error by default.

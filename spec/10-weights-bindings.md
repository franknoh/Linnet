# 10. Weights and bindings

Weights live outside source and bind to parameters through a declarative manifest.

## 10.1 Separation

Linnet source describes structure and computation; tensor payloads live in external artifacts. SafeTensors is a recommended first-class weight format, but the language is not semantically tied to one storage format.

## 10.2 Binding manifest

A binding manifest maps canonical parameter paths (§9.5) to external tensor keys. A path absent from the manifest binds to the tensor of the same name. Linnet's tools read a flat JSON object from expanded parameter paths, one per block-array element, to tensor keys:

```json
{
  "embedding.weight": "model.embed_tokens.weight",
  "layers.0.attention.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
  "layers.1.attention.q_proj.weight": "model.layers.1.self_attn.q_proj.weight"
}
```

A richer serialization of the same model, naming several sources and templating array indices, MAY be supported, for example as YAML:

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

## 10.3 Manifests are not programs

A binding manifest only selects sources and maps keys. It MUST NOT contain executable expressions, arbitrary transformations, imports, shell interpolation, Python tags, custom YAML object tags, callbacks, or conditions. Tensor transformations (transpose, reshape, dequantization, concatenation) belong in Linnet computation or a separate conversion tool.

## 10.4 Safe YAML subset

Implementations that accept YAML MUST parse it in a safe mode that permits only scalar, mapping, and sequence nodes. Arbitrary tagged object construction is forbidden.

## 10.5 Validation

When weight metadata is available, a binder SHOULD validate required key existence, duplicate bindings, tensor rank, known concrete dimensions, dtype compatibility, optional parameter absence, and unexpected external tensors according to configured strictness. Shape or dtype mismatch is an error by default.

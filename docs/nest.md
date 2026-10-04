# Nest

Nest is the Linnet model zoo: each model is a checked `.linnet` source plus a
SafeTensors checkpoint on the Hugging Face Hub. Browse it at
[nest.franknoh.dev](https://nest.franknoh.dev); the registry is
[github.com/franknoh/nest](https://github.com/franknoh/nest).

## Load a model

Install the `nest` extra (`uv add "linnet-lang[nest,torch]"`), then:

```python
from linnet import nest

model = nest.load("tinyllama-1.1b-chat", backend="torch", numerics="fast", compile="inductor")
logits = model(tokens)
```

`load` downloads the model and its checkpoint, binds the weights, and returns
the backend's object. Other keyword arguments go to the loader; `generics=`
overrides the card's values. A local model directory works in place of a
name.

| `backend=` | Loader |
| --- | --- |
| `"torch"` | `linnet.torch.load` |
| `"jax"` | `linnet.jax.load` |
| `"jax_source"` | `linnet.jax.load_source` |
| `"jax_model"` | `linnet.jax.load_model` |
| `"onnx_model"` | `linnet.onnx.load_model` |
| `"nnx"` | `linnet.jax.load_nnx` |

## Model directory

```text
models/<name>/
  nest.toml        the card
  README.md        what it is, how to load it, provenance
  bindings.json    Linnet parameter path -> checkpoint tensor name
  src/ or *.linnet the architecture
  preview.svg      the main entry, drawn by `linnet.diagram`
```

The card names the source, root block, generics, and Hub checkpoint. Its
full schema is in the registry's README.

## Checks

```bash
python -m linnet.nest check models/gpt2
python -m linnet.nest preview models/gpt2 -o models/gpt2/preview.svg
python -m linnet.nest index . -o index.json
```

Registry pull requests must pass `check`:

- the README and the card's required fields;
- the source compiles with the card's generics;
- every required parameter has a checkpoint tensor of the same shape and
  dtype, read from the SafeTensors headers on the Hub;
- the main entry exports to StableHLO, ONNX, PyTorch source, and JAX source.

`index` writes the registry document that `load` and the site read. In
Python: `nest.check`, `nest.index`, `nest.preview`, `nest.describe`,
`nest.Card.read`.

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

`load` takes a model from three places:

| `name_or_dir` | Where |
| --- | --- |
| `"tinyllama-1.1b-chat"` | a Nest name, fetched from the registry |
| `"org/name"`, `"hf://org/name@revision"` | a Hugging Face Hub repo with `nest.toml` at its root |
| `"path/to/model"` | a model directory on disk |

It reads the card, binds the weights and returns the backend's object. The
weights are the checkpoint beside the card when the directory or repo holds
it, or else the Hub checkpoint the card names. Other keyword arguments go to
the loader; `generics=` overrides the card's values, and `weights=` uses a
checkpoint already on disk.

Loading a repo runs no code from it. The compiler checks the `.linnet`
source and generates the backend's code itself.

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
<name>/
  nest.toml          the card
  README.md          what it is, how to load it, provenance
  bindings.json      Linnet parameter path -> checkpoint tensor name
  src/ or *.linnet   the architecture
  *.safetensors      the checkpoint, when it is not on the Hub
  preview.svg        the main entry, drawn by `linnet.diagram`
```

The card names the source, root block, generics and checkpoint. In
`[weights]`, `repo` names the Hub repo that holds `files`; leave it out to
keep the files beside the card. Its full schema is in the registry's README.

## Share a model

A model directory is a Hub repo as it is. Check it, then upload it:

```bash
python -m linnet.nest check my-model
python -m linnet.nest push my-model me/my-model      # nest.push in Python
```

Anyone can then load it with `nest.load("me/my-model")`.
`python -m linnet.nest pull <name>` downloads a model directory from the
registry or the Hub and prints where it is.

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
  dtype, read from the SafeTensors headers on the Hub or beside the card;
- the main entry exports to StableHLO, ONNX, PyTorch source, and JAX source.

`index` writes the registry document that `load` and the site read. In
Python: `nest.check`, `nest.index`, `nest.preview`, `nest.describe`,
`nest.push`, `nest.resolve`, `nest.Card.read`.

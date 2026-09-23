# Nest

Nest is the Linnet model zoo: a registry of models whose architecture is a
checked `.linnet` source and whose weights are a SafeTensors checkpoint on
the Hugging Face Hub. The registry lives at
[github.com/franknoh/nest](https://github.com/franknoh/nest); browse the
models at [nest.franknoh.dev](https://nest.franknoh.dev).

```python
from linnet import nest

model = nest.load("tinyllama-1.1b-chat", backend="torch", numerics="fast", compile="inductor")
logits = model(tokens)
```

`load` fetches the model's directory from the registry, downloads the
checkpoint files from the Hub (cached by `huggingface_hub`), binds them
through the card's `bindings.json`, and returns the backend's object:
`linnet.torch.load` for `"torch"`, `linnet.jax.load` for `"jax"`,
`load_source` for `"jax_source"`, `load_nnx` for `"nnx"`. Keyword arguments
go to that loader; `generics=` overrides the card's values. A local model
directory works in place of a name. Install with the `nest` extra
(`pip install "linnet-lang[nest]"`).

## A model

```text
models/<name>/
  nest.toml        the card
  README.md        what it is, how to load it, provenance
  bindings.json    Linnet parameter path -> checkpoint tensor name
  src/ or *.linnet the architecture
  preview.svg      the main entry, drawn by `linnet.diagram`
```

The card names the source and its root block, binds the root generics for
this checkpoint, points at the Hub repository and files, and carries the
title, license, summary, and links (`huggingface` is required; `github`,
`arxiv`, `homepage` when they exist). The full schema is in the registry's
README.

## Checks

```bash
python -m linnet.nest check models/gpt2
python -m linnet.nest preview models/gpt2 -o models/gpt2/preview.svg
python -m linnet.nest index . -o index.json
```

`check` is what a pull request to the registry must pass:

- the README and the card's required fields;
- the source compiles with the card's generics;
- every parameter of the manifest has a tensor of the same shape and dtype
  in the checkpoint, read from the SafeTensors headers on the Hub without
  downloading (optional parameters may be absent);
- the main entry exports to StableHLO, ONNX, PyTorch source, and JAX
  source with the `[check]` bindings.

`index` writes the registry document `load` and the site read: every card
with its entries, parameter count, and files. From Python, `nest.Card.read`,
`nest.check`, `nest.describe`, `nest.index`, and `nest.preview` do the same.

# linnet-lang

The Python side of Linnet: one package, `linnet`, with a backend per
framework behind an extra.

```bash
uv add "linnet-lang[torch] @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"   # linnet.torch
```

The other extras are `jax` (`linnet.jax`), `flax` (`linnet.jax.load_nnx`),
`onnx` (`linnet.onnx`) and `nest` (`linnet.nest`). The package is not on
PyPI yet; uv builds it, and the compiler it carries, from the repository.

```python
from linnet.torch import load

model = load("src/model.linnet", generics={"H": 64, "Layers": 2}, weights="weights/")
```

The core needs only NumPy, and each backend imports its framework on first
use. The platform wheels carry the `linnet` compiler and its standard
library; otherwise the compiler comes from `LINNET_BIN` or `PATH`.
Documentation:
[linnet.franknoh.dev/docs/python](https://linnet.franknoh.dev/docs/python).

Development: `uv sync --all-extras`, then `uv run pytest`, `uv run basedpyright`
and `uv run ruff check src tests`.

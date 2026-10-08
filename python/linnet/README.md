# linnet-lang

The Python side of Linnet: one package, `linnet`, with a backend per
framework behind an extra.

```bash
uv add "linnet-lang[torch]"      # linnet.torch
uv add "linnet-lang[jax]"        # linnet.jax
uv add "linnet-lang[flax]"       # linnet.jax.load_nnx
uv add "linnet-lang[onnx]"       # linnet.onnx
uv add "linnet-lang[nest]"       # linnet.nest
```

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

# linnet-lang

The Python side of Linnet: one package, `linnet`, with a backend per
framework behind an extra.

```bash
pip install "linnet-lang[torch]"      # linnet.torch
pip install "linnet-lang[jax]"        # linnet.jax
pip install "linnet-lang[flax]"       # linnet.jax.load_nnx
pip install "linnet-lang[onnx]"       # linnet.onnx
```

```python
from linnet.torch import load

model = load("src/model.linnet", generics={"H": 64, "Layers": 2}, weights="weights/")
```

`linnet` itself runs the compiler (`linnet plan`), reads plans and
SafeTensors checkpoints, and needs only NumPy. `linnet.torch`, `linnet.jax`,
and `linnet.onnx` import their framework on first use. The compiler binary
comes from `LINNET_BIN` or `PATH`.

Development: `uv sync --all-extras` installs every backend and the test
dependencies; `uv run pytest`, `uv run pyright`, `uv run ruff check src tests`.

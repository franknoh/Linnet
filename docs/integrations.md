# Serving and tools

A Linnet model is a checked source plus a checkpoint, so serving systems
and node editors get it through the same two things every framework does.
This page covers Triton Inference Server and ComfyUI; the framework
adapters are on their own pages.

## Triton Inference Server

```bash
python -m linnet.triton export gpt2 -o model_repository --bind B=1 --bind S=64
tritonserver --model-repository=model_repository
```

`linnet.triton.export` writes one model directory the server loads as it
is: `config.pbtxt` with the entry's inputs and outputs (names from the
signature, dtypes, static dims from the bindings) and a version directory
holding the model. `model` is a Nest name or directory (the card supplies
source, generics, checkpoint, and bindings) or a `.linnet` file with
`--weights` and `--bindings`.

| Backend | What is written | Use it for |
| --- | --- | --- |
| `--backend onnx` (default) | `1/model.onnx`, self-contained: `linnet onnx` output with the checkpoint as initializers (external data beyond 1 GB), `platform: "onnxruntime_onnx"` | stateless entries such as `forward` |
| `--backend python` | `1/model.py` plus the source and weights; `backend: "python"`, runs the entry through `linnet.torch` | entries with `state` (KV caches), any dtype, `numerics="fast"` |

Shapes are static (`max_batch_size: 0`); export one model per shape you
serve, or one per batch size. Entry generics (`B`, `S`) come from `--bind`
with a file, or from the card's `[check]` table with a Nest model, and
`--bind` overrides both. The Python backend needs `linnet-lang[torch]` and
the compiler in the server's environment (`LINNET_BIN`, `LINNET_STD`); it
returns `bf16` results as `f32`, which the config declares.

The same packaging is available directly: `linnet.onnx.export_model(source,
generics=..., weights=..., entry=...)` returns the ONNX model with its
inputs and outputs, and `.save(path)`.

```python
from linnet import triton

repository = triton.export("models/tinyllama-1.1b-chat", "model_repository",
                           generics={"S": 128}, backend="python", numerics="fast")
repository.inputs      # (Tensor(name='tokens', dtype='i32', dims=(1, 128)),)
```

## vLLM, SGLang, TGI

```bash
python -m linnet.hf export tinyllama-1.1b-chat -o serve/tinyllama
vllm serve serve/tinyllama
```

The LLM serving stacks implement a fixed set of architectures and load
them from Transformers checkpoint directories. `linnet.hf.export` writes
that directory for a Linnet model whose structure is one of them: it
recognizes the family from the parameter paths and generics of the typed
program (`llama`: grouped-query attention, RMS norms, SwiGLU; `gpt2`), derives
`config.json` from the generics and the module constants (`THETA` becomes
`rope_theta`), streams the checkpoint into `model.safetensors` under the
Transformers tensor names, and copies the tokenizer files from the card's
Hub repository. vLLM, SGLang, TGI, and `transformers` load the result as
they load any model of that family.

The same command takes a `.linnet` file with `--weights`, `--bind`, and
`--tokenizer <repo>`, so a model trained or modified in Linnet ships the
same way as long as it keeps a known structure. A structure outside the
recognized families is refused with the paths that did not fit; a general
out-of-tree vLLM model class for arbitrary Linnet programs is not part of
this release.

`transformers` agrees with the Linnet interpreter on the exported model to
1e-4 (the tests export tiny GPT-2 and Llama configurations and compare
logits).

## llama.cpp and Ollama

```bash
python -m linnet.gguf export tinyllama-1.1b-chat -o serve/tinyllama \
    --converter ~/llama.cpp/convert_hf_to_gguf.py --outtype q8_0
ollama create tinyllama -f serve/tinyllama/Modelfile
```

llama.cpp runs the same fixed set of architectures from GGUF files, and
its `convert_hf_to_gguf.py` is the one place that knows their tensor layout,
tokenizer metadata, and quantization. `linnet.gguf.export` writes the
Transformers checkpoint as above, runs that converter on it (`--converter`,
or `LLAMA_CPP` pointing at a llama.cpp checkout), and writes an Ollama
`Modelfile` next to the GGUF file, with the chat template translated when
the tokenizer's template is one Ollama has an equivalent for (ChatML,
Zephyr, Llama 2). The family restriction is the same as for vLLM: `llama`
and `gpt2`. A model outside them cannot run in llama.cpp anyway, and Linnet
does not pretend otherwise.

## ComfyUI

[linnet-comfyui](https://github.com/franknoh/linnet-comfyui) is a custom
node pack. Install it into `custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/franknoh/linnet-comfyui
pip install -r linnet-comfyui/requirements.txt
```

| Node | |
| --- | --- |
| Load Linnet model | a `.linnet` file or package, generics as JSON, a SafeTensors path, `numerics`, `compile`, device; outputs a `LINNET_MODEL` |
| Load from Nest | a registry name; downloads the source and checkpoint |
| Run entry | a model, an entry name, up to four `TENSOR` inputs, entry generics as JSON; outputs the results |
| Tensor from list / to string | JSON lists and readable summaries for token ids and small tensors |
| Image to tensor / Tensor to image | ComfyUI `IMAGE` (`[B, H, W, C]` float) to `[B, C, H, W]` in a dtype, and back |
| Argmax | the token ids of logits |
| Model info | `print(model)`: the block tree with names and shapes |
| Reset state | zeroes the model's `state` members between generations |

The nodes are thin: each maps to one call on `linnet.torch` or
`linnet.nest`, and the entry's signature is checked before it runs, so a
wrong shape is an error at the node, not a crash inside a kernel. The
compiler comes from `LINNET_BIN` or `PATH`.

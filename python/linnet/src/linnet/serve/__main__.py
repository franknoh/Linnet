"""`python -m linnet.serve MODEL`: a Nest decoder served over HTTP with
OpenAI's API (see `linnet.serve.server`), its tokenizer from Transformers."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from typing import Any


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m linnet.serve",
        description="Serve a Nest decoder over HTTP with OpenAI's completions API.",
    )
    parser.add_argument("model", help="a Nest model: a registry name or a card's directory")
    parser.add_argument("--backend", choices=("torch", "jax", "onnx"), default="torch")
    parser.add_argument("--device", help="the PyTorch device (default: cuda when there is one)")
    parser.add_argument("--batch", type=int, default=32, help="requests in flight (`Batch`)")
    parser.add_argument(
        "--max-seq", type=int, default=2048, help="the longest prompt plus completion (`MaxSeq`)"
    )
    parser.add_argument("--dtype", help="the `T` generic (default: the card's; f16 on ONNX)")
    parser.add_argument("--weights", help="a checkpoint on disk instead of the card's")
    parser.add_argument(
        "--tokenizer", help="a Hub id or directory (default: the card's weights repository)"
    )
    parser.add_argument("--name", help="the model's name in the API (default: the card's)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--warmup",
        type=int,
        nargs="*",
        default=[],
        metavar="LENGTH",
        help="prompt lengths to compile before serving (others compile when first seen)",
    )
    args = parser.parse_args(argv)

    from .. import nest
    from . import Engine
    from .server import Server

    card = nest.resolve(args.model)
    generics: dict[str, int | str] = {"Batch": args.batch, "MaxSeq": args.max_seq}
    if args.dtype:
        generics["T"] = args.dtype
    elif args.backend == "onnx":
        generics["T"] = "f16"
    common: dict[str, Any] = {"generics": generics, "weights": args.weights, "cast_dtype": True}
    if args.backend == "torch":
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = nest.load(
            card.directory, backend="torch", device=device, numerics="fast", compile=True, **common
        )
    elif args.backend == "jax":
        model = nest.load(card.directory, backend="jax_model", **common)
    else:
        model = nest.load(card.directory, backend="onnx_model", numerics="fast", **common)

    import transformers as transformers_module

    transformers: Any = transformers_module
    source = args.tokenizer or (card.weights.repo if card.weights else None)
    if source is None:
        parser.error("the card names no weights repository: pass --tokenizer")
    revision = card.weights.revision if card.weights and not args.tokenizer else None
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, revision=revision)
    # The end-of-sequence tokens: the tokenizer's, and the generation config's
    # (Llama 3's instruct models end a turn with one of their own).
    ids: list[Any] = [tokenizer.eos_token_id]
    try:
        generation = transformers.GenerationConfig.from_pretrained(source, revision=revision)
        more: Any = generation.eos_token_id
        ids += more if isinstance(more, list) else [more]
    except OSError:
        pass  # no generation config
    eos = {int(i) for i in ids if i is not None}

    engine = Engine(model)
    engine.warmup(args.warmup)
    server = Server(
        engine, tokenizer, name=args.name or card.name, eos=eos, host=args.host, port=args.port
    )
    host, port = server.address
    print(f"serving {args.name or card.name} at http://{host}:{port}/v1", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

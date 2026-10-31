"""`linnet serve MODEL` (`python -m linnet.serve MODEL`): a decoder served
over HTTP with OpenAI's API (see `linnet.serve.server`), its tokenizer from
Transformers. MODEL is anything `linnet.nest.load` takes: a Nest name, a
Hub repo (with a card, or a `transformers` checkpoint of a family Nest
knows), or a model directory.

A card with an `embed_padded` or `embed` entry and no decoding entries is
an embedding model: it is served at `/v1/embeddings` instead (see
`linnet.serve.embeddings`), with PyTorch."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

if TYPE_CHECKING:
    from .. import nest
    from .server import Server, Tokenizer

    class _Common(TypedDict):
        """The keywords every backend's `nest.load` takes here."""

        generics: dict[str, int | str]
        weights: str | None
        cast_dtype: bool

    # The part of Transformers `main` uses, typed: its own stubs leave
    # `from_pretrained` partially unknown.
    class _HfTokenizer(Tokenizer, Protocol):
        eos_token_id: int | None

    class _AutoTokenizer(Protocol):
        def from_pretrained(self, source: str, /, *, revision: str | None) -> _HfTokenizer: ...

    class _Generation(Protocol):
        eos_token_id: int | list[int] | None

    class _GenerationConfig(Protocol):
        def from_pretrained(self, source: str, /, *, revision: str | None) -> _Generation: ...

    class _Transformers(Protocol):
        AutoTokenizer: _AutoTokenizer
        GenerationConfig: _GenerationConfig


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        # ^C, while loading or serving: said, not a traceback.
        print("\nexiting...", file=sys.stderr, flush=True)
        return 0


def _main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(
        prog="linnet serve",
        description="Serve a decoder over HTTP with OpenAI's completions and chat API, "
        "or an embedding model with its embeddings API.",
    )
    parser.add_argument(
        "model", help="a Nest name, a Hugging Face Hub repo (org/name), or a model directory"
    )
    parser.add_argument("--backend", choices=("torch", "jax", "onnx"), default="torch")
    parser.add_argument("--device", help="the PyTorch device (default: cuda when there is one)")
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="requests in flight (`Batch`); for embeddings, the most inputs in one pass",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        help="the longest prompt plus completion (`MaxSeq`, default 2048); for embeddings, "
        "the longest input (default: the largest power of two up to the card's `MaxPositions`, "
        "else 512)",
    )
    parser.add_argument(
        "--pool",
        type=int,
        metavar="POSITIONS",
        help="serve from pages of one pool of POSITIONS cached positions (the card's paged "
        "entries, PyTorch): --batch is then the requests in flight, --max-seq the longest "
        "(default: on PyTorch, when the card has them, --batch times --max-seq positions)",
    )
    parser.add_argument(
        "--fixed-rows",
        action="store_true",
        help="keep a cache row of --max-seq positions for each of --batch requests instead "
        "of pages (each step then reads every row whole)",
    )
    parser.add_argument("--dtype", help="the `T` generic (default: the card's; f16 on ONNX)")
    parser.add_argument("--weights", help="a checkpoint on disk instead of the card's")
    parser.add_argument(
        "--tokenizer", help="a Hub id or directory (default: the card's weights repository)"
    )
    parser.add_argument(
        "--name", help="the model's name in the API (default: the Hub repo, or the card's name)"
    )
    parser.add_argument(
        "--plans",
        action="store_true",
        help="run the step and packed passes as execution plans over the backends found "
        "(PyTorch on CUDA; planned while warming up)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--warmup",
        type=int,
        nargs="*",
        metavar="LENGTH",
        help="prompt lengths to compile for before serving (default: every pass size; "
        "with lengths, the passes longer prompts need compile when first seen)",
    )
    args = parser.parse_args(argv)

    from .. import nest
    from . import Engine
    from .server import Server

    if args.pool and args.backend != "torch":
        parser.error("--pool serves from pages with the PyTorch backend")
    if args.pool and args.fixed_rows:
        parser.error("--pool serves from pages, --fixed-rows from rows: give one")
    card = nest.resolve(args.model)
    entries = {entry.short_name for entry in card.program().entries()}
    if entries & {"embed_padded", "embed"} and not entries & {"decode_rows", "decode_paged"}:
        return _serve_embeddings(parser, args, card)
    if args.max_seq is None:
        args.max_seq = 2048
    # Pages by default: a step reads the rows in flight to their lengths,
    # where fixed rows are read whole, every one of them.
    pages = {"prefill_paged", "decode_paged"} <= entries
    if args.backend == "torch" and pages and not args.pool and not args.fixed_rows:
        args.pool = args.batch * args.max_seq
    generics: dict[str, int | str] = (
        {"Batch": 1, "MaxSeq": args.pool}
        if args.pool
        else {"Batch": args.batch, "MaxSeq": args.max_seq}
    )
    if args.dtype:
        generics["T"] = args.dtype
    elif args.backend == "onnx":
        generics["T"] = "f16"
    common: _Common = {"generics": generics, "weights": args.weights, "cast_dtype": True}
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

    transformers, source, revision = _tokenizer_source(parser, args, card)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, revision=revision)
    # The end-of-sequence tokens: the tokenizer's, and the generation config's
    # (Llama 3's instruct models end a turn with one of their own).
    ids: list[int | None] = [tokenizer.eos_token_id]
    try:
        generation = transformers.GenerationConfig.from_pretrained(source, revision=revision)
        more = generation.eos_token_id
        ids += more if isinstance(more, list) else [more]
    except OSError:
        pass  # no generation config
    eos = {int(i) for i in ids if i is not None}

    name = _name(args, card)
    engine = (
        Engine(model, paged=True, rows=args.batch, max_len=args.max_seq, plans=args.plans)
        if args.pool
        else Engine(model, plans=args.plans)
    )
    # Every pass size by default: a pass first compiled while serving holds
    # up every request in flight for as long.
    engine.warmup(args.warmup or [])
    return _serve(Server(engine, tokenizer, name=name, eos=eos, host=args.host, port=args.port))


def _serve_embeddings(
    parser: argparse.ArgumentParser, args: argparse.Namespace, card: nest.Card
) -> int:
    """Serves the card's embedding entry at `/v1/embeddings`."""
    from .. import nest
    from ..torch import LinnetModule
    from .embeddings import Embedder
    from .server import Server

    if args.backend != "torch":
        parser.error("embeddings are served with the PyTorch backend")
    if args.pool or args.fixed_rows or args.plans:
        parser.error("--pool, --fixed-rows and --plans are for decoders")
    import torch

    positions = card.generics.get("MaxPositions")
    limit = positions if isinstance(positions, int) else None
    # The longest power of two the card's positions hold (512 of RoBERTa's
    # 514, whose first two rows are reserved), else 512.
    max_len = args.max_seq or (1 << (limit.bit_length() - 1) if limit else 512)
    if limit is not None and max_len > limit:
        parser.error(f"--max-seq is at most the card's MaxPositions, {limit}")
    generics: dict[str, int | str] = {"T": args.dtype} if args.dtype else {}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = nest.load(
        card.directory,
        backend="torch",
        device=device,
        numerics="fast",
        compile=True,
        generics=generics,
        weights=args.weights,
        cast_dtype=True,
    )
    assert isinstance(model, LinnetModule)
    transformers, source, revision = _tokenizer_source(parser, args, card)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, revision=revision)
    embedder = Embedder(model, max_len=max_len, batch=args.batch)
    # Every padded shape, so no request waits for one to compile.
    embedder.warmup()
    name = _name(args, card)
    return _serve(Server(embedder, tokenizer, name=name, host=args.host, port=args.port))


def _tokenizer_source(
    parser: argparse.ArgumentParser, args: argparse.Namespace, card: nest.Card
) -> tuple[_Transformers, str, str | None]:
    """Transformers, and where the tokenizer is: `--tokenizer`, or the card's
    weights repository at its revision."""
    import transformers as transformers_module

    transformers = cast("_Transformers", transformers_module)
    source = args.tokenizer or (card.weights.repo if card.weights else None)
    if source is None:
        parser.error("the card names no weights repository: pass --tokenizer")
    revision = card.weights.revision if card.weights and not args.tokenizer else None
    return transformers, source, revision


def _name(args: argparse.Namespace, card: nest.Card) -> str:
    """The model's name in the API: `--name`, else a Hub repo's id, as other
    servers name it, else the card's name."""
    from .. import nest

    hub = nest.HUB_REPO.match(args.model)
    return args.name or (hub["repo"] if hub and not Path(args.model).exists() else card.name)


def _serve(server: Server) -> int:
    host, port = server.address
    print(f"serving {server.name} at http://{host}:{port}/v1", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Nest, the Linnet model zoo: cards, validation, previews, and loading.

A model in Nest is a directory with a `nest.toml` card, a README, the
model's Linnet source, and (usually) a `bindings.json` naming the tensors of
a SafeTensors checkpoint on the Hugging Face Hub. `check` validates all of
that against the compiler and the Hub's tensor metadata without downloading
weights; `index` builds the registry document a site or `load` reads;
`load` fetches the weights and materializes the model in a backend.

    python -m linnet.nest check models/tinyllama-1.1b-chat
    python -m linnet.nest index models -o index.json
    python -m linnet.nest preview models/gpt2 -o models/gpt2/preview.svg
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tomllib
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from . import diagram, ir
from .compiler import LinnetError, run_compiler, std_arguments
from .weights import read_bindings

REGISTRY = "https://raw.githubusercontent.com/franknoh/nest/main"
EXPORTS = ("stablehlo", "onnx", "torch", "jax")
SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "I8": "i8",
    "I16": "i16",
    "I32": "i32",
    "I64": "i64",
    "U8": "u8",
    "U16": "u16",
    "U32": "u32",
    "U64": "u64",
    "F16": "f16",
    "BF16": "bf16",
    "F32": "f32",
    "F64": "f64",
}


class NestError(LinnetError):
    """A card is malformed or a model cannot be fetched."""


@dataclass(frozen=True, slots=True)
class Links:
    huggingface: str | None = None
    github: str | None = None
    arxiv: str | None = None
    homepage: str | None = None


@dataclass(frozen=True, slots=True)
class Weights:
    """A SafeTensors checkpoint on the Hub: `files` of `repo` at `revision`."""

    repo: str
    files: tuple[str, ...]
    revision: str | None = None
    bindings: str | None = None


@dataclass(frozen=True, slots=True)
class Card:
    """`nest.toml`, read from a model directory."""

    directory: Path
    name: str
    title: str
    summary: str
    license: str
    family: str | None
    tags: tuple[str, ...]
    links: Links
    source: str
    root: str | None
    entry: str | None
    generics: Mapping[str, int | str]
    check: Mapping[str, int | str]
    weights: Weights | None
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    # The published parameter count, when counting the checkpoint's tensors
    # would get it wrong: 4-bit weights packed two to a byte (gpt-oss's
    # MXFP4) count half their parameters as elements.
    parameters: int | None = None

    @staticmethod
    def read(directory: str | Path) -> Card:
        path = Path(directory)
        card_path = path / "nest.toml"
        if not card_path.exists():
            raise NestError(f"{path} has no nest.toml")
        try:
            document = tomllib.loads(card_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise NestError(f"{card_path}: {error}") from None
        model = _table(document, "model")
        links = _table(document, "links")
        source = _table(document, "source")
        weights_table = document.get("weights")
        weights = None
        if isinstance(weights_table, dict):
            table = cast(dict[str, Any], weights_table)
            files = table.get("files", [])
            weights = Weights(
                repo=str(table.get("repo", "")),
                files=tuple(str(f) for f in cast(list[Any], files)),
                revision=_optional_str(table, "revision"),
                bindings=_optional_str(table, "bindings"),
            )
        return Card(
            directory=path,
            name=str(model.get("name", path.name)),
            title=str(model.get("title", "")),
            summary=str(model.get("summary", "")),
            license=str(model.get("license", "")),
            family=_optional_str(model, "family"),
            tags=tuple(str(t) for t in cast(list[Any], model.get("tags", []))),
            links=Links(
                huggingface=_optional_str(links, "huggingface"),
                github=_optional_str(links, "github"),
                arxiv=_optional_str(links, "arxiv"),
                homepage=_optional_str(links, "homepage"),
            ),
            source=str(source.get("path", "src/lib.linnet")),
            root=_optional_str(source, "root"),
            entry=_optional_str(source, "entry"),
            generics=MappingProxyType(_generic_values(_table(document, "generics"))),
            check=MappingProxyType(_generic_values(_table(document, "check"))),
            weights=weights,
            extra=MappingProxyType({k: v for k, v in document.items() if k not in _KNOWN_TABLES}),
            parameters=int(model["parameters"]) if "parameters" in model else None,
        )

    @property
    def source_path(self) -> Path:
        return self.directory / self.source

    @property
    def bindings_path(self) -> Path | None:
        if self.weights is None or self.weights.bindings is None:
            return None
        return self.directory / self.weights.bindings

    def program(self, std_root: str | Path | None = None) -> ir.Program:
        return ir.load_program(self.source_path, root=self.root, std_root=std_root)


_KNOWN_TABLES = {"model", "links", "source", "generics", "check", "weights"}


def _table(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise NestError(f"`[{name}]` must be a table")
    return cast(dict[str, Any], value)


def _optional_str(table: Mapping[str, Any], key: str) -> str | None:
    value = table.get(key)
    return None if value is None else str(value)


def _generic_values(table: Mapping[str, Any]) -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    for key, value in table.items():
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise NestError(f"generic `{key}` must be an integer or a dtype name")
        out[str(key)] = value
    return out


# -------------------------------------------------------------------- check


def check(
    card: Card,
    *,
    std_root: str | Path | None = None,
    hub: bool = True,
    exports: bool = True,
) -> list[str]:
    """Everything a Nest model must satisfy; returns the problems found.

    The README and the card's required fields; the source compiles with the
    card's generics; every parameter the manifest lists has a tensor of the
    same shape and dtype in the Hub checkpoint (read from the SafeTensors
    headers, nothing is downloaded); and the main entry exports to every
    format.
    """
    problems: list[str] = []
    readme = card.directory / "README.md"
    if not readme.exists() or len(readme.read_text(encoding="utf-8").strip()) < 200:
        problems.append("README.md is missing or shorter than a paragraph")
    if card.name != card.directory.name:
        problems.append(
            f"model.name `{card.name}` differs from the directory `{card.directory.name}`"
        )
    for key, value in (("title", card.title), ("summary", card.summary), ("license", card.license)):
        if not value:
            problems.append(f"model.{key} is required")
    if card.links.huggingface is None:
        problems.append("links.huggingface is required (the checkpoint's home)")
    if card.weights is None:
        problems.append("[weights] is required: a SafeTensors checkpoint on the Hub")
    elif not card.weights.files or not card.weights.repo:
        problems.append("weights.repo and weights.files are required")
    elif not all(f.endswith(".safetensors") for f in card.weights.files):
        problems.append("every weights file must be .safetensors")
    if not card.source_path.exists():
        problems.append(f"source `{card.source}` does not exist")
        return problems

    try:
        program = card.program(std_root)
    except LinnetError as error:
        problems.append(f"the source does not compile: {error}")
        return problems
    try:
        bindings = ir.bind_generics(program.root.generics, card.generics)
    except LinnetError as error:
        problems.append(f"[generics]: {error}")
        return problems
    try:
        entry = program.entry(card.entry)
    except LinnetError as error:
        problems.append(str(error))
        return problems

    if card.weights is not None and card.weights.repo and hub:
        problems += _check_weights(card, program, bindings)
    if exports:
        problems += _check_exports(card, program, entry, std_root)
    return problems


def expand_paths(entry: ir.ManifestEntry, bindings: ir.Bindings) -> list[str]:
    """`layers[*].w` with repeat (2,) becomes `layers.0.w`, `layers.1.w`."""
    paths = [entry.path]
    for repeat in entry.repeat:
        count = ir.evaluate_dim(repeat, bindings)
        paths = [p.replace("[*]", f".{i}", 1) for p in paths for i in range(count)]
    return paths


def hub_safetensors_header(repo: str, filename: str, revision: str | None = None) -> dict[str, Any]:
    """The SafeTensors header of one file in a Hub repository.

    Read with two ranged requests: the eight bytes holding the header's
    length, then the header itself. No tensor data crosses the network.

    `huggingface_hub.get_safetensors_metadata` would be the obvious call, but
    it only knows the `transformers` naming (`model.safetensors` and its
    index), so it cannot see a checkpoint named the way `diffusers` names
    one. A card already says which files it means, so read those.
    """
    from huggingface_hub import hf_hub_url  # type: ignore[import-untyped]
    from huggingface_hub.utils import get_session  # type: ignore[import-untyped]

    url = hf_hub_url(repo, filename, revision=revision)
    session = get_session()
    prefix = session.get(url, headers={"Range": "bytes=0-7"}, timeout=30)
    prefix.raise_for_status()
    if prefix.status_code != 206 or len(prefix.content) != 8:
        raise LinnetError(f"{repo}/{filename}: the Hub did not honour a ranged request")
    (size,) = struct.unpack("<Q", prefix.content)
    body = session.get(url, headers={"Range": f"bytes=8-{7 + size}"}, timeout=60)
    body.raise_for_status()
    return cast(dict[str, Any], json.loads(body.content.decode("utf-8")))


def _check_weights(card: Card, program: ir.Program, bindings: ir.Bindings) -> list[str]:
    assert card.weights is not None
    try:
        import huggingface_hub  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return ["huggingface-hub is not installed (pip install 'linnet-lang[nest]')"]
    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    for filename in card.weights.files:
        try:
            header = hub_safetensors_header(
                card.weights.repo, filename, revision=card.weights.revision
            )
        except Exception as error:  # any Hub failure is one problem
            return [f"cannot read the headers of {card.weights.repo}/{filename}: {error}"]
        for name, info in header.items():
            if name == "__metadata__":
                continue
            entry = cast(dict[str, Any], info)
            shape = tuple(int(d) for d in cast(list[Any], entry["shape"]))
            tensors[str(name)] = (shape, str(entry["dtype"]))
    mapping: dict[str, str] = {}
    bindings_path = card.bindings_path
    if bindings_path is not None:
        if not bindings_path.exists():
            return [f"weights.bindings `{card.weights.bindings}` does not exist"]
        mapping = read_bindings(bindings_path)

    problems: list[str] = []
    for entry in program.manifest:
        if entry.kind != "param":
            continue
        try:
            shape = ir.evaluate_shape(entry.shape, bindings)
            dtype = ir.evaluate_dtype(entry.dtype, bindings)
        except LinnetError as error:
            problems.append(f"{entry.path}: {error}")
            continue
        for path in expand_paths(entry, bindings):
            source = mapping.get(path, path)
            if source not in tensors:
                if not entry.optional:
                    problems.append(f"missing tensor `{source}` for `{path}`")
                continue
            found_shape, found_dtype = tensors[source]
            if found_shape != shape:
                problems.append(
                    f"`{source}` has shape {list(found_shape)}, `{path}` needs {list(shape)}"
                )
            elif SAFETENSORS_DTYPES.get(found_dtype) != dtype:
                problems.append(f"`{source}` is {found_dtype}, `{path}` needs {dtype}")
    return problems


def _check_exports(
    card: Card, program: ir.Program, entry: ir.Function, std_root: str | Path | None
) -> list[str]:
    binds = [f"{k}={v}" for k, v in {**card.generics, **card.check}.items()]
    missing = [g.name for g in entry.generics if g.name not in card.check and g.default is None]
    if missing:
        names = ", ".join(missing)
        return [f"[check] must bind the entry generics {names} to export `{entry.short_name}`"]
    problems: list[str] = []
    for target in EXPORTS:
        arguments = [target, "--root", program.root.name, "--entry", entry.short_name]
        for bind in binds:
            arguments += ["--bind", bind]
        try:
            run_compiler(*arguments, *std_arguments(std_root), str(card.source_path))
        except LinnetError as error:
            first = str(error).strip().splitlines()
            problems.append(f"`linnet {target}` fails: {first[-1] if first else error}")
    return problems


# ------------------------------------------------------------ index/preview


def describe(card: Card, std_root: str | Path | None = None) -> dict[str, Any]:
    """The card plus what the compiler knows: entries, parameter count, files."""
    program = card.program(std_root)
    parameters: int | None = card.parameters
    if parameters is None:
        try:
            parameters = parameter_count(card, program)
        except LinnetError:
            parameters = None
    files = sorted(
        str(p.relative_to(card.directory)).replace(os.sep, "/")
        for p in card.directory.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(card.directory).parts)
    )
    return {
        "name": card.name,
        "title": card.title,
        "summary": card.summary,
        "license": card.license,
        "family": card.family,
        "tags": list(card.tags),
        "links": {k: v for k, v in asdict(card.links).items() if v is not None},
        "source": card.source,
        "root": program.root.name,
        "module": program.module,
        "entry": program.entry(card.entry).short_name,
        "generics": dict(card.generics),
        "check": dict(card.check),
        "parameters": parameters,
        "entries": [
            {
                "name": e.short_name,
                "signature": ir.format_signature(e),
                "inputs": [{"name": p.name, "type": ir.format_type(p.type)} for p in e.params],
                "results": [ir.format_type(t) for t in e.results],
                "states": list(e.states),
            }
            for e in program.entries()
        ],
        "blocks": {name: program.describe(name) for name in sorted(program.blocks)},
        "weights": None
        if card.weights is None
        else {
            "repo": card.weights.repo,
            "files": list(card.weights.files),
            "revision": card.weights.revision,
            "bindings": card.weights.bindings,
        },
        "files": files,
    }


def parameter_count(card: Card, program: ir.Program) -> int:
    """The model's parameters, counting a tensor two paths share only once.

    Tied embeddings are two parameters of the program bound to one tensor of
    the checkpoint, and the published figure counts that tensor once.
    """
    bindings = card.bindings_path
    mapping = read_bindings(bindings) if bindings is not None and bindings.exists() else {}
    values = ir.bind_generics(program.root.generics, card.generics)
    counted: set[str] = set()
    total = 0
    for entry in program.manifest:
        if entry.kind != "param":
            continue
        elements = 1
        for size in ir.evaluate_shape(entry.shape, values):
            elements *= size
        for path in expand_paths(entry, values):
            source = mapping.get(path, path)
            if source in counted:
                continue
            counted.add(source)
            total += elements
    return total


def index(root: str | Path, std_root: str | Path | None = None) -> dict[str, Any]:
    """The registry document: every model under `root/models`, described."""
    base = Path(root)
    models_dir = base / "models" if (base / "models").is_dir() else base
    models = [
        describe(Card.read(d), std_root)
        for d in sorted(models_dir.iterdir())
        if (d / "nest.toml").exists()
    ]
    return {"version": 1, "models": models}


def preview(
    card: Card,
    *,
    std_root: str | Path | None = None,
    expand: int = 1,
    theme: Literal["light", "dark"] = "light",
) -> str:
    """The SVG diagram of the card's main entry."""
    program = card.program(std_root)
    return diagram.render(program, card.entry, format="svg", expand=expand, theme=theme)


def format_parameters(count: int | None) -> str:
    if count is None:
        return "?"
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if count >= size:
            return f"{count / size:.1f}{unit}".replace(".0", "")
    return str(count)


# --------------------------------------------------------------------- load


def cache_dir() -> Path:
    override = os.environ.get("LINNET_NEST_CACHE")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "linnet" / "nest"


def fetch(name: str, *, registry: str = REGISTRY, cache: str | Path | None = None) -> Path:
    """Downloads a model's directory from the registry; returns the local path."""
    target = Path(cache) if cache is not None else cache_dir()
    with urllib.request.urlopen(f"{registry}/index.json") as response:
        document = json.loads(response.read().decode("utf-8"))
    models = cast(list[dict[str, Any]], cast(dict[str, Any], document)["models"])
    match = next((m for m in models if m["name"] == name), None)
    if match is None:
        raise NestError(f"Nest has no model `{name}`")
    directory = target / name
    directory.mkdir(parents=True, exist_ok=True)
    for relative in cast(list[str], match["files"]):
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{registry}/models/{name}/{relative}") as response:
            destination.write_bytes(response.read())
    return directory


def resolve(name_or_dir: str | Path, **fetch_options: Any) -> Card:
    """A local model directory, or a registry name fetched into the cache."""
    path = Path(name_or_dir)
    if (path / "nest.toml").exists():
        return Card.read(path)
    return Card.read(fetch(str(name_or_dir), **fetch_options))


def download_weights(card: Card) -> Path:
    """Fetches the card's checkpoint files from the Hub; returns their directory."""
    if card.weights is None:
        raise NestError(f"`{card.name}` names no weights")
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError:
        raise NestError(
            "huggingface-hub is not installed (pip install 'linnet-lang[nest]')"
        ) from None
    paths = [
        Path(hf_hub_download(card.weights.repo, filename, revision=card.weights.revision))
        for filename in card.weights.files
    ]
    return paths[0] if len(paths) == 1 else paths[0].parent


def load(
    name_or_dir: str | Path,
    *,
    backend: Literal["torch", "jax", "jax_source", "jax_model", "nnx"] = "torch",
    std_root: str | Path | None = None,
    generics: Mapping[str, int | str] | None = None,
    weights: str | Path | None = None,
    **options: Any,
) -> Any:
    """Materializes a Nest model in a backend with its published weights.

    `generics` overrides the card's values (a different `Batch`, say);
    `weights` uses a checkpoint already on disk instead of downloading the
    card's; `options` go to the backend's loader (`numerics`, `compile`,
    `device`, `trainable`, ...). `"jax_model"` is every entry over one copy of
    the weights with the state kept on the device (`linnet.jax.load_model`),
    which decoding and serving need.
    """
    card = resolve(name_or_dir)
    values = {**card.generics, **(generics or {})}
    if weights is None:
        weights = download_weights(card)
    bindings = card.bindings_path
    common: dict[str, Any] = {
        "generics": values,
        "root": card.root,
        "std_root": std_root,
        "weights": weights,
        "bindings": None if bindings is None else str(bindings),
    }
    if backend == "torch":
        from . import torch as torch_backend

        return torch_backend.load(card.source_path, **common, **options)
    from . import jax as jax_backend

    if backend == "jax_model":
        return jax_backend.load_model(card.source_path, **common, **options)
    entry = options.pop("entry", card.entry)
    if backend == "jax":
        return jax_backend.load(card.source_path, entry=entry, **common, **options)
    if backend == "jax_source":
        return jax_backend.load_source(card.source_path, entry=entry, **common, **options)
    if backend == "nnx":
        return jax_backend.load_nnx(card.source_path, entry=entry, **common, **options)
    raise NestError(f"unknown backend `{backend}`")


# ----------------------------------------------------------------------- CLI


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m linnet.nest", description="Nest model zoo tools."
    )
    parser.add_argument("--std", help="the standard library directory")
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="validate model directories")
    check_parser.add_argument("directories", nargs="+")
    check_parser.add_argument(
        "--no-hub", action="store_true", help="skip the checkpoint comparison"
    )
    check_parser.add_argument("--no-exports", action="store_true", help="skip the export checks")
    index_parser = commands.add_parser("index", help="write the registry index")
    index_parser.add_argument("root")
    index_parser.add_argument("-o", "--output")
    preview_parser = commands.add_parser("preview", help="draw a model's main entry")
    preview_parser.add_argument("directory")
    preview_parser.add_argument("-o", "--output")
    preview_parser.add_argument("--expand", type=int, default=1)
    preview_parser.add_argument("--theme", choices=["light", "dark"], default="light")
    pull_parser = commands.add_parser("pull", help="fetch a model directory from the registry")
    pull_parser.add_argument("name")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "check":
            failed = 0
            for directory in cast(Sequence[str], args.directories):
                card = Card.read(directory)
                problems = check(
                    card, std_root=args.std, hub=not args.no_hub, exports=not args.no_exports
                )
                status = "ok" if not problems else f"{len(problems)} problem(s)"
                print(f"{card.name}: {status}")
                for problem in problems:
                    print(f"  - {problem}")
                failed += bool(problems)
            return 1 if failed else 0
        if args.command == "index":
            text = json.dumps(index(args.root, args.std), indent=2) + "\n"
            _write(args.output, text)
            return 0
        if args.command == "preview":
            text = preview(
                Card.read(args.directory), std_root=args.std, expand=args.expand, theme=args.theme
            )
            _write(args.output, text)
            return 0
        print(fetch(args.name))
        return 0
    except LinnetError as error:
        print(str(error), file=sys.stderr)
        return 1


def _write(output: str | None, text: str) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    raise SystemExit(main())

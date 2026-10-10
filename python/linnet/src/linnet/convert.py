"""Hugging Face `transformers` checkpoints as Nest model directories.

A checkpoint of a family Nest has a card for (Llama and Mistral, Qwen2,
Qwen3, Phi-3, GPT-2) becomes a model directory built from that card: the
same Linnet source with the checkpoint's constants, generics from its
`config.json`, and bindings for its number of layers. The weights stay on
the Hub. Before it is returned, the directory is checked against the
checkpoint's SafeTensors headers (no tensor data is downloaded): every
parameter must find a tensor of its shape and dtype, and every tensor must
be bound. Optional parameters the checkpoint has (biases beside their
weights) bind too. A setting the family's source does not express (another
rope scaling, a different epsilon, a tensor nothing reads) is refused, never
approximated.

    python -m linnet.nest convert Qwen/Qwen2.5-7B-Instruct -o qwen2.5-7b
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from . import ir, nest
from .compiler import LinnetError
from .dtypes import from_safetensors
from .hf import DECODER_KEYS, LLAMA3_SCALING
from .home import home
from .weights import header_tensors, read_bindings, write_bindings

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo, RepoSibling

# A `config.json`, whose numbers `_int` and `_float` read.
Config = Mapping[str, ir.JsonValue]

# The cache length a converted card asks for when the checkpoint allows
# longer: caches are allocated whole, so 128K positions would cost gigabytes
# per layer. `generics={"MaxSeq": ...}` at load raises it.
MAX_SEQ = 8192

# Checkpoint tensors no Linnet source needs: buffers older `transformers`
# versions saved (rotary frequencies, GPT-2's causal masks).
IGNORED = re.compile(r"(\.rotary_emb\.inv_freq|\.attn\.bias|\.attn\.masked_bias)$")


@dataclass(frozen=True, slots=True)
class Family:
    """How one `model_type` maps onto a Nest card's source."""

    card: Callable[[Config], str]
    generics: Callable[[Config], dict[str, int]]
    # file -> constant -> value, for the `const` declarations to rewrite
    constants: Callable[[Config], dict[str, dict[str, float]]]
    problems: Callable[[Config], list[str]]
    # The tensor a tied output head binds, or None when the card has no
    # separate head.
    embedding: str | None = "model.embed_tokens.weight"


def _number(config: Config, key: str, default: float | None) -> int | float | str:
    """`config[key]`, or `default` when it is missing, as `int` and `float` read it."""
    if key not in config:
        if default is None:
            raise nest.NestError(f"the config has no `{key}`")
        return default
    value = config[key]
    if value is None or isinstance(value, list | dict):
        raise nest.NestError(f"`{key}` is {value!r}, not a number")
    return value


def _int(config: Config, key: str, default: int | None = None) -> int:
    return int(_number(config, key, default))


def _float(config: Config, key: str, default: float | None = None) -> float:
    return float(_number(config, key, default))


def _optional_int(config: Config, key: str) -> int | None:
    """`config[key]` as an integer; None when it is missing or null."""
    return None if config.get(key) is None else _int(config, key)


def _expect(
    config: Config, key: str, expected: str | float, default: str | float | None = None
) -> list[str]:
    value = config.get(key, default)
    if isinstance(expected, float) and isinstance(value, int | float):
        same = math.isclose(float(value), expected, rel_tol=1e-9)
    else:
        same = value == expected
    return [] if same else [f"`{key}` is {value!r}; the source needs {expected!r}"]


def _decoder(config: Config) -> dict[str, int]:
    heads = _int(config, "num_attention_heads")
    defaults = {"KvHeads": heads, "MaxSeq": MAX_SEQ}
    generics = {
        name: (_optional_int(config, key) or defaults[name])
        if name in defaults
        else _int(config, key)
        for name, key in DECODER_KEYS.items()
    }
    return generics | {"Batch": 1, "MaxSeq": min(generics["MaxSeq"], MAX_SEQ)}


def _head_dim(config: Config) -> list[str]:
    head_dim = _optional_int(config, "head_dim")
    width = _int(config, "hidden_size") // _int(config, "num_attention_heads")
    if head_dim is None or head_dim == width:
        return []
    return [f"`head_dim` is {head_dim}, not hidden_size / num_attention_heads ({width})"]


def _rope_scaling(config: Config) -> Config:
    """`rope_scaling`: empty when it is missing or null."""
    scaling = config.get("rope_scaling") or {}
    if not isinstance(scaling, dict):
        raise nest.NestError(f"`rope_scaling` is {scaling!r}, not an object")
    return scaling


def _rope_type(config: Config) -> str | None:
    scaling = _rope_scaling(config)
    if not scaling:
        return None
    return str(scaling.get("rope_type", scaling.get("type")))


def _rope_problems(config: Config, *computed: str) -> list[str]:
    """A rope scaling other than none and the `computed` ones."""
    if _rope_type(config) in (None, "default", *computed):
        return []
    return [f"rope scaling `{_rope_type(config)}` is not one the source computes"]


def _theta(config: Config) -> float:
    return _float(config, "rope_theta", 10000.0)


def _llama_card(config: Config) -> str:
    return "llama-3.1-8b-instruct" if _rope_type(config) == "llama3" else "tinyllama-1.1b-chat"


def _llama_constants(config: Config) -> dict[str, dict[str, float]]:
    rope: dict[str, float] = {"THETA": _theta(config)}
    if _rope_type(config) == "llama3":
        scaling = _rope_scaling(config)
        rope |= {name: _float(scaling, key) for key, name in LLAMA3_SCALING.items()}
    return {"src/rope.linnet": rope}


def _llama_problems(config: Config) -> list[str]:
    problems = (
        _expect(config, "hidden_act", "silu")
        + _expect(config, "rms_norm_eps", 1e-5)
        + _expect(config, "pretraining_tp", 1, 1)
        + _head_dim(config)
        + _rope_problems(config, "llama3")
    )
    window = _optional_int(config, "sliding_window")
    if window is not None and window < _int(config, "max_position_embeddings", 0):
        problems.append(f"`sliding_window` is {window}; the source attends to every position")
    return problems


def _qwen_problems(config: Config) -> list[str]:
    return (
        _expect(config, "hidden_act", "silu")
        + _expect(config, "rms_norm_eps", 1e-6)
        + _expect(config, "use_sliding_window", False, False)
        + _rope_problems(config)
    )


def _qwen2_problems(config: Config) -> list[str]:
    return _qwen_problems(config) + _head_dim(config)


def _qwen3_generics(config: Config) -> dict[str, int]:
    width = _int(config, "hidden_size") // _int(config, "num_attention_heads")
    return _decoder(config) | {"HeadDim": _optional_int(config, "head_dim") or width}


def _phi3_generics(config: Config) -> dict[str, int]:
    generics = _decoder(config)
    del generics["KvHeads"]
    return generics


def _phi3_constants(config: Config) -> dict[str, dict[str, float]]:
    window = _optional_int(config, "sliding_window") or _int(
        config, "max_position_embeddings", MAX_SEQ
    )
    return {
        "src/rope.linnet": {"THETA": _theta(config)},
        "src/attention.linnet": {"WINDOW": float(window)},
    }


def _phi3_problems(config: Config) -> list[str]:
    problems = (
        _expect(config, "hidden_act", "silu")
        + _expect(config, "rms_norm_eps", 1e-5)
        + _expect(config, "partial_rotary_factor", 1.0, 1.0)
        + _head_dim(config)
        + _rope_problems(config)
    )
    heads = _int(config, "num_attention_heads")
    if (_optional_int(config, "num_key_value_heads") or heads) != heads:
        problems.append("the source has as many key/value heads as query heads")
    return problems


def _gpt2_generics(config: Config) -> dict[str, int]:
    positions = _int(config, "n_positions", 1024)
    return {
        "Vocab": _int(config, "vocab_size"),
        "MaxPositions": positions,
        "H": _int(config, "n_embd"),
        "Heads": _int(config, "n_head"),
        "Layers": _int(config, "n_layer"),
        "Batch": 1,
        "MaxSeq": min(positions, MAX_SEQ),
    }


def _gpt2_problems(config: Config) -> list[str]:
    problems = (
        _expect(config, "activation_function", "gelu_new", "gelu_new")
        + _expect(config, "layer_norm_epsilon", 1e-5, 1e-5)
        + _expect(config, "scale_attn_weights", True, True)
        + _expect(config, "scale_attn_by_inverse_layer_idx", False, False)
        + _expect(config, "reorder_and_upcast_attn", False, False)
    )
    inner = _optional_int(config, "n_inner")
    if inner is not None and inner != 4 * _int(config, "n_embd"):
        problems.append(f"`n_inner` is {inner}; the source's MLP is 4 * n_embd wide")
    return problems


def _no_constants(config: Config) -> dict[str, dict[str, float]]:
    return {}


FAMILIES: dict[str, Family] = {
    "llama": Family(_llama_card, _decoder, _llama_constants, _llama_problems),
    "mistral": Family(_llama_card, _decoder, _llama_constants, _llama_problems),
    "qwen2": Family(
        lambda _: "qwen2.5-0.5b-instruct",
        _decoder,
        lambda config: {"src/rope.linnet": {"THETA": _theta(config)}},
        _qwen2_problems,
    ),
    "qwen3": Family(
        lambda _: "qwen3-8b",
        _qwen3_generics,
        lambda config: {"src/rope.linnet": {"THETA": _theta(config)}},
        _qwen_problems,
    ),
    "phi3": Family(
        lambda _: "phi-3-mini-4k-instruct", _phi3_generics, _phi3_constants, _phi3_problems
    ),
    "gpt2": Family(lambda _: "gpt2", _gpt2_generics, _no_constants, _gpt2_problems, None),
}


def convert(
    repo: str,
    *,
    revision: str | None = None,
    output: str | Path | None = None,
    std_root: str | Path | None = None,
) -> Path:
    """Writes a Nest model directory for a `transformers` checkpoint on the
    Hub and returns it. Without `output` it goes to `$LINNET_HOME/converted`,
    where a later call for the same commit finds it."""
    hub = nest.huggingface_hub()
    try:
        info = hub.HfApi().model_info(repo, revision=revision)
    except Exception as error:  # a missing or gated repo, or no network
        raise nest.NestError(f"cannot read the Hub repo `{repo}`: {error}") from error
    sha = str(info.sha)
    # A card is named after its directory, so `check` passes on it as it is.
    target = (
        Path(output) if output is not None else home() / "converted" / sha / repo.replace("/", "--")
    )
    name = target.name
    if output is None and (target / "nest.toml").exists():
        return target
    if output is not None and target.exists() and any(target.iterdir()):
        raise nest.NestError(f"{target} already exists")
    siblings: list[RepoSibling] = info.siblings or []
    files = {str(sibling.rfilename) for sibling in siblings}
    if "config.json" not in files:
        raise nest.NestError(f"`{repo}` has neither a nest.toml nor a transformers config.json")
    config: ir.JsonValue = json.loads(
        Path(hub.hf_hub_download(repo, "config.json", revision=sha)).read_text("utf-8")
    )
    if not isinstance(config, dict):
        raise nest.NestError(f"`{repo}`'s config.json is not a JSON object")
    model_type = str(config.get("model_type"))
    family = FAMILIES.get(model_type)
    if family is None:
        known = ", ".join(sorted(FAMILIES))
        raise nest.NestError(f"`{repo}` is a `{model_type}` model; Linnet converts {known}")
    problems = family.problems(config)
    if problems:
        raise nest.NestError(f"cannot convert `{repo}`:\n  - " + "\n  - ".join(problems))

    weights = _checkpoint_files(hub, repo, sha, files)
    tensors = _tensors(repo, sha, weights)
    base = nest.resolve(family.card(config))

    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch) / name
        _copy_source(base, directory)
        for relative, constants in family.constants(config).items():
            _rewrite_constants(directory / relative, constants)
        bindings = _bindings(base, _int(config, "num_hidden_layers", _int(config, "n_layer", 0)))
        if family.embedding is not None:
            # A tied head reads the embedding, unless the checkpoint stores
            # the head anyway (Qwen3's small models do).
            tied = bool(config.get("tie_word_embeddings", False))
            own = _present("lm_head.weight", tensors) in tensors
            bindings["lm_head.weight"] = family.embedding if tied and not own else "lm_head.weight"
        bindings = {path: _present(tensor, tensors) for path, tensor in bindings.items()}
        write_bindings(directory / "bindings.json", bindings)

        generics: dict[str, int | str] = dict(family.generics(config))
        generics["T"] = _dtype(tensors, next(iter(bindings.values())))
        (directory / "nest.toml").write_text(
            _card(name, repo, sha, model_type, base, generics, weights, info), "utf-8"
        )
        (directory / "README.md").write_text(_readme(repo, sha, base), "utf-8")

        card = nest.Card.read(directory)
        program = card.program(std_root)
        values = ir.bind_generics(program.root.generics, card.generics)
        bindings = _complete_bindings(program, values, bindings, tensors)
        write_bindings(directory / "bindings.json", bindings)
        unbound = _unbound(program, values, tensors, bindings)
        if unbound:
            shown = ", ".join(f"`{t}`" for t in unbound[:5])
            more = f" and {len(unbound) - 5} more" if len(unbound) > 5 else ""
            raise nest.NestError(
                f"cannot convert `{repo}`: the {model_type} source binds no parameter to "
                f"{shown}{more}"
            )
        problems = nest.check(card, std_root=std_root, exports=False)
        if problems:
            raise nest.NestError(f"cannot convert `{repo}`:\n  - " + "\n  - ".join(problems))
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(directory, target)
    return target


class _Hub(Protocol):
    """`huggingface_hub`, as `_checkpoint_files` uses it."""

    def hf_hub_download(
        self, repo_id: str, filename: str, *, revision: str | None = None
    ) -> str: ...


def _checkpoint_files(hub: _Hub, repo: str, sha: str, files: set[str]) -> list[str]:
    """The SafeTensors files `transformers` loads: the index's shards, or
    the single `model.safetensors`."""
    if "model.safetensors.index.json" in files:
        index = json.loads(
            Path(hub.hf_hub_download(repo, "model.safetensors.index.json", revision=sha)).read_text(
                "utf-8"
            )
        )
        return sorted(set(cast(dict[str, str], index["weight_map"]).values()))
    if "model.safetensors" in files:
        return ["model.safetensors"]
    raise nest.NestError(f"`{repo}` has no SafeTensors checkpoint (model.safetensors)")


def _tensors(repo: str, sha: str, files: list[str]) -> dict[str, tuple[tuple[int, ...], str]]:
    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    for filename in files:
        try:
            header = nest.hub_safetensors_header(repo, filename, revision=sha)
        except Exception as error:  # any Hub failure
            raise nest.NestError(
                f"cannot read the headers of {repo}/{filename}: {error}"
            ) from error
        tensors.update(header_tensors(header))
    return tensors


def _copy_source(base: nest.Card, directory: Path) -> None:
    source = Path(base.source)
    top = source.parts[0] if len(source.parts) > 1 else source.name
    destination = directory / top
    if (base.directory / top).is_dir():
        shutil.copytree(base.directory / top, destination)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(base.directory / top, destination)
    if (base.directory / "linnet.toml").exists():
        shutil.copy(base.directory / "linnet.toml", directory / "linnet.toml")


def _literal(value: float, dtype: str) -> str:
    if dtype.startswith(("i", "u")):
        return str(int(value))
    if value.is_integer() and abs(value) < 1e16:
        return f"{int(value)}.0"
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", repr(value)).replace("e+", "e")


def _rewrite_constants(path: Path, constants: Mapping[str, float]) -> None:
    text = path.read_text("utf-8")
    for name, value in constants.items():
        pattern = re.compile(rf"^((?:pub )?const {name}: (\w+) = ).*$", re.MULTILINE)
        found = list(pattern.finditer(text))
        if len(found) != 1:
            raise LinnetError(f"{path.name} does not declare `const {name}` once")
        match = found[0]
        text = text[: match.start()] + match[1] + _literal(value, match[2]) + text[match.end() :]
    path.write_text(text, "utf-8")


def _bindings(base: nest.Card, layers: int) -> dict[str, str]:
    """The base card's bindings with its layer 0 repeated for `layers`."""
    assert base.bindings_path is not None
    mapping = read_bindings(base.bindings_path)
    out: dict[str, str] = {}
    for path, tensor in mapping.items():
        layer = re.match(r"^\w+\.(\d+)\.", path)
        if layer is None:
            out[path] = tensor
        elif layer[1] == "0":
            if not re.search(r"(^|\.)0\.", tensor):
                raise LinnetError(f"cannot tell the layer in the tensor name `{tensor}`")
            for i in range(layers):
                at = path.replace(".0.", f".{i}.", 1)
                out[at] = re.sub(r"(^|\.)0\.", rf"\g<1>{i}.", tensor, count=1)
    return out


def _present(tensor: str, tensors: Mapping[str, object]) -> str:
    """The checkpoint's own spelling of a tensor name: with or without the
    `model.` or `transformer.` prefix a family's checkpoints differ by."""
    if tensor in tensors:
        return tensor
    for prefix in ("model.", "transformer."):
        if prefix + tensor in tensors:
            return prefix + tensor
        if tensor.startswith(prefix) and tensor.removeprefix(prefix) in tensors:
            return tensor.removeprefix(prefix)
    return tensor


def _dtype(tensors: Mapping[str, tuple[tuple[int, ...], str]], tensor: str) -> str:
    if tensor not in tensors:
        raise nest.NestError(f"the checkpoint has no `{tensor}`")
    dtype = from_safetensors(tensors[tensor][1]) or ""
    if dtype not in ("bf16", "f16", "f32"):
        raise nest.NestError(f"`{tensor}` is {tensors[tensor][1]}, not a float checkpoint")
    return dtype


def _complete_bindings(
    program: ir.Program,
    values: ir.Bindings,
    bindings: Mapping[str, str],
    tensors: Mapping[str, object],
) -> dict[str, str]:
    """Binds what the base card leaves to its path: a parameter whose tensor
    this checkpoint spells with a prefix (`transformer.ln_f.weight`), and an
    optional one the checkpoint has beside a bound sibling (`q_proj.bias`
    next to the tensor `q_proj.weight`, as Llama variants with
    `attention_bias` have)."""
    out = dict(bindings)
    for entry in program.manifest:
        if entry.kind not in ("param", "buffer"):
            continue
        for path in ir.expand_paths(entry, values):
            if path in out:
                continue
            if path not in tensors and _present(path, tensors) in tensors:
                out[path] = _present(path, tensors)
                continue
            if entry.kind != "param" or not entry.optional or "." not in path:
                continue
            stem, leaf = path.rsplit(".", 1)
            sibling = out.get(f"{stem}.weight")
            if sibling is None or not sibling.endswith(".weight"):
                continue
            candidate = f"{sibling.removesuffix('.weight')}.{leaf}"
            if candidate in tensors:
                out[path] = candidate
    return out


def _unbound(
    program: ir.Program,
    values: ir.Bindings,
    tensors: Mapping[str, object],
    bindings: Mapping[str, str],
) -> list[str]:
    """The checkpoint's tensors no parameter reads."""
    used: set[str] = set()
    for entry in program.manifest:
        if entry.kind in ("param", "buffer"):
            used.update(bindings.get(p, p) for p in ir.expand_paths(entry, values))
    return sorted(t for t in tensors if t not in used and not IGNORED.search(t))


def _card(
    name: str,
    repo: str,
    sha: str,
    model_type: str,
    base: nest.Card,
    generics: Mapping[str, int | str],
    files: list[str],
    info: ModelInfo,
) -> str:
    card_data = getattr(info, "card_data", None)
    license_name = getattr(card_data, "license", None) or "unknown"
    summary = f"{repo}, converted from its transformers checkpoint with the source of {base.name}."
    q = json.dumps
    lines = [
        "[model]",
        f"name = {q(name)}",
        f"title = {q(repo)}",
        f"summary = {q(summary)}",
        f"license = {q(str(license_name))}",
        f"family = {q(model_type)}",
        'tags = ["converted"]',
        "",
        "[links]",
        f"huggingface = {q(f'https://huggingface.co/{repo}')}",
        "",
        "[source]",
        f"path = {q(base.source)}",
    ]
    if base.root is not None:
        lines.append(f"root = {q(base.root)}")
    if base.entry is not None:
        lines.append(f"entry = {q(base.entry)}")
    lines += ["", "[generics]"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in generics.items()]
    lines += ["", "[check]"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in base.check.items()]
    lines += [
        "",
        "[weights]",
        f"repo = {q(repo)}",
        f"revision = {q(sha)}",
        "files = [" + ", ".join(json.dumps(f) for f in files) + "]",
        'bindings = "bindings.json"',
        "",
    ]
    return "\n".join(lines)


def _readme(repo: str, sha: str, base: nest.Card) -> str:
    return (
        f"# {repo}\n\n"
        f"[{repo}](https://huggingface.co/{repo}) at commit `{sha}`, converted by "
        "`linnet.convert` from its `config.json`. The Linnet source is the Nest card "
        f"[{base.name}](https://nest.franknoh.dev/models/{base.name})'s, with this "
        "checkpoint's constants; the generics come from the config, and `bindings.json` "
        "names the checkpoint's tensors. Every parameter was checked against the "
        "checkpoint's SafeTensors headers, and every tensor is bound. The weights stay on "
        "the Hub; `linnet.nest.load` downloads them.\n"
    )

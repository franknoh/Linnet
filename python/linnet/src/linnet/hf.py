"""Hugging Face format checkpoints from Linnet models, for vLLM and friends.

A Linnet model whose structure is one of the architectures the serving
stacks already implement (Llama, Qwen2, Qwen3, Phi-3, GPT-2) can be written
as a Transformers checkpoint directory: `config.json` with the
hyperparameters read from the program's generics, constants, and
normalization epsilon, `model.safetensors` under the Transformers tensor
names, and the tokenizer files of its Hub repository. vLLM, SGLang,
TGI, and `transformers` load that directory as they load any other model.

    python -m linnet.hf export tinyllama-1.1b-chat -o serve/tinyllama
    vllm serve serve/tinyllama

Structures that are not one of the recognized families are refused with
the paths that did not fit; nothing is guessed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from . import ir, nest
from .compiler import LinnetError
from .dtypes import DTYPES
from .weights import (
    TensorLocation,
    check_problems,
    match_checkpoint,
    read_bindings,
    safetensors_index,
    write_safetensors,
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "generation_config.json",
)


@dataclass(frozen=True, slots=True)
class Traits:
    """What decides a checkpoint beyond its family: the optional tensors it
    has (paths with `[*]`), the normalization epsilon the program uses, and
    whether its output head is its embedding table."""

    present: frozenset[str]
    eps: float
    tied: bool


@dataclass(frozen=True, slots=True)
class Family:
    """One architecture the serving stacks implement, and how Linnet spells it."""

    name: str
    architecture: str
    model_type: str
    # Linnet parameter path pattern (with `[*]`) -> Transformers name template
    # (with `{i}` for the layer index). Every required path must be present.
    names: tuple[tuple[str, str], ...]
    optional: frozenset[str]
    generics: frozenset[str]
    config: Callable[
        [Mapping[str, int | str], Mapping[str, float], Traits], dict[str, ir.JsonValue]
    ]
    # Optional paths one Transformers flag switches on together: a
    # checkpoint has all of them or none.
    together: tuple[frozenset[str], ...] = ()
    # The normalization op whose epsilon `config.json` declares, and the
    # position of that operand.
    norm: tuple[str, int] = ("std.nn.norm::rms_norm", 2)
    # The Transformers names of the embedding table and the output head,
    # which a checkpoint may bind to one tensor.
    tied: tuple[str, str] | None = ("model.embed_tokens.weight", "lm_head.weight")

    def hf_name(self, path: str) -> str | None:
        """The Transformers name of an expanded Linnet path, or None if it has none."""
        for pattern, template in self.names:
            regex = "^" + re.escape(pattern).replace(r"\[\*\]", r"\.(\d+)") + "$"
            match = re.match(regex, path)
            if match:
                return template.format(i=match.group(1)) if match.groups() else template
        return None

    def mismatch(self, present: frozenset[str], generics: frozenset[str]) -> list[str]:
        """Why a program with these generics, whose checkpoint has the
        tensors `present`, is not this family; empty when it is."""
        patterns = {pattern for pattern, _ in self.names}
        detail: list[str] = []
        missing = sorted(patterns - self.optional - present)
        if missing:
            detail.append("missing " + ", ".join(missing[:4]))
        extra = sorted(present - patterns)
        if extra:
            detail.append("unexpected " + ", ".join(extra[:4]))
        for group in self.together:
            if group & present and not group <= present:
                detail.append("only some of " + ", ".join(sorted(group)))
        # Extra generics (a `Batch` for the KV cache) are fine; missing ones are not.
        wrong_generics = sorted(self.generics - generics)
        if wrong_generics:
            detail.append("missing generics " + ", ".join(wrong_generics[:6]))
        return detail


# The Llama-shaped families' generics and the config keys that hold them
# (`linnet.convert` reads them back).
DECODER_KEYS = {
    "Vocab": "vocab_size",
    "H": "hidden_size",
    "Heads": "num_attention_heads",
    "KvHeads": "num_key_value_heads",
    "Inner": "intermediate_size",
    "Layers": "num_hidden_layers",
    "MaxSeq": "max_position_embeddings",
}


def _decoder_config(
    generics: Mapping[str, int | str], constants: Mapping[str, float], traits: Traits
) -> dict[str, ir.JsonValue]:
    """What the Llama-shaped families' configs share."""
    dtype = str(generics.get("T", "bf16"))
    # Without `KvHeads` (Phi-3), every query head has its own.
    sizes = {**generics, "KvHeads": generics.get("KvHeads", generics["Heads"])}
    return {
        **{key: sizes[name] for name, key in DECODER_KEYS.items()},
        "rope_theta": constants.get("THETA", 10000.0),
        "rms_norm_eps": traits.eps,
        "hidden_act": "silu",
        "tie_word_embeddings": traits.tied,
        "torch_dtype": DTYPES[dtype].torch if dtype in DTYPES else dtype,
        "transformers_version": "4.0.0",
    }


def _llama_config(
    generics: Mapping[str, int | str], constants: Mapping[str, float], traits: Traits
) -> dict[str, ir.JsonValue]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        **_decoder_config(generics, constants, traits),
        "head_dim": int(generics["H"]) // int(generics["Heads"]),
        "attention_bias": traits.present >= _LLAMA_ATTENTION_BIASES,
        "mlp_bias": traits.present >= _LLAMA_MLP_BIASES,
        **_rope_scaling(constants),
    }


# Llama 3.1's `llama3` rope scaling, when the program defines all four of
# its constants (the Nest card's `crate.rope` does); without it a server
# would rotate with the base frequency alone and drift past 8192 positions.
LLAMA3_SCALING = {
    "factor": "FACTOR",
    "low_freq_factor": "LOW_FREQ_FACTOR",
    "high_freq_factor": "HIGH_FREQ_FACTOR",
    "original_max_position_embeddings": "ORIGINAL_MAX_POSITION_EMBEDDINGS",
}


def _rope_scaling(constants: Mapping[str, float]) -> dict[str, ir.JsonValue]:
    if not all(name in constants for name in LLAMA3_SCALING.values()):
        return {}
    scaling: dict[str, ir.JsonValue] = {"rope_type": "llama3"}
    for key, name in LLAMA3_SCALING.items():
        value = constants[name]
        scaling[key] = int(value) if key == "original_max_position_embeddings" else value
    return {"rope_scaling": scaling}


def _qwen2_config(
    generics: Mapping[str, int | str], constants: Mapping[str, float], traits: Traits
) -> dict[str, ir.JsonValue]:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        **_decoder_config(generics, constants, traits),
        # Every layer attends over the whole sequence, as the program does.
        "use_sliding_window": False,
        "sliding_window": None,
        "max_window_layers": generics["Layers"],
    }


def _qwen3_config(
    generics: Mapping[str, int | str], constants: Mapping[str, float], traits: Traits
) -> dict[str, ir.JsonValue]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        **_decoder_config(generics, constants, traits),
        "head_dim": generics["HeadDim"],
        "attention_bias": False,
        "use_sliding_window": False,
        "sliding_window": None,
        "max_window_layers": generics["Layers"],
    }


def _phi3_config(
    generics: Mapping[str, int | str], constants: Mapping[str, float], traits: Traits
) -> dict[str, ir.JsonValue]:
    return {
        "architectures": ["Phi3ForCausalLM"],
        "model_type": "phi3",
        **_decoder_config(generics, constants, traits),
        "original_max_position_embeddings": generics["MaxSeq"],
        "rope_scaling": None,
        "partial_rotary_factor": 1.0,
        # No window: every position attends over all before it, as the
        # program does (Phi-3's own config slides one over 2047).
        "sliding_window": None,
        "attention_bias": False,
        # The program has no padding token; Transformers' default is an id
        # outside a small vocabulary.
        "pad_token_id": None,
    }


def _gpt2_config(
    generics: Mapping[str, int | str], constants: Mapping[str, float], traits: Traits
) -> dict[str, ir.JsonValue]:
    dtype = str(generics.get("T", "f32"))
    return {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": generics["Vocab"],
        "n_positions": generics["MaxPositions"],
        "n_ctx": generics["MaxPositions"],
        "n_embd": generics["H"],
        "n_head": generics["Heads"],
        "n_layer": generics["Layers"],
        "activation_function": "gelu_new",
        "layer_norm_epsilon": traits.eps,
        "tie_word_embeddings": True,
        "torch_dtype": DTYPES[dtype].torch if dtype in DTYPES else dtype,
        "transformers_version": "4.0.0",
    }


def _layer_names(
    block: str, theirs: str, members: Iterable[tuple[str, str]], suffix: str
) -> tuple[tuple[str, str], ...]:
    """`layers[*].<block>.<ours>.<suffix>` -> `model.layers.{i}.<theirs>.<name>.<suffix>`."""
    return tuple(
        (f"layers[*].{block}.{ours}.{suffix}", f"model.layers.{{i}}.{theirs}.{name}.{suffix}")
        for ours, name in members
    )


# The embedding, the final norm, the head, and the two norms of each layer,
# as every Llama-shaped family names them.
_DECODER = (
    ("embedding.weight", "model.embed_tokens.weight"),
    ("norm.weight", "model.norm.weight"),
    ("lm_head.weight", "lm_head.weight"),
    ("layers[*].attention_norm.weight", "model.layers.{i}.input_layernorm.weight"),
    ("layers[*].mlp_norm.weight", "model.layers.{i}.post_attention_layernorm.weight"),
)
_QKV = (("q_proj", "q_proj"), ("k_proj", "k_proj"), ("v_proj", "v_proj"))
_ATTENTION = (*_QKV, ("o_proj", "o_proj"))
_MLP = (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj"))
_LLAMA_ATTENTION_BIASES = frozenset(p for p, _ in _layer_names("attention", "", _ATTENTION, "bias"))
_LLAMA_MLP_BIASES = frozenset(p for p, _ in _layer_names("mlp", "", _MLP, "bias"))
_LLAMA_GENERICS = frozenset({"Vocab", "H", "Heads", "KvHeads", "Inner", "Layers", "MaxSeq", "T"})

FAMILIES: tuple[Family, ...] = (
    Family(
        name="llama",
        architecture="LlamaForCausalLM",
        model_type="llama",
        names=(
            *_DECODER,
            *_layer_names("attention", "self_attn", _ATTENTION, "weight"),
            *_layer_names("mlp", "mlp", _MLP, "weight"),
            *_layer_names("attention", "self_attn", _ATTENTION, "bias"),
            *_layer_names("mlp", "mlp", _MLP, "bias"),
        ),
        optional=_LLAMA_ATTENTION_BIASES | _LLAMA_MLP_BIASES,
        generics=_LLAMA_GENERICS,
        config=_llama_config,
        # `attention_bias` gives all four projections a bias, `mlp_bias` all three.
        together=(_LLAMA_ATTENTION_BIASES, _LLAMA_MLP_BIASES),
    ),
    Family(
        # Llama's layout with biases on the query, key, and value projections only.
        name="qwen2",
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        names=(
            *_DECODER,
            *_layer_names("attention", "self_attn", _ATTENTION, "weight"),
            *_layer_names("mlp", "mlp", _MLP, "weight"),
            *_layer_names("attention", "self_attn", _QKV, "bias"),
        ),
        optional=frozenset(),
        generics=_LLAMA_GENERICS,
        config=_qwen2_config,
    ),
    Family(
        # An RMSNorm over each query and key head, and a head width of its own.
        name="qwen3",
        architecture="Qwen3ForCausalLM",
        model_type="qwen3",
        names=(
            *_DECODER,
            *_layer_names("attention", "self_attn", _ATTENTION, "weight"),
            *_layer_names(
                "attention", "self_attn", (("q_norm", "q_norm"), ("k_norm", "k_norm")), "weight"
            ),
            *_layer_names("mlp", "mlp", _MLP, "weight"),
        ),
        optional=frozenset(),
        generics=_LLAMA_GENERICS | {"HeadDim"},
        config=_qwen3_config,
    ),
    Family(
        # One projection for the query, key, and value, one for the gate and
        # the up projection; as many key/value heads as query heads.
        name="phi3",
        architecture="Phi3ForCausalLM",
        model_type="phi3",
        names=(
            *_DECODER,
            *_layer_names(
                "attention", "self_attn", (("qkv", "qkv_proj"), ("o_proj", "o_proj")), "weight"
            ),
            *_layer_names(
                "mlp", "mlp", (("gate_up", "gate_up_proj"), ("down", "down_proj")), "weight"
            ),
        ),
        optional=frozenset(),
        generics=frozenset({"Vocab", "H", "Heads", "Inner", "Layers", "MaxSeq", "T"}),
        config=_phi3_config,
    ),
    Family(
        name="gpt2",
        architecture="GPT2LMHeadModel",
        model_type="gpt2",
        names=(
            ("wte", "wte.weight"),
            ("wpe", "wpe.weight"),
            ("ln_f.weight", "ln_f.weight"),
            ("ln_f.bias", "ln_f.bias"),
            ("blocks[*].ln_1.weight", "h.{i}.ln_1.weight"),
            ("blocks[*].ln_1.bias", "h.{i}.ln_1.bias"),
            ("blocks[*].ln_2.weight", "h.{i}.ln_2.weight"),
            ("blocks[*].ln_2.bias", "h.{i}.ln_2.bias"),
            ("blocks[*].attn.qkv.weight", "h.{i}.attn.c_attn.weight"),
            ("blocks[*].attn.qkv.bias", "h.{i}.attn.c_attn.bias"),
            ("blocks[*].attn.out.weight", "h.{i}.attn.c_proj.weight"),
            ("blocks[*].attn.out.bias", "h.{i}.attn.c_proj.bias"),
            ("blocks[*].mlp.up.weight", "h.{i}.mlp.c_fc.weight"),
            ("blocks[*].mlp.up.bias", "h.{i}.mlp.c_fc.bias"),
            ("blocks[*].mlp.down.weight", "h.{i}.mlp.c_proj.weight"),
            ("blocks[*].mlp.down.bias", "h.{i}.mlp.c_proj.bias"),
        ),
        optional=frozenset(),
        generics=frozenset({"Vocab", "MaxPositions", "H", "Heads", "Layers", "T"}),
        config=_gpt2_config,
        norm=("std.nn.norm::layer_norm", 3),
        # The output head is the embedding table by construction.
        tied=None,
    ),
)


def recognize(program: ir.Program, present: Iterable[str] | None = None) -> Family:
    """The family whose parameter paths and generics the program has, exactly.
    `present` are the parameter paths (with `[*]`) its checkpoint has; by
    default every path the program declares."""
    if present is None:
        present = (e.path for e in program.manifest if e.kind == "param")
    have = frozenset(present)
    generics = frozenset(g.name for g in program.root.generics)
    reasons: list[str] = []
    for family in FAMILIES:
        detail = family.mismatch(have, generics)
        if not detail:
            return family
        reasons.append(f"{family.name}: " + "; ".join(detail))
    raise LinnetError(
        "the model is not one of the architectures the serving stacks implement "
        f"({', '.join(f.name for f in FAMILIES)}):\n  " + "\n  ".join(reasons)
    )


def constants_of(program: ir.Program) -> dict[str, float]:
    """Numeric module constants by short name (`THETA`, `EPS`), when they are literals."""
    out: dict[str, float] = {}
    for constant in program.constants:
        ops = constant.body.ops
        if len(ops) == 2 and ops[0].kind in ("const.float", "const.int") and ops[1].kind == "yield":
            value = ops[0].attrs.get("value")
            if isinstance(value, int | float) and not isinstance(value, bool):
                out[constant.name.rsplit("::", 1)[-1]] = float(value)
    return out


def norm_epsilon(program: ir.Program, op: str, operand: int) -> float:
    """The epsilon every call of the normalization `op` passes as its
    `operand`-th argument, among the functions the root's entries reach.
    Each must be a literal, directly or through the arguments of the
    functions that pass it on, and all must agree."""
    reached: list[ir.Function] = []
    seen: set[str] = set()
    pending = [f.name for f in program.entries()]
    callers: dict[str, list[tuple[ir.Function, ir.Op]]] = {}
    while pending:
        name = pending.pop()
        if name in seen or name not in program.functions:
            continue
        seen.add(name)
        function = program.functions[name]
        reached.append(function)
        for inner in function.body.walk():
            callee = inner.attrs.get("callee")
            if isinstance(callee, str) and inner.kind in ("call", "semantic.call"):
                callers.setdefault(callee, []).append((function, inner))
                pending.append(callee)

    def literals(function: ir.Function, value: int, depth: int) -> set[float] | None:
        for inner in function.body.walk():
            if any(result.id == value for result in inner.results):
                literal = inner.attrs.get("value")
                if inner.kind in ("const.float", "const.int") and isinstance(literal, int | float):
                    return {float(literal)}
                return None
        # An argument: whatever each caller passes for it.
        params = list(function.body.args)
        position = next((i for i, arg in enumerate(params) if arg.id == value), None)
        if position is None or depth > 8 or function.name not in callers:
            return None
        found: set[float] = set()
        for caller, call in callers[function.name]:
            if position >= len(call.operands):
                return None
            passed = literals(caller, call.operands[position], depth + 1)
            if passed is None:
                return None
            found |= passed
        return found

    values: set[float] = set()
    for function in reached:
        for inner in function.body.walk():
            if inner.kind != "semantic.call" or inner.attrs.get("callee") != op:
                continue
            passed = (
                literals(function, inner.operands[operand], 0)
                if operand < len(inner.operands)
                else None
            )
            if passed is None:
                raise LinnetError(
                    f"the epsilon `{function.name}` passes to `{op}` is not a literal; "
                    "config.json needs its value"
                )
            values |= passed
    if len(values) != 1:
        found = ", ".join(str(v) for v in sorted(values)) or "none"
        raise LinnetError(f"the model's `{op}` calls need one epsilon; found {found}")
    return values.pop()


@dataclass(frozen=True, slots=True)
class Exported:
    directory: Path
    family: Family
    config: Mapping[str, ir.JsonValue]
    tensors: int
    tokenizer_files: tuple[str, ...]


def export(
    model: str | Path,
    output: str | Path,
    *,
    generics: Mapping[str, int | str] | None = None,
    weights: str | Path | None = None,
    bindings: str | Path | None = None,
    root: str | None = None,
    std_root: str | Path | None = None,
    tokenizer: str | None = None,
) -> Exported:
    """Writes a Transformers checkpoint directory for a Linnet model.

    `model` is a Nest model directory or registry name (card supplies source,
    generics, checkpoint, bindings, and the tokenizer's repository), or a
    `.linnet` file with `generics`, `weights`, and `bindings` given here.
    `tokenizer` names a Hub repository whose tokenizer files are copied in.
    """
    resolved = nest.resolve_model(model, root=root, weights=weights, bindings=bindings)
    card, source, root = resolved.card, resolved.source, resolved.root
    weights, bindings = resolved.weights, resolved.bindings
    if tokenizer is None and card is not None and card.weights is not None:
        tokenizer = card.weights.repo
    values = {**resolved.generics, **(generics or {})}
    if weights is None:
        raise LinnetError("an export needs weights: a SafeTensors file or directory")

    program = ir.load_program(source, root=root, std_root=std_root)
    root_values = {g.name: values[g.name] for g in program.root.generics if g.name in values}
    bound = ir.bind_generics(program.root.generics, root_values)
    resolved = {
        g.name: (
            bound.dtypes[g.id]
            if g.kind == "dtype"
            else bound.dims[g.id]
            if g.kind == "dim"
            else values[g.name]
        )
        for g in program.root.generics
    }
    mapping = read_bindings(bindings) if bindings is not None else {}
    index = safetensors_index(weights)

    # Every parameter's tensors under the checkpoint's names; an optional
    # one (a bias) counts as present when every layer has it. The tensors
    # are copied as they are, and `config.json` declares the program's
    # dtype: they must agree.
    shapes = {name: (location.shape, location.dtype) for name, location in index.items()}
    matched, problems = match_checkpoint(program, bound, shapes, mapping)
    found: list[tuple[str, str, TensorLocation]] = []
    present: set[str] = set()
    for entry, located in matched:
        if entry.optional and located and len(located) != len(ir.expand_paths(entry, bound)):
            problems.append(f"`{entry.path}` is in the checkpoint for some layers only")
        if located:
            present.add(entry.path)
            found += [(path, source, index[source]) for path, source in located]
    check_problems(problems)

    family = recognize(program, present)
    tensors: list[tuple[str, str, tuple[int, ...], TensorLocation | bytes]] = []
    sources: dict[str, str] = {}
    for linnet_path, source_name, location in found:
        hf_name = family.hf_name(linnet_path)
        assert hf_name is not None  # every present path is one of the family's
        sources[hf_name] = source_name
        tensors.append((hf_name, location.dtype, location.shape, location))
    # A head bound to the embedding's own tensor is written once, as
    # Transformers ties them.
    tied = False
    if family.tied is not None:
        embedding, head = family.tied
        tied = embedding in sources and sources.get(head) == sources[embedding]
        if tied:
            tensors = [t for t in tensors if t[0] != head]
    traits = Traits(frozenset(present), norm_epsilon(program, *family.norm), tied)

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    copied = _copy_tokenizer(tokenizer, directory) if tokenizer else ()
    config = family.config(resolved, constants_of(program), traits)
    for key, value in _special_tokens(directory).items():
        config.setdefault(key, value)
    directory.joinpath("config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_safetensors(directory / "model.safetensors", tensors, metadata={"format": "pt"})
    directory.joinpath("README.md").write_text(
        _readme(program, family, directory, tokenizer), encoding="utf-8"
    )
    return Exported(directory, family, config, len(tensors), copied)


def _copy_tokenizer(repo: str, directory: Path) -> tuple[str, ...]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore[import-untyped]
    except ImportError:
        raise LinnetError(
            "huggingface-hub is not installed (pip install 'linnet-lang[nest]')"
        ) from None
    available = set(list_repo_files(repo))
    copied: list[str] = []
    for name in TOKENIZER_FILES:
        if name in available:
            downloaded = Path(hf_hub_download(repo, name))
            directory.joinpath(name).write_bytes(downloaded.read_bytes())
            copied.append(name)
    return tuple(copied)


def _special_tokens(directory: Path) -> dict[str, ir.JsonValue]:
    """The copied tokenizer's `bos_token_id` and `eos_token_id`, from its
    `generation_config.json`: servers and converters read them from
    `config.json`."""
    path = directory / "generation_config.json"
    if not path.exists():
        return {}
    generation: dict[str, ir.JsonValue] = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: generation[key]
        for key in ("bos_token_id", "eos_token_id")
        if generation.get(key) is not None
    }


def _readme(program: ir.Program, family: Family, directory: Path, tokenizer: str | None) -> str:
    lines = [
        f"# {program.module}::{program.root.name} as {family.architecture}",
        "",
        "Written by `linnet.hf` from the Linnet source; the hyperparameters in",
        "`config.json` come from the program's generics and constants, the tensor",
        f"names follow Transformers' `{family.model_type}`.",
        "",
        "```bash",
        f"vllm serve {directory}",
        "```",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM",
        f'model = AutoModelForCausalLM.from_pretrained("{directory}")',
        "```",
    ]
    if tokenizer:
        lines += ["", f"Tokenizer files come from `{tokenizer}`."]
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m linnet.hf", description="Transformers-format checkpoints."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser(
        "export", help="write a checkpoint directory vLLM and transformers load"
    )
    nest.add_model_arguments(exp)
    exp.add_argument("-o", "--output", required=True)
    exp.add_argument("--tokenizer", help="a Hub repository to take tokenizer files from")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        exported = export(
            args.model, args.output, tokenizer=args.tokenizer, **nest.model_options(args)
        )
    except LinnetError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"wrote {exported.directory}: {exported.family.architecture}, {exported.tensors} tensors")
    if exported.tokenizer_files:
        print("  tokenizer: " + ", ".join(exported.tokenizer_files))
    print(f"  vllm serve {exported.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

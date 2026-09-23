"""Hugging Face format checkpoints from Linnet models, for vLLM and friends.

A Linnet model whose structure is one of the architectures the serving
stacks already implement (Llama, GPT-2) can be written as a Transformers
checkpoint directory: `config.json` with the hyperparameters read from the
program's generics and constants, `model.safetensors` under the Transformers
tensor names, and the tokenizer files of its Hub repository. vLLM, SGLang,
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
from typing import Any

from . import ir, nest
from .compiler import LinnetError
from .weights import TensorLocation, read_bindings, safetensors_index, write_safetensors

TORCH_DTYPE_NAMES = {"bf16": "bfloat16", "f16": "float16", "f32": "float32", "f64": "float64"}
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
    config: Callable[[Mapping[str, int | str], Mapping[str, float], bool], dict[str, Any]]

    def hf_name(self, path: str) -> str | None:
        """The Transformers name of an expanded Linnet path, or None if it has none."""
        for pattern, template in self.names:
            regex = "^" + re.escape(pattern).replace(r"\[\*\]", r"\.(\d+)") + "$"
            match = re.match(regex, path)
            if match:
                return template.format(i=match.group(1)) if match.groups() else template
        return None


def _llama_config(
    generics: Mapping[str, int | str], constants: Mapping[str, float], biases: bool
) -> dict[str, Any]:
    dtype = str(generics.get("T", "bf16"))
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": generics["Vocab"],
        "hidden_size": generics["H"],
        "num_attention_heads": generics["Heads"],
        "num_key_value_heads": generics["KvHeads"],
        "head_dim": int(generics["H"]) // int(generics["Heads"]),
        "intermediate_size": generics["Inner"],
        "num_hidden_layers": generics["Layers"],
        "max_position_embeddings": generics["MaxSeq"],
        "rope_theta": constants.get("THETA", 10000.0),
        "rms_norm_eps": constants.get("EPS", 1e-5),
        "hidden_act": "silu",
        "attention_bias": biases,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "torch_dtype": TORCH_DTYPE_NAMES.get(dtype, dtype),
        "transformers_version": "4.0.0",
    }


def _gpt2_config(
    generics: Mapping[str, int | str], constants: Mapping[str, float], biases: bool
) -> dict[str, Any]:
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
        "layer_norm_epsilon": constants.get("EPS", 1e-5),
        "tie_word_embeddings": True,
        "torch_dtype": TORCH_DTYPE_NAMES.get(dtype, dtype),
        "transformers_version": "4.0.0",
    }


FAMILIES: tuple[Family, ...] = (
    Family(
        name="llama",
        architecture="LlamaForCausalLM",
        model_type="llama",
        names=(
            ("embedding.weight", "model.embed_tokens.weight"),
            ("norm.weight", "model.norm.weight"),
            ("lm_head.weight", "lm_head.weight"),
            ("layers[*].attention_norm.weight", "model.layers.{i}.input_layernorm.weight"),
            ("layers[*].mlp_norm.weight", "model.layers.{i}.post_attention_layernorm.weight"),
            ("layers[*].attention.q_proj.weight", "model.layers.{i}.self_attn.q_proj.weight"),
            ("layers[*].attention.k_proj.weight", "model.layers.{i}.self_attn.k_proj.weight"),
            ("layers[*].attention.v_proj.weight", "model.layers.{i}.self_attn.v_proj.weight"),
            ("layers[*].attention.o_proj.weight", "model.layers.{i}.self_attn.o_proj.weight"),
            ("layers[*].attention.q_proj.bias", "model.layers.{i}.self_attn.q_proj.bias"),
            ("layers[*].attention.k_proj.bias", "model.layers.{i}.self_attn.k_proj.bias"),
            ("layers[*].attention.v_proj.bias", "model.layers.{i}.self_attn.v_proj.bias"),
            ("layers[*].attention.o_proj.bias", "model.layers.{i}.self_attn.o_proj.bias"),
            ("layers[*].mlp.gate.weight", "model.layers.{i}.mlp.gate_proj.weight"),
            ("layers[*].mlp.up.weight", "model.layers.{i}.mlp.up_proj.weight"),
            ("layers[*].mlp.down.weight", "model.layers.{i}.mlp.down_proj.weight"),
            ("layers[*].mlp.gate.bias", "model.layers.{i}.mlp.gate_proj.bias"),
            ("layers[*].mlp.up.bias", "model.layers.{i}.mlp.up_proj.bias"),
            ("layers[*].mlp.down.bias", "model.layers.{i}.mlp.down_proj.bias"),
            ("lm_head.bias", "lm_head.bias"),
        ),
        optional=frozenset(
            {
                "layers[*].attention.q_proj.bias",
                "layers[*].attention.k_proj.bias",
                "layers[*].attention.v_proj.bias",
                "layers[*].attention.o_proj.bias",
                "layers[*].mlp.gate.bias",
                "layers[*].mlp.up.bias",
                "layers[*].mlp.down.bias",
                "lm_head.bias",
            }
        ),
        generics=frozenset({"Vocab", "H", "Heads", "KvHeads", "Inner", "Layers", "MaxSeq", "T"}),
        config=_llama_config,
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
    ),
)


def recognize(program: ir.Program) -> Family:
    """The family whose parameter paths and generics the program has, exactly."""
    paths = {e.path for e in program.manifest if e.kind == "param"}
    generics = {g.name for g in program.root.generics}
    reasons: list[str] = []
    for family in FAMILIES:
        patterns = {pattern for pattern, _ in family.names}
        required = patterns - family.optional
        missing = sorted(required - paths)
        extra = sorted(paths - patterns)
        # Extra generics (a `Batch` for the KV cache) are fine; missing ones are not.
        wrong_generics = sorted(family.generics - generics)
        if not missing and not extra and not wrong_generics:
            return family
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing[:4]))
        if extra:
            detail.append("unexpected " + ", ".join(extra[:4]))
        if wrong_generics:
            detail.append("missing generics " + ", ".join(wrong_generics[:6]))
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


@dataclass(frozen=True, slots=True)
class Exported:
    directory: Path
    family: Family
    config: Mapping[str, Any]
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
    card: nest.Card | None = None
    path = Path(model)
    if (path / "nest.toml").exists() or not path.exists():
        card = nest.resolve(model)
    values: dict[str, int | str] = {}
    if card is not None:
        values.update(card.generics)
        source = card.source_path
        root = card.root if root is None else root
        if weights is None:
            weights = nest.download_weights(card)
        if bindings is None:
            bindings = card.bindings_path
        if tokenizer is None and card.weights is not None:
            tokenizer = card.weights.repo
    else:
        source = path
    values.update(generics or {})
    if weights is None:
        raise LinnetError("an export needs weights: a SafeTensors file or directory")

    program = ir.load_program(source, root=root, std_root=std_root)
    family = recognize(program)
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

    tensors: list[tuple[str, str, tuple[int, ...], TensorLocation | bytes]] = []
    problems: list[str] = []
    biases = False
    for entry in program.manifest:
        if entry.kind != "param":
            continue
        shape = ir.evaluate_shape(entry.shape, bound)
        for linnet_path in nest.expand_paths(entry, bound):
            source_name = mapping.get(linnet_path, linnet_path)
            location = index.get(source_name)
            if location is None:
                if not entry.optional:
                    problems.append(f"missing tensor `{source_name}` for `{linnet_path}`")
                continue
            if location.shape != shape:
                found, needs = list(location.shape), list(shape)
                problems.append(f"`{source_name}` has shape {found}, `{linnet_path}` needs {needs}")
                continue
            hf_name = family.hf_name(linnet_path)
            if hf_name is None:
                problems.append(f"`{linnet_path}` has no {family.architecture} counterpart")
                continue
            if ".bias" in hf_name and "attn" in hf_name and family.name == "llama":
                biases = True
            tensors.append((hf_name, location.dtype, location.shape, location))
    if problems:
        raise LinnetError("checkpoint does not match the model:\n  " + "\n  ".join(problems))

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    config = family.config(resolved, constants_of(program), biases)
    directory.joinpath("config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_safetensors(directory / "model.safetensors", tensors, metadata={"format": "pt"})
    copied = _copy_tokenizer(tokenizer, directory) if tokenizer else ()
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
    exp.add_argument("model", help="a Nest model directory or name, or a .linnet file")
    exp.add_argument("-o", "--output", required=True)
    exp.add_argument("--root")
    exp.add_argument("--std")
    exp.add_argument("--bind", action="append", default=[], metavar="NAME=VALUE")
    exp.add_argument("--weights")
    exp.add_argument("--bindings")
    exp.add_argument("--tokenizer", help="a Hub repository to take tokenizer files from")
    args = parser.parse_args(list(argv) if argv is not None else None)
    generics: dict[str, int | str] = {}
    for bind in args.bind:
        key, _, value = bind.partition("=")
        generics[key] = int(value) if value.lstrip("-").isdigit() else value
    try:
        exported = export(
            args.model,
            args.output,
            generics=generics,
            weights=args.weights,
            bindings=args.bindings,
            root=args.root,
            std_root=args.std,
            tokenizer=args.tokenizer,
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

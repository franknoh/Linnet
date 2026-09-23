"""GGUF files and Ollama Modelfiles from Linnet models.

llama.cpp runs a fixed set of architectures from GGUF files, and its own
`convert_hf_to_gguf.py` knows how to write them (tensor layout, tokenizer
metadata, quantization). `export` therefore goes through the Transformers
checkpoint `linnet.hf` writes and hands it to that converter, then writes a
Modelfile so `ollama create` picks the result up. Only the `llama` and
`gpt2` families qualify, as with `linnet.hf`; nothing else is attempted.

    python -m linnet.gguf export tinyllama-1.1b-chat -o serve/tinyllama \\
        --converter ~/llama.cpp/convert_hf_to_gguf.py --outtype q8_0
    ollama create tinyllama -f serve/tinyllama/Modelfile
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from . import hf
from .compiler import LinnetError

OUTTYPES = ("f32", "f16", "bf16", "q8_0", "auto")


@dataclass(frozen=True, slots=True)
class Exported:
    directory: Path
    gguf: Path
    modelfile: Path
    checkpoint: hf.Exported


def find_converter(converter: str | Path | None = None) -> Path:
    """llama.cpp's `convert_hf_to_gguf.py`: the argument, `LLAMA_CPP_CONVERTER`, or
    `LLAMA_CPP/convert_hf_to_gguf.py`."""
    candidates: list[Path] = []
    if converter is not None:
        candidates.append(Path(converter))
    if os.environ.get("LLAMA_CPP_CONVERTER"):
        candidates.append(Path(os.environ["LLAMA_CPP_CONVERTER"]))
    if os.environ.get("LLAMA_CPP"):
        candidates.append(Path(os.environ["LLAMA_CPP"]) / "convert_hf_to_gguf.py")
    for candidate in candidates:
        if candidate.is_dir():
            candidate = candidate / "convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate
    raise LinnetError(
        "llama.cpp's convert_hf_to_gguf.py was not found; pass --converter, or set "
        "LLAMA_CPP to a llama.cpp checkout (pip install -r llama.cpp/requirements.txt)"
    )


def modelfile_text(
    gguf: Path,
    *,
    template: str | None = None,
    system: str | None = None,
    parameters: Mapping[str, str] | None = None,
) -> str:
    """An Ollama Modelfile next to the GGUF file."""
    lines = [f"FROM ./{gguf.name}"]
    for key, value in (parameters or {}).items():
        lines.append(f"PARAMETER {key} {value}")
    if template:
        lines.append('TEMPLATE """' + template + '"""')
    if system:
        lines.append('SYSTEM """' + system + '"""')
    return "\n".join(lines) + "\n"


def chat_template_of(directory: Path) -> str | None:
    """The tokenizer's chat template, translated for Ollama when it is a known shape.

    Ollama templates are Go templates, not Jinja, so only the templates
    Ollama has an equivalent for are translated; others are left out and
    the model runs as a plain completion model.
    """
    config = directory / "tokenizer_config.json"
    if not config.exists():
        return None
    try:
        template = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(template, str):
        return None
    if "<|im_start|>" in template:
        return (
            "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
            "{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}"
            "<|im_start|>assistant\n{{ .Response }}<|im_end|>\n"
        )
    if "<|user|>" in template and "<|assistant|>" in template:  # Zephyr / TinyLlama chat
        return (
            "{{ if .System }}<|system|>\n{{ .System }}</s>\n{{ end }}"
            "{{ if .Prompt }}<|user|>\n{{ .Prompt }}</s>\n{{ end }}"
            "<|assistant|>\n{{ .Response }}</s>\n"
        )
    if "[INST]" in template:
        return "[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST] {{ .Response }}"
    return None


def export(
    model: str | Path,
    output: str | Path,
    *,
    converter: str | Path | None = None,
    outtype: str = "f16",
    generics: Mapping[str, int | str] | None = None,
    weights: str | Path | None = None,
    bindings: str | Path | None = None,
    root: str | None = None,
    std_root: str | Path | None = None,
    tokenizer: str | None = None,
    name: str | None = None,
    keep_checkpoint: bool = False,
) -> Exported:
    """Writes `<output>/<name>.gguf` and `<output>/Modelfile`.

    The Transformers checkpoint is written under `output/hf/` first and
    removed afterwards unless `keep_checkpoint`.
    """
    if outtype not in OUTTYPES:
        raise LinnetError(f"outtype must be one of {', '.join(OUTTYPES)}")
    script = find_converter(converter)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = hf.export(
        model,
        directory / "hf",
        generics=generics,
        weights=weights,
        bindings=bindings,
        root=root,
        std_root=std_root,
        tokenizer=tokenizer,
    )
    stem = name or (Path(model).name if Path(model).exists() else str(model))
    stem = stem.removesuffix(".linnet") or "model"
    gguf_path = directory / f"{stem}.gguf"
    command = [
        sys.executable,
        str(script),
        str(checkpoint.directory),
        "--outfile",
        str(gguf_path),
        "--outtype",
        outtype,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not gguf_path.exists():
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-8:]
        raise LinnetError("convert_hf_to_gguf.py failed:\n  " + "\n  ".join(tail))
    modelfile = directory / "Modelfile"
    modelfile.write_text(
        modelfile_text(gguf_path, template=chat_template_of(checkpoint.directory)),
        encoding="utf-8",
    )
    if not keep_checkpoint:
        shutil.rmtree(checkpoint.directory, ignore_errors=True)
    return Exported(directory, gguf_path, modelfile, checkpoint)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m linnet.gguf", description="GGUF files for llama.cpp and Ollama."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export", help="write a GGUF file and a Modelfile")
    exp.add_argument("model", help="a Nest model directory or name, or a .linnet file")
    exp.add_argument("-o", "--output", required=True)
    exp.add_argument("--converter", help="llama.cpp's convert_hf_to_gguf.py (or its checkout)")
    exp.add_argument("--outtype", choices=OUTTYPES, default="f16")
    exp.add_argument("--name", help="the GGUF file's stem")
    exp.add_argument("--root")
    exp.add_argument("--std")
    exp.add_argument("--bind", action="append", default=[], metavar="NAME=VALUE")
    exp.add_argument("--weights")
    exp.add_argument("--bindings")
    exp.add_argument("--tokenizer")
    exp.add_argument(
        "--keep-checkpoint", action="store_true", help="keep the Transformers directory"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    generics: dict[str, int | str] = {}
    for bind in args.bind:
        key, _, value = bind.partition("=")
        generics[key] = int(value) if value.lstrip("-").isdigit() else value
    try:
        exported = export(
            args.model,
            args.output,
            converter=args.converter,
            outtype=args.outtype,
            generics=generics,
            weights=args.weights,
            bindings=args.bindings,
            root=args.root,
            std_root=args.std,
            tokenizer=args.tokenizer,
            name=args.name,
            keep_checkpoint=args.keep_checkpoint,
        )
    except LinnetError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"wrote {exported.gguf} and {exported.modelfile}")
    print(f"  ollama create {exported.gguf.stem} -f {exported.modelfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

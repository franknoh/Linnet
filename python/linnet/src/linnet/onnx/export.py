"""Self-contained ONNX models: `linnet onnx` output with the weights inside.

The compiler exports parameters as graph inputs named in the metadata
(`linnet.path.param0` = `layers.0.up.weight`). `export_model` turns those
into initializers from a SafeTensors checkpoint, so the result runs in any
ONNX runtime with nothing else, and `save` writes it with external data
when it is too large for one protobuf.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..compiler import LinnetError, run_compiler, std_arguments
from ..weights import iter_safetensors, read_bindings, safetensors_index

# SafeTensors dtype names to ONNX TensorProto data types.
ONNX_DTYPES = {
    "BOOL": 9,
    "I8": 3,
    "I16": 5,
    "I32": 6,
    "I64": 7,
    "U8": 2,
    "U16": 4,
    "U32": 12,
    "U64": 13,
    "F16": 10,
    "BF16": 16,
    "F32": 1,
    "F64": 11,
}
# ONNX data types to Linnet dtype names, for reporting.
LINNET_DTYPES = {v: k.lower() for k, v in ONNX_DTYPES.items()}


@dataclass(frozen=True, slots=True)
class Port:
    name: str
    dtype: str  # a Linnet dtype name
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Exported:
    """A packaged model and the interface that remains: inputs, states, results."""

    model: Any  # onnx.ModelProto
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    parameters: tuple[str, ...]  # the paths now embedded

    def save(self, path: str | Path, *, external_threshold: int = 1 << 30) -> Path:
        """Writes the model; weights beyond `external_threshold` bytes go next to it."""
        import onnx

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        size = sum(len(t.raw_data) for t in self.model.graph.initializer)
        if size > external_threshold:
            onnx.save_model(
                self.model,
                str(target),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=target.name + ".data",
            )
        else:
            onnx.save_model(self.model, str(target))
        return target


def export_model(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path,
    entry: str | None = None,
    root: str | None = None,
    std_root: str | Path | None = None,
    numerics: str = "equivalent",
    bindings: str | Path | None = None,
    optionals: str = "auto",
) -> Exported:
    """Compiles one entry to ONNX and embeds the checkpoint as initializers.

    `generics` binds the root block's and the entry's generics (`--bind`).
    Every parameter the graph names must be in `weights` with the shape and
    dtype the graph declares; `bindings` maps parameter paths to tensor names.

    `optionals="auto"` (the default) includes each optional parameter exactly
    when the checkpoint has it. Checkpoints mix them -- attention projections
    with a bias beside ones without, convolutions without one before a
    classifier with one -- so neither `"present"` nor `"absent"` for the whole
    model is right in general; those two remain for a caller who knows.
    """
    import onnx
    from onnx import parser

    mapping = read_bindings(bindings) if bindings is not None else {}
    arguments = ["onnx", "--numerics", numerics]
    arguments += ["--optionals", "present" if optionals == "auto" else optionals]
    if root is not None:
        arguments += ["--root", root]
    if entry is not None:
        arguments += ["--entry", entry]
    for name, value in generics.items():
        arguments += ["--bind", f"{name}={value}"]
    text = run_compiler(*arguments, *std_arguments(std_root), str(source))
    model = parser.parse_model(text)
    if optionals == "auto":
        # Every optional was compiled in; the ones the checkpoint lacks are
        # compiled out again, by path. A required parameter the checkpoint
        # lacks stays in the graph and is reported as missing below.
        available = set(safetensors_index(weights))
        missing = sorted(
            p.value
            for p in model.metadata_props
            if p.key.startswith("linnet.path.") and mapping.get(p.value, p.value) not in available
        )
        if missing:
            with tempfile.TemporaryDirectory() as work:
                listing = Path(work) / "absent.txt"
                listing.write_text("\n".join(missing) + "\n", encoding="utf-8")
                text = run_compiler(
                    *arguments,
                    "--absent-file",
                    str(listing),
                    *std_arguments(std_root),
                    str(source),
                )
            model = parser.parse_model(text)

    paths = {
        p.key.removeprefix("linnet.path."): p.value
        for p in model.metadata_props
        if p.key.startswith("linnet.path.")
    }
    wanted = {mapping.get(path, path): (input_name, path) for input_name, path in paths.items()}
    declared = {i.name: i for i in model.graph.input}

    found: dict[str, Any] = {}
    problems: list[str] = []
    for tensor in iter_safetensors(weights):
        if tensor.name not in wanted:
            continue
        input_name, path = wanted[tensor.name]
        info = declared[input_name].type.tensor_type
        shape = tuple(d.dim_value for d in info.shape.dim)
        if shape != tensor.shape:
            problems.append(
                f"`{tensor.name}` has shape {list(tensor.shape)}, `{path}` needs {list(shape)}"
            )
            continue
        if ONNX_DTYPES.get(tensor.dtype) != info.elem_type:
            needs = LINNET_DTYPES.get(info.elem_type, str(info.elem_type))
            problems.append(f"`{tensor.name}` is {tensor.dtype}, `{path}` needs {needs}")
            continue
        initializer = onnx.TensorProto()
        initializer.name = input_name
        initializer.data_type = info.elem_type
        initializer.dims.extend(tensor.shape)
        initializer.raw_data = tensor.data
        found[input_name] = initializer
    missing = [path for input_name, path in wanted.values() if input_name not in found]
    if missing:
        problems += [f"missing tensor for `{path}`" for path in missing]
    if problems:
        raise LinnetError("checkpoint does not match the model:\n  " + "\n  ".join(problems))

    model.graph.initializer.extend(found[name] for name in paths if name in found)
    remaining = [i for i in model.graph.input if i.name not in found]
    del model.graph.input[:]
    model.graph.input.extend(remaining)
    onnx.checker.check_model(model)
    return Exported(
        model=model,
        inputs=tuple(_port(i) for i in model.graph.input),
        outputs=tuple(_port(o) for o in model.graph.output),
        parameters=tuple(paths.values()),
    )


def _port(value: Any) -> Port:
    info = value.type.tensor_type
    return Port(
        name=value.name,
        dtype=LINNET_DTYPES.get(info.elem_type, str(info.elem_type)),
        shape=tuple(d.dim_value for d in info.shape.dim),
    )

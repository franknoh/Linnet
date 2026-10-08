"""Self-contained ONNX models: `linnet onnx` output with the weights inside,
and module-level entries (functions, which have none) on their own.

The compiler exports parameters as graph inputs named in the metadata
(`linnet.path.param0` = `layers.0.up.weight`). `export_model` turns those
into initializers from a SafeTensors checkpoint, so the result runs in any
ONNX runtime with nothing else, and `save` writes it with external data
when it is too large for one protobuf.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import tempfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..compiler import LinnetError, bind_arguments, run_compiler, std_arguments
from ..dtypes import BY_ONNX, BY_SAFETENSORS
from ..weights import (
    RawTensor,
    decode_floats,
    encode_floats,
    read_bindings,
    safetensors_index,
)


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
    # With `embed=False`: the checked weights as initializers-to-be, by input name.
    weights: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

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
    cast_dtype: bool = False,
    embed: bool = True,
    held: Collection[str] = (),
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

    `cast_dtype=True` converts floating-point weights to the dtype the graph
    declares for them (the generics' `T`), so a checkpoint published in f32
    exports as an f16 or bf16 model. `embed=False` checks the weights but
    leaves them graph inputs, for a runtime that binds one copy to several
    entries' graphs (`linnet.onnx.load_model`); `Exported.weights` then holds
    them by input name. With it, `held` names parameter paths whose weights
    the caller already has: they are checked against the checkpoint's
    header and not read, so a runtime compiling one more shape does not read
    the checkpoint again.
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
    arguments += bind_arguments(generics)
    text = run_compiler(*arguments, *std_arguments(std_root), str(source))
    model = parser.parse_model(text)
    index = safetensors_index(weights)
    if optionals == "auto":
        # Every optional was compiled in; the ones the checkpoint lacks are
        # compiled out again, by path. A required parameter the checkpoint
        # lacks stays in the graph and is reported as missing below.
        available = set(index)
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
    # A checkpoint tensor may serve several parameters (an output head tied
    # to the embedding): each gets its own initializer.
    declared = {i.name: i for i in model.graph.input}
    skipped = set(held) if not embed else set()
    wanted: dict[str, list[tuple[str, str]]] = {}
    found: dict[str, Any] = {}
    checked: set[str] = set()  # held: header checked, bytes not read
    problems: list[str] = []
    for input_name, path in paths.items():
        name = mapping.get(path, path)
        location = index.get(name)
        if location is None:
            problems.append(f"missing tensor for `{path}`")
        elif path in skipped:
            if _matches(
                location.dtype,
                location.shape,
                name,
                input_name,
                path,
                declared,
                cast_dtype,
                problems,
            ):
                checked.add(input_name)
        else:
            wanted.setdefault(name, []).append((input_name, path))
    # Only the tensors the graph reads, in file order.
    for name, location in index.items():
        for input_name, path in wanted.get(name, []):
            tensor = RawTensor(name, location.dtype, location.shape, location.read())
            _take(tensor, input_name, path, declared, cast_dtype, found, problems)
    if problems:
        raise LinnetError("checkpoint does not match the model:\n  " + "\n  ".join(problems))

    if not embed:
        onnx.checker.check_model(model)
        return Exported(
            model=model,
            inputs=tuple(
                _port(i) for i in model.graph.input if i.name not in found and i.name not in checked
            ),
            outputs=tuple(_port(o) for o in model.graph.output),
            parameters=tuple(paths.values()),
            weights={name: found[name] for name in paths if name in found},
        )
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


def export_function(
    source: str | Path,
    name: str | None = None,
    *,
    generics: Mapping[str, int | str],
    std_root: str | Path | None = None,
    numerics: str = "equivalent",
) -> Exported:
    """Compiles a module-level entry -- a function of its inputs alone, such
    as a loss, a preprocessing step, or a reward -- to a self-contained ONNX
    model. It has no weights, so nothing is embedded. `name` picks the entry
    (the only one when omitted); `generics` binds every generic, the graph's
    shapes being static."""
    import onnx
    from onnx import parser

    from ..plan import compile_plan

    plan = compile_plan(source, std_root=std_root, optimize=False, functions=True)
    entry = plan.module_entry(name).name.rsplit("::", 1)[1]
    arguments = ["onnx", "--numerics", numerics, "--entry", entry]
    arguments += bind_arguments(generics)
    model = parser.parse_model(run_compiler(*arguments, *std_arguments(std_root), str(source)))
    onnx.checker.check_model(model)
    return Exported(
        model=model,
        inputs=tuple(_port(i) for i in model.graph.input),
        outputs=tuple(_port(o) for o in model.graph.output),
        parameters=(),
    )


def _matches(
    dtype: str,
    shape: tuple[int, ...],
    name: str,
    input_name: str,
    path: str,
    declared: dict[str, Any],
    cast_dtype: bool,
    problems: list[str],
) -> bool:
    """Whether checkpoint tensor `name` (`dtype`, `shape`) can be graph input
    `input_name` (parameter `path`); if not, why is added to `problems`."""
    info = declared[input_name].type.tensor_type
    wanted = tuple(d.dim_value for d in info.shape.dim)
    if wanted != shape:
        problems.append(f"`{name}` has shape {list(shape)}, `{path}` needs {list(wanted)}")
        return False
    found, needed = BY_SAFETENSORS.get(dtype), BY_ONNX.get(info.elem_type)
    if found is None or found.onnx != info.elem_type:
        floats = found is not None and found.is_float and needed is not None and needed.is_float
        if not (cast_dtype and floats):
            needs = str(info.elem_type) if needed is None else needed.name
            problems.append(f"`{name}` is {dtype}, `{path}` needs {needs}")
            return False
    return True


def _take(
    tensor: Any,
    input_name: str,
    path: str,
    declared: dict[str, Any],
    cast_dtype: bool,
    found: dict[str, Any],
    problems: list[str],
) -> None:
    """`tensor` as the initializer for graph input `input_name` (parameter
    `path`), checked against its declared shape and dtype."""
    import onnx

    if not _matches(
        tensor.dtype, tensor.shape, tensor.name, input_name, path, declared, cast_dtype, problems
    ):
        return
    info = declared[input_name].type.tensor_type
    data = tensor.data
    stored = BY_SAFETENSORS.get(tensor.dtype)
    if stored is None or stored.onnx != info.elem_type:
        data = encode_floats(decode_floats(tensor.dtype, data), BY_ONNX[info.elem_type].safetensors)
    initializer = onnx.TensorProto()
    initializer.name = input_name
    initializer.data_type = info.elem_type
    initializer.dims.extend(tensor.shape)
    initializer.raw_data = data
    found[input_name] = initializer


def _port(value: Any) -> Port:
    info = value.type.tensor_type
    return Port(
        name=value.name,
        dtype=BY_ONNX[info.elem_type].name if info.elem_type in BY_ONNX else str(info.elem_type),
        shape=tuple(d.dim_value for d in info.shape.dim),
    )

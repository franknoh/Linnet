"""Every entry of a Linnet block on ONNX Runtime, over one copy of the weights.

`load_model` is to ONNX Runtime what `linnet.jax.load_model` is to XLA: each
entry is exported (`linnet onnx`) for the shapes it is called with and run by
its own inference session, but the weights are not embedded in any of them.
They are placed on the device once, as `OrtValue`s, and bound to every
session by I/O binding; the block's `state` (a decoder's KV caches) stays on
the device between calls, each call's new state bound as the next call's
input. An entry with no state -- an encoder's, a classifier's -- takes them
as initializers instead, from the same host copy, so ONNX Runtime can fold
what depends only on them (a batch norm into its convolution); each such
session keeps its own device copy.

    from linnet.onnx import load_model

    model = load_model("model.linnet", generics={..., "T": "f16"},
                       weights="model.safetensors", cast_dtype=True)
    logits = model.run_entry("prefill", [tokens, np.int32(0)])
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..compiler import LinnetError, run_compiler, std_arguments
from .export import export_model

# ONNX element types and the NumPy dtypes their bytes are read as. bf16 has
# no NumPy type; its bytes travel as uint16.
_NUMPY = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    16: np.uint16,
}
_ELEMENTS = {"f32": 1, "f16": 10, "bf16": 16, "i32": 6, "i64": 7, "bool": 9}


class OnnxModel:
    """A root block's entries on ONNX Runtime, sharing weights and state.

    `run_entry(name, inputs)` runs one entry on NumPy inputs, or on
    `OrtValue`s already on the device, and returns its results as NumPy
    arrays (one result bare, several as a tuple); with `keep_on_device=True`,
    as `OrtValue`s left on the device. `state`
    holds the block's state members by path, on the device; `reset_state()`
    clears it.
    """

    def __init__(
        self,
        source: Path,
        plan: dict[str, Any],
        options: dict[str, Any],
        providers: Sequence[Any],
    ) -> None:
        import onnxruntime  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]

        self._ort = onnxruntime
        self._source = source
        self.plan = plan
        self._options = options
        self.providers = list(providers)
        # Providers are names or (name, options); a GPU one keeps the
        # weights and state in GPU memory.
        names = [p if isinstance(p, str) else p[0] for p in self.providers]
        gpu = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        self._device = "cuda" if gpu & set(names) else "cpu"
        root = str(plan["root"]["name"])
        self._signatures = {
            str(f["name"]).rsplit(".", 1)[1]: f
            for f in plan["functions"]
            if f["kind"] == "entry" and f["block"] == root
        }
        self.entries = list(self._signatures)
        self.generics = dict(options["generics"])
        self._sessions: dict[tuple[Any, ...], _Session] = {}
        self._weights: dict[str, Any] = {}  # path -> OrtValue on the device
        self._host: dict[str, tuple[np.ndarray, Any]] = {}  # path -> (bytes, CPU OrtValue)
        self._prepared: dict[str, Any] = {}  # key -> weight-only value on the device
        self.state: dict[str, Any] = {}  # path -> OrtValue on the device

    # ---- entries

    def run_entry(self, name: str, inputs: Sequence[Any], keep_on_device: bool = False) -> Any:
        if name not in self._signatures:
            raise LinnetError(f"the block has no entry `{name}`")
        arrays = [v if isinstance(v, self._ort.OrtValue) else np.asarray(v) for v in inputs]
        bindings = self._bindings(name, arrays)
        key = (name, tuple(sorted(bindings.items())))
        if key not in self._sessions:
            self._sessions[key] = self._session(name, bindings)
        session = self._sessions[key]
        binding = session.session.io_binding()
        for port, array in zip(session.inputs, arrays, strict=True):
            element = session.elements[port]
            if isinstance(array, self._ort.OrtValue):
                # Already where it runs, in the graph's type.
                binding.bind_ortvalue_input(port, array)
            elif element == 16:
                # bf16 inputs travel as their bits, typed as bf16.
                value = self._ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                    _encode(array, element), 16
                )
                binding.bind_ortvalue_input(port, value)
            else:
                binding.bind_cpu_input(port, _encode(array, element))
        for port, path in session.parameters.items():
            binding.bind_ortvalue_input(port, self._weights[path])
        for port, key in session.prepared.items():
            binding.bind_ortvalue_input(port, self._prepared[key])
        for port, (path, shape, element) in session.states.items():
            if path not in self.state:
                zeros = np.zeros(shape, dtype=_NUMPY[element])
                self.state[path] = _to_device(self._ort, zeros, element, self._device)
            binding.bind_ortvalue_input(port, self.state[path])
        # bf16 results land in host buffers of their bits: ONNX Runtime has
        # no NumPy type to hand them back as.
        buffers: dict[int, np.ndarray] = {}
        for i, port in enumerate(session.results):
            if keep_on_device:
                binding.bind_output(port, self._device, 0)
            elif session.result_elements[i] == 16:
                buffers[i] = np.empty(session.result_shapes[i], dtype=np.uint16)
                target = self._ort.OrtValue.ortvalue_from_numpy_with_onnx_type(buffers[i], 16)
                binding.bind_ortvalue_output(port, target)
            else:
                binding.bind_output(port, "cpu")
        for port in session.next_states:
            binding.bind_output(port, self._device, 0)
        session.session.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        # bf16 results arrive as their bits; they are returned as f32.
        results = [
            outputs[i]
            if keep_on_device
            else _decode(buffers[i], 16)
            if i in buffers
            else outputs[i].numpy()
            for i in range(len(session.results))
        ]
        for i, path in enumerate(session.next_states.values()):
            self.state[path] = outputs[len(session.results) + i]
        return results[0] if len(results) == 1 else tuple(results)

    def reset_state(self) -> None:
        self.state = {}

    def place(self, array: Any, dtype: str) -> Any:
        """`array` as an `OrtValue` on the model's device in `dtype` (`f32`,
        `f16`, `bf16`, `i32`, ...): an input reused call after call, bound
        without a copy from the host each time."""
        element = _ELEMENTS[dtype]
        return _to_device(self._ort, _encode(np.asarray(array), element), element, self._device)

    # ---- compilation

    def _session(self, name: str, bindings: Mapping[str, int | str]) -> _Session:
        options = self._options
        exported = export_model(
            self._source,
            generics={**options["generics"], **bindings},
            weights=options["weights"],
            entry=name,
            root=options["root"],
            std_root=options["std_root"],
            numerics=options["numerics"],
            bindings=options["bindings"],
            cast_dtype=options["cast_dtype"],
            embed=False,
        )
        metadata = {p.key: p.value for p in exported.model.metadata_props}
        parameters = {
            key.removeprefix("linnet.path."): path
            for key, path in metadata.items()
            if key.startswith("linnet.path.")
        }
        stateless = not any(
            key.startswith(("linnet.state.", "linnet.next_state.")) for key in metadata
        )
        settings = self._ort.SessionOptions()
        settings.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._device == "cuda":
            # Every session allocates from one arena: each of its own grows
            # to the largest call it has seen and keeps it, which for a
            # server's dozen shapes is gigabytes held apart.
            _share_cuda_arena(self._ort)
            settings.add_session_config_entry("session.use_env_allocators", "1")
        if stateless:
            self._fold_weights(exported, parameters, settings)
            parameters = {}
        for port, path in parameters.items():
            if path not in self._weights:
                tensor = exported.weights[port]
                array = np.frombuffer(tensor.raw_data, dtype=_NUMPY[tensor.data_type]).reshape(
                    tuple(tensor.dims)
                )
                self._weights[path] = _to_device(self._ort, array, tensor.data_type, self._device)
        prepared: dict[str, str] = {}
        if not stateless:
            prepared = self._prepare(exported, parameters)
            used = _consumed(exported.model.graph)
            parameters = {port: path for port, path in parameters.items() if port in used}
        graph_inputs = {i.name: i.type.tensor_type for i in exported.model.graph.input}
        states = {}
        for key, path in metadata.items():
            if key.startswith("linnet.state."):
                port = key.removeprefix("linnet.state.")
                info = graph_inputs[port]
                shape = tuple(d.dim_value for d in info.shape.dim)
                states[port] = (path, shape, info.elem_type)
        next_states = {
            key.removeprefix("linnet.next_state."): path
            for key, path in metadata.items()
            if key.startswith("linnet.next_state.")
        }
        outputs = [o.name for o in exported.model.graph.output]
        results = [o for o in outputs if o not in next_states]
        inputs = [
            i.name
            for i in exported.model.graph.input
            if i.name not in parameters and i.name not in states and i.name not in prepared
        ]
        session = self._ort.InferenceSession(
            exported.model.SerializeToString(), settings, providers=self.providers
        )
        elements = {port: graph_inputs[port].elem_type for port in inputs}
        graph_outputs = {o.name: o.type.tensor_type for o in exported.model.graph.output}
        result_elements = [graph_outputs[port].elem_type for port in results]
        result_shapes = [
            tuple(d.dim_value for d in graph_outputs[port].shape.dim) for port in results
        ]
        return _Session(
            session,
            inputs,
            elements,
            parameters,
            states,
            results,
            result_elements,
            result_shapes,
            next_states,
            prepared,
        )

    def _prepare(self, exported: Any, parameters: Mapping[str, str]) -> dict[str, str]:
        """Runs what reads only weights -- a dequantizer's unpacking of
        every expert -- once, in a session of its own, and makes each such
        value an input of the entry's graph instead. Values are keyed by
        their computation, so entries computing the same one share it, as
        `--prepare` does for torch and JAX. Returns graph input -> key."""
        import onnx

        model = exported.model
        graph = model.graph
        split = _split_weight_only(graph, parameters)
        if not split.boundary:
            return {}
        typed = onnx.shape_inference.infer_shapes(model)
        types = {v.name: v.type for v in list(typed.graph.value_info) + list(typed.graph.input)}
        missing = [name for name in split.boundary if split.keys[name] not in self._prepared]
        if missing:
            self._run_prepare(model, split, missing, parameters, types)
        # The entry's graph: without the weight-only nodes it no longer
        # needs, with each prepared value an input.
        kept = [node for index, node in enumerate(graph.node) if index in split.main_nodes]
        del graph.node[:]
        graph.node.extend(kept)
        used = _consumed(graph)
        inputs = [i for i in graph.input if i.name not in parameters or i.name in used]
        del graph.input[:]
        graph.input.extend(inputs)
        for name in split.boundary:
            value = onnx.ValueInfoProto()
            value.name = name
            value.type.CopyFrom(types[name])
            graph.input.append(value)
        return {name: split.keys[name] for name in split.boundary}

    def _run_prepare(
        self,
        model: Any,
        split: _WeightOnly,
        missing: Sequence[str],
        parameters: Mapping[str, str],
        types: Mapping[str, Any],
    ) -> None:
        import onnx

        needed = _producers(model.graph, missing, set(parameters))
        graph = onnx.GraphProto()
        graph.name = "prepare"
        graph.node.extend(node for index, node in enumerate(model.graph.node) if index in needed)
        reads = _consumed(graph)
        graph.input.extend(i for i in model.graph.input if i.name in parameters and i.name in reads)
        for name in missing:
            value = onnx.ValueInfoProto()
            value.name = name
            value.type.CopyFrom(types[name])
            graph.output.append(value)
        prepare = onnx.helper.make_model(graph, opset_imports=model.opset_import)
        prepare.ir_version = model.ir_version
        # TensorRT takes no part: the unpacking works on bytes it does not
        # import, and it runs once. The session's arena grows only by what
        # it is asked for and gives back what the run no longer holds, so
        # only the prepared values stay.
        providers: list[Any] = []
        for provider in self.providers:
            name = provider if isinstance(provider, str) else provider[0]
            if name == "TensorrtExecutionProvider":
                continue
            if name == "CUDAExecutionProvider":
                options = {} if isinstance(provider, str) else dict(provider[1])
                options["arena_extend_strategy"] = "kSameAsRequested"
                provider = (name, options)
            providers.append(provider)
        settings = self._ort.SessionOptions()
        session = self._ort.InferenceSession(
            prepare.SerializeToString(), settings, providers=providers
        )
        binding = session.io_binding()
        for value in graph.input:
            binding.bind_ortvalue_input(value.name, self._weights[parameters[value.name]])
        for name in missing:
            binding.bind_output(name, self._device, 0)
        run = self._ort.RunOptions()
        run.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage", "gpu:0" if self._device == "cuda" else "cpu:0"
        )
        session.run_with_iobinding(binding, run)
        for name, value in zip(missing, binding.get_outputs(), strict=True):
            self._prepared[split.keys[name]] = value

    def _fold_weights(self, exported: Any, parameters: Mapping[str, str], settings: Any) -> None:
        """Makes the graph's parameters initializers, their bytes the shared
        host copy: ONNX Runtime folds and fuses what reads only constants,
        which it cannot do for inputs bound at run time."""
        import onnx

        graph = exported.model.graph
        used = _consumed(graph)
        kept = [i for i in graph.input if i.name not in parameters]
        del graph.input[:]
        graph.input.extend(kept)
        names: list[str] = []
        values: list[Any] = []
        for port, path in parameters.items():
            if port not in used:
                continue
            tensor = exported.weights[port]
            if path not in self._host:
                array = np.frombuffer(tensor.raw_data, dtype=_NUMPY[tensor.data_type]).reshape(
                    tuple(tensor.dims)
                )
                self._host[path] = (array, _to_device(self._ort, array, tensor.data_type, "cpu"))
            # An initializer whose data is external: the session takes it
            # from `values`, not from a file.
            placeholder = onnx.TensorProto()
            placeholder.name = port
            placeholder.data_type = tensor.data_type
            placeholder.dims.extend(tensor.dims)
            placeholder.data_location = onnx.TensorProto.EXTERNAL
            for key, value in (
                ("location", "linnet-weights"),
                ("offset", "0"),
                ("length", str(len(tensor.raw_data))),
            ):
                entry = placeholder.external_data.add()
                entry.key, entry.value = key, value
            graph.initializer.append(placeholder)
            names.append(port)
            values.append(self._host[path][1])
        if names:
            settings.add_external_initializers(names, values)

    def _bindings(self, name: str, inputs: Sequence[Any]) -> dict[str, int]:
        """The entry's own generics, from the shapes of its inputs."""
        arguments = self._signatures[name]["body"]["args"][1:]
        if len(arguments) != len(inputs):
            raise LinnetError(f"entry `{name}` takes {len(arguments)} inputs, got {len(inputs)}")
        bindings: dict[str, int] = {}
        for argument, value in zip(arguments, inputs, strict=True):
            declared = argument["type"]
            if declared.get("kind") != "tensor":
                continue
            shape = value.shape() if isinstance(value, self._ort.OrtValue) else value.shape
            for unit, size in zip(declared["shape"], shape, strict=True):
                if isinstance(unit, dict) and "sym" in unit and unit["name"] not in self.generics:
                    previous = bindings.get(unit["name"])
                    if previous is not None and previous != size:
                        raise LinnetError(
                            f"dimension `{unit['name']}` is both {previous} and {size}"
                        )
                    bindings[str(unit["name"])] = int(size)
        return bindings


class _Session:
    def __init__(
        self,
        session: Any,
        inputs: list[str],
        elements: dict[str, int],
        parameters: dict[str, str],
        states: dict[str, tuple[str, tuple[int, ...], int]],
        results: list[str],
        result_elements: list[int],
        result_shapes: list[tuple[int, ...]],
        next_states: dict[str, str],
        prepared: dict[str, str],
    ) -> None:
        self.session = session
        self.inputs = inputs
        self.elements = elements  # graph input -> ONNX element type
        self.result_elements = result_elements
        self.result_shapes = result_shapes
        self.parameters = parameters  # graph input -> parameter path
        self.states = states  # graph input -> (state path, shape, element type)
        self.results = results  # graph outputs, in order
        self.next_states = next_states  # graph output -> state path
        self.prepared = prepared  # graph input -> prepared value key


# Operators that only move or relabel what they read: a weight passed
# through them alone is not worth a prepared copy.
_MOVES = {
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Identity",
    "Cast",
    "Expand",
    "Slice",
    "Concat",
    "Gather",
    "Constant",
}
_VIEWS = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Identity"}


class _WeightOnly:
    """The weight-only values an entry's graph reads outside that part:
    `boundary` (tensor names), their `keys`, and `main_nodes`, the indices of
    the nodes the entry keeps."""

    def __init__(self, boundary: list[str], keys: dict[str, str], main_nodes: set[int]) -> None:
        self.boundary = boundary
        self.keys = keys
        self.main_nodes = main_nodes


def _split_weight_only(graph: Any, parameters: Mapping[str, str]) -> _WeightOnly:
    import hashlib

    weight_only: set[str] = set(parameters)
    reads_weights: dict[str, bool] = {port: True for port in parameters}
    computes: dict[str, bool] = {port: False for port in parameters}
    digest: dict[str, str] = {port: "param:" + path for port, path in parameters.items()}
    producer: dict[str, int] = {}
    nodes = list(graph.node)
    for index, node in enumerate(nodes):
        for output in node.output:
            producer[output] = index
        inputs = [name for name in node.input if name]
        subgraphs = any(a.g.node or a.graphs for a in node.attribute)
        if subgraphs or not all(name in weight_only for name in inputs):
            continue
        attributes = b"".join(a.SerializeToString() for a in node.attribute)
        for index, output in enumerate(node.output):
            weight_only.add(output)
            reads_weights[output] = any(reads_weights[name] for name in inputs)
            computes[output] = node.op_type not in _MOVES or any(computes[name] for name in inputs)
            text = "|".join(
                [node.op_type, attributes.hex(), str(index), *(digest[i] for i in inputs)]
            )
            digest[output] = hashlib.sha1(text.encode()).hexdigest()
    outside: set[str] = {o.name for o in graph.output}
    for node in graph.node:
        if not all(o in weight_only for o in node.output):
            outside.update(name for name in node.input if name)
    boundary: list[str] = []
    for name in sorted(outside):
        if name not in weight_only or name in parameters:
            continue
        if not (reads_weights[name] and computes[name]):
            continue
        # A view of a prepared value stays in the entry, reading the value.
        while nodes[producer[name]].op_type in _VIEWS and computes[nodes[producer[name]].input[0]]:
            name = nodes[producer[name]].input[0]
        if name not in boundary:
            boundary.append(name)
    keys = {name: digest[name] for name in boundary}
    # The entry keeps every node it needs but those behind the boundary.
    needed: set[int] = set()
    pending = [name for name in outside]
    for index, node in enumerate(nodes):
        if not all(o in weight_only for o in node.output):
            needed.add(index)
    stop = set(boundary) | set(parameters)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen or name in stop or name not in producer:
            continue
        seen.add(name)
        needed.add(producer[name])
        pending.extend(i for i in nodes[producer[name]].input if i)
    return _WeightOnly(boundary, keys, needed)


def _producers(graph: Any, outputs: Sequence[str], stop: set[str]) -> set[int]:
    """The indices of the nodes `outputs` are computed by, back to `stop`."""
    nodes = list(graph.node)
    producer = {output: index for index, node in enumerate(nodes) for output in node.output}
    needed: set[int] = set()
    pending = list(outputs)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen or name in stop or name not in producer:
            continue
        seen.add(name)
        needed.add(producer[name])
        pending.extend(i for i in nodes[producer[name]].input if i)
    return needed


_SHARED_ARENA: list[bool] = []


def _share_cuda_arena(ort: Any) -> None:
    """Registers one CUDA arena for the process, growing only by what is
    asked (not to the next power of two); sessions opt in to it."""
    if _SHARED_ARENA:
        return
    memory = ort.OrtMemoryInfo(
        "Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT
    )
    arena = ort.OrtArenaCfg({"arena_extend_strategy": 1})
    ort.create_and_register_allocator_v2("CUDAExecutionProvider", memory, {}, arena)
    _SHARED_ARENA.append(True)


def _consumed(graph: Any) -> set[str]:
    """Every name a node of `graph` or of its subgraphs reads."""
    names: set[str] = set()
    for node in graph.node:
        names.update(node.input)
        for attribute in node.attribute:
            if attribute.g.node:
                names |= _consumed(attribute.g)
            for subgraph in attribute.graphs:
                names |= _consumed(subgraph)
    return names


def _encode(array: np.ndarray, element: int) -> np.ndarray:
    """An input as the graph's element type; bf16 as its rounded bits."""
    if element == 16:
        bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
        return (rounded >> 16).astype(np.uint16)
    return np.ascontiguousarray(array.astype(_NUMPY[element]))


def _decode(array: np.ndarray, element: int) -> np.ndarray:
    """A result as NumPy can hold it: bf16 bits widened to f32."""
    if element == 16:
        return (np.asarray(array).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return array


def _to_device(ort: Any, array: np.ndarray, element: int, device: str) -> Any:
    """`array` as an OrtValue on `device`; bf16 arrives as its uint16 bytes
    and is typed as bf16."""
    array = np.ascontiguousarray(array)
    if element != 16:
        return ort.OrtValue.ortvalue_from_numpy(array, device, 0)
    if device == "cpu":
        return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(array, 16)
    value = ort.OrtValue.ortvalue_from_shape_and_type(list(array.shape), 16, device, 0)
    value.update_inplace(array)
    return value


def load_model(
    source: str | Path,
    *,
    generics: Mapping[str, int | str],
    weights: str | Path,
    root: str | None = None,
    bindings: str | Path | None = None,
    std_root: str | Path | None = None,
    numerics: str = "fast",
    cast_dtype: bool = False,
    providers: Sequence[Any] | None = None,
) -> OnnxModel:
    """Every entry of the root block on ONNX Runtime over one copy of the
    weights, with its state kept on the device. `providers` defaults to CUDA
    where ONNX Runtime has it, else the CPU; it takes what `InferenceSession`
    does (names, or `(name, options)` pairs, TensorRT's among them).
    `cast_dtype=True` converts the checkpoint to the dtype the generics ask
    for (`T`)."""
    try:
        import onnxruntime  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
    except ImportError:
        raise LinnetError(
            "onnxruntime is not installed: pip install onnxruntime, or onnxruntime-gpu "
            "for CUDA and TensorRT"
        ) from None

    arguments = ["plan", "--no-optimize", *(["--root", root] if root is not None else [])]
    plan = json.loads(run_compiler(*arguments, *std_arguments(std_root), str(source)))
    if providers is None:
        available = onnxruntime.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
    options = {
        "generics": dict(generics),
        "weights": weights,
        "root": root,
        "bindings": bindings,
        "std_root": std_root,
        "numerics": numerics,
        "cast_dtype": cast_dtype,
    }
    return OnnxModel(Path(source), plan, options, providers)

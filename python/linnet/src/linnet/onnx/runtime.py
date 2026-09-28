"""Every entry of a Linnet block on ONNX Runtime, over one copy of the weights.

`load_model` is to ONNX Runtime what `linnet.jax.load_model` is to XLA: each
entry is exported (`linnet onnx`) for the shapes it is called with and run by
its own inference session, but the weights are not embedded in any of them.
They are placed on the device once, as `OrtValue`s, and bound to every
session by I/O binding; the block's `state` (a decoder's KV caches) stays on
the device between calls, each call's new state bound as the next call's
input.

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
_NUMPY = {1: np.float32, 10: np.float16, 16: np.uint16, 6: np.int32, 7: np.int64, 9: np.bool_}


class OnnxModel:
    """A root block's entries on ONNX Runtime, sharing weights and state.

    `run_entry(name, inputs)` runs one entry on NumPy inputs and returns its
    results as NumPy arrays (one result bare, several as a tuple). `state`
    holds the block's state members by path, on the device; `reset_state()`
    clears it.
    """

    def __init__(
        self,
        source: Path,
        plan: dict[str, Any],
        options: dict[str, Any],
        providers: Sequence[str],
    ) -> None:
        import onnxruntime  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]

        self._ort = onnxruntime
        self._source = source
        self.plan = plan
        self._options = options
        self.providers = list(providers)
        self._device = "cuda" if self.providers[0] == "CUDAExecutionProvider" else "cpu"
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
        self.state: dict[str, Any] = {}  # path -> OrtValue on the device

    # ---- entries

    def run_entry(self, name: str, inputs: Sequence[Any]) -> Any:
        if name not in self._signatures:
            raise LinnetError(f"the block has no entry `{name}`")
        arrays = [np.asarray(value) for value in inputs]
        bindings = self._bindings(name, arrays)
        key = (name, tuple(sorted(bindings.items())))
        if key not in self._sessions:
            self._sessions[key] = self._session(name, bindings)
        session = self._sessions[key]
        binding = session.session.io_binding()
        for port, array in zip(session.inputs, arrays, strict=True):
            binding.bind_cpu_input(port, np.ascontiguousarray(array.astype(session.dtypes[port])))
        for port, path in session.parameters.items():
            binding.bind_ortvalue_input(port, self._weights[path])
        for port, (path, shape, element) in session.states.items():
            if path not in self.state:
                zeros = np.zeros(shape, dtype=_NUMPY[element])
                self.state[path] = _to_device(self._ort, zeros, element, self._device)
            binding.bind_ortvalue_input(port, self.state[path])
        for port in session.results:
            binding.bind_output(port, "cpu")
        for port in session.next_states:
            binding.bind_output(port, self._device, 0)
        session.session.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        results = [outputs[i].numpy() for i in range(len(session.results))]
        for i, path in enumerate(session.next_states.values()):
            self.state[path] = outputs[len(session.results) + i]
        return results[0] if len(results) == 1 else tuple(results)

    def reset_state(self) -> None:
        self.state = {}

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
        for port, path in parameters.items():
            if path not in self._weights:
                tensor = exported.weights[port]
                array = np.frombuffer(tensor.raw_data, dtype=_NUMPY[tensor.data_type]).reshape(
                    tuple(tensor.dims)
                )
                self._weights[path] = _to_device(self._ort, array, tensor.data_type, self._device)
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
            if i.name not in parameters and i.name not in states
        ]
        settings = self._ort.SessionOptions()
        settings.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = self._ort.InferenceSession(
            exported.model.SerializeToString(), settings, providers=self.providers
        )
        dtypes = {port: _NUMPY[graph_inputs[port].elem_type] for port in inputs}
        return _Session(session, inputs, dtypes, parameters, states, results, next_states)

    def _bindings(self, name: str, inputs: Sequence[np.ndarray]) -> dict[str, int]:
        """The entry's own generics, from the shapes of its inputs."""
        arguments = self._signatures[name]["body"]["args"][1:]
        if len(arguments) != len(inputs):
            raise LinnetError(f"entry `{name}` takes {len(arguments)} inputs, got {len(inputs)}")
        bindings: dict[str, int] = {}
        for argument, value in zip(arguments, inputs, strict=True):
            declared = argument["type"]
            if declared.get("kind") != "tensor":
                continue
            for unit, size in zip(declared["shape"], value.shape, strict=True):
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
        dtypes: dict[str, Any],
        parameters: dict[str, str],
        states: dict[str, tuple[str, tuple[int, ...], int]],
        results: list[str],
        next_states: dict[str, str],
    ) -> None:
        self.session = session
        self.inputs = inputs
        self.dtypes = dtypes
        self.parameters = parameters  # graph input -> parameter path
        self.states = states  # graph input -> (state path, shape, element type)
        self.results = results  # graph outputs, in order
        self.next_states = next_states  # graph output -> state path


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
    providers: Sequence[str] | None = None,
) -> OnnxModel:
    """Every entry of the root block on ONNX Runtime over one copy of the
    weights, with its state kept on the device. `providers` defaults to CUDA
    where ONNX Runtime has it, else the CPU. `cast_dtype=True` converts the
    checkpoint to the dtype the generics ask for (`T`)."""
    import onnxruntime  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]

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

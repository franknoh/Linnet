"""Linnet's runtime, from Python (include/linnet/runtime/runtime.h).

A `Runtime` owns one device's memory and stream. A `NativePlan` runs calls of
executables over tensors in its arena, in order; on CUDA it can capture them
into one graph and replay it. An executable is native (a kernel the runtime
launches) or a Python `callback`.
"""

from __future__ import annotations

import ctypes
import functools
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .compiler import LinnetError, find_compiler

OK, ERROR, UNAVAILABLE = 0, 1, 2
CPU, CUDA = 1, 2
_KINDS = {"cpu": CPU, "cuda": CUDA}

# DLPack's type codes and widths, by Linnet dtype.
DTYPES: dict[str, tuple[int, int]] = {
    "bool": (6, 8),
    "i8": (0, 8),
    "i16": (0, 16),
    "i32": (0, 32),
    "i64": (0, 64),
    "u8": (1, 8),
    "u16": (1, 16),
    "u32": (1, 32),
    "u64": (1, 64),
    "f16": (2, 16),
    "bf16": (4, 16),
    "f32": (2, 32),
    "f64": (2, 64),
}
_NAMES = {code_bits: name for name, code_bits in DTYPES.items()}


class NativeError(LinnetError):
    """The runtime refused a call."""


class UnavailableError(NativeError):
    """What was asked for has no runtime on this device or platform."""


class _Device(ctypes.Structure):
    _fields_ = (("type", ctypes.c_int32), ("id", ctypes.c_int32))


class _DataType(ctypes.Structure):
    _fields_ = (("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16))


class _Tensor(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int32),
        ("dtype", _DataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    )


class _Context(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("device", _Device),
        ("stream", ctypes.c_void_p),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_size_t),
        ("capturing", ctypes.c_int32),
    )


class _Argument(ctypes.Structure):
    _fields_ = (
        ("kind", ctypes.c_int32),
        ("index", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("size", ctypes.c_uint32),
    )


_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_Tensor),
    ctypes.c_size_t,
    ctypes.POINTER(_Tensor),
    ctypes.c_size_t,
    ctypes.POINTER(_Context),
)

_LIBRARY = {"win32": "linnet_runtime.dll", "darwin": "liblinnet_runtime.dylib"}.get(
    sys.platform, "liblinnet_runtime.so"
)


def find_library() -> Path:
    """The runtime library: LINNET_RUNTIME, then the one beside the compiler
    (a build tree) or installed with it (`share/linnet/lib`, where the wheel
    carries it too)."""
    candidate = os.environ.get("LINNET_RUNTIME")
    if candidate and Path(candidate).is_file():
        return Path(candidate)
    compiler = Path(find_compiler()).resolve().parent
    for directory in (compiler, compiler.parent / "share" / "linnet" / "lib"):
        if (directory / _LIBRARY).is_file():
            return directory / _LIBRARY
    raise LinnetError(f"cannot find {_LIBRARY}; set LINNET_RUNTIME to its path")


@functools.cache
def library() -> ctypes.CDLL:
    """The runtime library, loaded once, its functions typed."""
    lib = ctypes.CDLL(str(find_library()))
    status = ctypes.c_int32
    pointer = ctypes.c_void_p
    signatures: dict[str, tuple[type[object] | None, list[type[object]]]] = {
        "ln_last_error": (ctypes.c_char_p, []),
        "ln_runtime_create": (status, [_Device, ctypes.POINTER(pointer)]),
        "ln_runtime_destroy": (None, [pointer]),
        "ln_runtime_stream": (pointer, [pointer]),
        "ln_alloc": (status, [pointer, ctypes.c_size_t, ctypes.POINTER(pointer)]),
        "ln_free": (status, [pointer, pointer]),
        "ln_plan_create": (status, [pointer, ctypes.c_size_t, ctypes.POINTER(pointer)]),
        "ln_plan_destroy": (None, [pointer]),
        "ln_plan_arena": (pointer, [pointer]),
        "ln_plan_add": (
            status,
            [
                pointer,
                pointer,
                ctypes.POINTER(_Tensor),
                ctypes.c_size_t,
                ctypes.POINTER(_Tensor),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
            ],
        ),
        "ln_plan_run": (status, [pointer]),
        "ln_plan_capture": (status, [pointer]),
        "ln_plan_synchronize": (status, [pointer]),
        "ln_callback_create": (status, [_CALLBACK, pointer, ctypes.POINTER(pointer)]),
        "ln_kernel_create": (
            status,
            [
                pointer,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
                ctypes.POINTER(_Argument),
                ctypes.c_size_t,
                ctypes.POINTER(pointer),
            ],
        ),
        "ln_cublaslt_linear_create": (
            status,
            [
                pointer,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                _DataType,
                ctypes.POINTER(pointer),
            ],
        ),
        "ln_executable_prepare": (status, [pointer, ctypes.POINTER(_Context)]),
        "ln_executable_execute": (
            status,
            [
                pointer,
                ctypes.POINTER(_Tensor),
                ctypes.c_size_t,
                ctypes.POINTER(_Tensor),
                ctypes.c_size_t,
                ctypes.POINTER(_Context),
            ],
        ),
        "ln_executable_destroy": (None, [pointer]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(lib, name)
        function.restype = result
        function.argtypes = arguments
    return lib


def _check(code: int, what: str) -> None:
    if code == OK:
        return
    message: bytes | None = library().ln_last_error()
    text = f"{what}: {(message or b'').decode()}"
    raise (UnavailableError if code == UNAVAILABLE else NativeError)(text)


@dataclass(frozen=True, slots=True)
class TensorRef:
    """A tensor in memory: its first element's address, shape, dtype, the
    device it is on (`cpu` or `cuda`, and an index), and its strides in
    elements (None: contiguous)."""

    address: int
    shape: tuple[int, ...]
    dtype: str
    device: tuple[str, int] = ("cpu", 0)
    strides: tuple[int, ...] | None = None

    @property
    def nbytes(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count * DTYPES[self.dtype][1] // 8


def _native(tensor: TensorRef, keep: list[object]) -> _Tensor:
    shape = (ctypes.c_int64 * len(tensor.shape))(*tensor.shape)
    keep.append(shape)
    strides = None
    if tensor.strides is not None:
        strides = (ctypes.c_int64 * len(tensor.strides))(*tensor.strides)
        keep.append(strides)
    code, bits = DTYPES[tensor.dtype]
    native = _Tensor()
    native.data = tensor.address
    native.device = _Device(_KINDS[tensor.device[0]], tensor.device[1])
    native.ndim = len(tensor.shape)
    native.dtype = _DataType(code, bits, 1)
    native.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
    if strides is not None:
        native.strides = ctypes.cast(strides, ctypes.POINTER(ctypes.c_int64))
    return native


def _view(native: _Tensor) -> TensorRef:
    rank = int(native.ndim)
    shape = tuple(int(native.shape[i]) for i in range(rank))
    strides = None if not native.strides else tuple(int(native.strides[i]) for i in range(rank))
    kind = "cuda" if native.device.type == CUDA else "cpu"
    return TensorRef(
        int(native.data or 0) + int(native.byte_offset),
        shape,
        _NAMES[(int(native.dtype.code), int(native.dtype.bits))],
        (kind, int(native.device.id)),
        strides,
    )


@dataclass(frozen=True, slots=True)
class CallContext:
    """What a callback runs with: its stream (0 on the CPU), its workspace,
    and whether a CUDA graph records the call."""

    stream: int
    workspace: int
    workspace_bytes: int
    capturing: bool


Callback = Callable[[list[TensorRef], list[TensorRef], CallContext], None]


class Executable:
    """An executable the runtime made; `close` frees it."""

    def __init__(self, handle: int, keep: Sequence[object] = ()) -> None:
        self.handle = handle
        self._keep = list(keep)
        self.failure: BaseException | None = None

    def prepare(self, device: tuple[str, int], stream: int = 0) -> None:
        """Runs the executable's `prepare` on `stream`."""
        context = _context(device, stream, 0, 0)
        _check(library().ln_executable_prepare(self.handle, ctypes.byref(context)), "prepare")

    def execute(
        self,
        inputs: Sequence[TensorRef],
        outputs: Sequence[TensorRef],
        stream: int = 0,
        workspace: tuple[int, int] = (0, 0),
    ) -> None:
        """Enqueues one call on `stream` (0: the CPU, or CUDA's default),
        outside any plan."""
        keep: list[object] = []
        native_in = (_Tensor * len(inputs))(*(_native(t, keep) for t in inputs))
        native_out = (_Tensor * len(outputs))(*(_native(t, keep) for t in outputs))
        device = (inputs or outputs)[0].device if (inputs or outputs) else ("cpu", 0)
        context = _context(device, stream, *workspace)
        code = library().ln_executable_execute(
            self.handle, native_in, len(inputs), native_out, len(outputs), ctypes.byref(context)
        )
        failure, self.failure = self.failure, None
        if failure is not None:
            raise NativeError("execute: a callback raised") from failure
        _check(code, "execute")

    def close(self) -> None:
        if self.handle:
            library().ln_executable_destroy(self.handle)
            self.handle = 0


def _context(
    device: tuple[str, int], stream: int, workspace: int, workspace_bytes: int
) -> _Context:
    context = _Context()
    context.abi_version = 1
    context.device = _Device(_KINDS[device[0]], device[1])
    context.stream = stream or None
    context.workspace = workspace or None
    context.workspace_bytes = workspace_bytes
    context.capturing = 0
    return context


def callback(function: Callback) -> Executable:
    """An executable that runs `function(inputs, outputs, context)`, which
    writes its outputs in place. An exception it raises fails the run, and
    the run raises it again."""
    holder: list[Executable] = []

    def run(
        user: int | None,
        inputs: ctypes._Pointer[_Tensor],  # pyright: ignore[reportPrivateUsage]
        input_count: int,
        outputs: ctypes._Pointer[_Tensor],  # pyright: ignore[reportPrivateUsage]
        output_count: int,
        context: ctypes._Pointer[_Context],  # pyright: ignore[reportPrivateUsage]
    ) -> int:
        try:
            native = context.contents
            function(
                [_view(inputs[i]) for i in range(input_count)],
                [_view(outputs[i]) for i in range(output_count)],
                CallContext(
                    int(native.stream or 0),
                    int(native.workspace or 0),
                    int(native.workspace_bytes),
                    bool(native.capturing),
                ),
            )
        except BaseException as error:  # carried across the C boundary, raised again by the run
            holder[0].failure = error
            return 1
        return 0

    trampoline = _CALLBACK(run)
    handle = ctypes.c_void_p()
    _check(library().ln_callback_create(trampoline, None, ctypes.byref(handle)), "callback")
    executable = Executable(handle.value or 0, [trampoline])
    holder.append(executable)
    return executable


@dataclass(frozen=True, slots=True)
class Argument:
    """A kernel argument: an input's or an output's address (`index`), or
    up to 8 bytes passed as they are (`value`, `size`)."""

    kind: str  # `input`, `output`, `bytes`
    index: int = 0
    value: int = 0
    size: int = 0


class Runtime:
    """One device's memory and stream."""

    def __init__(self, kind: str = "cpu", index: int = 0) -> None:
        handle = ctypes.c_void_p()
        _check(
            library().ln_runtime_create(_Device(_KINDS[kind], index), ctypes.byref(handle)),
            f"a runtime for {kind}:{index}",
        )
        self.handle: int = handle.value or 0
        self.device = (kind, index)

    @property
    def stream(self) -> int:
        """The runtime's stream (a CUstream), 0 on the CPU."""
        return int(library().ln_runtime_stream(self.handle) or 0)

    def alloc(self, nbytes: int) -> int:
        pointer = ctypes.c_void_p()
        _check(library().ln_alloc(self.handle, nbytes, ctypes.byref(pointer)), "alloc")
        return pointer.value or 0

    def free(self, address: int) -> None:
        _check(library().ln_free(self.handle, address), "free")

    def plan(self, arena_bytes: int) -> NativePlan:
        return NativePlan(self, arena_bytes)

    def kernel(
        self,
        image: bytes,
        function: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        arguments: Sequence[Argument],
        shared_bytes: int = 0,
    ) -> Executable:
        """A kernel of a cubin or PTX `image`, launched natively."""
        kinds = {"input": 0, "output": 1, "bytes": 2}
        native = (_Argument * len(arguments))(
            *(_Argument(kinds[a.kind], a.index, a.value, a.size) for a in arguments)
        )
        handle = ctypes.c_void_p()
        _check(
            library().ln_kernel_create(
                self.handle,
                image,
                function.encode(),
                (ctypes.c_uint32 * 3)(*grid),
                (ctypes.c_uint32 * 3)(*block),
                shared_bytes,
                native,
                len(arguments),
                ctypes.byref(handle),
            ),
            f"kernel {function}",
        )
        return Executable(handle.value or 0)

    def cublaslt_linear(self, rows: int, n: int, k: int, dtype: str) -> Executable:
        """A linear layer's product by cuBLASLt: its output ([rows, n]) is its
        input ([rows, k]) times its weight ([n, k]) transposed."""
        code, bits = DTYPES[dtype]
        handle = ctypes.c_void_p()
        _check(
            library().ln_cublaslt_linear_create(
                self.handle, rows, n, k, _DataType(code, bits, 1), ctypes.byref(handle)
            ),
            f"cuBLASLt product of {rows}x{k} by {n}x{k}",
        )
        return Executable(handle.value or 0)

    def close(self) -> None:
        if self.handle:
            library().ln_runtime_destroy(self.handle)
            self.handle = 0


class NativePlan:
    """Calls of executables over tensors, run in order; `arena` is the
    address of the plan's own memory."""

    def __init__(self, runtime: Runtime, arena_bytes: int) -> None:
        handle = ctypes.c_void_p()
        _check(library().ln_plan_create(runtime.handle, arena_bytes, ctypes.byref(handle)), "plan")
        self.handle: int = handle.value or 0
        self.runtime = runtime
        self.arena = int(library().ln_plan_arena(self.handle) or 0)
        self._executables: list[Executable] = []

    def add(
        self,
        executable: Executable,
        inputs: Sequence[TensorRef],
        outputs: Sequence[TensorRef],
        workspace: tuple[int, int] = (0, 0),
    ) -> None:
        """Appends a call of `executable` over `inputs` and `outputs`; its
        workspace is the arena's `workspace[1]` bytes at `workspace[0]`."""
        keep: list[object] = []
        native_in = (_Tensor * len(inputs))(*(_native(t, keep) for t in inputs))
        native_out = (_Tensor * len(outputs))(*(_native(t, keep) for t in outputs))
        _check(
            library().ln_plan_add(
                self.handle,
                executable.handle,
                native_in,
                len(inputs),
                native_out,
                len(outputs),
                workspace[0],
                workspace[1],
            ),
            "plan.add",
        )
        self._executables.append(executable)

    def _raise_failure(self, code: int, what: str) -> None:
        for executable in self._executables:
            failure, executable.failure = executable.failure, None
            if failure is not None:
                raise NativeError(f"{what}: a callback raised") from failure
        _check(code, what)

    def run(self) -> None:
        """Enqueues every call (the captured graph, once captured)."""
        self._raise_failure(library().ln_plan_run(self.handle), "plan.run")

    def capture(self) -> None:
        """Records the calls into a CUDA graph that later runs replay."""
        self._raise_failure(library().ln_plan_capture(self.handle), "plan.capture")

    def synchronize(self) -> None:
        _check(library().ln_plan_synchronize(self.handle), "plan.synchronize")

    def close(self) -> None:
        if self.handle:
            library().ln_plan_destroy(self.handle)
            self.handle = 0

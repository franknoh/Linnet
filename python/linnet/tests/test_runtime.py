"""The runtime from Python: a CPU plan of callbacks over its arena."""

from __future__ import annotations

import ctypes
import struct
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from linnet.runtime import (
    Argument,
    Callback,
    CallContext,
    NativeError,
    Runtime,
    TensorRef,
    UnavailableError,
    callback,
)
from tests.helpers import requires_cuda

if TYPE_CHECKING:
    import torch


def _floats(tensor: TensorRef) -> npt.NDArray[np.float32]:
    count = tensor.nbytes // 4
    buffer = (ctypes.c_float * count).from_address(tensor.address)
    return np.frombuffer(buffer, dtype=np.float32).reshape(tensor.shape)


def _scale(factor: float) -> Callback:
    """A callback that writes its input times `factor` to its output."""

    def run(inputs: list[TensorRef], outputs: list[TensorRef], context: CallContext) -> None:
        assert context.stream == 0 and not context.capturing
        _floats(outputs[0])[:] = _floats(inputs[0]) * factor

    return run


def test_a_cpu_plan_runs_its_calls_in_order_over_its_arena() -> None:
    runtime = Runtime("cpu")
    assert runtime.stream == 0
    plan = runtime.plan(3 * 16)
    a, b, c = (TensorRef(plan.arena + 16 * i, (4,), "f32") for i in range(3))
    doubled = callback(_scale(2.0))
    tenfold = callback(_scale(10.0))
    plan.add(doubled, [a], [b])
    plan.add(tenfold, [b], [c])
    _floats(a)[:] = np.arange(4, dtype=np.float32)
    plan.run()
    plan.synchronize()
    assert _floats(c).tolist() == [0.0, 20.0, 40.0, 60.0]
    # A CPU plan has no CUDA graph to capture.
    with pytest.raises(UnavailableError):
        plan.capture()
    plan.close()
    doubled.close()
    tenfold.close()
    runtime.close()


def test_a_callback_sees_what_it_was_given() -> None:
    runtime = Runtime("cpu")
    plan = runtime.plan(64)
    seen: list[tuple[list[TensorRef], list[TensorRef], CallContext]] = []
    record = callback(lambda i, o, x: seen.append((i, o, x)))
    given = TensorRef(plan.arena, (2, 3), "bf16", strides=(1, 2))
    plan.add(record, [given], [], workspace=(16, 32))
    plan.run()
    ((inputs, outputs, context),) = seen
    assert inputs == [given] and outputs == []
    assert context.workspace == plan.arena + 16 and context.workspace_bytes == 32
    plan.close()
    record.close()
    runtime.close()


def test_an_exception_in_a_callback_fails_the_run_and_comes_back() -> None:
    def fail(inputs: list[TensorRef], outputs: list[TensorRef], context: CallContext) -> None:
        raise ValueError("no")

    runtime = Runtime("cpu")
    plan = runtime.plan(16)
    failing = callback(fail)
    plan.add(failing, [], [])
    with pytest.raises(NativeError) as raised:
        plan.run()
    assert isinstance(raised.value.__cause__, ValueError)
    # The runtime checks what a plan is given.
    with pytest.raises(NativeError, match="outside the arena"):
        plan.add(failing, [], [], workspace=(8, 16))
    plan.close()
    failing.close()
    runtime.close()


# out[i] = in[i] * factor for i < n, one thread an element.
_SCALE_PTX = b"""
.version 7.0
.target sm_70
.address_size 64

.visible .entry scale(.param .u64 in_ptr, .param .u64 out_ptr, .param .f32 factor, .param .u32 n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.param.f32 %f1, [factor];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f3, %f2, %f1;
    add.s64 %rd7, %rd4, %rd5;
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
\0"""


@requires_cuda
def test_a_cuda_plan_launches_kernels_and_replays_its_capture() -> None:
    import torch

    device = ("cuda", torch.cuda.current_device())
    runtime = Runtime(*device)
    assert runtime.stream != 0
    count = 1000
    plan = runtime.plan(3 * count * 4)
    a, b, c = (TensorRef(plan.arena + i * count * 4, (count,), "f32", device) for i in range(3))
    factor = struct.unpack("<I", struct.pack("<f", 3.0))[0]
    scale = runtime.kernel(
        _SCALE_PTX,
        "scale",
        ((count + 255) // 256, 1, 1),
        (256, 1, 1),
        [
            Argument("input", 0),
            Argument("output", 0),
            Argument("bytes", value=factor, size=4),
            Argument("bytes", value=count, size=4),
        ],
    )

    def add_one(inputs: list[TensorRef], outputs: list[TensorRef], context: CallContext) -> None:
        # PyTorch on the runtime's stream, into the output it was given.
        with torch.cuda.stream(torch.cuda.ExternalStream(context.stream)):
            torch.add(_cuda(inputs[0]), 1.0, out=_cuda(outputs[0]))

    plus = callback(add_one)
    plan.add(scale, [a], [b])
    plan.add(plus, [b], [c])
    _cuda(a).copy_(torch.arange(count, dtype=torch.float32))
    torch.cuda.synchronize()
    plan.run()
    plan.synchronize()
    expected = torch.arange(count, dtype=torch.float32, device="cuda") * 3 + 1
    torch.testing.assert_close(_cuda(c), expected)
    plan.capture()
    _cuda(a).fill_(2.0)
    torch.cuda.synchronize()
    plan.run()  # the graph, replayed
    plan.synchronize()
    torch.testing.assert_close(_cuda(c), torch.full_like(expected, 7.0))
    plan.close()
    scale.close()
    plus.close()
    runtime.close()


class _CudaArray:
    """A tensor of the runtime's, seen by PyTorch through
    `__cuda_array_interface__`."""

    def __init__(self, tensor: TensorRef) -> None:
        self.__cuda_array_interface__ = {
            "shape": tensor.shape,
            "typestr": "<f4",
            "data": (tensor.address, False),
            "version": 3,
            "strides": None,
        }


def _cuda(tensor: TensorRef) -> torch.Tensor:
    import torch

    return torch.as_tensor(_CudaArray(tensor), device="cuda")

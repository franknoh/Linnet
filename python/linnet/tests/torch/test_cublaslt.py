"""cuBLASLt as a backend: its product, and a plan around it captured as one
CUDA graph."""

from __future__ import annotations

from pathlib import Path

import torch

from linnet.backends import Context
from linnet.torch import load
from linnet.torch.compiled import CompiledLinnetModule
from linnet.torch.cublaslt import CublasLt, runtime_for, tensor_ref
from linnet.torch.planned import Captured
from tests.helpers import LLAMA, STDLIB, TINY_LLAMA, random_llama, requires_cuda


@requires_cuda
def test_the_product_is_a_linear_layers() -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(3, 64, device="cuda", generator=generator).bfloat16()
    weight = torch.randn(32, 64, device="cuda", generator=generator).bfloat16()
    out = torch.empty(3, 32, device="cuda", dtype=torch.bfloat16)
    product = runtime_for(torch.cuda.current_device()).cublaslt_linear(3, 32, 64, "bf16")
    product.execute(
        [tensor_ref(x), tensor_ref(weight)],
        [tensor_ref(out)],
        torch.cuda.current_stream().cuda_stream,
    )
    torch.testing.assert_close(out, torch.nn.functional.linear(x, weight))
    product.close()


@requires_cuda
def test_a_plan_with_cublaslt_replays_as_one_graph(tmp_path: Path) -> None:
    weights = random_llama(tmp_path / "model.safetensors")
    generics = {**TINY_LLAMA, "T": "bf16"}
    planned, direct = (
        load(
            LLAMA,
            generics=generics,
            std_root=STDLIB,
            weights=weights,
            compile=True,
            device="cuda",
            cast_dtype=True,
        )
        for _ in range(2)
    )
    assert isinstance(planned, CompiledLinnetModule)
    token = torch.tensor([[3]], dtype=torch.int32, device="cuda")
    plan = planned.plan_for(
        "decode", [token, torch.tensor(0, dtype=torch.int32, device="cuda")], backends=[CublasLt()]
    )
    # Each layer's q, k and v joined, its o, its gate and up joined, its
    # down; and the head.
    assert [step.backend for step in plan.steps].count("cublaslt") == 2 * 4 + 1
    parameters = {
        name.removeprefix("root."): parameter.detach()
        for name, parameter in planned.named_parameters()
    }
    runner = Captured(plan, Context(plan.target, parameters))
    with torch.no_grad():
        for pos in range(4):  # as it is, captured, then replayed twice
            step = torch.tensor(pos, dtype=torch.int32, device="cuda")
            (logits,) = runner([token, step])
            expected: object = direct.run_entry("decode", [token, step])
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(logits, expected, rtol=2e-2, atol=2e-2)

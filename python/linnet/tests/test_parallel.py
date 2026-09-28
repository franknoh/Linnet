"""Tensor parallelism: a decoder split across two devices computes what it
computes on one. JAX runs on two host devices XLA is told to simulate, in a
process of its own (the device count is fixed when JAX starts); PyTorch runs
two `gloo` processes with the weights and KV caches as DTensors, attention as
its canonical body (PyTorch's CPU attention kernel has no DTensor rule)."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # type: ignore[import-untyped]

from .test_serve import GENERICS, SOURCE

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "serve.linnet"
    source.write_text(SOURCE, encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "embedding.weight": torch.randn(64, 32, generator=generator),
        "positions.weight": torch.randn(48, 32, generator=generator),
        "head.weight": torch.randn(64, 32, generator=generator),
    }
    for name in ("q_proj", "k_proj", "v_proj"):
        tensors[f"{name}.weight"] = torch.randn(32, 32, generator=generator) * 0.3
    weights = tmp_path / "model.safetensors"
    save_file(tensors, str(weights))
    return source, weights


JAX_SCRIPT = """
import os, sys
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp
from linnet.jax import load_model
source, weights, std, generics = sys.argv[1], sys.argv[2], sys.argv[3], eval(sys.argv[4])
kw = dict(generics=generics, weights=weights, std_root=std)
one, two = load_model(source, **kw), load_model(source, mesh=2, **kw)
assert tuple(two.weights["q_proj.weight"].sharding.spec)[0] == "model"
tok = jnp.asarray([[3, 1, 4, 1, 5, 9, 2, 6]], jnp.int32)
forward = one.run_entry("forward", [tok]) - two.run_entry("forward", [tok])
assert float(jnp.abs(forward).max()) < 1e-4
prompts = [jnp.asarray([[3, 1, 4, 1], [5, 9, 2, 6], [2, 7, 1, 8]], jnp.int32),
           jnp.asarray([0, 1, 2], jnp.int32), jnp.asarray([4, 3, 2], jnp.int32)]
step = [jnp.asarray([[7], [8], [9]], jnp.int32), jnp.asarray([4, 3, 2], jnp.int32)]
for m in (one, two):
    m.run_entry("prefill_slots", prompts)
# One model's state lives on one device, the other's on the mesh: compare
# on the host.
import numpy as np
single, split = (np.asarray(m.run_entry("decode_rows", step)) for m in (one, two))
diff = np.abs(single - split).max()
assert float(diff) < 1e-4
assert "'model'" in str(two.state["cache_k"].sharding.spec)
print("ok")
"""


def test_jax_on_two_devices(files: tuple[Path, Path]) -> None:
    pytest.importorskip("jax")
    source, weights = files
    completed = subprocess.run(
        [sys.executable, "-c", JAX_SCRIPT, str(source), str(weights), str(STDLIB), repr(GENERICS)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
        timeout=600,
    )
    assert completed.returncode == 0 and "ok" in completed.stdout, completed.stderr[-2000:]


def _torch_rank(rank: int, world: int, port: int, source: str, weights: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from linnet.torch import load

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        kw = {
            "generics": GENERICS,
            "std_root": STDLIB,
            "weights": weights,
            "compile": True,
            "numerics": "exact",
        }
        one = load(source, **kw)
        two = load(source, tensor_parallel=mesh, **kw)
        split = dict(two.named_parameters())["root.q_proj.weight"]
        assert tuple(split.to_local().shape) == (32 // world, 32)  # pyright: ignore[reportAttributeAccessIssue]
        tokens = torch.tensor([[3, 1, 4, 1, 5, 9]], dtype=torch.int32)
        torch.testing.assert_close(
            two.run_entry("forward", [tokens]),
            one.run_entry("forward", [tokens]),
            atol=1e-4,
            rtol=1e-4,
        )

        def ints(values: list[object]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32)

        for model in (one, two):
            model.run_entry(
                "prefill_slots",
                [
                    ints([[3, 1, 4, 1], [5, 9, 2, 6], [2, 7, 1, 8]]),
                    ints([0, 1, 2]),
                    ints([4, 3, 2]),
                ],
            )
        step = [ints([[7], [8], [9]]), ints([4, 3, 2])]
        torch.testing.assert_close(
            two.run_entry("decode_rows", step),
            one.run_entry("decode_rows", step),
            atol=1e-4,
            rtol=1e-4,
        )
    finally:
        dist.destroy_process_group()


def test_torch_on_two_processes(files: tuple[Path, Path]) -> None:
    source, weights = files
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(_torch_rank, args=(2, port, str(source), str(weights)), nprocs=2)

"""Benchmarks the Linnet backends against hand-written PyTorch.

For each model configuration the harness times a PyTorch reference (eager and
`torch.compile`d), the Linnet source materialized as a `torch.nn.Module` with
native kernels (`numerics="equivalent"`, and the `"fast"` tier that skips f32
accumulation), and the same source compiled through StableHLO and run by XLA
via `linnet.jax`. Outputs are compared against the reference so every row
carries a `max |Δ|`. The result is a JSON document the
documentation site renders (`site/benchmarks.md`).

    LINNET_BIN=build/release/linnet python bench/run.py --device cuda \\
        --configs small,medium --out bench/results/latest.json

Run it from an environment with `linnet-lang[torch]` installed and, for the
XLA rows, the `jax` extra with a `jax` that sees the same device.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
STDLIB = REPO / "stdlib"
LLAMA = REPO / "examples/05-llama/src/lib.linnet"

# Generated-source rows: `torch.compile` backend (None runs the source as is),
# numerics tier, and the label. The fast tier runs softmax, normalization,
# and attention in the input dtype, as the PyTorch reference does.
GENERATED_VARIANTS: tuple[tuple[str | None, str, str], ...] = (
    (None, "equivalent", "generated source"),
    ("inductor", "equivalent", "generated source + torch.compile"),
    ("inductor", "fast", "generated source + torch.compile, numerics=fast"),
    ("reduce-overhead", "fast", "generated source + CUDA graphs, numerics=fast"),
)

# Llama-shaped configurations: hidden width, query heads, key/value heads,
# MLP width, depth, vocabulary, and the sequence the forward pass runs over.
CONFIGS: dict[str, dict[str, int]] = {
    "tiny": dict(H=64, Heads=4, KvHeads=2, Inner=128, Layers=2, Vocab=256, S=16),
    "small": dict(H=512, Heads=8, KvHeads=8, Inner=1376, Layers=8, Vocab=32000, S=512),
    "medium": dict(H=2048, Heads=32, KvHeads=8, Inner=5632, Layers=22, Vocab=32000, S=512),
    "large": dict(H=4096, Heads=32, KvHeads=8, Inner=14336, Layers=32, Vocab=32000, S=512),
}


@dataclass
class Variant:
    name: str
    latency_ms: float | None
    throughput: float | None
    max_abs_diff: float | None
    note: str = ""
    kernels: int | None = None  # CUDA kernels launched by one call, when profiled

    def __post_init__(self) -> None:
        # `time_call` profiles the call it just timed; the variant built right
        # after it takes that count.
        global _last_kernels
        if self.kernels is None and self.latency_ms is not None:
            self.kernels = _last_kernels
        _last_kernels = None


_last_kernels: int | None = None


@dataclass
class Run:
    model: str
    config: str
    entry: str
    unit: str
    variants: list[Variant] = field(default_factory=list)


# ---------------------------------------------------------------- reference


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def rope_tables(seq: int, dim: int, device: torch.device, dtype: torch.dtype) -> tuple[Any, Any]:
    positions = torch.arange(seq, dtype=torch.float32, device=device)[:, None]
    inv_freq = torch.exp(
        -(torch.arange(0, dim // 2, dtype=torch.float32, device=device) * 2.0 / dim)
        * torch.log(torch.tensor(500000.0, device=device))
    )
    angles = positions * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype)
    return cos, sin


def llama_reference(cfg: dict[str, int], w: dict[str, torch.Tensor]) -> Callable[..., torch.Tensor]:
    """`forward(tokens) -> logits`, the example's architecture in plain PyTorch."""
    H, heads, kv_heads, layers = cfg["H"], cfg["Heads"], cfg["KvHeads"], cfg["Layers"]  # noqa: N806
    D = H // heads  # noqa: N806

    def forward(tokens: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape  # noqa: N806
        embedding = w["embedding.weight"]
        cos, sin = rope_tables(S, D, tokens.device, embedding.dtype)

        def rope(x: torch.Tensor) -> torch.Tensor:
            first, second = x[..., : D // 2], x[..., D // 2 :]
            return x * cos + torch.cat([-second, first], dim=-1) * sin

        def split(x: torch.Tensor, n: int) -> torch.Tensor:
            return x.reshape(B, S, n, D).permute(0, 2, 1, 3)

        x = embedding[tokens.long()]
        for i in range(layers):
            p = f"layers.{i}."
            a = p + "attention."
            h = rms_norm(x, w[p + "attention_norm.weight"])
            q = rope(split(h @ w[a + "q_proj.weight"].T, heads))
            k = rope(split(h @ w[a + "k_proj.weight"].T, kv_heads))
            v = split(h @ w[a + "v_proj.weight"].T, kv_heads)
            k = k.repeat_interleave(heads // kv_heads, dim=1)
            v = v.repeat_interleave(heads // kv_heads, dim=1)
            mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=D**-0.5)
            x = x + mixed.permute(0, 2, 1, 3).reshape(B, S, H) @ w[a + "o_proj.weight"].T
            h = rms_norm(x, w[p + "mlp_norm.weight"])
            gate, up = h @ w[p + "mlp.gate.weight"].T, h @ w[p + "mlp.up.weight"].T
            x = x + (F.silu(gate) * up) @ w[p + "mlp.down.weight"].T
        return rms_norm(x, w["norm.weight"]) @ w["lm_head.weight"].T

    return forward


# ------------------------------------------------------------------ timing


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn: Callable[[], Any], device: torch.device, warmup: int, iters: int) -> float:
    """Median wall time of `fn` in milliseconds, device-synchronized. On CUDA
    the call is also profiled once for the number of kernels it launches."""
    global _last_kernels
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)
    _last_kernels = count_kernels(fn, device)
    return statistics.median(samples)


def count_kernels(fn: Callable[[], Any], device: torch.device) -> int | None:
    """How many CUDA kernels one call launches (None off CUDA or if profiling fails)."""
    if device.type != "cuda":
        return None
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as trace:
            fn()
            synchronize(device)
        return sum(1 for event in trace.events() if str(event.device_type).endswith("CUDA"))
    except Exception:  # profiling is a diagnostic; its failure is not a result
        return None


def diff(actual: Any, expected: torch.Tensor) -> float:
    a = torch.as_tensor(actual).float().cpu() if not isinstance(actual, torch.Tensor) else actual
    return float((a.float().cpu() - expected.float().cpu()).abs().max())


def compiler() -> str:
    return os.environ.get("LINNET_BIN", "linnet")


# ------------------------------------------------------------------- bench


def random_weights(
    skeleton: torch.nn.Module, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """A tensor per parameter path, skipping the standard library's optional
    biases so the reference and the Linnet model see the same parameters."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in skeleton.named_parameters():
        path = name.removeprefix("root.")
        if path.endswith(".bias"):
            continue
        weights[path] = (torch.randn(parameter.shape, generator=generator) * 0.02).to(device, dtype)
    return weights


def bench_config(
    name: str,
    cfg: dict[str, int],
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    timings: list[dict[str, Any]],
    with_xla: bool,
) -> list[Run]:
    from linnet.torch import load

    dtype_name = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}[dtype]
    B, S = 1, cfg["S"]  # noqa: N806
    generics: dict[str, int | str] = {
        key: cfg[key] for key in ("Vocab", "H", "Heads", "KvHeads", "Inner", "Layers")
    }
    generics.update({"Batch": B, "MaxSeq": S, "T": dtype_name})
    label = f"H={cfg['H']} L={cfg['Layers']} heads={cfg['Heads']}/{cfg['KvHeads']} B={B} S={S} {dtype_name}"
    print(f"== {name}: {label}", flush=True)

    start = time.perf_counter()
    skeleton = load(LLAMA, generics=generics, std_root=STDLIB, device=device)
    timings.append({"name": f"linnet.torch.load ({name})", "seconds": time.perf_counter() - start})
    weights = random_weights(skeleton, device, dtype)
    linnet_model = load(
        LLAMA, generics=generics, std_root=STDLIB, device=device, numerics="equivalent"
    )
    state = {f"root.{path}": tensor for path, tensor in weights.items()}
    linnet_model.load_state_dict(state, strict=False)
    for module in linnet_model.modules():
        # Optional biases stay absent, as in the reference.
        if hasattr(module, "absent_params"):
            module.absent_params = set(module.optional_params)

    tokens = torch.randint(0, cfg["Vocab"], (B, S), dtype=torch.int32, device=device)
    reference = llama_reference(cfg, weights)
    with torch.no_grad():
        expected = reference(tokens)
    runs: list[Run] = []

    # ---- forward over the sequence
    forward = Run("Llama-style decoder", label, "forward", "tokens/s")
    with torch.no_grad():
        ms = time_call(lambda: reference(tokens), device, warmup, iters)
    forward.variants.append(Variant("PyTorch reference (eager)", ms, B * S / ms * 1e3, 0.0))

    try:
        compiled = torch.compile(reference)
        with torch.no_grad():
            start = time.perf_counter()
            out = compiled(tokens)
            synchronize(device)
            timings.append(
                {"name": f"torch.compile ({name})", "seconds": time.perf_counter() - start}
            )
            ms = time_call(lambda: compiled(tokens), device, warmup, iters)
        forward.variants.append(
            Variant("PyTorch reference (torch.compile)", ms, B * S / ms * 1e3, diff(out, expected))
        )
    except Exception as error:  # noqa: BLE001 - reported in the table
        forward.variants.append(
            Variant("PyTorch reference (torch.compile)", None, None, None, str(error)[:120])
        )

    with torch.no_grad():
        out = linnet_model(tokens)
        ms = time_call(lambda: linnet_model(tokens), device, warmup, iters)
    forward.variants.append(
        Variant(
            "Linnet → PyTorch (numerics=equivalent)",
            ms,
            B * S / ms * 1e3,
            diff(out, expected),
            "Core IR interpreted per call; library ops on native kernels",
        )
    )

    for backend, numerics, label_suffix in GENERATED_VARIANTS:
        try:
            start = time.perf_counter()
            generated = load(
                LLAMA,
                generics=generics,
                std_root=STDLIB,
                device=device,
                numerics=numerics,
                compile=backend or True,
            )
            generated.load_state_dict(state, strict=False)
            for module in generated.modules():
                if hasattr(module, "absent_params"):
                    module.absent_params = set(module.optional_params)
            with torch.no_grad():
                out = generated(tokens)
                synchronize(device)
                timings.append(
                    {
                        "name": f"linnet torch + first call ({name}, {label_suffix})",
                        "seconds": time.perf_counter() - start,
                    }
                )
                ms = time_call(lambda: generated(tokens), device, warmup, iters)
            forward.variants.append(
                Variant(
                    f"Linnet → PyTorch ({label_suffix})",
                    ms,
                    B * S / ms * 1e3,
                    diff(out, expected),
                    "straight-line PyTorch from `linnet torch`, native kernels",
                )
            )
        except Exception as error:  # noqa: BLE001 - reported in the table
            forward.variants.append(
                Variant(f"Linnet → PyTorch ({label_suffix})", None, None, None, str(error)[:120])
            )
    for numerics in ("equivalent", "fast") if with_xla else ():
        forward.variants.append(
            xla_variant(
                generics,
                weights,
                [tokens],
                expected,
                "forward",
                numerics,
                device,
                warmup,
                iters,
                timings,
                name,
                B * S,
            )
        )
    runs.append(forward)

    # ---- decode: one token per step through the KV caches
    decode = Run("Llama-style decoder", label, "decode", "steps/s")
    half = S // 2
    prefix = tokens[:, : half + 1]
    with torch.no_grad():
        expected_step = reference(prefix)[:, -1, :]
        ms = time_call(lambda: reference(prefix), device, warmup, iters)
    decode.variants.append(
        Variant(
            "PyTorch reference (eager, recomputes prefix)",
            ms,
            1e3 / ms,
            0.0,
            f"no KV cache: full forward over {half + 1} tokens",
        )
    )

    def linnet_decode() -> torch.Tensor:
        return linnet_model.run_entry(
            "decode",
            [tokens[:, half : half + 1], torch.tensor(half, dtype=torch.int32, device=device)],
        )

    with torch.no_grad():
        linnet_model.reset_state()
        for pos in range(half):
            linnet_model.run_entry(
                "decode",
                [tokens[:, pos : pos + 1], torch.tensor(pos, dtype=torch.int32, device=device)],
            )
        out = linnet_decode()
        ms = time_call(linnet_decode, device, warmup, iters)
    decode.variants.append(
        Variant(
            "Linnet → PyTorch (numerics=equivalent)",
            ms,
            1e3 / ms,
            diff(out, expected_step),
            f"KV cache at position {half} of {S}",
        )
    )
    for backend, numerics, label_suffix in GENERATED_VARIANTS:
        try:
            generated = load(
                LLAMA,
                generics=generics,
                std_root=STDLIB,
                device=device,
                numerics=numerics,
                compile=backend or True,
            )
            generated.load_state_dict(state, strict=False)
            for module in generated.modules():
                if hasattr(module, "absent_params"):
                    module.absent_params = set(module.optional_params)

            def generated_decode() -> torch.Tensor:
                return generated.run_entry(
                    "decode",
                    [
                        tokens[:, half : half + 1],
                        torch.tensor(half, dtype=torch.int32, device=device),
                    ],
                )

            with torch.no_grad():
                for pos in range(half):
                    generated.run_entry(
                        "decode",
                        [
                            tokens[:, pos : pos + 1],
                            torch.tensor(pos, dtype=torch.int32, device=device),
                        ],
                    )
                out = generated_decode()
                ms = time_call(generated_decode, device, warmup, iters)
            decode.variants.append(
                Variant(
                    f"Linnet → PyTorch ({label_suffix})",
                    ms,
                    1e3 / ms,
                    diff(out, expected_step),
                    f"KV cache at position {half} of {S}",
                )
            )
        except Exception as error:  # noqa: BLE001 - reported in the table
            decode.variants.append(
                Variant(f"Linnet → PyTorch ({label_suffix})", None, None, None, str(error)[:120])
            )
    for numerics in ("equivalent", "fast") if with_xla else ():
        decode.variants.append(
            xla_decode_variant(
                generics,
                weights,
                tokens,
                half,
                expected_step,
                numerics,
                device,
                warmup,
                iters,
                timings,
                name,
            )
        )
    runs.append(decode)
    return runs


def jax_weights(weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    import jax.numpy as jnp

    out: dict[str, Any] = {}
    for path, tensor in weights.items():
        array = jnp.asarray(tensor.float().cpu().numpy())
        if tensor.dtype == torch.bfloat16:
            array = array.astype(jnp.bfloat16)
        elif tensor.dtype == torch.float16:
            array = array.astype(jnp.float16)
        out[path] = array
    return out


def xla_variant(
    generics: dict[str, int | str],
    weights: dict[str, torch.Tensor],
    inputs: list[torch.Tensor],
    expected: torch.Tensor,
    entry: str,
    numerics: str,
    device: torch.device,
    warmup: int,
    iters: int,
    timings: list[dict[str, Any]],
    name: str,
    tokens_per_call: int,
) -> Variant:
    try:
        import jax
        import jax.numpy as jnp

        from linnet.jax import load as load_jax

        function = load_jax(
            LLAMA,
            generics=generics,
            weights=jax_weights(weights),
            entry=entry,
            std_root=STDLIB,
            numerics=numerics,
        )
        arrays = [jnp.asarray(tensor.cpu().numpy()) for tensor in inputs]
        start = time.perf_counter()
        out = function(*arrays)
        jax.block_until_ready(out)
        timings.append(
            {
                "name": f"linnet stablehlo + XLA compile ({name}, {entry}, numerics={numerics})",
                "seconds": time.perf_counter() - start,
            }
        )

        def call() -> None:
            jax.block_until_ready(function(*arrays))

        ms = time_call(call, torch.device("cpu"), warmup, iters)
        result = torch.tensor(jax.device_get(out).astype("float32"))
        backend = jax.default_backend()
        tier = "" if numerics == "equivalent" else f", numerics={numerics}"
        return Variant(
            f"Linnet → XLA (jax, {backend}{tier})",
            ms,
            tokens_per_call / ms * 1e3,
            diff(result, expected),
            "whole-program compile of the StableHLO export",
        )
    except Exception as error:  # noqa: BLE001 - reported in the table
        return Variant("Linnet → XLA (jax)", None, None, None, str(error)[:120])


def xla_decode_variant(
    generics: dict[str, int | str],
    weights: dict[str, torch.Tensor],
    tokens: torch.Tensor,
    half: int,
    expected: torch.Tensor,
    numerics: str,
    device: torch.device,
    warmup: int,
    iters: int,
    timings: list[dict[str, Any]],
    name: str,
) -> Variant:
    try:
        import jax
        import jax.numpy as jnp

        from linnet.jax import load as load_jax

        function = load_jax(
            LLAMA,
            generics=generics,
            weights=jax_weights(weights),
            entry="decode",
            std_root=STDLIB,
            numerics=numerics,
        )
        ids = jnp.asarray(tokens.cpu().numpy())
        start = time.perf_counter()
        state: Any = None
        for pos in range(half):
            _, state = function(ids[:, pos : pos + 1], jnp.int32(pos), state=state)
        jax.block_until_ready(state)
        timings.append(
            {
                "name": f"linnet stablehlo + XLA compile ({name}, decode, numerics={numerics})"
                f" incl. {half} steps",
                "seconds": time.perf_counter() - start,
            }
        )
        out, _ = function(ids[:, half : half + 1], jnp.int32(half), state=state)

        def step() -> None:
            jax.block_until_ready(function(ids[:, half : half + 1], jnp.int32(half), state=state))

        ms = time_call(step, torch.device("cpu"), warmup, iters)
        result = torch.tensor(jax.device_get(out).astype("float32"))
        tier = "" if numerics == "equivalent" else f", numerics={numerics}"
        return Variant(
            f"Linnet → XLA (jax, {jax.default_backend()}{tier})",
            ms,
            1e3 / ms,
            diff(result, expected),
            f"KV cache threaded as inputs/outputs, position {half}",
        )
    except Exception as error:  # noqa: BLE001 - reported in the table
        return Variant("Linnet → XLA (jax)", None, None, None, str(error)[:120])


def environment(device: torch.device) -> dict[str, str]:
    env = {
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }
    try:
        import jax

        env["jax"] = f"{jax.__version__} ({jax.default_backend()})"
    except Exception:  # noqa: BLE001
        pass
    version = subprocess.run([compiler(), "--version"], capture_output=True, text=True, check=False)
    env["linnet"] = version.stdout.strip() or "unknown"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", default=None, help="f32, bf16, or f16 (default: bf16 on cuda, f32 on cpu)"
    )
    parser.add_argument(
        "--configs", default="small,medium", help="comma-separated subset of " + ",".join(CONFIGS)
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-xla", action="store_true", help="skip the linnet.jax rows")
    parser.add_argument("--out", type=Path, default=REPO / "bench/results/latest.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_name = args.dtype or ("bf16" if device.type == "cuda" else "f32")
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype_name]
    torch.manual_seed(0)

    timings: list[dict[str, Any]] = []
    start = time.perf_counter()
    check = subprocess.run(
        [compiler(), "check", "--std", str(STDLIB), str(LLAMA.parent.parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        raise SystemExit(check.stderr)
    timings.append(
        {"name": "linnet check examples/05-llama", "seconds": time.perf_counter() - start}
    )

    runs: list[Run] = []
    for name in args.configs.split(","):
        runs += bench_config(
            name, CONFIGS[name], device, dtype, args.warmup, args.iters, timings, not args.no_xla
        )

    result = {
        "measured_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        "environment": environment(device),
        "runs": [asdict(run) for run in runs],
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for run in runs:
        print(f"\n{run.model} · {run.config} · {run.entry}")
        for variant in run.variants:
            latency = "—" if variant.latency_ms is None else f"{variant.latency_ms:9.2f} ms"
            kernels = "" if variant.kernels is None else f"  {variant.kernels} kernels"
            print(f"  {variant.name:48s} {latency}{kernels}  Δ={variant.max_abs_diff}  {variant.note}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""The command line of `linnet memory` and `linnet fit`
(`python -m linnet.resources ...`).

linnet memory llama-3.1-8b-instruct --batch 8 --seq-len 8192
linnet memory model.linnet --bind H=1024 --training --optimizer adamw --checkpoint block
linnet fit llama-3.1-8b-instruct --device-memory 80GiB --seq-len 8192 --maximize batch
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .. import sizes
from ..compiler import LinnetError, parse_binding
from ..sizes import format_bytes
from .analysis import MemoryModel
from .config import ExecutionConfig, Numerics
from .kvcache import ContiguousLayout, KVLayout, PagedLayout
from .planner import ExecutionPlanner, ResourceConstraint, Target
from .result import format_result
from .training import OPTIMIZERS, CheckpointPolicy, TrainingConfig


def parse_size(text: str) -> int:
    """`48GiB`, `80GB`, `512MiB`, or bytes, as an argument."""
    try:
        return sizes.parse_size(text)
    except LinnetError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _binding(text: str) -> tuple[str, int | str]:
    try:
        return parse_binding(text)
    except LinnetError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _checkpoint(text: str) -> CheckpointPolicy:
    if text == "none":
        return CheckpointPolicy()
    if text in ("block", "blocks"):
        return CheckpointPolicy("blocks")
    return CheckpointPolicy("regions", tuple(p for p in text.split(",") if p))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model", help="a .linnet file, a Nest name, a Hub repo, or a model directory"
    )
    parser.add_argument("--entry", help="the entry to analyze (default: the only one, or forward)")
    parser.add_argument("--root", help="the root block of a .linnet file with several")
    parser.add_argument("--std", help="the standard library directory")
    parser.add_argument("--backend", choices=("cuda", "generic"), default="cuda")
    parser.add_argument("--numerics", choices=("exact", "equivalent", "fast"), default="fast")
    parser.add_argument("--dtype", help="the compute dtype (binds the model's float dtype generic)")
    parser.add_argument("--batch", type=int, help="the batch size (binds B and Batch)")
    parser.add_argument(
        "--seq-len", "--context", dest="context", type=int, help="the sequence length (binds S, P)"
    )
    parser.add_argument(
        "--cache-len",
        dest="cache",
        type=int,
        help="cache positions (binds MaxSeq; default: the sequence length)",
    )
    parser.add_argument("--bind", action="append", type=_binding, default=[], metavar="G=VALUE")
    parser.add_argument(
        "--optional",
        action="append",
        default=[],
        metavar="PATH",
        help="an optional parameter or block that is present",
    )
    parser.add_argument("--kv-dtype", help="the dtype the backend stores caches in")
    parser.add_argument("--kv-block-size", type=int, help="paged caches: positions per block")
    parser.add_argument(
        "--training", action="store_true", help="a training step of the entry (a loss)"
    )
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="adamw")
    parser.add_argument("--master-dtype", help="master weights' dtype, such as f32")
    parser.add_argument("--gradient-dtype", help="gradients' dtype")
    parser.add_argument(
        "--trainable",
        action="append",
        default=[],
        metavar="PATTERN",
        help="trainable parameter paths (default: all)",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_checkpoint,
        default=[],
        metavar="POLICY",
        help="none, block, or block paths such as layers.*; repeat to compare",
    )
    parser.add_argument(
        "--shards", type=int, default=1, help="FSDP: devices the training state is split across"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        metavar="N",
        help="processes each weight is split across (binds the model's Shards)",
    )
    parser.add_argument(
        "--context-bytes", type=parse_size, help="the CUDA context's size, if known"
    )
    parser.add_argument("--json", action="store_true", help="one JSON document instead of text")


def config_from_args(args: argparse.Namespace, checkpoint: CheckpointPolicy) -> ExecutionConfig:
    layout: KVLayout = (
        ContiguousLayout() if args.kv_block_size is None else PagedLayout(args.kv_block_size)
    )
    training = None
    if args.training:
        training = TrainingConfig(
            optimizer=OPTIMIZERS[args.optimizer],
            gradient_dtype=args.gradient_dtype,
            master_dtype=args.master_dtype,
            trainable=tuple(args.trainable) or ("*",),
            checkpoint=checkpoint,
            shards=args.shards,
        )
    numerics: Numerics = args.numerics
    return ExecutionConfig(
        backend=args.backend,
        entry=args.entry,
        numerics=numerics,
        dtype=args.dtype,
        kv_dtype=args.kv_dtype,
        kv_layout=layout,
        batch=args.batch,
        context=args.context,
        cache=args.cache,
        bindings=dict(args.bind),
        optionals=frozenset(args.optional),
        training=training,
        tensor_parallel=args.tensor_parallel,
        data_parallel=args.shards,
        sharding="fsdp" if args.shards > 1 else "none",
        context_bytes=args.context_bytes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linnet", description="Static memory analysis.")
    commands = parser.add_subparsers(dest="command", required=True)
    memory = commands.add_parser("memory", help="memory a configuration needs, before running it")
    add_arguments(memory)
    fit = commands.add_parser("fit", help="the largest batch, context or cache that fits a device")
    add_arguments(fit)
    fit.add_argument("--device-memory", type=parse_size, required=True, metavar="SIZE")
    fit.add_argument("--maximize", choices=("batch", "context", "kv-cache"), required=True)
    fit.add_argument("--reserve", type=parse_size, default=0, metavar="SIZE")
    fit.add_argument("--reserve-percent", type=float, default=0.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        policies: list[CheckpointPolicy] = args.checkpoint or [CheckpointPolicy()]
        if args.command == "memory":
            results: list[dict[str, Any]] = []
            texts: list[str] = []
            for policy in policies:
                result = MemoryModel(
                    args.model, config_from_args(args, policy), root=args.root, std_root=args.std
                ).analyze()
                results.append(result.to_dict())
                heading = f"Checkpointing: {_policy_name(policy)}\n" if len(policies) > 1 else ""
                texts.append(heading + format_result(result))
            if args.json:
                json.dump(results[0] if len(results) == 1 else results, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                print("\n\n".join(texts))
            return 0
        constraint = ResourceConstraint(args.device_memory, args.reserve, args.reserve_percent)
        target: Target = args.maximize
        found = ExecutionPlanner(args.model, root=args.root, std_root=args.std).maximize(
            config_from_args(args, policies[0]), constraint, target
        )
        if args.json:
            json.dump(
                {
                    "target": found.target,
                    "value": found.value,
                    "budget": found.budget,
                    "headroom": found.headroom,
                    "limited_by": found.limited_by,
                    "result": None if found.result is None else found.result.to_dict(),
                },
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
            return 0 if found.value is not None else 1
        label = {"batch": "batch size", "context": "context length", "kv-cache": "cache length"}[
            found.target
        ]
        if found.value is None:
            print(f"No {label} fits: even 1 needs more than {format_bytes(found.budget)}.")
            if found.result is not None:
                print("\n" + format_result(found.result))
            return 1
        assert found.result is not None
        print(f"Maximum {label}: {found.value}")
        print(f"\nEstimated peak memory: {format_bytes(found.result.expected_peak)}")
        print(f"Headroom: {format_bytes(found.headroom)} of a {format_bytes(found.budget)} budget")
        if found.limited_by == "model":
            print("Stopped by the model's `where` clauses, not by memory.")
        elif found.limited_by == "search":
            print("Stopped at the search limit, not by memory.")
        print("\n" + format_result(found.result))
        return 0
    except LinnetError as error:
        print(f"linnet: {error}", file=sys.stderr)
        return 1


def _policy_name(policy: CheckpointPolicy) -> str:
    return policy.kind if policy.kind != "regions" else ", ".join(policy.patterns)

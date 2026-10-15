"""What many tests share: the repository's paths, the skips for tests that
need a GPU, the compiler run for its exit status, and a Llama small enough
for the CPU with the data its trainers are tested on."""

from __future__ import annotations

import itertools
import random
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from linnet.compiler import find_compiler
from linnet.packing import Prompt
from linnet.train import Example
from linnet.train.dpo import Pair

if TYPE_CHECKING:
    from linnet.torch import LinnetModule

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"


def _capability(least: tuple[int, int]) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= least


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
requires_sm90 = pytest.mark.skipif(
    not _capability((9, 0)), reason="needs a GPU of compute capability 9.0 or later"
)


def compiler_ok(*args: str) -> None:
    """`linnet <args>` succeeds; its output is the failure's message."""
    completed = subprocess.run(
        [find_compiler(), *args], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


LLAMA = REPO / "examples/01-llama/src/lib.linnet"
TINY_LLAMA: dict[str, int | str] = {
    "Vocab": 11,
    "H": 8,
    "Heads": 4,
    "KvHeads": 2,
    "Inner": 16,
    "Layers": 2,
    "Batch": 1,
    "MaxSeq": 8,
    "T": "f32",
}


def random_llama(path: Path, seed: int = 0) -> Path:
    """`TINY_LLAMA`'s weights drawn at random (normal, times 0.3, from
    `seed`), without biases, written to `path` as SafeTensors."""
    from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

    from linnet.torch import load

    skeleton = load(LLAMA, generics=TINY_LLAMA, std_root=STDLIB)
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        name.removeprefix("root."): torch.randn(parameter.shape, generator=generator) * 0.3
        for name, parameter in skeleton.named_parameters()
        if not name.endswith(".bias")
    }
    save_file(tensors, str(path))
    return path


def trainable_llama(weights: Path) -> LinnetModule:
    """`TINY_LLAMA` with `weights`, compiled for training."""
    from linnet.torch import load

    return load(
        LLAMA, generics=TINY_LLAMA, std_root=STDLIB, weights=weights, compile=True, trainable=True
    )


def tiny_examples() -> list[Example]:
    """Four prompted examples of 4 to 7 tokens, two of them the prompt."""
    generator = torch.Generator().manual_seed(1)
    return [
        Example.prompted(
            torch.randint(0, 11, (2,), generator=generator).tolist(),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            torch.randint(0, 11, (n - 2,), generator=generator).tolist(),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        )
        for n in [5, 6, 4, 7]
    ]


def tiny_pairs(count: int) -> list[Pair]:
    """Prompts of two tokens; the preferred answer repeats 3, the other 7."""
    generator = torch.Generator().manual_seed(2)
    return [
        Pair(torch.randint(0, 11, (2,), generator=generator).tolist(), [3, 3, 3, 3], [7, 7, 7, 7])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        for _ in range(count)
    ]


def tiny_prompts() -> itertools.cycle[Prompt]:
    """Eight prompts of two tokens, round and round."""
    rng = random.Random(0)
    return itertools.cycle([Prompt([rng.randrange(11), rng.randrange(11)]) for _ in range(8)])


def threes(_: Prompt, completion: list[int]) -> float:
    """A reward: the share of the completion's tokens that are 3."""
    return sum(token == 3 for token in completion) / len(completion)

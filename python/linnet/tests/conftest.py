"""Shared fixtures: the compiler comes from LINNET_BIN or a local build."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]

# Without a GPU, Triton kernels run in Triton's interpreter. Triton reads the
# switch as it defines its own functions (`tl.sum` is one), so it is set
# before any test imports Triton.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


@pytest.fixture(autouse=True, scope="session")
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break

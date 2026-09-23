"""Shared fixtures: the compiler comes from LINNET_BIN or a local build."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True, scope="session")
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break

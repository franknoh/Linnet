"""LANGUAGE.md's complete modules check strictly and are formatted."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from linnet import run_compiler

REPO = Path(__file__).resolve().parents[3]
STDLIB = REPO / "stdlib"
MODULES = [
    block
    for block in re.findall(r"```linnet\n(.*?)```", (REPO / "LANGUAGE.md").read_text("utf-8"), re.S)
    if block.startswith("module ")
]


def test_the_document_has_modules() -> None:
    assert len(MODULES) >= 5


@pytest.mark.parametrize("source", MODULES, ids=[m.split()[1] for m in MODULES])
def test_module_checks_and_is_formatted(source: str, tmp_path: Path) -> None:
    path = tmp_path / "example.linnet"
    path.write_text(source, encoding="utf-8")
    run_compiler("check", "--strict", "--std", str(STDLIB), str(path))
    assert run_compiler("fmt", "-", stdin=source) == source

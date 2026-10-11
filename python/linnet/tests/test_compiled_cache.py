"""Compiler outputs and generated modules under `$LINNET_HOME/compiled`: the
same call again reads what was kept, and a change to any file the source can
read compiles anew."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from linnet import compiler
from linnet.generated import import_generated, write_generated
from linnet.plan import compile_plan
from tests.helpers import STDLIB

SOURCE = """\
module cached

pub entry double(x: Tensor[4; f32]) -> Tensor[4; f32] {
    return x * 2.0
}
"""


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINNET_HOME", str(tmp_path / "home"))


def _counting(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many times the compiler runs from here on."""
    runs = [0]
    run = cast("Callable[..., subprocess.CompletedProcess[str]]", subprocess.run)

    def counted(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        runs[0] += 1
        return run(*args, **kwargs)

    monkeypatch.setattr(compiler.subprocess, "run", counted)
    return runs


def test_a_plan_is_compiled_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    (package / "src").mkdir(parents=True)
    (package / "linnet.toml").write_text(
        '[package]\nname = "cached"\nversion = "0.1.0"\nlanguage = "0.1"\n', encoding="utf-8"
    )
    library = package / "src" / "lib.linnet"
    library.write_text(SOURCE, encoding="utf-8")
    runs = _counting(monkeypatch)
    first = compile_plan(library, functions=True, std_root=STDLIB)
    second = compile_plan(library, functions=True, std_root=STDLIB)
    assert runs[0] == 1
    assert first.functions.keys() == second.functions.keys()
    assert list((tmp_path / "home" / "compiled" / "outputs").rglob("*"))
    # Any file of the package changes the key.
    (package / "src" / "other.linnet").write_text(
        "module cached.other\n\npub fn one() -> f32 {\n    return 1.0\n}\n", encoding="utf-8"
    )
    compile_plan(library, functions=True, std_root=STDLIB)
    assert runs[0] == 2
    library.write_text(SOURCE.replace("2.0", "3.0"), encoding="utf-8")
    compile_plan(library, functions=True, std_root=STDLIB)
    assert runs[0] == 3


def test_a_failure_is_not_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broken = tmp_path / "broken.linnet"
    broken.write_text("module broken\n\npub fn f( -> f32 {\n}\n", encoding="utf-8")
    runs = _counting(monkeypatch)
    for _ in range(2):
        with pytest.raises(compiler.LinnetError):
            compile_plan(broken, functions=True, std_root=STDLIB)
    assert runs[0] == 2


def test_generated_modules_share_a_file_not_a_module(tmp_path: Path) -> None:
    directory = tmp_path / "modules"
    source = "STATE = []\n"
    first = import_generated(directory, "entry", source)
    second = import_generated(directory, "entry", source)
    assert first.__linnet_path__ == second.__linnet_path__
    assert first is not second
    first.STATE.append(1)
    assert second.STATE == []
    assert write_generated(directory, "entry", source + "# other\n") != first.__linnet_path__

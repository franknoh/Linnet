"""Which `linnet` executable the package runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from linnet import compiler


def test_linnet_bin_then_the_wheel_then_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, wheel, path = (tmp_path / name for name in ("from-env", "from-wheel", "from-path"))
    for fake in (env, wheel, path):
        fake.write_text("")
    found: dict[str, Path | None] = {"wheel": wheel, "path": path}

    def which(name: str) -> str | None:
        return None if found["path"] is None else str(found["path"])

    def installed() -> Path | None:
        return found["wheel"]

    monkeypatch.setattr(compiler.shutil, "which", which)
    monkeypatch.setattr(compiler, "installed_compiler", installed)
    monkeypatch.setenv("LINNET_BIN", str(env))
    assert compiler.find_compiler() == str(env)
    monkeypatch.delenv("LINNET_BIN")
    assert compiler.find_compiler() == str(wheel)
    found["wheel"] = None
    assert compiler.find_compiler() == str(path)
    found["path"] = None
    with pytest.raises(compiler.LinnetError, match="LINNET_BIN"):
        compiler.find_compiler()


def test_a_source_install_has_no_compiler_of_its_own() -> None:
    assert compiler.installed_compiler() is None

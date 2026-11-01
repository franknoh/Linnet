"""`linnet self update` and `linnet serve`'s notice of a newer release, from
the record `install.sh` writes; GitHub is never asked."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from linnet import _self


def _record(root: Path, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "install").write_text(text)


def test_the_record_is_what_install_sh_wrote(tmp_path: Path) -> None:
    assert _self.installed(tmp_path) is None
    _record(tmp_path, "tag=dev-2026-11-01\noptions=--with jax,onnx --no-modify-path\n")
    assert _self.installed(tmp_path) == _self.Installed(
        "dev-2026-11-01", ["--with", "jax,onnx", "--no-modify-path"]
    )
    _record(tmp_path, "tag=\noptions=\n")
    assert _self.installed(tmp_path) == _self.Installed("", [])


def test_a_newer_release_is_said_once_a_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LINNET_NO_UPDATE_CHECK", raising=False)
    asked: list[bool] = []

    def latest(timeout: float = 3.0) -> str:
        asked.append(True)
        return "dev-2026-11-05"

    monkeypatch.setattr(_self, "latest_tag", latest)
    # Not installed by install.sh: nothing to say, nothing asked.
    assert _self.update_notice(tmp_path) is None and not asked
    _record(tmp_path, "tag=dev-2026-11-01\noptions=\n")
    notice = _self.update_notice(tmp_path)
    assert notice is not None and "dev-2026-11-05" in notice and "dev-2026-11-01" in notice
    # Asked again only a day later.
    assert _self.update_notice(tmp_path) is None and len(asked) == 1
    stamp = (tmp_path / "update-check").stat().st_mtime
    assert _self.update_notice(tmp_path, now=stamp + 25 * 3600) is not None
    assert len(asked) == 2
    _record(tmp_path, "tag=dev-2026-11-05\noptions=\n")
    assert _self.update_notice(tmp_path, now=stamp + 50 * 3600) is None
    monkeypatch.setenv("LINNET_NO_UPDATE_CHECK", "1")
    assert _self.update_notice(tmp_path, now=stamp + 75 * 3600) is None and len(asked) == 3


def test_update_runs_the_installer_with_the_recorded_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LINNET_HOME", str(tmp_path))
    assert _self.main(["update"]) == 1
    assert "not installed by install.sh" in capsys.readouterr().err
    _record(tmp_path, "tag=dev-2026-11-01\noptions=--with jax --no-modify-path\n")
    monkeypatch.setattr(_self, "latest_tag", lambda timeout=3.0: "dev-2026-11-01")
    assert _self.main(["update", "--check"]) == 0
    assert "is the latest release" in capsys.readouterr().out
    ran: list[list[str]] = []

    def exec_(file: str, args: list[str]) -> None:
        ran.append(args)
        raise SystemExit(0)

    # A `curl` that hands over an installer echoing its options.
    tools = tmp_path / "bin"
    tools.mkdir()
    fake = tools / "curl"
    fake.write_text("#!/bin/sh\necho 'echo installed \"$@\"'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(_self.os, "execvp", exec_)
    with pytest.raises(SystemExit):
        _self.main(["update"])
    (command,) = ran
    assert command[:2] == ["sh", "-c"]
    assert command[2].endswith(f"{_self.INSTALLER} | sh -s -- --with jax --no-modify-path")
    # Run as `sh` runs it (dash on Ubuntu).
    ran_sh = subprocess.run(["sh", "-c", command[2]], capture_output=True, text=True, check=True)
    assert ran_sh.stdout.strip() == "installed --with jax --no-modify-path"

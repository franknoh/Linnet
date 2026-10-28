#!/usr/bin/env python3
"""Checks an installed platform wheel of linnet-lang or linnet-cli, outside
the checkout.

The package must run the compiler its wheel carries, at the package's own
version, and that compiler must find the standard library carried beside
it, whether it is started by path or by name from PATH. linnet-lang puts no
`linnet` on PATH; linnet-cli (`--cli`) puts its entry point there, which runs
that compiler.

    python scripts/check-wheel.py <repository> [--cli]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import linnet
from linnet.compiler import installed_compiler


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    llama = repo / "examples/01-llama/src/lib.linnet"

    compiler = installed_compiler()
    assert compiler is not None, "the wheel installed no compiler"
    assert Path(linnet.find_compiler()) == compiler, linnet.find_compiler()
    assert repo not in compiler.parents, f"{compiler} is in the checkout"

    version = subprocess.run(
        [str(compiler), "--version"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert version == ["linnet", linnet.__version__], (version, linnet.__version__)

    program = linnet.load_program(llama)
    entries = {entry.short_name for entry in program.entries()}
    assert {"forward", "decode"} <= entries, entries

    # By name from PATH, in another directory: argv[0] is then only "linnet".
    # (Windows looks the name up in this process's PATH, not the child's.)
    os.environ["PATH"] = os.pathsep.join([str(compiler.parent), os.environ.get("PATH", "")])
    with tempfile.TemporaryDirectory() as elsewhere:
        subprocess.run(["linnet", "check", str(llama)], cwd=elsewhere, check=True)
    scripts = Path(sysconfig.get_path("scripts"))
    command = shutil.which("linnet", path=str(scripts))
    if "--cli" in sys.argv:
        assert command is not None, f"no linnet in {scripts}"
        with tempfile.TemporaryDirectory() as elsewhere:
            out = subprocess.run(
                [command, "--version"], cwd=elsewhere, capture_output=True, text=True, check=True
            ).stdout.split()
            assert out == ["linnet", linnet.__version__], out
            subprocess.run([command, "check", str(llama)], cwd=elsewhere, check=True)
    else:
        assert command is None, f"linnet-lang put {command} on PATH"
    print(f"linnet {linnet.__version__}: {compiler}")


if __name__ == "__main__":
    main()

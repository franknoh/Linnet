#!/usr/bin/env python3
"""Checks an installed platform wheel of linnet-lang, outside the checkout.

The package must run the compiler its wheel installed, at the package's own
version, and that compiler must find the standard library installed beside
it, whether it is started by path or by name from PATH.

    python scripts/check-wheel.py <repository>
"""

from __future__ import annotations

import os
import subprocess
import sys
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
    print(f"linnet {linnet.__version__}: {compiler}")


if __name__ == "__main__":
    main()

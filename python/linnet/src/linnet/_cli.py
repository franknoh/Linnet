"""The `linnet` command of `linnet-cli`: the compiler its wheel carries, run
with this environment's Python for `linnet serve`, `memory` and `fit`."""

from __future__ import annotations

import os
import subprocess
import sys

from .compiler import installed_compiler


def main() -> None:
    compiler = os.environ.get("LINNET_BIN") or installed_compiler()
    if compiler is None:
        sys.exit("linnet: this install carries no compiler; set LINNET_BIN")
    # The subcommands that run `python -m linnet...` need this environment,
    # which has the package and the frameworks serving takes.
    os.environ.setdefault("LINNET_PYTHON", sys.executable)
    arguments = [str(compiler), *sys.argv[1:]]
    if os.name == "nt":
        sys.exit(subprocess.call(arguments))
    os.execv(arguments[0], arguments)

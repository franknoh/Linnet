"""Where Linnet keeps what it downloads and computes: `$LINNET_HOME`, by
default `~/.linnet`, laid out as Hugging Face's cache is.

    ~/.linnet/
      git/        git dependencies: a bare clone per repository and the
                  files of each commit used (`linnet fetch`)
      nest/       Nest model directories (`linnet.nest.fetch`)
      converted/  Hub checkpoints converted to cards (`linnet.convert`),
                  by commit
      devices/    device profiles measured on this machine
                  (`linnet.resources.calibrate`)
      compiled/   compiler outputs by everything they were compiled from
                  (`outputs/`), and the generated modules imported
                  (`modules/`, named by their content)

Checkpoints themselves stay in Hugging Face's cache (`HF_HOME`), shared
with every other tool that reads them.
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """`$LINNET_HOME`, else `~/.linnet`."""
    chosen = os.environ.get("LINNET_HOME")
    return Path(chosen).expanduser() if chosen else Path.home() / ".linnet"


__all__ = ["home"]

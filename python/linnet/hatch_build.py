"""Puts a built compiler into the wheel.

With LINNET_PREFIX naming a `cmake --install` prefix (relative to this
directory), the wheel installs `linnet` beside the Python scripts and
`share/linnet` (the standard library, where the executable looks for it, and
the Vim files) under the environment's prefix, and is tagged for this
platform. Without it the wheel is pure Python and takes the compiler from
LINNET_BIN or PATH.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CompilerHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        prefix = os.environ.get("LINNET_PREFIX")
        if self.target_name != "wheel" or not prefix:
            return
        root = Path(self.root, prefix).resolve()
        name = "linnet.exe" if os.name == "nt" else "linnet"
        executable = root / "bin" / name
        share = root / "share" / "linnet"
        if not executable.is_file() or not (share / "stdlib").is_dir():
            raise RuntimeError(f"LINNET_PREFIX={prefix} holds no installed compiler")
        build_data["shared_scripts"][str(executable)] = name
        build_data["shared_data"][str(share)] = "share/linnet"
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform_tag()}"


def platform_tag() -> str:
    """This platform's wheel tag; macOS takes MACOSX_DEPLOYMENT_TARGET's
    version, which the compiler was built for."""
    platform = sysconfig.get_platform()
    target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
    if platform.startswith("macosx-") and target:
        platform = f"macosx-{target}-{platform.rsplit('-', 1)[1]}"
    return platform.replace("-", "_").replace(".", "_")

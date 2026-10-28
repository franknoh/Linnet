"""Puts a built compiler into the wheel.

With LINNET_PREFIX naming a `cmake --install` prefix (relative to this
directory), the wheel installs `linnet` beside the Python scripts and
`share/linnet` (the standard library, where the executable looks for it, and
the Vim files) under the environment's prefix, and is tagged for this
platform.

Without it, a wheel built inside a checkout of the repository -- what
`uv add "linnet-lang @ git+https://github.com/franknoh/Linnet#subdirectory=python/linnet"`
builds -- compiles the compiler first (CMake and Ninja come from PyPI; a
C++23 compiler must be installed) and carries it the same way. An editable
install, a build from the source archive, or LINNET_BUILD_COMPILER=0 gives a
pure Python wheel, which takes the compiler from LINNET_BIN or PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CompilerHook(BuildHookInterface):
    def dependencies(self) -> list[str]:
        return ["cmake>=3.25", "ninja"] if self._builds_compiler("standard") else []

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return
        prefix = os.environ.get("LINNET_PREFIX")
        if prefix:
            root = Path(self.root, prefix).resolve()
        elif self._builds_compiler(version):
            root = build_compiler(self._checkout())
        else:
            return
        name = "linnet.exe" if os.name == "nt" else "linnet"
        executable = root / "bin" / name
        share = root / "share" / "linnet"
        if not executable.is_file() or not (share / "stdlib").is_dir():
            raise RuntimeError(f"{root} holds no installed compiler")
        build_data["shared_scripts"][str(executable)] = name
        build_data["shared_data"][str(share)] = "share/linnet"
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform_tag()}"

    def _checkout(self) -> Path:
        return Path(self.root).resolve().parent.parent

    def _builds_compiler(self, version: str) -> bool:
        return (
            self.target_name == "wheel"
            and version != "editable"
            and not os.environ.get("LINNET_PREFIX")
            and os.environ.get("LINNET_BUILD_COMPILER", "1") != "0"
            and (self._checkout() / "CMakeLists.txt").is_file()
            and (self._checkout() / "stdlib").is_dir()
        )


def build_compiler(checkout: Path) -> Path:
    """Configures, builds and installs the compiler from `checkout` into
    `build/wheel-local/prefix` there; returns the prefix."""
    cmake = _tool("cmake")
    work = checkout / "build" / "wheel-local"
    prefix = work / "prefix"
    configure = [cmake, "-S", str(checkout), "-B", str(work / "build")]
    configure += ["-DCMAKE_BUILD_TYPE=Release", "-DLINNET_BUILD_TESTS=OFF"]
    if os.name != "nt":
        # Windows keeps CMake's own choice (Visual Studio), which finds MSVC
        # without a developer prompt.
        configure += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={_tool('ninja')}"]
    subprocess.run(configure, check=True)
    jobs = str(os.cpu_count() or 2)
    subprocess.run(
        [cmake, "--build", str(work / "build"), "--config", "Release", "-j", jobs], check=True
    )
    subprocess.run(
        [cmake, "--install", str(work / "build"), "--config", "Release", "--prefix", str(prefix)],
        check=True,
    )
    return prefix


def _tool(name: str) -> str:
    """CMake or Ninja from its PyPI package in the build environment, else
    from PATH."""
    try:
        module = __import__(name)
        directory = getattr(module, "CMAKE_BIN_DIR" if name == "cmake" else "BIN_DIR")
        candidate = Path(directory, name + (".exe" if os.name == "nt" else ""))
        if candidate.is_file():
            return str(candidate)
    except (ImportError, AttributeError):
        pass
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(f"building the compiler needs `{name}`")
    return found


def platform_tag() -> str:
    """This platform's wheel tag; macOS takes MACOSX_DEPLOYMENT_TARGET's
    version, which the compiler was built for."""
    platform = sysconfig.get_platform()
    target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
    if platform.startswith("macosx-") and target:
        platform = f"macosx-{target}-{platform.rsplit('-', 1)[1]}"
    return platform.replace("-", "_").replace(".", "_")

#!/usr/bin/env python3
"""Checks that the compiler, the Python package and the VS Code extension
carry one version, and that it is the release tag's when one is given.

    python scripts/check-version.py [v0.1.0]
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    cmake = re.search(r"project\(linnet VERSION (\S+)", (REPO / "CMakeLists.txt").read_text())
    assert cmake is not None, "CMakeLists.txt has no project version"
    versions = {
        "CMakeLists.txt": cmake.group(1),
        "python/linnet/pyproject.toml": tomllib.loads(
            (REPO / "python/linnet/pyproject.toml").read_text()
        )["project"]["version"],
        "editors/vscode/package.json": json.loads(
            (REPO / "editors/vscode/package.json").read_text()
        )["version"],
    }
    if len(sys.argv) > 1:
        versions["tag"] = sys.argv[1].removeprefix("v")
    for where, version in versions.items():
        print(f"{version:>10}  {where}")
    if len(set(versions.values())) != 1:
        sys.exit("the versions differ")


if __name__ == "__main__":
    main()

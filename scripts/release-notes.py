#!/usr/bin/env python3
"""Prints a version's section of CHANGELOG.md, the body of its GitHub
release; fails when the changelog has none.

    python scripts/release-notes.py v0.1.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    version = sys.argv[1].removeprefix("v")
    text = (REPO / "CHANGELOG.md").read_text("utf-8")
    found = re.search(rf"^## {re.escape(version)}\b.*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if found is None:
        sys.exit(f"CHANGELOG.md has no section for {version}")
    print(found[1].strip())


if __name__ == "__main__":
    main()

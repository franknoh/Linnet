#!/usr/bin/env python3
"""Packs the compiler a platform wheel carries into a release archive, laid
out as `cmake --install` lays it out: bin/linnet and share/linnet. The
archive and the wheel then hold the same build.

    python scripts/pack-compiler.py <wheel> <name>   # <name>.tar.gz; .zip on Windows
"""

from __future__ import annotations

import shutil
import stat
import sys
import zipfile
from pathlib import Path


def main() -> None:
    wheel, name = Path(sys.argv[1]), sys.argv[2]
    out = Path(name)
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if len(parts) < 3 or not parts[0].endswith(".data"):
                continue
            if parts[1] == "scripts":
                target = out / "bin" / Path(*parts[2:])
            elif parts[1] == "data":
                target = out / Path(*parts[2:])
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
            mode = info.external_attr >> 16
            if mode:
                target.chmod(stat.S_IMODE(mode))
    if not list(out.glob("bin/linnet*")) or not (out / "share/linnet/stdlib").is_dir():
        sys.exit(f"{wheel} carries no compiler")
    kind = "zip" if sys.platform == "win32" else "gztar"
    print(shutil.make_archive(name, kind, root_dir=".", base_dir=name))


if __name__ == "__main__":
    main()

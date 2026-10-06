"""Memory sizes as people write them (`48GiB`, `80GB`, `512MiB`, bytes) and
as reports print them."""

from __future__ import annotations

import re

from .compiler import LinnetError

_SIZE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-z]*)\s*$", re.IGNORECASE)
_UNITS = {
    "": 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 1 << 10,
    "mib": 1 << 20,
    "gib": 1 << 30,
    "tib": 1 << 40,
}


def parse_size(size: int | str) -> int:
    """Bytes from `12345`, `"20GiB"`, or `"20GB"`."""
    if isinstance(size, int):
        return size
    match = _SIZE.match(size)
    if match is None or match[2].lower() not in _UNITS:
        raise LinnetError(f"`{size}` is not a size such as 48GiB or 80GB")
    return int(float(match[1]) * _UNITS[match[2].lower()])


def format_bytes(nbytes: int | None) -> str:
    """`17.98 GiB`, `512.00 MiB`, `40 B`; `unknown` for None."""
    if nbytes is None:
        return "unknown"
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if nbytes >= scale:
            return f"{nbytes / scale:.2f} {unit}"
    return f"{nbytes} B"


__all__ = ["format_bytes", "parse_size"]

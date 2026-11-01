"""`linnet self update` (`python -m linnet._self`): the `linnet` command
installed again from the latest release, with the options `install.sh` was
run with; `linnet self update --check` only says whether one is newer.

`install.sh` records what it installed in `$LINNET_HOME/install`: the
release's tag and the options. `update_notice` reads it for `linnet serve`,
which says once a day when a newer release is out (`LINNET_NO_UPDATE_CHECK=1`
keeps it quiet).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import shutil
import sys
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .home import home

INSTALLER = "https://linnet.franknoh.dev/install.sh"
LATEST = "https://api.github.com/repos/franknoh/Linnet/releases/latest"
# How often `linnet serve` asks for the latest release.
_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class Installed:
    """What `install.sh` installed: the release's tag (empty for a build
    from source) and the options it ran with."""

    tag: str
    options: list[str]


def installed(root: Path | None = None) -> Installed | None:
    """The record `install.sh` wrote, or None for another install (`uv add`,
    a wheel, a checkout)."""
    path = (root or home()) / "install"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    fields = dict(line.split("=", 1) for line in lines if "=" in line)
    return Installed(fields.get("tag", ""), shlex.split(fields.get("options", "")))


def latest_tag(timeout: float = 3.0) -> str | None:
    """The latest release's tag, or None when GitHub cannot be reached."""
    request = urllib.request.Request(LATEST, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            tag = json.load(response).get("tag_name")
    except (OSError, ValueError):
        return None
    return tag if isinstance(tag, str) else None


def update_notice(root: Path | None = None, now: float | None = None) -> str | None:
    """A line saying a newer release is out, at most once a day; None when
    there is none, when it was said today, or for an install `install.sh`
    did not make."""
    if os.environ.get("LINNET_NO_UPDATE_CHECK"):
        return None
    root = root or home()
    record = installed(root)
    if record is None or not record.tag:
        return None
    stamp = root / "update-check"
    now = time.time() if now is None else now
    try:
        if now - stamp.stat().st_mtime < _DAY:
            return None
    except OSError:
        pass
    latest = latest_tag()
    if latest is None:
        return None
    with contextlib.suppress(OSError):
        stamp.touch()
    if latest == record.tag:
        return None
    return f"linnet: {latest} is out (this is {record.tag}); `linnet self update` installs it"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="linnet self", description="Update the linnet command installed by install.sh."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser("update", help="install the latest release")
    update.add_argument(
        "--check", action="store_true", help="only say whether a newer release is out"
    )
    args = parser.parse_args(argv)
    assert args.command == "update"
    record = installed()
    if record is None:
        print(
            "linnet: this linnet was not installed by install.sh; update it as it was installed "
            f"(uv add, a wheel), or install it with `curl -LsSf {INSTALLER} | sh`",
            file=sys.stderr,
        )
        return 1
    if args.check:
        latest = latest_tag()
        if latest is None:
            print("linnet: GitHub cannot be reached", file=sys.stderr)
            return 1
        if not record.tag:
            print(f"linnet: built from source; the latest release is {latest}")
        elif latest == record.tag:
            print(f"linnet: {record.tag} is the latest release")
        else:
            print(f"linnet: {latest} is out (this is {record.tag})")
        return 0
    if os.name == "nt":
        print(
            "linnet: on Windows, install the latest release's win_amd64 wheel: "
            '`uv tool install --force "linnet-cli @ <its URL>"`',
            file=sys.stderr,
        )
        return 1
    fetch = (
        f"curl -LsSf {INSTALLER}" if shutil.which("curl") is not None else f"wget -qO- {INSTALLER}"
    )
    command = f"{fetch} | sh"
    if record.options:
        command += " -s -- " + shlex.join(record.options)
    print(f"linnet: {command}", flush=True)
    # This process's environment is the one replaced: the installer runs in
    # its place.
    os.execvp("sh", ["sh", "-c", f"set -o pipefail 2>/dev/null; {command}"])


if __name__ == "__main__":
    sys.exit(main())

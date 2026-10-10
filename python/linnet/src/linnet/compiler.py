"""Running the `linnet` compiler as a subprocess."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import tomllib
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import cast

from .home import home


class LinnetError(Exception):
    """The compiler rejected the request, or the toolchain is missing."""


class PlanError(LinnetError):
    """A plan could not be produced, read, or instantiated."""


def find_compiler() -> str:
    """The `linnet` executable: LINNET_BIN, then the one this package's wheel
    installed, then PATH."""
    candidate = os.environ.get("LINNET_BIN")
    if candidate and Path(candidate).exists():
        return candidate
    installed = installed_compiler()
    if installed is not None:
        return str(installed)
    found = shutil.which("linnet")
    if found is None:
        raise LinnetError(
            "cannot find the `linnet` executable; set LINNET_BIN or add it to PATH "
            "(https://linnet.franknoh.dev/guide/installation)"
        )
    return found


def installed_compiler() -> Path | None:
    """The executable a platform wheel of `linnet-lang` installs beside the
    Python scripts, or None for a pure or source install."""
    try:
        files = metadata.distribution("linnet-lang").files or []
    except metadata.PackageNotFoundError:
        return None
    for file in files:
        if file.name in ("linnet", "linnet.exe") and file.parent.name in ("bin", "Scripts"):
            path = Path(str(file.locate())).resolve()
            if path.is_file():
                return path
    return None


# Commands whose output depends only on their arguments, their standard
# input, the compiler, and the source files they read: kept in
# `$LINNET_HOME/compiled/outputs` (`run_compiler`).
_CACHED = frozenset(("plan", "torch", "jax", "stablehlo", "onnx"))


def run_compiler(
    *arguments: str, stdin: str | None = None, error: type[LinnetError] | None = None
) -> str:
    """Runs one compiler command and returns its standard output. A failure
    raises `error` (a `LinnetError` by default) with what the compiler said.

    `plan` and the exporters are kept in `$LINNET_HOME/compiled/outputs`, by
    the compiler, the arguments, standard input, and every file the source
    (the last argument) can read: the standard library, its package and the
    packages it depends on by path, and `linnet.lock`, which pins the git
    dependencies. The same call again reads what was kept."""
    compiler = find_compiler()
    kept: Path | None = None
    if arguments and arguments[0] in _CACHED and Path(arguments[-1]).exists():
        key = _output_key(compiler, arguments, stdin)
        kept = home() / "compiled" / "outputs" / key[:2] / key
        try:
            return kept.read_text(encoding="utf-8")
        except OSError:
            pass
    completed = subprocess.run(
        [compiler, *arguments],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise (error or LinnetError)(
            f"linnet {' '.join(arguments)} failed:\n{completed.stderr}{completed.stdout}".rstrip()
        )
    if kept is not None:
        write_atomically(kept, completed.stdout)
    return completed.stdout


def write_atomically(path: Path, text: str) -> None:
    """`text` at `path`, written beside it first and renamed into place, so
    that a reader in another process never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _output_key(compiler: str, arguments: Sequence[str], stdin: str | None) -> str:
    digest = hashlib.sha256()
    stat = Path(compiler).stat()
    for part in (str(Path(compiler).resolve()), str(stat.st_size), str(stat.st_mtime_ns)):
        digest.update(part.encode() + b"\0")
    for argument in arguments:
        digest.update(argument.encode() + b"\0")
    digest.update(b"stdin\0" + (stdin or "").encode() + b"\0")
    for path in _source_files(compiler, arguments):
        digest.update(str(path).encode() + b"\0" + _file_digest(path) + b"\0")
    return digest.hexdigest()


def _source_files(compiler: str, arguments: Sequence[str]) -> list[Path]:
    """Every file a compilation of `arguments[-1]` can read: the standard
    library the compiler would use, and the source's package with the
    packages it depends on by path. A git dependency's files are its
    commit's, which `linnet.lock` names."""
    files: set[Path] = set()
    std = _std_root(compiler, arguments)
    if std is not None:
        files.update(std.rglob("*.linnet"))
    source = Path(arguments[-1]).resolve()
    root = next((d for d in (source, *source.parents) if (d / "linnet.toml").is_file()), None)
    if root is None:
        files.update(source.rglob("*.linnet") if source.is_dir() else [source])
        return sorted(files)
    pending = [root]
    seen: set[Path] = set()
    while pending:
        package = pending.pop().resolve()
        if package in seen or not (package / "linnet.toml").is_file():
            continue
        seen.add(package)
        files.update(package.joinpath("src").rglob("*.linnet"))
        files.add(package / "linnet.toml")
        try:
            manifest = tomllib.loads((package / "linnet.toml").read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        dependencies: object = manifest.get("dependencies", {})
        if not isinstance(dependencies, dict):
            continue
        for dependency in cast("dict[str, object]", dependencies).values():
            if isinstance(dependency, dict):
                path = cast("dict[str, object]", dependency).get("path")
                if isinstance(path, str):
                    pending.append(package / path)
    if (root / "linnet.lock").is_file():
        files.add(root / "linnet.lock")
    return sorted(files)


def _std_root(compiler: str, arguments: Sequence[str]) -> Path | None:
    """The standard library directory the compiler finds, as it finds it:
    `--std`, then `LINNET_STD`, then `share/linnet/stdlib` or `stdlib` up to
    four directories above the executable."""
    for index, argument in enumerate(arguments[:-1]):
        if argument == "--std":
            return Path(arguments[index + 1])
    if os.environ.get("LINNET_STD"):
        return Path(os.environ["LINNET_STD"])
    directory = Path(compiler).resolve().parent
    for _ in range(4):
        for candidate in ("share/linnet/stdlib", "stdlib"):
            if (directory / candidate).is_dir():
                return directory / candidate
        directory = directory.parent
    return None


# A file's digest by its path, size and modification time: read once a
# process for as long as it is unchanged.
_digests: dict[Path, tuple[int, int, bytes]] = {}


def _file_digest(path: Path) -> bytes:
    try:
        stat = path.stat()
    except OSError:
        return b"missing"
    known = _digests.get(path)
    if known is not None and known[:2] == (stat.st_size, stat.st_mtime_ns):
        return known[2]
    digest = hashlib.sha256(path.read_bytes()).digest()
    _digests[path] = (stat.st_size, stat.st_mtime_ns, digest)
    return digest


def parse_binding(text: str) -> tuple[str, int | str]:
    """`NAME=VALUE` as a generic's binding: a whole number as an `int`, any
    other value (a dtype) as text."""
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise LinnetError(f"`{text}` is not NAME=VALUE")
    return name, int(value) if value.lstrip("-").isdigit() else value


def bind_arguments(bindings: Mapping[str, object]) -> list[str]:
    """`--bind NAME=VALUE` for each generic binding."""
    return [part for name, value in bindings.items() for part in ("--bind", f"{name}={value}")]


# How exactly compiled kernels follow the canonical decompositions (`linnet
# plan --numerics`).
NUMERICS = ("exact", "equivalent", "fast")


def check_numerics(numerics: str) -> None:
    """Raises a `PlanError` unless `numerics` is one of `NUMERICS`."""
    if numerics not in NUMERICS:
        raise PlanError('numerics must be "exact", "equivalent", or "fast"')


def std_arguments(std_root: str | Path | None) -> list[str]:
    """`--std <dir>` when a standard library directory is given."""
    return [] if std_root is None else ["--std", str(std_root)]


def lora_arguments(lora: tuple[Sequence[str], int, float] | None) -> list[str]:
    """`--lora` for each pattern of `(patterns, rank, alpha)`, with its rank
    and alpha; none without adapters."""
    if lora is None:
        return []
    patterns, rank, alpha = lora
    arguments = [part for pattern in patterns for part in ("--lora", pattern)]
    return [*arguments, "--lora-rank", str(rank), "--lora-alpha", repr(float(alpha))]

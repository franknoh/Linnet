"""What the importers share (`linnet.torch.export_linnet`,
`linnet.jax.export_linnet`, `linnet.onnx.import_onnx`): Linnet identifiers
for foreign names, the blocks a tree of parameters becomes, and writing the
result as Linnet source the compiler has checked.

Each importer reads its own tree (a module, a parameter dict, initializer
names) into `Member`s; `Hierarchy` turns them into blocks, one per distinct
shape of subtree, and says how a parameter's type is written.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compiler import LinnetError, run_compiler, std_arguments

KEYWORDS = frozenset(
    {
        "as", "const", "type", "struct", "enum", "fn", "op", "block", "entry", "param", "buffer",
        "state", "sub", "let", "var", "return", "if", "else", "match", "static", "for", "in",
        "while", "where", "true", "false", "none", "some", "extern", "module", "use", "pub",
    }
)  # fmt: skip
PRELUDE = frozenset(
    {
        "Tensor", "Dim", "Shape", "DType", "Numeric", "Integer", "Float", "cast", "reshape",
        "permute", "broadcast_to", "concat", "pad", "iota", "fill", "gather", "scatter", "exp",
        "log", "sqrt", "rsqrt", "sin", "cos", "tanh", "abs", "select", "min", "max", "sum", "prod",
        "any", "all", "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16",
        "f32", "f64",
    }
)  # fmt: skip
RESERVED = KEYWORDS | PRELUDE


def identifier(name: str) -> str:
    """A Linnet identifier for a foreign attribute, class or tensor name."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not clean or clean[0].isdigit():
        clean = "_" + clean
    if clean in RESERVED:
        clean += "_"
    return clean


@dataclass
class Block:
    """A Linnet block synthesized from a subtree."""

    name: str
    members: list[dict[str, Any]] = field(default_factory=lambda: [])


@dataclass
class Member:
    """How one node of the foreign tree maps onto a block member."""

    kind: str  # "param" | "buffer" | "sub"
    name: str
    block: Block | None = None  # sub: the child's block
    length: int | None = None  # sub: array length when the children are uniform
    element: Block | None = None  # sub: the array element block
    children: dict[str, Member] = field(default_factory=lambda: {})  # sub: by foreign name
    leaf: Any = None  # param: the importer's own description of its tensor


class Hierarchy:
    """The blocks of an imported model, one per distinct subtree, named
    after the foreign class or key and kept unique."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.blocks: dict[str, Block] = {}
        self._by_signature: dict[Any, Block] = {}
        # Linnet path -> the foreign name, where they differ (`bindings.json`).
        self.renamed: dict[str, str] = {}

    def leaf_type(self, member: Member) -> dict[str, Any] | None:
        """A parameter's plan type; None for one the importer fills in later."""
        return None

    def block_for(self, class_name: str, signature: Any, member: Member) -> Block:
        """The block for `member`'s subtree: the same one for every subtree of
        the same class name and signature."""
        key = (class_name, signature)
        if key in self._by_signature:
            return self._by_signature[key]
        base = identifier(class_name)
        name = base
        for suffix in range(2, 1000):
            if name not in self.blocks:
                break
            name = f"{base}_{suffix}"
        block = Block(name)
        self.blocks[name] = block
        self._by_signature[key] = block
        for child in member.children.values():
            kind_type = self.member_type(child) if child.kind == "sub" else self.leaf_type(child)
            block.members.append({"name": child.name, "kind": child.kind, "type": kind_type})
        return block

    def block_type(self, block: Block) -> dict[str, Any]:
        return {"kind": "block", "name": block.name, "module": self.module, "args": []}

    def member_type(self, member: Member) -> dict[str, Any]:
        """The plan type of a `sub` member: a block, or an array of one."""
        if member.length is not None:
            assert member.element is not None
            return {
                "kind": "array",
                "element": self.block_type(member.element),
                "length": member.length,
            }
        assert member.block is not None
        return self.block_type(member.block)


def write_source(
    plan: dict[str, Any],
    output: Path,
    std_root: str | Path | None,
    error: type[LinnetError],
    what: str = "exported",
) -> None:
    """Prints `plan` as Linnet source at `output` (`linnet emit`) and checks
    it (`linnet check`); `error` says what went wrong."""
    try:
        text = run_compiler("emit", "-", stdin=json.dumps(plan))
    except LinnetError as failure:
        raise error(f"the compiler rejected the {what} plan:\n{failure}") from None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    try:
        run_compiler("check", *std_arguments(std_root), str(output))
    except LinnetError as failure:
        raise error(f"the {what} source does not check:\n{failure}") from None


__all__ = [
    "KEYWORDS",
    "PRELUDE",
    "RESERVED",
    "Block",
    "Hierarchy",
    "Member",
    "identifier",
    "write_source",
]

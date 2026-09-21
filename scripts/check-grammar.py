#!/usr/bin/env python3
"""Validate the TextMate grammar: well-formed JSON, every include resolves, and
every pattern compiles. Python's regex dialect is close enough to Oniguruma for
the constructs this grammar uses."""

import json
import pathlib
import re
import sys

path = pathlib.Path(__file__).resolve().parent.parent / "editors/textmate/linnet.tmLanguage.json"
grammar = json.loads(path.read_text(encoding="utf-8"))
repository = grammar.get("repository", {})
problems = []


def walk(node, where):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("match", "begin", "end"):
                try:
                    re.compile(value)
                except re.error as error:
                    problems.append(f"{where}.{key}: {error}")
            elif key == "include":
                if not value.startswith("#") or value[1:] not in repository:
                    problems.append(f"{where}: unknown include {value}")
            else:
                walk(value, f"{where}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            walk(value, f"{where}[{index}]")


if grammar.get("scopeName") != "source.linnet":
    problems.append("scopeName must be source.linnet")
walk(grammar, "grammar")
for problem in problems:
    print(problem, file=sys.stderr)
sys.exit(1 if problems else 0)

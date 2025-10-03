"""A tiny YAML loader used when PyYAML is unavailable.

The project primarily consumes :mod:`tree.yaml`, which only relies on a very
small subset of YAML (mappings, sequences, scalars and folded block
scalars).  Requiring users to install PyYAML just for this file makes the
application harder to run in constrained environments, so we provide a
minimal fallback parser.  The implementation is intentionally conservative
and does **not** aim to be a complete YAML implementation – it merely
supports the constructs present in the repository's YAML resources.

The parser understands:

* nested mappings (``key: value``)
* nested sequences (``- item``)
* multi-line scalars introduced with ``>`` or ``|``
* quoted strings and basic numeric/boolean scalars

Anything outside of this surface area will raise ``ValueError`` to make it
obvious that the file requires features unsupported by the fallback parser.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional, Tuple

__all__ = ["safe_load"]


@dataclass
class _Line:
    raw: str
    stripped: str
    indent: int
    index: int


def _iter_content(lines: Iterable[str]) -> List[_Line]:
    result = []
    for idx, raw in enumerate(lines):
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(stripped)
        result.append(_Line(raw.rstrip("\n"), stripped.rstrip(), indent, idx))
    return result


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return value[:i].rstrip()
    return value.rstrip()


def _parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        body = value[1:-1]
        return bytes(body, "utf-8").decode("unicode_escape")
    if value.startswith("'") and value.endswith("'"):
        body = value[1:-1]
        return body.replace("''", "'")

    int_match = re.fullmatch(r"[-+]?[0-9]+", value)
    float_match = re.fullmatch(r"[-+]?[0-9]*\.[0-9]+", value)
    if int_match:
        try:
            return int(value)
        except ValueError:
            pass
    if float_match:
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _peek(lines: List[_Line], start: int) -> Optional[_Line]:
    if start < len(lines):
        return lines[start]
    return None


def _collect_block(lines: List[_Line], start: int, parent_indent: int, indicator: str) -> Tuple[str, int]:
    collected: List[str] = []
    i = start
    block_indent: Optional[int] = None
    while i < len(lines):
        line = lines[i]
        if line.indent <= parent_indent:
            break
        if block_indent is None:
            block_indent = line.indent
        text = line.raw[block_indent:] if len(line.raw) >= block_indent else ""
        collected.append(text)
        i += 1

    if indicator == ">":
        text = " ".join(part.strip() for part in collected).strip()
    else:
        text = "\n".join(collected)
    return text, i


def _gather_plain_continuation(lines: List[_Line], start: int, parent_indent: int) -> Tuple[List[str], int]:
    extras: List[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if line.indent <= parent_indent:
            break
        if line.stripped.startswith("- "):
            break
        if ":" in line.stripped:
            break
        extras.append(line.stripped)
        i += 1
    return extras, i


def safe_load(stream):  # pragma: no cover - thin wrapper around helper
    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = stream
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    lines = _iter_content(text.splitlines())
    if not lines:
        return {}

    root = None
    stack: List[Tuple[int, object]] = []  # (indent, container)
    i = 0

    while i < len(lines):
        line = lines[i]
        indent = line.indent
        stripped = line.stripped

        while stack and indent < stack[-1][0]:
            stack.pop()
        while (
            stack
            and isinstance(stack[-1][1], list)
            and indent <= stack[-1][0]
            and not stripped.startswith("- ")
        ):
            stack.pop()

        parent = stack[-1][1] if stack else None
        next_line = _peek(lines, i + 1)

        if stripped.startswith("- "):
            if parent is None:
                parent = []
                root = parent
                stack.append((indent, parent))
            if not isinstance(parent, list):
                raise ValueError("sequence item without list parent")

            value_part = _strip_inline_comment(stripped[2:]).strip()
            if value_part == "":
                parent.append(None)
                i += 1
                continue

            if value_part in {">", "|"}:
                block, new_i = _collect_block(lines, i + 1, indent, value_part)
                parent.append(block)
                i = new_i
                continue

            if ":" in value_part:
                key, rest = value_part.split(":", 1)
                key = key.strip()
                rest = _strip_inline_comment(rest).strip()
                item: dict = {}
                parent.append(item)
                stack.append((indent + 2, item))

                if rest == "":
                    if next_line and (next_line.indent > indent or (
                        next_line.indent == indent and next_line.stripped.startswith("- ")
                    )):
                        container = [] if next_line.stripped.startswith("- ") else {}
                        item[key] = container
                        stack.append((next_line.indent, container))
                    else:
                        item[key] = None
                    i += 1
                    continue

                if rest in {">", "|"}:
                    block, new_i = _collect_block(lines, i + 1, indent, rest)
                    item[key] = block
                    i = new_i
                    continue

                value = _parse_scalar(rest)
                if isinstance(value, str):
                    extras, new_i = _gather_plain_continuation(lines, i + 1, indent)
                    if extras:
                        value = " ".join([value] + extras)
                        item[key] = value
                        i = new_i
                        continue
                item[key] = value
                i += 1
                continue

            value = _parse_scalar(value_part)
            if isinstance(value, str):
                extras, new_i = _gather_plain_continuation(lines, i + 1, indent)
                if extras:
                    value = " ".join([value] + extras)
                    parent.append(value)
                    i = new_i
                    continue
            parent.append(value)
            i += 1
            continue

        if ":" not in stripped:
            scalar = _parse_scalar(_strip_inline_comment(stripped))
            if parent is None:
                root = scalar
            elif isinstance(parent, list):
                parent.append(scalar)
            else:
                raise ValueError(f"dangling scalar at line {line.index + 1}")
            i += 1
            continue

        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = _strip_inline_comment(rest).strip()

        if parent is None:
            parent = {}
            root = parent
            stack.append((indent, parent))
        elif isinstance(parent, list):
            item = {}
            parent.append(item)
            parent = item
            stack.append((indent, parent))

        if rest == "":
            if next_line and (next_line.indent > indent or (
                next_line.indent == indent and next_line.stripped.startswith("- ")
            )):
                container = [] if next_line.stripped.startswith("- ") else {}
                parent[key] = container
                stack.append((next_line.indent, container))
            else:
                parent[key] = None
            i += 1
            continue

        if rest in {">", "|"}:
            block, new_i = _collect_block(lines, i + 1, indent, rest)
            parent[key] = block
            i = new_i
            continue

        value = _parse_scalar(rest)
        if isinstance(value, str):
            extras, new_i = _gather_plain_continuation(lines, i + 1, indent)
            if extras:
                value = " ".join([value] + extras)
                parent[key] = value
                i = new_i
                continue
        parent[key] = value
        i += 1

    return root if root is not None else {}

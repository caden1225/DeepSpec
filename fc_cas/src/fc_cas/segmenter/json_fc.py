"""JSON Function-Call segment classifier (minimal FSM).

This module classifies the *tail* of a growing assistant serialization that
contains OpenAI-style tool_calls JSON. It is intentionally dependency-free.
"""

from __future__ import annotations

import re
from typing import Any

from fc_cas.schema.guard import enum_allowed
from fc_cas.types import Region, SegmentInfo

_TOOL_HINT = re.compile(r'"tool_calls"|"function"|"arguments"')


def classify_prefix(
    text: str,
    tools: list[dict[str, Any]],
    active_fn: str | None = None,
) -> SegmentInfo:
    if not text or not _TOOL_HINT.search(text):
        return SegmentInfo(region=Region.NL)

    fn_name = active_fn or _infer_fn_name(text)
    arg_key, in_value, in_key = _arguments_cursor(text)
    if in_value and arg_key:
        allowed = enum_allowed(tools, fn_name, arg_key) if fn_name else None
        if allowed:
            return SegmentInfo(
                region=Region.TOOL_ENUM,
                allowed_strings=tuple(allowed),
                json_pointer=f"/arguments/{arg_key}",
            )
        return SegmentInfo(
            region=Region.TOOL_FREE,
            json_pointer=f"/arguments/{arg_key}",
        )
    if in_key or '"arguments"' in text:
        return SegmentInfo(region=Region.TOOL_SKELETON, json_pointer="/arguments")

    return SegmentInfo(region=Region.TOOL_SKELETON, json_pointer="/")


def _infer_fn_name(text: str) -> str | None:
    m = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None


def _arguments_cursor(text: str) -> tuple[str | None, bool, bool]:
    """Return (current_arg_key, in_value, in_key) for the arguments object.

    Supports both a literal arguments object and the JSON-encoded arguments
    string emitted by OpenAI-compatible tool-call serializers.
    """
    idx = text.rfind('"arguments"')
    if idx < 0:
        return None, False, False
    colon = text.find(":", idx + len('"arguments"'))
    if colon < 0:
        return None, False, True
    value = text[colon + 1 :].lstrip()
    if not value:
        return None, False, True
    if value.startswith("{"):
        return _scan_arguments_object(value)
    if value.startswith('"'):
        # The outer arguments string may be incomplete.  Unescaping quotes is
        # sufficient for this shallow FSM because it only reads top-level keys.
        return _scan_arguments_object(value[1:].replace('\\"', '"'))
    return None, False, True


def _scan_arguments_object(fragment: str) -> tuple[str | None, bool, bool]:
    """Classify the cursor within a partial, top-level JSON arguments object."""
    if not fragment.startswith("{"):
        return None, False, True

    pos = 1
    length = len(fragment)
    while True:
        pos = _skip_whitespace(fragment, pos)
        if pos >= length or fragment[pos] == "}":
            return None, False, True
        if fragment[pos] != '"':
            return None, False, True

        key_end = _json_string_end(fragment, pos)
        if key_end is None:
            return None, False, True
        key = fragment[pos + 1 : key_end - 1]
        pos = _skip_whitespace(fragment, key_end)
        if pos >= length or fragment[pos] != ":":
            return key, False, True
        pos = _skip_whitespace(fragment, pos + 1)
        if pos >= length:
            return key, False, True

        if fragment[pos] == '"':
            value_end = _json_string_end(fragment, pos)
            if value_end is None:
                return key, True, False
            pos = _skip_whitespace(fragment, value_end)
        else:
            value_end = _json_value_end(fragment, pos)
            if value_end == length:
                return key, True, False
            if value_end is None:
                return key, True, False
            pos = _skip_whitespace(fragment, value_end)

        if pos >= length or fragment[pos] == "}":
            return key, False, True
        if fragment[pos] != ",":
            return key, False, True
        pos += 1


def _skip_whitespace(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _json_string_end(text: str, start: int) -> int | None:
    pos = start + 1
    escaped = False
    while pos < len(text):
        char = text[pos]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return pos + 1
        pos += 1
    return None


def _json_value_end(text: str, start: int) -> int | None:
    """Return the end of a JSON value, or None while it remains incomplete."""
    if text[start] not in "[{":
        pos = start
        while pos < len(text) and text[pos] not in ",}":
            pos += 1
        return pos if pos < len(text) else None

    stack = [text[start]]
    pos = start + 1
    while pos < len(text):
        char = text[pos]
        if char == '"':
            pos = _json_string_end(text, pos)
            if pos is None:
                return None
            continue
        if char in "[{":
            stack.append(char)
        elif char in "]}":
            expected = "{" if char == "}" else "["
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return pos + 1
        pos += 1
    return None

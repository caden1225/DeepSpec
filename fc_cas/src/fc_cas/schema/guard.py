from __future__ import annotations

from typing import Any


def _find_function_schema(tools: list[dict[str, Any]], fn_name: str) -> dict[str, Any] | None:
    for item in tools:
        fn = (item.get("function") or {}) if isinstance(item, dict) else {}
        if fn.get("name") == fn_name:
            return fn
    return None


def enum_allowed(
    tools: list[dict[str, Any]],
    fn_name: str,
    arg_key: str,
) -> list[str] | None:
    """Return enum values for tools[fn].parameters.properties[arg_key], else None."""
    fn = _find_function_schema(tools, fn_name)
    if fn is None:
        return None
    props = ((fn.get("parameters") or {}).get("properties") or {})
    field = props.get(arg_key) or {}
    enum_vals = field.get("enum")
    if not enum_vals:
        return None
    return [str(v) for v in enum_vals]


def next_template_prefix(
    prefix: str,
    tools: list[dict[str, Any]],
    fn_name: str | None,
) -> str | None:
    """Suggest a deterministic continuation for JSON skeleton positions.

    Minimal v0: if prefix ends with an open object awaiting a known required key
    that has not appeared yet, suggest `"<key>":`. Full FSM lives in segmenter.
    """
    if not fn_name:
        return None
    fn = _find_function_schema(tools, fn_name)
    if fn is None:
        return None
    params = fn.get("parameters") or {}
    required = list(params.get("required") or [])
    if not required:
        return None
    # If arguments object just opened and first required key missing in prefix
    if '"arguments"' in prefix and prefix.rstrip().endswith("{"):
        key = required[0]
        return f'"{key}": '
    return None

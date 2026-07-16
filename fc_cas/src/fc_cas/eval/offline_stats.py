"""Character-level offline coverage statistics for assistant responses."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from fc_cas.segmenter import classify_prefix
from fc_cas.types import Region

_REGIONS = tuple(region.value for region in Region)
_COVERABLE_REGIONS = {Region.TOOL_SKELETON, Region.TOOL_ENUM}


def serialize_assistant(message: Mapping[str, Any]) -> str:
    """Return the deterministic text used to analyze an assistant message.

    Content is emitted first (when present), followed directly by compact JSON
    with canonical field order for OpenAI-compatible ``tool_calls``.
    """
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        text += json.dumps(
            {"tool_calls": [_canonical_tool_call(call) for call in tool_calls]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return text


def _canonical_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI tool-call field order for reproducible prefixes."""
    function = call.get("function")
    function = function if isinstance(function, Mapping) else {}
    return {
        "id": call.get("id"),
        "type": call.get("type"),
        "function": {
            "name": function.get("name"),
            "arguments": function.get("arguments"),
        },
    }


def compute_offline_stats(
    examples: Iterable[Mapping[str, Any]], *, split: str
) -> dict[str, Any]:
    """Compute character-level region coverage for assistant messages.

    Each character is assigned the region returned by ``classify_prefix`` for
    the serialized prefix ending at that character.  This is an intentionally
    tokenizer-independent approximation, rather than a model-token statistic.
    """
    region_counts: Counter[str] = Counter({region: 0 for region in _REGIONS})
    sample_counts: Counter[str] = Counter({"nl": 0, "tool": 0})

    for example in examples:
        tools = example.get("tools", [])
        for message in example.get("messages", []):
            if message.get("role") != "assistant":
                continue

            has_tool_calls = bool(message.get("tool_calls"))
            sample_counts["tool" if has_tool_calls else "nl"] += 1
            text = serialize_assistant(message)
            for end in range(1, len(text) + 1):
                prefix = text[:end]
                region = classify_prefix(
                    prefix, tools, active_fn=_active_function_name(prefix)
                ).region
                region_counts[region.value] += 1

    total_chars = sum(region_counts.values())
    coverable_chars = sum(
        region_counts[region.value] for region in _COVERABLE_REGIONS
    )
    return {
        "split": split,
        "approximation": (
            "Character-level: each serialized character is labeled from its "
            "inclusive prefix; this is not tokenizer-token coverage."
        ),
        "assistant_sample_counts": dict(sample_counts),
        "assistant_samples_total": sum(sample_counts.values()),
        "total_serialized_chars": total_chars,
        "region_char_counts": dict(region_counts),
        "region_char_shares": {
            region: region_counts[region] / total_chars if total_chars else 0.0
            for region in _REGIONS
        },
        "template_enum_coverable_chars": coverable_chars,
        "template_enum_coverable_char_share": (
            coverable_chars / total_chars if total_chars else 0.0
        ),
    }


def _active_function_name(prefix: str) -> str | None:
    """Return the latest fully emitted function name in a tool-call sequence."""
    matches = re.findall(r'"name":"([^"]+)"', prefix)
    return matches[-1] if matches else None

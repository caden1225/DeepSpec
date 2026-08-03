#!/usr/bin/env python3
"""Convert processed_FC_dataset JSON splits into DeepSpec conversations jsonl.

DeepSpec GeneralParser already forwards assistant ``tool_calls`` into
``tokenizer.apply_chat_template``.  Tool schemas are inlined into a system
message so the template still sees candidate tools without a ``tools=`` kwarg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_arguments(arguments: Any) -> Any:
    """Qwen chat template requires function.arguments to be a mapping."""
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": arguments}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    if arguments is None:
        return {}
    return arguments


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for call in tool_calls:
        call = dict(call)
        fn = dict(call.get("function") or {})
        fn["arguments"] = _normalize_arguments(fn.get("arguments"))
        call["function"] = fn
        if "type" not in call:
            call["type"] = "function"
        normalized.append(call)
    return normalized


def convert_example(example: dict[str, Any]) -> dict[str, Any] | None:
    messages = example.get("messages") or []
    if not messages:
        return None
    # Do not inline tools into system text — pass ``tools`` through to
    # tokenizer.apply_chat_template via DeepSpec parser.
    conversations: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        out: dict[str, Any] = {"role": role}
        content = msg.get("content")
        if content is not None:
            out["content"] = content
        elif role == "assistant" and msg.get("tool_calls"):
            out["content"] = ""
        else:
            out["content"] = "" if content is None else content
        if role == "assistant" and msg.get("tool_calls"):
            out["tool_calls"] = _normalize_tool_calls(msg["tool_calls"])
        if role == "tool":
            if "tool_call_id" in msg:
                out["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                out["name"] = msg["name"]
        conversations.append(out)

    roles = [m["role"] for m in conversations if m["role"] != "system"]
    if not roles or roles[0] != "user":
        return None
    return {
        "id": example.get("id"),
        "source": example.get("source"),
        "conversations": conversations,
        "tools": example.get("tools") or [],
    }


def convert_split(
    *,
    input_path: Path,
    output_path: Path,
    limit: int | None,
) -> int:
    data = json.loads(input_path.read_text())
    if limit is not None:
        data = data[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in data:
            row = convert_example(example)
            if row is None:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "processed_FC_dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "deepspec_jsonl",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    counts = {}
    for split, limit in (
        ("train", args.limit_train),
        ("validation", args.limit_val),
        ("test", args.limit_test),
    ):
        src = args.data_root / f"{split}.json"
        dst = args.out_dir / f"fc_{split}.jsonl"
        counts[split] = convert_split(input_path=src, output_path=dst, limit=limit)
        print(f"{split}: {counts[split]} -> {dst}")
    (args.out_dir / "convert_manifest.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

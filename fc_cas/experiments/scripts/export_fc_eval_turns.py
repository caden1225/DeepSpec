#!/usr/bin/env python3
"""Export FC test user turns into DeepSpec eval jsonl (``turns`` field)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("fc_cas/data/deepspec_jsonl/fc_test.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval_datasets/fc_cockpit.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.src.open() as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if n >= args.limit:
                break
            row = json.loads(line)
            user_text = None
            for msg in row.get("conversations") or []:
                if msg.get("role") == "user":
                    user_text = msg.get("content")
                    break
            if not user_text:
                continue
            dst.write(
                json.dumps(
                    {"id": row.get("id"), "turns": [user_text]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Task 5 entrypoint — offline region coverage stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fc_cas.eval.offline_stats import compute_offline_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_FC_dataset",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "offline_stats_test.json",
    )
    args = parser.parse_args()
    path = args.data_root / f"{args.split}.json"
    data = json.loads(path.read_text())
    out = compute_offline_stats(data, split=args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

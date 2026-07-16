#!/usr/bin/env python3
"""Compare global and segmented proposal policies on FC-CAS samples.

Mock verification accepts the gold next characters.  It validates policy routing
and reporting only; it is not a model-quality or speed measurement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fc_cas.eval.offline_stats import serialize_assistant
from fc_cas.runtime import cas_proposal_hparams
from fc_cas.segmenter import classify_prefix
from fc_cas.strategy import StrategyTable
from fc_cas.types import DraftMode

BASELINE_HPARAMS = {
    "block_size": 8,
    "confidence_threshold": 0.4,
    "mode": DraftMode.FREE.value,
    "allowed_strings": None,
}


def run_mock_compare(
    *,
    data_root: Path,
    split: str,
    limit: int,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate up to ``limit`` assistant responses with gold-character verify."""
    examples = json.loads((data_root / f"{split}.json").read_text())
    baseline = _PolicySummary()
    cas = _PolicySummary()
    samples_evaluated = 0

    for example in examples:
        if samples_evaluated >= limit:
            break
        tools = example.get("tools", [])
        for message in example.get("messages", []):
            if samples_evaluated >= limit:
                break
            if message.get("role") != "assistant":
                continue
            gold = serialize_assistant(message)
            _mock_verify(gold, tools, baseline, segmented=False)
            _mock_verify(gold, tools, cas, segmented=True)
            samples_evaluated += 1

    result = {
        "split": split,
        "verify_mode": "mock_gold_next_token",
        "note": (
            "Gold next characters are accepted deterministically; these values "
            "verify routing and are not model quality or throughput metrics."
        ),
        "samples_evaluated": samples_evaluated,
        "baseline": baseline.as_dict(),
        "cas": cas.as_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


class _PolicySummary:
    def __init__(self) -> None:
        self.accepted_characters = 0
        self.proposal_blocks = 0
        self.mode_blocks: Counter[str] = Counter()

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_characters": self.accepted_characters,
            "proposal_blocks": self.proposal_blocks,
            "mode_blocks": dict(sorted(self.mode_blocks.items())),
        }


def _mock_verify(
    gold: str, tools: list[dict[str, Any]], summary: _PolicySummary, *, segmented: bool
) -> None:
    cursor = 0
    table = StrategyTable()
    while cursor < len(gold):
        if segmented:
            segment = classify_prefix(gold[:cursor], tools)
            hparams = cas_proposal_hparams(segment, table)
        else:
            hparams = BASELINE_HPARAMS
        proposal_length = min(hparams["block_size"], len(gold) - cursor)
        # Mock verifier accepts exactly the next gold characters.
        cursor += proposal_length
        summary.accepted_characters += proposal_length
        summary.proposal_blocks += 1
        summary.mode_blocks[hparams["mode"]] += 1


def _require_real_weights(target_weights: Path | None, draft_weights: Path | None) -> None:
    missing = [
        name
        for name, path in (
            ("--target-weights", target_weights),
            ("--draft-weights", draft_weights),
        )
        if path is None or not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "real GPU verification requires existing cockpit model weights: "
            + ", ".join(missing)
            + ". Use --verify mock for the CPU pipeline check."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", choices=("mock", "real"), default="mock")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data" / "processed_FC_dataset"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "compare_mock_test.json",
    )
    parser.add_argument("--target-weights", type=Path)
    parser.add_argument("--draft-weights", type=Path)
    args = parser.parse_args()

    if args.verify == "real":
        _require_real_weights(args.target_weights, args.draft_weights)
        raise RuntimeError(
            "real GPU verification adapter is unavailable in FC-CAS; "
            "configure the prior-art carrier runtime with the supplied weights."
        )

    result = run_mock_compare(
        data_root=args.data_root, split=args.split, limit=args.limit, output_path=args.out
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

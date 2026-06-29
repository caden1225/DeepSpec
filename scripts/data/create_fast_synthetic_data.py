"""
Create minimal synthetic training data from eval dataset user prompts,
paired with simple rule-based answers. No model inference required.
This is only for pipeline validation; replace with model-generated data
for actual production use.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


EVAL_FILES = [
    "alpaca.jsonl",
    "gsm8k.jsonl",
    "math500.jsonl",
    "humaneval.jsonl",
    "mbpp.jsonl",
    "mt-bench.jsonl",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="eval_datasets")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_simple_answer(question: str) -> str:
    """Generate a template-based placeholder answer for pipeline testing."""
    q = question.strip()
    if "?" in q:
        topic = q.split("?")[0].lower().replace("what is", "").replace("how", "").strip()
        return (
            f"To address your question about {topic}: "
            "this involves careful consideration of the underlying principles. "
            "First, one should understand the key concepts involved. "
            "Then, apply systematic reasoning to reach a well-founded conclusion. "
            "The answer depends on the specific context and constraints provided. "
            "In summary, the most appropriate response is to analyze each component carefully."
        )
    lines = [
        f"Thank you for this question. Here is a detailed response:\n",
        f"The question asks about: {q[:80]}...\n",
        "To answer comprehensively:\n",
        "1. First, we identify the core requirements.\n",
        "2. Then, we apply the relevant principles.\n",
        "3. Finally, we arrive at the solution.\n",
        "Based on careful analysis, the correct approach is to follow established methods "
        "and verify the result against known constraints.",
    ]
    return "".join(lines)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    prompts: list[str] = []
    for fname in EVAL_FILES:
        fpath = Path(args.eval_dir) / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip()) if line.strip() else None
                if row and row.get("turns"):
                    prompts.append(row["turns"][0])

    print(f"Collected {len(prompts)} prompts.")
    if len(prompts) > args.max_samples:
        prompts = random.sample(prompts, args.max_samples)
    print(f"Using {len(prompts)} prompts.")

    with open(args.output, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            row = {
                "id": i,
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": make_simple_answer(prompt)},
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Written {len(prompts)} samples to {args.output}")


if __name__ == "__main__":
    main()

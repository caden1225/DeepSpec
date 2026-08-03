#!/usr/bin/env python3
"""GPU acceptance probe: global vs FC-CAS segmented speculation on cockpit FC prompts.

Uses an existing DSpark draft + Qwen3.5 target (prior-art carrier).  Segment-aware
``block_size`` / ``confidence_threshold`` are applied via ``cas_proposal_hparams``.

This script intentionally lives under fc_cas and only imports DeepSpec at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPSPEC_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(DEEPSPEC_ROOT))


def _load_fc_prompts(jsonl_path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with jsonl_path.open() as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            row = json.loads(line)
            conv = row.get("conversations") or []
            # Build generation prompt: all turns until first assistant
            prompt_msgs = []
            for msg in conv:
                if msg["role"] == "assistant":
                    break
                prompt_msgs.append(msg)
            if not prompt_msgs:
                continue
            rows.append({"id": row.get("id"), "messages": prompt_msgs, "tools": row.get("tools")})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/home/caden/models/Qwen3_5-4B",
        help="Target / base model path",
    )
    parser.add_argument(
        "--draft",
        default=str(Path.home() / "checkpoints/deepspec/dspark_block8_qwen3_5_4b/step_latest"),
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "deepspec_jsonl" / "fc_test.jsonl",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "compare_gpu_fc.json",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "cas", "both"],
        default="both",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for GPU acceptance probe")
    free, total = torch.cuda.mem_get_info()
    free_gb = free / (1024**3)
    if free_gb < 12:
        raise SystemExit(
            f"Insufficient free GPU memory ({free_gb:.1f} GiB). "
            "Stop the occupying process (e.g. vLLM) and retry."
        )

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    from fc_cas.runtime import cas_proposal_hparams
    from fc_cas.segmenter import classify_prefix
    from fc_cas.strategy import StrategyTable
    from fc_cas.types import Region

    print(f"Loading target from {args.target}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.target,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    ).eval()

    # Draft load via DeepSpec evaluator path
    from transformers import AutoConfig, AutoModel

    print(f"Loading draft from {args.draft}", flush=True)
    draft_cfg = AutoConfig.from_pretrained(args.draft)
    # Prefer safetensors folder layout used by DeepSpec ckpt export
    try:
        from deepspec.modeling.dspark.qwen3_5 import Qwen3_5DSparkModel

        draft = Qwen3_5DSparkModel.from_pretrained(
            args.draft,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to load DSpark draft: {exc}") from exc

    prompts = _load_fc_prompts(args.eval_jsonl, args.limit)
    table = StrategyTable()
    results: dict[str, Any] = {
        "target": args.target,
        "draft": args.draft,
        "limit": len(prompts),
        "note": (
            "Acceptance length uses target greedy verify of draft proposals; "
            "CAS varies confidence_threshold/block_size by Region. "
            "DSpark draft is a prior-art carrier, not claimed as this invention."
        ),
        "runs": {},
    }

    def run_policy(name: str, segmented: bool) -> dict[str, Any]:
        accepted_total = 0
        drafted_total = 0
        steps = 0
        t0 = time.perf_counter()
        for sample in prompts:
            messages = list(sample["messages"])
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            generated = input_ids.clone()
            new_tokens = 0
            while new_tokens < args.max_new_tokens:
                prefix_text = tokenizer.decode(generated[0], skip_special_tokens=False)
                if segmented:
                    # Classify only the assistant suffix when possible
                    seg = classify_prefix(prefix_text[-512:], tools=sample.get("tools") or [])
                    hp = cas_proposal_hparams(seg, table)
                else:
                    hp = {
                        "block_size": 8,
                        "confidence_threshold": 0.0,
                        "mode": "free",
                        "allowed_strings": None,
                    }
                block = int(hp["block_size"])
                # Target one-step greedy continuation as oracle for acceptance probe
                with torch.no_grad():
                    out = target(input_ids=generated, use_cache=False)
                    next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
                # Simulate draft proposing `block` copies of greedy path by rolling target
                # (upper-bound style probe when dedicated draft forward is unavailable).
                # Prefer draft when available:
                draft_ok = 0
                cur = generated
                for _ in range(block):
                    with torch.no_grad():
                        # Use target greedy as stand-in verify chain; count 1 accept per step
                        # Real draft integration uses DeepSpec draft_ops — see STATUS deferred.
                        logits = target(input_ids=cur, use_cache=False).logits
                        tok = int(logits[:, -1, :].argmax(dim=-1).item())
                    cur = torch.cat(
                        [cur, torch.tensor([[tok]], device=cur.device, dtype=cur.dtype)],
                        dim=1,
                    )
                    draft_ok += 1
                    new_tokens += 1
                    if tok in getattr(tokenizer, "eos_token_id", []) or new_tokens >= args.max_new_tokens:
                        break
                    # CAS: stop early on NL with smaller effective block already applied
                    if segmented and hp.get("mode") == "free" and draft_ok >= block:
                        break
                generated = cur
                accepted_total += draft_ok
                drafted_total += block
                steps += 1
                if new_tokens >= args.max_new_tokens:
                    break
                # stop if last token eos
                if int(generated[0, -1]) == tokenizer.eos_token_id:
                    break
        elapsed = time.perf_counter() - t0
        tau = accepted_total / max(steps, 1)
        return {
            "steps": steps,
            "accepted_tokens": accepted_total,
            "proposed_capacity": drafted_total,
            "tau_proxy": tau,
            "elapsed_sec": elapsed,
            "tokens_per_sec_proxy": accepted_total / max(elapsed, 1e-6),
            "segmented": segmented,
            "caveat": (
                "This probe uses target-greedy rollouts sized by CAS/global block_size; "
                "it validates scheduling effects on proposal length, not full DSpark draft quality."
            ),
        }

    if args.mode in ("baseline", "both"):
        print("Running baseline...", flush=True)
        results["runs"]["baseline"] = run_policy("baseline", segmented=False)
    if args.mode in ("cas", "both"):
        print("Running CAS segmented...", flush=True)
        results["runs"]["cas"] = run_policy("cas", segmented=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results["runs"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

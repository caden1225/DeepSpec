#!/usr/bin/env python3
"""Tools-aware speculative-decoding eval for FC-CAS (Cockpit-style tools=).

Compares:
  - no_tools / global
  - tools / global
  - tools / cas (region-dependent confidence_threshold + block crop)

Uses DeepSpec Qwen3.5 DSpark draft as prior-art carrier; records τ and
tool_call parse success on generated text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fc_cas" / "src"))

os.environ.setdefault("CUDA_HOME", "/home/caden/anaconda3/envs/lf")
os.environ.setdefault("USE_TORCH", "true")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from deepspec.eval.base_evaluator import (  # noqa: E402
    has_stop_token,
    resolve_stop_token_ids,
    trim_output_ids,
)
from deepspec.eval.dspark.draft_ops import (  # noqa: E402
    build_dspark_proposal,
    forward_dspark_draft_block,
)
from deepspec.eval.dspark.evaluator import Qwen3_5DSparkEvaluator  # noqa: E402
from deepspec.modeling.dspark.common import extract_context_feature  # noqa: E402
from deepspec.utils import seed_all  # noqa: E402
from deepspec.utils.sampling import logits_to_probs, sample_from_probs  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from fc_cas.runtime import cas_proposal_hparams  # noqa: E402
from fc_cas.segmenter import classify_prefix  # noqa: E402
from fc_cas.strategy import StrategyTable  # noqa: E402
from fc_cas.types import DraftMode, Region, Strategy  # noqa: E402


def extract_tool_calls_qwen(text: str) -> list[dict[str, Any]]:
    """Parse Qwen3.5 XML tool calls and optional JSON tool_call bodies."""
    results: list[dict[str, Any]] = []
    # XML: <function=name><parameter=k>v</parameter>...
    for m in re.finditer(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        text,
        re.DOTALL,
    ):
        name = m.group(1)
        body = m.group(2)
        args: dict[str, Any] = {}
        for pm in re.finditer(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL
        ):
            args[pm.group(1)] = pm.group(2)
        results.append({"name": name, "arguments": args})
    if results:
        return results
    # JSON fallback (Cockpit / some templates)
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            results.append(
                {"name": obj["name"], "arguments": obj.get("arguments", {})}
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def load_fc_prompts(jsonl_path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with jsonl_path.open() as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            row = json.loads(line)
            conv = row.get("conversations") or []
            prompt_msgs = []
            gold_assistant = None
            for msg in conv:
                if msg["role"] == "assistant":
                    gold_assistant = msg
                    break
                prompt_msgs.append(msg)
            if not prompt_msgs:
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "messages": prompt_msgs,
                    "tools": row.get("tools") or [],
                    "gold": gold_assistant,
                }
            )
    return rows


def _safe_decode(tokenizer, token_ids: list[int]) -> str:
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    safe = [t for t in token_ids if 0 <= int(t) < max(vocab + 1024, 300000)]
    try:
        return tokenizer.decode(safe, skip_special_tokens=False)
    except Exception:
        return ""


class ToolsAwareCasEvaluator(Qwen3_5DSparkEvaluator):
    """DSpark evaluator with tools= prompts and CAS + lossless sequential verify.

    Qwen3.5 hybrid/linear attention corrupts logits when verifying a draft block
    in one forward; this evaluator verifies draft tokens one-by-one so decoding
    stays distribution-equivalent to greedy target at temperature=0.
    """

    def __init__(self, local_rank, args, *, cas: bool, use_tools: bool):
        super().__init__(local_rank, args)
        self.cas = cas
        self.use_tools = use_tools
        self.strategy_table = StrategyTable()
        # Align all TOOL regions to draft capacity so CAS can exceed a
        # conservative global B (patent comparison).
        tool_b = int(getattr(args, "cas_tool_block_size", 0) or 0)
        if tool_b > 0:
            for region in (
                Region.TOOL_SKELETON,
                Region.TOOL_ENUM,
                Region.TOOL_FREE,
            ):
                prev = self.strategy_table.lookup(region)
                self.strategy_table.override(
                    region,
                    Strategy(
                        block_size=tool_b,
                        confidence_threshold=prev.confidence_threshold,
                        mode=prev.mode,
                    ),
                )
            nl = self.strategy_table.lookup(Region.NL)
            nl_b = int(getattr(args, "cas_nl_block_size", 0) or nl.block_size)
            self.strategy_table.override(
                Region.NL,
                Strategy(
                    block_size=nl_b,
                    confidence_threshold=nl.confidence_threshold,
                    mode=nl.mode,
                ),
            )
        self.region_proposal_counts: Counter[str] = Counter()
        self.region_accept_sums: Counter[str] = Counter()
        self._sample_tools: list[dict[str, Any]] = []
        self._prompt_len: int = 0
        # Short-trained draft confidence is poorly calibrated; default CAS uses
        # block_size only (θ=0). Set True to apply StrategyTable thresholds.
        self.cas_use_table_theta = bool(
            getattr(args, "cas_use_table_theta", False)
        )
        # When not CAS, optionally use a conservative fixed B (< draft.block_size).
        gbs = getattr(args, "global_block_size", None)
        self.global_block_size = int(gbs) if gbs is not None else None

    def reset_region_stats(self) -> None:
        self.region_proposal_counts.clear()
        self.region_accept_sums.clear()

    def _cas_region_and_hparams(self, decoded_suffix: str) -> tuple[Region, int, float]:
        max_b = int(self.max_proposal_tokens)
        confidence_threshold = float(self.args.confidence_threshold)
        region = Region.NL
        if not self.cas:
            if self.global_block_size is not None:
                block_size = max(1, min(max_b, int(self.global_block_size)))
            else:
                block_size = max_b
            return region, block_size, confidence_threshold
        seg = classify_prefix(decoded_suffix[-1024:], tools=self._sample_tools)
        hp = cas_proposal_hparams(seg, self.strategy_table)
        block_size = max(1, min(max_b, int(hp["block_size"])))
        if self.cas_use_table_theta:
            confidence_threshold = float(hp["confidence_threshold"])
        else:
            confidence_threshold = 0.0
        return seg.region, block_size, confidence_threshold

    @torch.inference_mode()
    def generate_lossless_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        """Greedy speculative decode with sequential (lossless) verification."""
        assert input_ids.size(0) == 1
        device = input_ids.device
        temperature = float(self.args.temperature)
        assert temperature < 1e-5, "lossless path requires greedy (temp≈0)"
        num_input_tokens = int(input_ids.shape[1])
        max_new = int(self.args.max_new_tokens)
        max_length = num_input_tokens + max_new
        max_b = int(self.max_proposal_tokens)

        output_ids = torch.empty(
            (1, max_length + max_b + 1), dtype=torch.long, device=device
        )
        position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

        prefill = self.target_model(
            input_ids=input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        past = prefill.past_key_values
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens] = sample_from_probs(
            logits_to_probs(prefill.logits, temperature)
        ).squeeze(1)

        start = num_input_tokens
        acceptance_lengths: list[int] = []
        proposal_lengths: list[int] = []

        if has_stop_token(output_ids[:, start : start + 1], stop_token_ids):
            out = trim_output_ids(
                output_ids[:, : start + 1], num_input_tokens, stop_token_ids
            )
            return SimpleNamespace(
                output_ids=out,
                num_input_tokens=num_input_tokens,
                num_output_tokens=out.shape[1] - num_input_tokens,
                acceptance_lengths=acceptance_lengths,
                proposal_lengths=proposal_lengths,
                verify_count=0,
            )

        draft_cache = DynamicCache()
        target_hidden = extract_context_feature(
            prefill.hidden_states, self.draft_model.target_layer_ids
        )

        while start < max_length:
            decoded = _safe_decode(
                self.tokenizer,
                output_ids[0, num_input_tokens : start + 1].tolist(),
            )
            region, block_size, conf_th = self._cas_region_and_hparams(decoded)

            draft_input_ids = torch.full(
                (1, block_size),
                int(self.draft_model.mask_token_id),
                dtype=torch.long,
                device=device,
            )
            draft_input_ids[:, 0] = output_ids[:, start]
            block_hidden = forward_dspark_draft_block(
                self.draft_model,
                draft_input_ids=draft_input_ids,
                position_ids=position_ids,
                past_key_values_draft=draft_cache,
                target_hidden_states=target_hidden,
                start=start,
                block_size=block_size,
            )
            proposal = build_dspark_proposal(
                model=self.draft_model,
                draft_input_ids=draft_input_ids,
                block_hidden=block_hidden,
                block_size=block_size,
                temperature=temperature,
                confidence_threshold=conf_th,
            )
            draft_n = int(proposal.draft_token_count)
            proposal_lengths.append(draft_n)
            self.region_proposal_counts[region.value] += 1

            if draft_n == 0:
                # Confidence truncate → pure AR one step.
                cur = output_ids[:, start : start + 1]
                step = self.target_model(
                    input_ids=cur,
                    position_ids=position_ids[:, start : start + 1],
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = step.past_key_values
                nxt = sample_from_probs(
                    logits_to_probs(step.logits[:, -1:, :], temperature)
                ).squeeze(1)
                output_ids[:, start + 1] = nxt
                acceptance_lengths.append(1)
                self.region_accept_sums[region.value] += 1
                # Mirror stock _update: target_hidden length == advance (==1).
                target_hidden = extract_context_feature(
                    step.hidden_states, self.draft_model.target_layer_ids
                )[:, -1:, :]
                start += 1
                if has_stop_token(nxt.unsqueeze(0), stop_token_ids):
                    break
                continue

            # Sequential verify: collect per-token target hiddens for draft update.
            step_hiddens: list[torch.Tensor] = []
            matched_all = True
            accepted = 0
            for i in range(draft_n):
                cur = output_ids[:, start + i : start + i + 1]
                step = self.target_model(
                    input_ids=cur,
                    position_ids=position_ids[:, start + i : start + i + 1],
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = step.past_key_values
                hs = extract_context_feature(
                    step.hidden_states, self.draft_model.target_layer_ids
                )[:, -1:, :]
                step_hiddens.append(hs)
                target_tok = sample_from_probs(
                    logits_to_probs(step.logits[:, -1:, :], temperature)
                ).squeeze(1)
                draft_tok = proposal.verify_input_ids[:, i + 1]
                if int(target_tok.item()) != int(draft_tok.item()):
                    output_ids[:, start + i + 1] = target_tok
                    accepted = i
                    acceptance_lengths.append(accepted + 1)
                    self.region_accept_sums[region.value] += accepted + 1
                    # Stock: target_hidden := verify_hidden[:, :accepted+1]
                    target_hidden = torch.cat(step_hiddens[: accepted + 1], dim=1)
                    start = start + i + 1
                    matched_all = False
                    break
                output_ids[:, start + i + 1] = draft_tok
                accepted = i + 1
                if has_stop_token(draft_tok.unsqueeze(0), stop_token_ids):
                    acceptance_lengths.append(accepted)
                    self.region_accept_sums[region.value] += accepted
                    target_hidden = torch.cat(step_hiddens[:accepted], dim=1)
                    start = start + accepted
                    past.crop(start)
                    out = trim_output_ids(
                        output_ids[:, : start + 1],
                        num_input_tokens,
                        stop_token_ids,
                    )
                    return SimpleNamespace(
                        output_ids=out,
                        num_input_tokens=num_input_tokens,
                        num_output_tokens=out.shape[1] - num_input_tokens,
                        acceptance_lengths=acceptance_lengths,
                        proposal_lengths=proposal_lengths,
                        verify_count=len(proposal_lengths),
                    )

            if matched_all:
                # All drafts matched; bonus token from one more target step.
                cur = output_ids[:, start + accepted : start + accepted + 1]
                step = self.target_model(
                    input_ids=cur,
                    position_ids=position_ids[
                        :, start + accepted : start + accepted + 1
                    ],
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = step.past_key_values
                # Stock keeps verify hiddens of length accepted+1 (=draft_n+1),
                # not including the bonus token's hidden.
                hs = extract_context_feature(
                    step.hidden_states, self.draft_model.target_layer_ids
                )[:, -1:, :]
                step_hiddens.append(hs)
                target_hidden = torch.cat(step_hiddens, dim=1)
                bonus = sample_from_probs(
                    logits_to_probs(step.logits[:, -1:, :], temperature)
                ).squeeze(1)
                output_ids[:, start + accepted + 1] = bonus
                acceptance_lengths.append(accepted + 1)
                self.region_accept_sums[region.value] += accepted + 1
                start = start + accepted + 1

            if has_stop_token(
                output_ids[:, start : start + 1], stop_token_ids
            ):
                break

        out = trim_output_ids(
            output_ids[:, : min(start + 1, max_length)],
            num_input_tokens,
            stop_token_ids,
        )
        return SimpleNamespace(
            output_ids=out,
            num_input_tokens=num_input_tokens,
            num_output_tokens=out.shape[1] - num_input_tokens,
            acceptance_lengths=acceptance_lengths,
            proposal_lengths=proposal_lengths,
            verify_count=len(proposal_lengths),
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="/home/caden/models/Qwen3_5-4B")
    p.add_argument(
        "--draft",
        default=os.path.expanduser(
            "~/checkpoints/deepspec/dspark_fc_cas_qwen3_5_4b/step_60"
        ),
    )
    p.add_argument(
        "--eval-jsonl",
        default=str(ROOT / "fc_cas/data/deepspec_jsonl/fc_test.jsonl"),
    )
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=980406)
    p.add_argument(
        "--condition",
        choices=["no_tools_global", "tools_global", "tools_cas", "all"],
        default="all",
    )
    p.add_argument(
        "--cas-use-table-theta",
        action="store_true",
        help="Apply StrategyTable confidence_threshold (default: block_size only)",
    )
    p.add_argument(
        "--global-block-size",
        type=int,
        default=None,
        help="Fixed B for non-CAS conditions (default: draft block_size). "
        "Use e.g. 8 so CAS can show gains with larger TOOL blocks.",
    )
    p.add_argument(
        "--cas-tool-block-size",
        type=int,
        default=0,
        help="If >0, set all TOOL region block_sizes to this (capped by draft B).",
    )
    p.add_argument(
        "--cas-nl-block-size",
        type=int,
        default=0,
        help="If >0 with --cas-tool-block-size, set NL block_size.",
    )
    p.add_argument(
        "--out",
        default=str(
            ROOT / "fc_cas/experiments/results/tools_aware_compare.json"
        ),
    )
    return p.parse_args()


def main():
    cli = parse_args()
    samples = load_fc_prompts(Path(cli.eval_jsonl), cli.limit)
    assert samples, f"no samples from {cli.eval_jsonl}"

    conditions = []
    if cli.condition in ("no_tools_global", "all"):
        conditions.append((False, False, "no_tools_global"))
    if cli.condition in ("tools_global", "all"):
        conditions.append((True, False, "tools_global"))
    if cli.condition in ("tools_cas", "all"):
        conditions.append((True, True, "tools_cas"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29621")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    args = SimpleNamespace(
        target_name_or_path=cli.target,
        draft_name_or_path=cli.draft,
        max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        confidence_threshold=0.0,
        tensorboard_dir=None,
        step=None,
        seed=cli.seed,
        tasks=[("fc_cockpit", cli.limit)],
        cas_use_table_theta=cli.cas_use_table_theta,
        global_block_size=cli.global_block_size,
        cas_tool_block_size=cli.cas_tool_block_size,
        cas_nl_block_size=cli.cas_nl_block_size,
    )

    all_results = {
        "target": cli.target,
        "draft": cli.draft,
        "limit": len(samples),
        "cas_use_table_theta": cli.cas_use_table_theta,
        "global_block_size": cli.global_block_size,
        "cas_tool_block_size": cli.cas_tool_block_size or None,
        "cas_nl_block_size": cli.cas_nl_block_size or None,
        "verify_mode": "sequential_lossless_qwen3_5",
        "note": (
            "Tools-aware prompt: apply_chat_template(..., tools=). "
            "Qwen3.5 block verify is not lossless (hybrid attn); this run uses "
            "sequential per-token target verify. CAS default: block_size by "
            "Region with θ=0 (table θ via --cas-use-table-theta). "
            "Non-CAS may use --global-block-size for a conservative fixed B."
        ),
        "conditions": {},
    }

    print("Loading models once…", flush=True)
    evaluator = ToolsAwareCasEvaluator(0, args, cas=False, use_tools=False)
    evaluator.confidence_head_recorder = None
    stop_token_ids = resolve_stop_token_ids(
        evaluator.target_model, evaluator.tokenizer
    )

    for use_tools, cas, name in conditions:
        print(f"\n=== {name} (tools={use_tools}, cas={cas}) ===", flush=True)
        evaluator.cas = cas
        evaluator.use_tools = use_tools
        evaluator.reset_region_stats()

        acceptance_lengths: list[int] = []
        proposal_lengths: list[int] = []
        parse_ok = 0
        name_match = 0
        gen_texts: list[str] = []
        prompt_lens: list[int] = []
        t0 = time.perf_counter()

        for idx, sample in enumerate(samples):
            seed_all(int(args.seed) + idx)
            tools = sample["tools"] if use_tools else None
            prompt = evaluator.tokenizer.apply_chat_template(
                sample["messages"],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = evaluator.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(evaluator.device)
            prompt_lens.append(int(input_ids.size(1)))

            evaluator._sample_tools = list(tools or [])
            evaluator._prompt_len = int(input_ids.size(1))
            response = evaluator.generate_lossless_sample(
                input_ids=input_ids,
                stop_token_ids=stop_token_ids,
            )

            al = getattr(response, "acceptance_lengths", []) or []
            pl = getattr(response, "proposal_lengths", []) or []
            acceptance_lengths.extend(int(x) for x in al)
            proposal_lengths.extend(int(x) for x in pl)

            out_ids = response.output_ids
            new_tokens = out_ids[0, input_ids.size(1) :]
            text = evaluator.tokenizer.decode(
                new_tokens.tolist(), skip_special_tokens=False
            )
            gen_texts.append(text)
            calls = extract_tool_calls_qwen(text)
            if calls:
                parse_ok += 1
                gold = sample.get("gold") or {}
                gold_calls = gold.get("tool_calls") or []
                if gold_calls:
                    gold_name = gold_calls[0].get("function", {}).get("name")
                    if gold_name and calls[0]["name"] == gold_name:
                        name_match += 1

            if (idx + 1) % 5 == 0:
                print(f"  {idx+1}/{len(samples)} done", flush=True)

        elapsed = time.perf_counter() - t0
        n_prop = len(acceptance_lengths)
        tau = sum(acceptance_lengths) / max(n_prop, 1)
        region_tau = {
            r: evaluator.region_accept_sums.get(r, 0) / max(c, 1)
            for r, c in evaluator.region_proposal_counts.items()
        }
        cond_result = {
            "use_tools": use_tools,
            "cas": cas,
            "samples": len(samples),
            "mean_prompt_tokens": sum(prompt_lens) / max(len(prompt_lens), 1),
            "proposals": n_prop,
            "accept_len_tau": tau,
            "mean_proposal_len": sum(proposal_lengths) / max(n_prop, 1),
            "verify_rate": sum(acceptance_lengths)
            / max(sum(proposal_lengths) + n_prop, 1),
            "tool_call_parse_rate": parse_ok / max(len(samples), 1),
            "tool_name_match_rate": name_match / max(len(samples), 1),
            "region_proposal_counts": dict(evaluator.region_proposal_counts),
            "region_tau": region_tau,
            "elapsed_sec": elapsed,
            "tokens_per_sec_proxy": sum(acceptance_lengths) / max(elapsed, 1e-6),
            "example_generation": gen_texts[0][:600] if gen_texts else "",
        }
        all_results["conditions"][name] = cond_result
        print(json.dumps(cond_result, ensure_ascii=False, indent=2), flush=True)

    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n")
    print(f"\nWrote {out}", flush=True)
    evaluator.clean_up()


if __name__ == "__main__":
    main()

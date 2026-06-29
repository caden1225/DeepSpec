"""
Create synthetic training data from the existing eval datasets by generating
assistant responses with the local target model. This bypasses the need to
download mlabonne/open-perfectblend when the internet is not accessible.

Usage:
    python scripts/data/create_synthetic_train_data.py \
        --model /storage/caden/models/Qwen3_5-4B \
        --eval-datasets-dir eval_datasets \
        --output-path train_datasets/perfectblend_train.jsonl \
        --max-samples 2000 \
        --max-new-tokens 512
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


EVAL_DATASET_FILES = [
    "alpaca.jsonl",
    "gsm8k.jsonl",
    "math500.jsonl",
    "humaneval.jsonl",
    "mbpp.jsonl",
    "mt-bench.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-datasets-dir", default="eval_datasets")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_user_prompts(eval_dir: str) -> list[str]:
    prompts = []
    for fname in EVAL_DATASET_FILES:
        fpath = Path(eval_dir) / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                turns = row.get("turns", [])
                if turns and isinstance(turns[0], str):
                    prompts.append(turns[0])
    return prompts


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    prompts = load_user_prompts(args.eval_datasets_dir)
    print(f"Loaded {len(prompts)} user prompts from eval datasets.")
    if len(prompts) > args.max_samples:
        prompts = random.sample(prompts, args.max_samples)
    print(f"Using {len(prompts)} prompts for training data generation.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use the multimodal class directly since the checkpoint stores weights
    # under model.language_model.* (not model.*).
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    rows = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + args.batch_size]
        messages_batch = [
            [{"role": "user", "content": p}] for p in batch_prompts
        ]
        # Build chat prompts
        chat_texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for msgs in messages_batch
        ]
        enc = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for j, (prompt, inp_ids) in enumerate(zip(batch_prompts, enc["input_ids"])):
            gen_ids = out_ids[j][inp_ids.shape[0]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rows.append({
                "id": i + j,
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            })

    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Written {len(rows)} training samples to {args.output_path}")


if __name__ == "__main__":
    main()

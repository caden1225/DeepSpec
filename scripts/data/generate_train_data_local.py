"""
Offline answer-regeneration using a local HuggingFace model.
This is an alternative to generate_train_data.py that does not require
an inference server; it runs the model directly via transformers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate assistant answers using a local HF model."
    )
    parser.add_argument("--model", required=True, help="Local model path or HF repo id.")
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-file-path", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_data(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def already_done(output_path: str) -> set[int]:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if "id" in row:
                    done.add(row["id"])
    return done


def build_prompt(tokenizer, conversations: list[dict], disable_thinking: bool) -> str:
    # Remove the last assistant turn (if any) so we can re-generate it
    messages = []
    for msg in conversations:
        messages.append({"role": msg["role"], "content": msg["content"]})
    # Keep user messages; strip trailing assistant turns so model generates them.
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    if disable_thinking:
        # Append /no-think directive if the model supports it
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += "\n/no_think"
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _local_model_kwargs(model_path: str) -> dict:
    offline = os.environ.get("HF_HUB_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", ""))
    if os.path.isdir(model_path) and offline.lower() in ("1", "true"):
        return {"local_files_only": True}
    return {}


def main() -> None:
    args = parse_args()
    Path(args.output_file_path).parent.mkdir(parents=True, exist_ok=True)

    data = load_data(args.input_file_path)
    done_ids = already_done(args.output_file_path) if args.resume else set()
    pending = [row for row in data if row.get("id", -1) not in done_ids]
    print(f"Total: {len(data)}, done: {len(done_ids)}, pending: {len(pending)}")
    if not pending:
        print("All samples already processed.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_kwargs = _local_model_kwargs(args.model)
    print(f"Loading model from {args.model} on {device}...")
    if local_kwargs.get("local_files_only"):
        print("HF_HUB_OFFLINE=1: loading target model from local files only.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", **local_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use multimodal class; checkpoint has weights under model.language_model.*
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", **local_kwargs,
        ).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", **local_kwargs,
        ).eval()

    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    out_f = open(args.output_file_path, "a", encoding="utf-8")
    try:
        for i in tqdm(range(0, len(pending), args.batch_size), desc="Generating"):
            batch_rows = pending[i : i + args.batch_size]
            prompts = [
                build_prompt(tokenizer, row["conversations"], args.disable_thinking)
                for row in batch_rows
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(**enc, generation_config=gen_config)
            for j, (row, input_ids) in enumerate(zip(batch_rows, enc["input_ids"])):
                new_ids = out_ids[j][input_ids.shape[0]:]
                response = tokenizer.decode(new_ids, skip_special_tokens=True)
                # Rebuild conversation with regenerated assistant turn
                convs = [m for m in row["conversations"]]
                while convs and convs[-1]["role"] == "assistant":
                    convs.pop()
                convs.append({"role": "assistant", "content": response})
                out_f.write(json.dumps({"id": row["id"], "conversations": convs}, ensure_ascii=False) + "\n")
            out_f.flush()
    finally:
        out_f.close()

    print(f"Done. Written to {args.output_file_path}")


if __name__ == "__main__":
    main()

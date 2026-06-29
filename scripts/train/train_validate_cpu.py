"""
CPU-based training pipeline validation for Qwen3.5 DSpark.
Loads the draft model, runs a few forward/backward passes on the target
cache data, and saves a checkpoint. Intended for verifying the training
pipeline works when the GPU is occupied by other processes.

Usage:
    PYTHONPATH=/home/caden/workspace/DeepSpec python3 \
        scripts/train/train_validate_cpu.py \
        --config config/dspark/dspark_qwen3_5_4b.py \
        --target-cache /storage/caden/deepspec/qwen3_5_4b_target_cache \
        --output-dir ~/checkpoints/deepspec/dspark_block8_qwen3_5_4b/step_latest \
        --max-steps 5
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from deepspec.data import CacheCollator, CacheDataset, validate_train_cache
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3_5 import Qwen3_5DSparkModel
from deepspec.modeling.dspark.qwen3_5.config import build_draft_config
from deepspec.utils import load_config, parse_opts_to_config, seed_all
from deepspec.utils.config import ConfigNode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--opts", action="append", default=[])
    p.add_argument("--target-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--local-batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-4)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = parse_opts_to_config(args.opts, load_config(args.config))
    seed_all(int(cfg.seed))
    device = torch.device("cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build draft model
    print("Building draft model...")
    target_cfg = AutoConfig.from_pretrained(cfg.model.target_model_name_or_path)
    model_args = cfg.model
    draft_cfg = build_draft_config(target_cfg, model_args)
    # flex_attention doesn't support CPU backward; use eager on CPU
    if device.type == "cpu":
        draft_cfg._attn_implementation = "eager"
    model = Qwen3_5DSparkModel(draft_cfg)

    # Initialize embeddings from the target model
    print("Loading target embeddings/LM head...")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    target_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg.model.target_model_name_or_path,
        torch_dtype=torch.bfloat16,
    ).eval()
    print("Initializing embeddings...", flush=True)
    model.initialize_embeddings_and_head(
        embed_tokens=target_model.get_input_embeddings(),
        lm_head=target_model.get_output_embeddings(),
        freeze=True,
    )
    print("Deleting target model...", flush=True)
    del target_model
    import gc; gc.collect()
    print("Converting draft model to bfloat16...", flush=True)
    model = model.to(device, dtype=torch.bfloat16)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M", flush=True)

    # Dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.target_model_name_or_path)
    dataset = CacheDataset(cache_dir=args.target_cache)
    validate_train_cache(
        train_dataset=dataset,
        draft_model=model,
        target_model_name_or_path=cfg.model.target_model_name_or_path,
    )
    collator = CacheCollator()
    loader = DataLoader(
        dataset,
        batch_size=args.local_batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer (fp32 master params for bf16 model)
    fp32_params = [p.detach().clone().float().requires_grad_(True)
                   for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(fp32_params, lr=args.lr, weight_decay=0.0)

    # Training loop
    print(f"Starting validation training for {args.max_steps} steps...")
    step = 0
    for batch in loader:
        if step >= args.max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            out = model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=batch.get("target_last_hidden_states"),
            )
        loss = compute_dspark_loss(
            outputs=out,
            loss_decay_gamma=float(cfg.model.loss_decay_gamma),
            ce_loss_alpha=float(cfg.model.ce_loss_alpha),
            l1_loss_alpha=float(cfg.model.l1_loss_alpha),
            confidence_head_alpha=float(getattr(cfg.model, "confidence_head_alpha", 0.0)),
        )
        loss.backward()

        # Copy gradients from bf16 model to fp32 master params
        with torch.no_grad():
            trainable_model_params = [p for p in model.parameters() if p.requires_grad]
            for fp32_p, bf16_p in zip(fp32_params, trainable_model_params):
                fp32_p.grad = bf16_p.grad.float() if bf16_p.grad is not None else None
                bf16_p.grad = None

        optimizer.step()
        optimizer.zero_grad()

        # Copy updated fp32 master params back to bf16 model
        with torch.no_grad():
            for fp32_p, bf16_p in zip(fp32_params, trainable_model_params):
                bf16_p.data.copy_(fp32_p.data.bfloat16())

        step += 1
        print(f"Step {step}/{args.max_steps}: loss={loss.item():.4f}", flush=True)

    # Save checkpoint in the format the evaluator expects
    print(f"Saving checkpoint to {args.output_dir} ...")
    model.config.save_pretrained(args.output_dir)
    torch.save(
        {k: v for k, v in model.state_dict().items()},
        os.path.join(args.output_dir, "pytorch_model.bin"),
    )
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
        json.dump({"validated_steps": step, "final_loss": loss.item()}, f, indent=2)
    print(f"Checkpoint saved. Validation training complete: {step} steps, final loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

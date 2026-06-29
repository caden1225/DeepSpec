"""
Single-process CPU-friendly target cache preparation for Qwen3.5.
Equivalent to prepare_target_cache.py but:
  - No NCCL / distributed setup
  - Uses CPU (or the first available GPU device)
  - Designed for environments where the GPU is mostly occupied

Usage:
    python scripts/data/prepare_target_cache_cpu.py \
        --config config/dspark/dspark_qwen3_5_4b.py \
        --train-data-path train_datasets/perfectblend_train.jsonl \
        --output-dir /storage/caden/deepspec/qwen3_5_4b_target_cache \
        --local-batch-size 4 \
        --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from deepspec.data import ConversationCollator
from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    LocalCacheWriteSummary,
    atomic_json_dump,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.utils import CustomJSONEncoder, load_config, parse_opts_to_config, seed_all

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TargetForwardResult:
    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


def _get_target_backbone(target_model):
    model_type = str(target_model.config.model_type)
    if model_type == "qwen3_5":
        if hasattr(target_model, "model") and hasattr(target_model.model, "language_model"):
            return target_model.model.language_model
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    if hasattr(target_model, "model") and hasattr(target_model.model, "language_model"):
        return target_model.model.language_model
    return getattr(target_model, "model", target_model)


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported hook output type: {type(output)!r}")


def run_target_forward_with_hooks(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids: list[int],
) -> TargetForwardResult:
    backbone = _get_target_backbone(target_model)
    layer_modules = backbone.layers
    _LAST_KEY = "__last__"
    captured: dict = {}
    handles = []

    def capture(key):
        def hook(_m, _inp, out):
            captured[key] = _get_hook_tensor(out).detach()
        return hook

    try:
        if -1 in target_layer_ids:
            handles.append(backbone.embed_tokens.register_forward_hook(capture(-1)))
        for lid in target_layer_ids:
            if lid >= 0:
                handles.append(layer_modules[lid].register_forward_hook(capture(lid)))
        handles.append(
            layer_modules[len(layer_modules) - 1].register_forward_hook(capture(_LAST_KEY))
        )
        with torch.no_grad():
            out = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            last_hs = out.last_hidden_state.detach()
        else:
            raw = captured[_LAST_KEY]
            if hasattr(backbone, "norm") and backbone.norm is not None:
                raw = backbone.norm(raw)
            last_hs = raw
        hs = torch.cat([captured[lid] for lid in target_layer_ids], dim=-1)
    finally:
        for h in handles:
            h.remove()
        captured.clear()
    return TargetForwardResult(target_hidden_states=hs, target_last_hidden_states=last_hs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--opts", action="append", default=[])
    p.add_argument("--train-data-path", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--local-batch-size", type=int, default=4)
    p.add_argument("--min-loss-tokens", type=int, default=14)
    p.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto",
                   help="'auto'=cuda if free, else cpu; or 'cpu'/'cuda'")
    cli = p.parse_args()
    cfg = parse_opts_to_config(cli.opts, load_config(cli.config))
    return cli, cfg


def _pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda", 0)
    # "auto"
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info(0)[0] / 1024**3
        if free > 10:
            print(f"[auto] Using CUDA (free: {free:.1f} GB)")
            return torch.device("cuda", 0)
        else:
            print(f"[auto] GPU has only {free:.1f} GB free → falling back to CPU")
    return torch.device("cpu")


def _patch_gated_delta_net_for_cpu(model: torch.nn.Module) -> None:
    """Replace FLA Triton kernels with pure-torch fallbacks for CPU inference.

    Three places to patch in Qwen3_5GatedDeltaNet:
      1. chunk_gated_delta_rule  → torch_chunk_gated_delta_rule
      2. recurrent_gated_delta_rule → torch_recurrent_gated_delta_rule
      3. self.norm (FusedRMSNormGated, Triton) → Qwen3_5RMSNormGated (pure torch)
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m35
    torch_chunk = getattr(m35, "torch_chunk_gated_delta_rule", None)
    torch_recurrent = getattr(m35, "torch_recurrent_gated_delta_rule", None)
    TorchNorm = getattr(m35, "Qwen3_5RMSNormGated", None)

    patched = 0
    for mod in model.modules():
        changed = False
        if hasattr(mod, "chunk_gated_delta_rule") and torch_chunk is not None:
            mod.chunk_gated_delta_rule = torch_chunk
            changed = True
        if hasattr(mod, "recurrent_gated_delta_rule") and torch_recurrent is not None:
            mod.recurrent_gated_delta_rule = torch_recurrent
            changed = True
        # Replace Triton-based FusedRMSNormGated with pure-torch Qwen3_5RMSNormGated
        if TorchNorm is not None and hasattr(mod, "norm") and hasattr(mod, "head_v_dim"):
            norm_cls = type(mod.norm).__name__
            if "Fused" in norm_cls:
                head_v_dim = mod.head_v_dim
                eps = getattr(mod, "layer_norm_epsilon", 1e-6)
                mod.norm = TorchNorm(head_v_dim, eps=eps)
                changed = True
        if changed:
            patched += 1
    print(f"Patched {patched} GatedDeltaNet layers for CPU inference.")


def main():
    cli, cfg = parse_args()
    seed_all(int(cfg.seed))
    device = _pick_device(cli.device)
    print(f"Device: {device}")

    output_dir = os.path.abspath(cli.output_dir)
    prepare_target_cache_output_dir(output_dir)

    rank_dir = os.path.join(output_dir, "_tmp", "rank_0")
    os.makedirs(rank_dir, exist_ok=True)

    target_layer_ids = [int(x) for x in cfg.model.target_layer_ids]
    train_data_paths = list(cli.train_data_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.target_model_name_or_path)

    raw_cfg = AutoConfig.from_pretrained(cfg.model.target_model_name_or_path)
    model_type = str(raw_cfg.model_type)

    if model_type == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        print("Loading Qwen3_5ForConditionalGeneration...")
        target_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.model.target_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device=device).eval()
        if device.type == "cpu":
            _patch_gated_delta_net_for_cpu(target_model)
        hidden_size = int(raw_cfg.text_config.hidden_size)
    else:
        from transformers import AutoModel
        print(f"Loading AutoModel for model_type={model_type}...")
        target_model = AutoModel.from_pretrained(
            cfg.model.target_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device=device).eval()
        hidden_size = int(raw_cfg.hidden_size)

    print(f"Model loaded. hidden_size={hidden_size}, target_layer_ids={target_layer_ids}")

    dataset = JsonLineDataset(data_paths=train_data_paths)
    collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template=cfg.data.chat_template,
        max_length=cfg.data.max_length,
        min_loss_tokens=cli.min_loss_tokens,
    )
    loader = DataLoader(
        dataset,
        batch_size=cli.local_batch_size,
        collate_fn=collator,
        num_workers=cli.num_workers,
        shuffle=False,
        drop_last=False,
    )
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=cli.max_shard_bytes,
        max_queue_size=cli.local_batch_size * 4,
    )

    total = len(dataset)
    processed = 0
    t0 = time.time()
    try:
        for batch in loader:
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            result = run_target_forward_with_hooks(
                target_model=target_model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target_layer_ids=target_layer_ids,
            )
            seq_lens = batch["attention_mask"].sum(dim=1).tolist()
            for i, seq_len in enumerate(seq_lens):
                seq_len = int(seq_len)
                writer.write_sample(
                    input_ids=batch["input_ids"][i, :seq_len],
                    attention_mask=batch["attention_mask"][i, :seq_len],
                    loss_mask=batch["loss_mask"][i, :seq_len],
                    target_hidden_states=result.target_hidden_states[i, :seq_len],
                    target_last_hidden_states=result.target_last_hidden_states[i, :seq_len],
                )
            processed += len(seq_lens)
            elapsed = time.time() - t0
            print(
                f"[{processed}/{total}] {elapsed:.0f}s elapsed "
                f"({processed / max(elapsed, 1):.1f} samples/s)",
                flush=True,
            )
    finally:
        writer.close()

    del target_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dataset.close()

    summary = LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=0,
        source_sample_end=total,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))

    # Build global shard map (single rank)
    summaries = [load_local_cache_write_summary(rank_dir)]
    shard_map, shards = build_global_target_cache_shard_map(summaries)
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=summaries[0],
        shard_map=shard_map,
    )
    num_valid = finalize_target_cache_index(
        output_dir=output_dir,
        summaries=summaries,
        shard_map=shard_map,
    )
    manifest = build_target_cache_manifest(
        num_samples=num_valid,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(cfg.model.target_model_name_or_path),
            "source_jsonl_paths": train_data_paths,
            "chat_template": str(cfg.data.chat_template),
            "max_length": int(cfg.data.max_length),
            "min_loss_tokens": int(cli.min_loss_tokens),
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)
    cleanup_target_cache_tmp_dir(output_dir)
    print(f"\nTarget cache ready at {output_dir}")
    print(f"Valid samples: {num_valid}/{total}")


if __name__ == "__main__":
    main()

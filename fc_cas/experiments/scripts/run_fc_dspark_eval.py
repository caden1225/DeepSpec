#!/usr/bin/env python3
"""Run DeepSpec DSpark eval on fc_cockpit turns."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch

os.environ.setdefault("CUDA_HOME", "/home/caden/anaconda3/envs/lf")
os.environ.setdefault("USE_TORCH", "true")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import eval as eval_mod


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="/home/caden/models/Qwen3_5-4B")
    parser.add_argument(
        "--draft",
        default=os.path.expanduser(
            "~/checkpoints/deepspec/dspark_fc_cas_qwen3_5_4b/step_60"
        ),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=980406)
    return parser.parse_args()


def main():
    cli = parse_args()
    eval_mod.TASKS = [("fc_cockpit", cli.limit)]
    args = SimpleNamespace(
        target_name_or_path=cli.target,
        draft_name_or_path=cli.draft,
        max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        confidence_threshold=cli.confidence_threshold,
        tensorboard_dir=None,
        step=None,
        seed=cli.seed,
        task_overrides=None,
        max_samples=cli.limit,
        tasks=[("fc_cockpit", cli.limit)],
    )
    print(json.dumps(vars(cli), ensure_ascii=False, indent=2), flush=True)
    torch.multiprocessing.spawn(
        eval_mod.main,
        args=(args,),
        nprocs=max(torch.cuda.device_count(), 1),
    )


if __name__ == "__main__":
    main()

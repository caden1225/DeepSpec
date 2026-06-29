#!/usr/bin/env bash
# Train a DSpark draft model targeting Qwen3.5-4B.
#
# train.py spawns one worker per visible GPU automatically.
# Adjust CUDA_VISIBLE_DEVICES to use fewer GPUs (e.g. "0" for single-GPU).
#
# The default global_batch_size in the config is 512; with 1 GPU and
# local_batch_size=1 the effective gradient_accumulation_steps=512.
# For faster iteration reduce global_batch_size via --opts:
#   --opts "train.global_batch_size=64"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

target_cache_dir=${target_cache_dir:-/storage/caden/deepspec/qwen3_5_4b_target_cache}

python train.py \
    --config config/dspark/dspark_qwen3_5_4b.py \
    --opts "data.target_cache_path=${target_cache_dir}" \
    --opts "train.global_batch_size=32" \
    --opts "train.local_batch_size=1" \
    --opts "train.torch_compile=false" \
    --opts "model.num_anchors=64"

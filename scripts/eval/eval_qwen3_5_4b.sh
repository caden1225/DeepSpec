#!/usr/bin/env bash
# Evaluate a trained DSpark draft model for Qwen3.5-4B.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

target_name_or_path=/storage/caden/models/Qwen3_5-4B

# Training writes checkpoints under ~/checkpoints/deepspec/dspark_block8_qwen3_5_4b/step_*
draft_name_or_path=${HOME}/checkpoints/deepspec/dspark_block8_qwen3_5_4b/step_latest

PYTHON_BIN=${PYTHON_BIN:-python}
"${PYTHON_BIN}" eval.py \
    --target_name_or_path "${target_name_or_path}" \
    --draft_name_or_path "${draft_name_or_path}"

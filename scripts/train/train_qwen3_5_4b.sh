#!/usr/bin/env bash
# Train a DSpark draft model targeting Qwen3.5-4B.
#
# train.py spawns one worker per visible GPU automatically.
# Adjust CUDA_VISIBLE_DEVICES to use fewer GPUs (e.g. "0" for single-GPU).
#
# Single-GPU tuning (override via env vars):
#   NUM_ANCHORS=64          # default 64 (config default is 512; lower saves VRAM)
#   GLOBAL_BATCH_SIZE=32    # effective grad accumulation = GLOBAL / LOCAL
#   LOCAL_BATCH_SIZE=1
#
# Example:
#   NUM_ANCHORS=128 GLOBAL_BATCH_SIZE=16 bash scripts/train/train_qwen3_5_4b.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

target_cache_dir=${target_cache_dir:-/storage/caden/deepspec/qwen3_5_4b_target_cache}
NUM_ANCHORS=${NUM_ANCHORS:-64}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
TORCH_COMPILE=${TORCH_COMPILE:-false}
PYTHON_BIN=${PYTHON_BIN:-/home/caden/anaconda3/envs/train/bin/python}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a conda env with requirements.txt installed." >&2
    exit 1
fi

echo "Training with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  target_cache_dir=${target_cache_dir}"
echo "  NUM_ANCHORS=${NUM_ANCHORS}"
echo "  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "  LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}"
echo "  TORCH_COMPILE=${TORCH_COMPILE}"

"${PYTHON_BIN}" train.py \
    --config config/dspark/dspark_qwen3_5_4b.py \
    --opts "data.target_cache_path=${target_cache_dir}" \
    --opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --opts "train.local_batch_size=${LOCAL_BATCH_SIZE}" \
    --opts "train.torch_compile=${TORCH_COMPILE}" \
    --opts "model.num_anchors=${NUM_ANCHORS}"

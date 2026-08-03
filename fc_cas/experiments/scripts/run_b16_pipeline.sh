#!/usr/bin/env bash
# v1 patent-friendly attempt: B=16 draft, 1k-sample cache, 200 steps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/home/caden/anaconda3/envs/lf/bin/python}
CONFIG=${CONFIG:-fc_cas/experiments/configs/dspark_fc_qwen3_5_4b_b16.py}
TRAIN_LIMIT=${TRAIN_LIMIT:-1000}
TRAIN_JSONL=${TRAIN_JSONL:-fc_cas/data/deepspec_jsonl/fc_train_1k.jsonl}
CACHE_DIR=${CACHE_DIR:-/storage/caden/deepspec/fc_cas_qwen3_5_4b_b16_target_cache}
MAX_STEPS=${MAX_STEPS:-200}
MASTER_PORT=${MASTER_PORT:-29631}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME=/home/caden/anaconda3/envs/lf
export PATH="${CUDA_HOME}/bin:${PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DISABLED=true

echo "=== [1/3] Convert train jsonl (limit=${TRAIN_LIMIT}) ==="
"${PYTHON_BIN}" fc_cas/experiments/scripts/convert_fc_to_deepspec_jsonl.py \
  --out-dir fc_cas/data/deepspec_jsonl \
  --limit-train "${TRAIN_LIMIT}"

# Keep a dedicated 1k train artifact; restore full test for eval if overwritten.
cp -f fc_cas/data/deepspec_jsonl/fc_train.jsonl "${TRAIN_JSONL}"
wc -l "${TRAIN_JSONL}"

echo "=== [2/3] Prepare target cache → ${CACHE_DIR} ==="
mkdir -p "$(dirname "${CACHE_DIR}")"
"${PYTHON_BIN}" scripts/data/prepare_target_cache.py \
  --config "${CONFIG}" \
  --train-data-path "${TRAIN_JSONL}" \
  --output-dir "${CACHE_DIR}" \
  --local-batch-size 1 \
  --num-workers 0 \
  --opts "model.target_model_name_or_path=/home/caden/models/Qwen3_5-4B" \
  --opts "data.max_length=2048"

echo "=== [3/3] Train draft B=16 steps=${MAX_STEPS} ==="
"${PYTHON_BIN}" train.py \
  --config "${CONFIG}" \
  --opts "data.target_cache_path=${CACHE_DIR}" \
  --opts "model.target_model_name_or_path=/home/caden/models/Qwen3_5-4B" \
  --opts "train.torch_compile=false" \
  --opts "model.num_anchors=64" \
  --opts "train.global_batch_size=32" \
  --opts "train.local_batch_size=1" \
  --opts "train.max_train_steps=${MAX_STEPS}" \
  --opts "train.num_train_epochs=50" \
  --opts "logging.checkpointing_steps=${MAX_STEPS}"

echo "Done. ckpt: ~/checkpoints/deepspec/dspark_fc_cas_qwen3_5_4b_b16/"

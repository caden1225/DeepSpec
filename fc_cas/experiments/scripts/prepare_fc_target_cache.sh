#!/usr/bin/env bash
# Prepare FC-CAS target cache (Qwen3.5-4B) then short-train a domain draft.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/home/caden/anaconda3/envs/lf/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/caden/models/Qwen3_5-4B}
TRAIN_JSONL=${TRAIN_JSONL:-fc_cas/data/deepspec_jsonl/fc_train.jsonl}
CACHE_DIR=${CACHE_DIR:-/storage/caden/deepspec/fc_cas_qwen3_5_4b_target_cache}
CONFIG=${CONFIG:-fc_cas/experiments/configs/dspark_fc_qwen3_5_4b.py}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29611}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_JSONL=${TRAIN_JSONL}"
echo "CACHE_DIR=${CACHE_DIR}"

mkdir -p "$(dirname "${CACHE_DIR}")"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/data/prepare_target_cache.py \
  --config "${CONFIG}" \
  --train-data-path "${TRAIN_JSONL}" \
  --output-dir "${CACHE_DIR}" \
  --local-batch-size 1 \
  --num-workers 0 \
  --opts "model.target_model_name_or_path=${MODEL_PATH}" \
  --opts "data.max_length=2048"

echo "Cache ready: ${CACHE_DIR}"

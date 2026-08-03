#!/usr/bin/env bash
# Short-train FC-CAS domain DSpark draft on Qwen3.5-4B.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/home/caden/anaconda3/envs/lf/bin/python}
CONFIG=${CONFIG:-fc_cas/experiments/configs/dspark_fc_qwen3_5_4b.py}
CACHE_DIR=${CACHE_DIR:-/storage/caden/deepspec/fc_cas_qwen3_5_4b_target_cache}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME=/home/caden/anaconda3/envs/lf
export PATH="${CUDA_HOME}/bin:${PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29612}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Training FC draft with cache=${CACHE_DIR}"
"${PYTHON_BIN}" train.py \
  --config "${CONFIG}" \
  --opts "data.target_cache_path=${CACHE_DIR}" \
  --opts "model.target_model_name_or_path=/home/caden/models/Qwen3_5-4B" \
  --opts "train.torch_compile=false" \
  --opts "model.num_anchors=64" \
  --opts "train.global_batch_size=32" \
  --opts "train.local_batch_size=1" \
  --opts "train.max_train_steps=60" \
  --opts "train.num_train_epochs=10"

echo "Done. Checkpoints under ~/checkpoints/deepspec/dspark_fc_cas_qwen3_5_4b/"

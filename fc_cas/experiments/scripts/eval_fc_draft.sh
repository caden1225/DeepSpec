#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/home/caden/anaconda3/envs/lf/bin/python}
TARGET=${TARGET:-/home/caden/models/Qwen3_5-4B}
DRAFT=${DRAFT:-$HOME/checkpoints/deepspec/dspark_fc_cas_qwen3_5_4b/step_60}
LIMIT=${LIMIT:-20}
CONF_TH=${CONF_TH:-0.0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME=/home/caden/anaconda3/envs/lf
export PATH="${CUDA_HOME}/bin:${PATH:-}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29613}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" fc_cas/experiments/scripts/export_fc_eval_turns.py --limit "${LIMIT}"
"${PYTHON_BIN}" fc_cas/experiments/scripts/run_fc_dspark_eval.py \
  --target "${TARGET}" \
  --draft "${DRAFT}" \
  --limit "${LIMIT}" \
  --confidence-threshold "${CONF_TH}"

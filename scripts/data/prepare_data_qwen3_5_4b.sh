#!/usr/bin/env bash
# End-to-end data prep for Qwen3.5-4B DSpark (ModelScope/HF dataset + local target model).
# Override via env: SAMPLE_SIZE, FORCE=1, REGEN_BATCH_SIZE, CACHE_BATCH_SIZE, MODEL_PATH, etc.
# Train after: target_cache_dir=${CACHE_DIR} NUM_ANCHORS=64 bash scripts/train/train_qwen3_5_4b.sh

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/storage/caden/models/Qwen3_5-4B}
CONFIG_PATH=${CONFIG_PATH:-config/dspark/dspark_qwen3_5_4b.py}
PYTHON_BIN=${PYTHON_BIN:-python}

DATA_SOURCE=${DATA_SOURCE:-modelscope}
DATASET_NAME=${DATASET_NAME:-mlabonne/open-perfectblend}
MODELSCOPE_DATASET_NAME=${MODELSCOPE_DATASET_NAME:-AI-ModelScope/open-perfectblend}
MODELSCOPE_CACHE_DIR=${MODELSCOPE_CACHE_DIR:-}

SAMPLE_SIZE=${SAMPLE_SIZE:-500}
TEST_SIZE=${TEST_SIZE:-0.05}

TRAIN_SPLIT_PATH=${TRAIN_SPLIT_PATH:-train_datasets/perfectblend_train.jsonl}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-eval_datasets}
TEST_SPLIT_PATH=${TEST_SPLIT_PATH:-${EVAL_DATA_DIR}/perfectblend.jsonl}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-train_datasets/qwen3_5_4b/perfectblend_train_regen.jsonl}

CACHE_DIR=${CACHE_DIR:-/storage/caden/deepspec/qwen3_5_4b_target_cache_${SAMPLE_SIZE}}

REGEN_BATCH_SIZE=${REGEN_BATCH_SIZE:-1}
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-1}
CACHE_NUM_WORKERS=${CACHE_NUM_WORKERS:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_LENGTH=${MAX_LENGTH:-4096}
FORCE=${FORCE:-0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
# Steps 2/3 load the target model from a local directory only.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Local model directory not found: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: Missing ${MODEL_PATH}/config.json" >&2
    exit 1
fi

mkdir -p train_datasets/qwen3_5_4b "$(dirname "${CACHE_DIR}")"

backup_if_exists() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        local backup="${path}.bak.$(date +%Y%m%d_%H%M%S)"
        echo "Backing up ${path} -> ${backup}"
        mv "${path}" "${backup}"
    fi
}

if [[ "${FORCE}" == "1" ]]; then
    echo "FORCE=1: backing up existing outputs before rebuild..."
    backup_if_exists "${TRAIN_SPLIT_PATH}"
    backup_if_exists "${TEST_SPLIT_PATH}"
    backup_if_exists "${TRAIN_DATA_PATH}"
    backup_if_exists "${CACHE_DIR}"
fi

echo "Config: source=${DATA_SOURCE} samples=${SAMPLE_SIZE} model=${MODEL_PATH} cache=${CACHE_DIR}"
echo ""
DOWNLOAD_ARGS=(
    --source "${DATA_SOURCE}"
    --sample-size "${SAMPLE_SIZE}"
    --test-size "${TEST_SIZE}"
    --train-output-path "${TRAIN_SPLIT_PATH}"
    --test-output-dir "${EVAL_DATA_DIR}"
    --skip-existing
)
if [[ "${DATA_SOURCE}" == "modelscope" ]]; then
    DOWNLOAD_ARGS+=(--modelscope-dataset-name "${MODELSCOPE_DATASET_NAME}")
    if [[ -n "${MODELSCOPE_CACHE_DIR}" ]]; then
        DOWNLOAD_ARGS+=(--modelscope-cache-dir "${MODELSCOPE_CACHE_DIR}")
    fi
else
    DOWNLOAD_ARGS+=(--dataset-name "${DATASET_NAME}")
fi

echo "========================================="
echo "Step 1/3: Downloading and splitting dataset (${SAMPLE_SIZE} samples via ${DATA_SOURCE})"
echo "========================================="
"${PYTHON_BIN}" scripts/data/download_and_split.py "${DOWNLOAD_ARGS[@]}"

echo ""
echo "========================================="
echo "Step 2/3: Regenerating answers with local Qwen3.5-4B"
echo "  model=${MODEL_PATH}"
echo "========================================="
"${PYTHON_BIN}" scripts/data/generate_train_data_local.py \
    --model "${MODEL_PATH}" \
    --input-file-path "${TRAIN_SPLIT_PATH}" \
    --output-file-path "${TRAIN_DATA_PATH}" \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${REGEN_BATCH_SIZE}" \
    --disable-thinking \
    --resume

echo ""
echo "========================================="
echo "Step 3/3: Preparing target cache"
echo "  model=${MODEL_PATH}"
echo "  output=${CACHE_DIR}"
echo "========================================="
"${PYTHON_BIN}" scripts/data/prepare_target_cache.py \
    --config "${CONFIG_PATH}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir "${CACHE_DIR}" \
    --local-batch-size "${CACHE_BATCH_SIZE}" \
    --num-workers "${CACHE_NUM_WORKERS}" \
    --opts "data.max_length=${MAX_LENGTH}" \
    --opts "model.target_model_name_or_path=${MODEL_PATH}"

echo ""
echo "========================================="
echo "Data preparation complete."
echo "========================================="
echo "Target cache: ${CACHE_DIR}"
echo ""
echo "Next, start training:"
echo "  target_cache_dir=${CACHE_DIR} NUM_ANCHORS=64 bash scripts/train/train_qwen3_5_4b.sh"

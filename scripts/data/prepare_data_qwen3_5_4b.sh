#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=/storage/caden/models/Qwen3_5-4B
CONFIG_PATH=config/dspark/dspark_qwen3_5_4b.py

DATASET_NAME=mlabonne/open-perfectblend
SAMPLE_SIZE=2000          # small subset to keep storage manageable
TEST_SIZE=0.05

TRAIN_SPLIT_PATH=train_datasets/perfectblend_train.jsonl
EVAL_DATA_DIR=eval_datasets
TRAIN_DATA_PATH=train_datasets/qwen3_5_4b/perfectblend_train_regen.jsonl

# Store cache on /storage to avoid filling the home partition
CACHE_DIR=/storage/caden/deepspec/qwen3_5_4b_target_cache

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

mkdir -p train_datasets/qwen3_5_4b

echo "========================================="
echo "Step 1/3: Downloading and splitting ${DATASET_NAME} (${SAMPLE_SIZE} samples)"
echo "========================================="
python scripts/data/download_and_split.py \
    --dataset-name "${DATASET_NAME}" \
    --sample-size "${SAMPLE_SIZE}" \
    --test-size "${TEST_SIZE}" \
    --train-output-path "${TRAIN_SPLIT_PATH}" \
    --test-output-dir "${EVAL_DATA_DIR}" \
    --skip-existing

echo ""
echo "========================================="
echo "Step 2/3: Regenerating answers with Qwen3.5-4B (local inference)"
echo "========================================="
python scripts/data/generate_train_data_local.py \
    --model "${MODEL_PATH}" \
    --input-file-path "${TRAIN_SPLIT_PATH}" \
    --output-file-path "${TRAIN_DATA_PATH}" \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --max-new-tokens 1024 \
    --batch-size 4 \
    --disable-thinking \
    --resume

echo ""
echo "========================================="
echo "Step 3/3: Preparing Qwen3.5-4B target cache: ${CACHE_DIR}"
echo "========================================="
python scripts/data/prepare_target_cache.py \
    --config "${CONFIG_PATH}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir "${CACHE_DIR}" \
    --local-batch-size 4

echo ""
echo "Data preparation complete."
echo "Target cache written to: ${CACHE_DIR}"

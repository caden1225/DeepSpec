#!/usr/bin/env bash
# Qwen3.5-9B + DFlash 文本推理（1× L20）
# 文档：docs/vllm_qwen35_9b_dflash_serve.md
set -euo pipefail

NAME="${NAME:-qwen35-9b-infer}"
IMAGE="${IMAGE:-ccr-53sfop7y-pub.cnc.su.baidubce.com/vllm/vllm-openai:0.21.5}"
MODELS_HOST="${MODELS_HOST:-/storage/caden/models}"
PORT="${PORT:-8000}"
NUM_SPEC="${NUM_SPEC:-15}"

docker rm -f "${NAME}" 2>/dev/null || true

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --shm-size 8g \
  -p "${PORT}:8000" \
  -v "${MODELS_HOST}:/models:ro" \
  --entrypoint vllm \
  "${IMAGE}" \
  serve /models/Qwen3_5-9B \
  --served-model-name Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/Qwen3_5-9B-DFlash\",\"num_speculative_tokens\":${NUM_SPEC},\"attention_backend\":\"FLASH_ATTN\"}"

echo "started: ${NAME}  port=${PORT}  num_speculative_tokens=${NUM_SPEC}"
echo "wait for: curl -s http://127.0.0.1:${PORT}/v1/models"

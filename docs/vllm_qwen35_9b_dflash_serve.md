# vLLM：Qwen3.5-9B + DFlash 文本推理启动记录

> 机器：Dell Pro Max Tower（1× NVIDIA L20 48GB）  
> 记录日期：2026-07-29  
> 容器名：`qwen35-9b-infer`  
> 目的：纯文本推理 + DFlash 投机解码，单卡低并发下最大化加速

配套脚本：[`scripts/serve_qwen35_9b_dflash.sh`](../scripts/serve_qwen35_9b_dflash.sh)

---

## 1. 模型与镜像

| 角色 | 宿主机路径 | 容器内路径 |
|------|------------|------------|
| Target（base） | `/home/caden/models/Qwen3_5-9B`（即 `/storage/caden/models/Qwen3_5-9B`） | `/models/Qwen3_5-9B` |
| Draft（DFlash） | `/home/caden/models/Qwen3_5-9B-DFlash` | `/models/Qwen3_5-9B-DFlash` |

- 镜像：`ccr-53sfop7y-pub.cnc.su.baidubce.com/vllm/vllm-openai:0.21.5`（实际引擎日志为 v0.21.0）
- 挂载：`-v /storage/caden/models:/models:ro`
- **必须** `--entrypoint vllm`：该镜像默认入口是定制 shell，会忽略 `serve` 参数并回落到 `/models/base`

---

## 2. 配置取舍（针对 1× L20）

| 项 | 取值 | 说明 |
|----|------|------|
| `--language-model-only` | 开 | 仅文本；Qwen3.5 为多模态架构，投机解码需关掉视觉 |
| `method` | `dflash` | 块扩散草稿，单次前向出整块 token |
| `num_speculative_tokens` | `15` | 对应 draft `block_size=16`；低并发吞吐最优 |
| `attention_backend`（draft） | `FLASH_ATTN` | DFlash 非因果 attention，需 Flash Attention |
| `--dtype` / `--kv-cache-dtype` | `bfloat16` / `auto` | DFlash 与 FP8 等量化 KV 不兼容 |
| `--max-model-len` | `32768` | 与现网一致；显存仍有余量 |
| `--gpu-memory-utilization` | `0.92` | base≈18GB + draft≈2.5GB，余量给 KV |
| `--max-num-seqs` | `32` | 兼顾少量并发；纯延迟可再降 |
| `--max-num-batched-tokens` | `32768` | 与 max-model-len 对齐 |
| thinking | `enable_thinking: false` | 关闭思维链，减输出长度与延迟 |
| `--shm-size` | `8g` | 与历史容器一致 |

高并发吞吐为主时，可将 `num_speculative_tokens` 改为 `7`（约等于 block=8）。

---

## 3. 启动命令

```bash
docker rm -f qwen35-9b-infer 2>/dev/null || true

docker run -d \
  --name qwen35-9b-infer \
  --gpus all \
  --shm-size 8g \
  -p 8000:8000 \
  -v /storage/caden/models:/models:ro \
  --entrypoint vllm \
  ccr-53sfop7y-pub.cnc.su.baidubce.com/vllm/vllm-openai:0.21.5 \
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
  --speculative-config '{"method":"dflash","model":"/models/Qwen3_5-9B-DFlash","num_speculative_tokens":15,"attention_backend":"FLASH_ATTN"}'
```

冷启动约 1.5–2 分钟（含 compile / cudagraph）。就绪标志：

```bash
curl -s http://127.0.0.1:8000/v1/models
# served id: Qwen3.5-9B
# 日志含：speculative_config=SpeculativeConfig(method='dflash', ..., num_spec_tokens=15)
```

---

## 4. 调用示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [{"role":"user","content":"你好"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

运维：

```bash
docker logs -f qwen35-9b-infer
docker stop qwen35-9b-infer
nvidia-smi
```

---

## 5. 排错备忘

| 现象 | 原因 / 处理 |
|------|-------------|
| 加载 `/models/base`、模型名 `base-qwen` | 未加 `--entrypoint vllm` |
| `unrecognized arguments: --disable-log-requests` | 该镜像参数集不支持，勿加 |
| Spec Decode + multimodal `NotImplementedError` | 缺少 `--language-model-only` |
| FlashInfer / non-causal 报错 | draft 侧指定 `attention_backend: FLASH_ATTN`；勿用 FP8 KV |
| 客户端 404 `base-qwen` | 请求 model 须为 `Qwen3.5-9B`（`--served-model-name`） |

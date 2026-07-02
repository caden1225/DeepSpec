# DSpark × Qwen3.5 复现与学习指南

> 本文目标：带你从零理解 DeepSpec 仓库中 **DSpark 投机解码 draft model** 的研究逻辑，
> 并完整复现「数据准备 → 训练 → 评测」三阶段，重点聚焦提交
> `62c9b6e (feat: add Qwen3.5 DSpark support)` 新增的 Qwen3.5 支持。
>
> 阅读建议：第 1–3 章建立概念，第 4 章逐块走读代码，第 5–6 章动手复现，
> 第 7–8 章用于移植与排错。

---

## 1. 背景：什么是投机解码，DSpark 解决什么问题

自回归大模型（target model）逐 token 生成，每个 token 都要一次完整前向，慢。
**投机解码（speculative decoding）** 的思路是：

1. 用一个轻量的 **draft model（草稿模型）** 一次性"猜"出未来连续 `k` 个 token；
2. target model 对这 `k` 个候选**并行验证**（一次前向）；
3. 接受最长的正确前缀，被拒处回退。

只要 draft 足够准、足够快，单位时间生成的 token 数显著提升，且**输出分布与 target 完全一致**（无损加速）。

DeepSpec 提供三种 draft 算法：`DSpark`、`DFlash`、`Eagle3`。本指南只讲 **DSpark**。

DSpark 的特点：
- draft 模型**复用 target 的若干层隐藏状态**作为输入（站在 target 肩膀上），并冻结共享 target 的 embedding 与 LM head；
- 采用 **anchor + block** 的训练范式，一条样本可并行产出数百个训练子样本，训练效率极高；
- 配备 **Markov head**（用前序 token 修正 logits）与 **Confidence head**（预测每个草稿 token 的被接受概率，推理时据此动态裁剪投机长度）。

---

## 2. 仓库代码地图

```
DeepSpec/
├── config/dspark/                 # 每个 target 一份训练配置
│   └── dspark_qwen3_5_4b.py        # ★ 本次新增：Qwen3.5-4B 配置
├── deepspec/
│   ├── data/                       # 数据集 + collator（target cache 读写）
│   ├── modeling/dspark/
│   │   ├── common.py               # ★ anchor 采样 / 注意力 mask / noise embed（算法骨架）
│   │   ├── loss.py                 # ★ CE + L1 + Confidence 三项损失
│   │   ├── markov_head.py          # Markov head
│   │   ├── qwen3/                   # Qwen3 draft（参考实现）
│   │   ├── gemma4/                  # Gemma4 draft
│   │   └── qwen3_5/                 # ★ 本次新增：Qwen3.5 draft（modeling/config）
│   ├── trainer/
│   │   ├── base_trainer.py          # 训练循环骨架
│   │   └── dspark_trainer.py        # ★ 新增 Qwen3_5DSparkTrainer
│   └── eval/
│       ├── base_evaluator.py        # ★ 投机解码评测骨架（KV cache 修复）
│       └── dspark/
│           ├── evaluator.py         # ★ 新增 Qwen3_5DSparkEvaluator
│           └── draft_ops.py         # 投机"提议—验证"核心
├── scripts/
│   ├── data/                        # 数据准备（含本次新增的 CPU/合成数据脚本）
│   ├── train/                       # 训练入口（含本次新增 CPU 验证脚本）
│   └── eval/                        # 评测入口
├── train.py / eval.py               # 顶层入口
└── DSpark_paper.pdf                 # 论文
```

标 ★ 的是本次提交涉及的文件。

---

## 3. DSpark 算法原理（看懂这章，代码自然通）

### 3.1 核心数据流

```
target 模型 ──prepare_target_cache──▶ ① target_hidden_states（中间若干层拼接）
                                       ② target_last_hidden_states（末层，算 target logits 用）
                                       （连同 input_ids / loss_mask 一起存为 cache）
                                                  │
                                                  ▼
        DSpark draft.forward(input_ids, target_hidden_states, loss_mask, target_last_hidden_states)
                                                  │
                                                  ▼
            draft_logits  ──compute_dspark_loss(CE + L1 + Confidence)──▶ 反向更新 draft
                                                  │
                                                  ▼（训练完成后）
            eval.py：draft 提议 k 个 token，target 验证，统计接受率 / 加速比 τ
```

> 关键认知：**训练阶段 target 不参与前向**，它的隐藏状态已被提前算好缓存在磁盘（target cache）。
> 这就是为什么数据准备阶段是整个复现的第一步、也是最耗存储的一步。

### 3.2 Anchor 与 Block（训练范式）

实现于 `deepspec/modeling/dspark/common.py`：

- **Anchor（锚点）**：在一条长度 `seq_len` 的序列里随机采样 `num_anchors`（如 512）个位置当起点。
  采样见 `sample_anchor_positions`；只有"自身有效且下一个位置也需要监督"的位置才是合法候选
  （`build_anchor_candidate_mask`）。不足时用 dummy 锚点补齐，并由 `block_keep_mask` 标记屏蔽。
- **Block（块）**：每个 anchor 之后并行预测 `block_size`（如 7）个未来 token。
- **专用注意力 mask**（`create_dspark_attention_mask`）：每个 block 的草稿位置只能看到
  ① 锚点之前的真实上下文；② 同一 block 内部。**块与块互不可见**——这才使得一条序列里
  512 个子样本可以一次前向、互不串扰。

```93:103:deepspec/modeling/dspark/common.py
    def dspark_mask_mod(b, h, q_idx, kv_idx):
        del h
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]
        is_context = kv_idx < seq_len
        mask_context = is_context & (kv_idx < anchor_pos)
        is_draft = kv_idx >= seq_len
        kv_block_id = (kv_idx - seq_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block
```

- **Noise embedding**（`create_noise_embed`）：每个 block 的输入起始放锚点真实 token，其余位置填
  `mask_token_id`（占位），由模型逐步"填空"预测。

### 3.3 draft 的前向（`qwen3_5/modeling.py`）

`forward` 串起整套流程：采样锚点 → 造 noise embedding → 造位置 id → 造 mask → 过 backbone →
取 block 隐藏态 → 算 draft_logits（经 Markov head 修正）→ 对齐 target logits → 算 confidence。

```446:451:deepspec/modeling/dspark/qwen3_5/modeling.py
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )
```

`_forward_backbone` 里，target 隐藏态先经 `fc + hidden_norm` 投影，再作为"交叉注意力的 KV 来源"
注入每一层（draft 的 query 只来自草稿位置，key/value 来自 [上下文 + 草稿]）：

```393:408:deepspec/modeling/dspark/qwen3_5/modeling.py
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        # Qwen3.5 rotary_emb expects (x, position_ids) where x provides device/dtype
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)
```

### 3.4 损失函数（`loss.py` → `compute_dspark_loss`）

总损失三项加权求和：

| 项 | 含义 | 权重（Qwen3.5-4B 配置） |
|----|------|------|
| **CE loss** | draft_logits 对齐真实 token（交叉熵） | `ce_loss_alpha=0.1` |
| **L1 loss** | draft 分布对齐 target 分布（全变差距离 ×2） | `l1_loss_alpha=0.9` |
| **Confidence loss** | confidence head 预测的接受率 ↔ 真实接受率（BCE） | `confidence_head_alpha=1.0` |

- 接受率用 **TV 距离**估计：`accept_rate = 1 - TV(draft, target)`，并对 vocab/anchor 分块计算以省显存
  （见 `_chunked_tv_distance`）。
- block 内靠后的位置按 `loss_decay_gamma` 指数降权（越靠后越难，权重越小）。
- 训练时还会记录 `accept_rate@k`、`tau_probabilistic`（期望投机步长）等指标。

```275:279:deepspec/modeling/dspark/loss.py
    return (
        ce_loss_alpha * ce_loss
        + l1_loss_alpha * l1_loss
        + confidence_head_alpha * confidence_loss
    ) * world_size
```

### 3.5 推理时的投机解码（`eval/dspark/draft_ops.py`）

每一步：draft 用 `_forward_backbone` 算一个 block 的隐藏态 → 采样 `block_size` 个候选 token →
若有 confidence head，用它把候选**截断到第一个低置信位置**（`_confident_prefix_length`）→
交给 target 验证。这正是 confidence head 的价值：动态决定"这次敢猜多长"。

```96:131:deepspec/eval/dspark/draft_ops.py
def build_dspark_proposal(
    model: DSparkModel,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    temperature: float,
    confidence_threshold: float,
) -> DSparkDraftProposal:
    assert draft_input_ids.size(0) == 1, "build_dspark_proposal requires batch_size=1"
    proposal_hidden_states = block_hidden[:, :block_size, :]
    base_draft_logits = model.compute_logits(proposal_hidden_states)
    sampled_tokens, draft_logits = model.sample_draft_tokens(
        base_draft_logits,
        first_prev_token_ids=draft_input_ids[:, 0],
        temperature=temperature,
        hidden_states=proposal_hidden_states,
    )
    proposal_draft_tokens = int(block_size)
    confidence_logits = None
    if model.confidence_head is not None:
        confidence_logits = _predict_confidence_logits(...)
        proposal_draft_tokens = _confident_prefix_length(
            confidence_logits, block_size=block_size, threshold=float(confidence_threshold),
        )
```

---

## 4. 本次提交逐块走读（Qwen3.5 支持）

提交 `62c9b6e` 共 22 文件、+1878 行。核心是把 DSpark 从纯文本 target 扩展到
**Qwen3.5（多模态 VLM + 混合注意力）**，并修复一批单卡/CPU 工程问题。
下面按"为什么改 / 改了什么"分组说明。

### 4.1 新增 Qwen3.5 draft 模型 `deepspec/modeling/dspark/qwen3_5/`

`modeling.py` 由 `qwen3/modeling.py` 复制改写（同为 533 行）。三处结构性适配：

**① 部分旋转 RoPE（partial rotary）**
Qwen3.5 只对 head_dim 的前 `rotary_dim` 维做旋转，其余维度原样穿过。新函数
`_apply_rotary_pos_emb_dspark` 同时保留 DSpark 的交叉注意力特性（q 取草稿位置、k 取全部位置）：

```55:63:deepspec/modeling/dspark/qwen3_5/modeling.py
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_embed = torch.cat(
        [(q_rot * cos_q) + (rotate_half(q_rot) * sin_q), q_pass], dim=-1
    )
    # Key uses all positions.
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    k_embed = torch.cat(
        [(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1
    )
```

**② 注意力输出门（output gate）**
Qwen3.5 的 full-attention 变体在输出处多了一个 sigmoid 门。故 `q_proj` 输出维度翻倍，
一半作 query、一半作 gate，最后 `attn_output *= sigmoid(gate)`：

```123:130:deepspec/modeling/dspark/qwen3_5/modeling.py
        q, gate = q_and_gate.chunk(2, dim=-1)  # [bsz, q_len, heads, head_dim] each
        gate = gate.reshape(bsz, q_len, self.num_attention_heads * self.head_dim)

        q = self.q_norm(q).transpose(1, 2)  # [bsz, heads, q_len, head_dim]
```

**③ 用 sdpa 而非 flex_attention（config.py）**
Qwen3.5 `head_dim=256`，flex_attention 在 L20 上共享内存不足；且 draft 全用 full attention：

```6:8:deepspec/modeling/dspark/qwen3_5/config.py
# flex_attention requires too much shared memory for head_dim=256 on L20 GPUs;
# sdpa uses Flash Attention 2 which handles large head dims correctly.
TRAIN_ATTN_IMPLEMENTATION = "sdpa"
```

**④ 多模态 config 抽取文本子配置**
Qwen3.5 config 嵌套，文本配置在 `text_config` 下：

```11:18:deepspec/modeling/dspark/qwen3_5/config.py
def build_draft_config(target_config, model_args):
    # target_config is Qwen3_5Config (multimodal); extract text config
    if hasattr(target_config, "text_config"):
        text_config = copy.deepcopy(target_config.text_config)
    else:
        text_config = copy.deepcopy(target_config)
```

### 4.2 `common.py`：支持 eager/sdpa 的稠密 mask

因不能用 flex_attention 的 `BlockMask`，`create_dspark_attention_mask` 新增 `use_block_mask` 开关。
为 `False` 时手动构造 `[bsz, 1, q_len, kv_len]` 浮点加性 mask（0=可见，-inf=屏蔽）：

```135:138:deepspec/modeling/dspark/common.py
    # Additive mask: 0 where attention is allowed, -inf where masked out
    float_mask = torch.zeros(bsz, 1, q_len, kv_len, device=device, dtype=torch.float32)
    float_mask.masked_fill_(~bool_mask.unsqueeze(1), float("-inf"))
    return float_mask
```

`forward` 据 `_attn_implementation` 自动选择路径（`use_block_mask = (impl == "flex_attention")`）。
另有两处 dtype 健壮性修复：`anchor_tokens.long()`、cross_entropy 的 `flat_targets.long()`。

### 4.3 `loss.py`：单卡 loss 同步修复

未初始化分布式（单卡/CPU）时 `dist.get_world_size()` 会崩，改为：

```295:295:deepspec/modeling/dspark/loss.py
    world_size = dist.get_world_size() if dist.is_initialized() else 1
```

### 4.4 Trainer：`Qwen3_5DSparkTrainer`

target 是多模态模型，权重 key 在 `model.language_model.*` 下，须用
`Qwen3_5ForConditionalGeneration` 加载，并**只抽出 embedding 与 lm_head 冻结复用**：

```77:91:deepspec/trainer/dspark_trainer.py
        target_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = target_model.get_input_embeddings()
        target_lm_head = target_model.get_output_embeddings()
        assert target_embed_tokens is not None and target_lm_head is not None
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        del target_model
        torch.cuda.empty_cache()
```

### 4.5 Evaluator：`Qwen3_5DSparkEvaluator`

强制 `EVAL_ATTN_IMPLEMENTATION = "eager"`（target 含 GatedDeltaNet 线性注意力，不支持 sdpa/flash），
并用多模态类加载 target。

`base_evaluator.py` 同步修复 **混合模型 KV cache 复用**：不再手动 `DynamicCache()`，
改为让模型自行初始化 config-aware 的 cache：

```350:355:deepspec/eval/base_evaluator.py
    # Use the cache the model initialized (handles hybrid models like Qwen3.5 that need
    # a config-aware cache with LinearAttention layer support).
    past_key_values_target = output.past_key_values
```

### 4.6 `eval.py`：CLI 任务过滤

新增 `--task NAME:N`（限定单任务+样本数）与 `--max-samples`（统一封顶），便于快速验证：

```62:71:eval.py
    if args.task_overrides:
        tasks = []
        for spec in args.task_overrides:
            name, _, n = spec.partition(":")
            tasks.append((name, int(n) if n else 500))
    if args.max_samples is not None:
        tasks = [(name, min(n, args.max_samples)) for name, n in tasks]
    args.tasks = tasks
```

### 4.7 `prepare_target_cache.py`：适配多模态 target

三处关键改动：
1. **定位文本 backbone**：多模态层在 `model.language_model` 下（`_get_target_backbone`）；
2. **hook 末层取 last_hidden_state**：`ForConditionalGeneration` 不直接暴露 `last_hidden_state`，
   于是给最后一层挂 forward hook，并补做 backbone 的 `norm`；
3. **加载方式**：用 `Qwen3_5ForConditionalGeneration` + `attn_implementation="eager"`。

```119:130:scripts/data/prepare_target_cache.py
        # Always hook the final backbone layer to capture last_hidden_state
        final_layer_idx = len(layer_modules) - 1
        handles.append(
            layer_modules[final_layer_idx].register_forward_hook(
                capture_layer(_LAST_LAYER_KEY)
            )
        )
```

### 4.8 新增的本地验证 / 合成数据脚本（复现"绕路工具"）

这些脚本不是论文主线，而是在**无外网 / GPU 被占用 / 无推理服务器**时打通流水线的替代品：

| 脚本 | 用途 |
|------|------|
| `scripts/data/create_fast_synthetic_data.py` | 规则生成极简训练数据，**无需模型推理**，纯验证流水线 |
| `scripts/data/create_synthetic_train_data.py` | 用本地 target 给 eval 数据集生成回答，替代下载 open-perfectblend |
| `scripts/data/generate_train_data_local.py` | 用本地 HF 模型离线重生成答案（替代推理服务器）|
| `scripts/data/prepare_target_cache_cpu.py` | **单进程 CPU 版** target cache 生成（无 NCCL）|
| `scripts/train/train_validate_cpu.py` | **CPU 版**训练验证，跑几个 step 存 ckpt，验证训练通路 |
| `scripts/data/prepare_data_qwen3_5_4b.sh` | 把数据准备串成一条龙 |
| `scripts/train/train_qwen3_5_4b.sh` | 单卡正式训练入口（已调小 batch / num_anchors）|
| `scripts/eval/eval_qwen3_5_4b.sh` | 单卡评测入口 |

---

## 5. 端到端复现流程

### 5.1 环境

```bash
# 推荐使用 conda 环境 312
conda activate 312
python -m pip install -r requirements.txt
```

### 5.2 阶段一：数据准备（产出 target cache）

正式路径（需 target 模型权重，可选外网下载 open-perfectblend）：

```bash
bash scripts/data/prepare_data_qwen3_5_4b.sh
```

脚本内默认参数（按需改）：
- `MODEL_PATH=/storage/caden/models/Qwen3_5-4B`
- `SAMPLE_SIZE=2000`（小子集，控制 cache 体积）
- `CACHE_DIR=/storage/caden/deepspec/qwen3_5_4b_target_cache`

> ⚠️ 存储警告：README 指出默认 Qwen3-4B 全量 target cache 约 **38 TB**。
> 务必用小 `SAMPLE_SIZE` 起步。无外网时改用 `create_synthetic_train_data.py` 自造训练数据。

CPU/无分布式环境，用单进程版：

```bash
python scripts/data/prepare_target_cache_cpu.py \
    --config config/dspark/dspark_qwen3_5_4b.py \
    --train-data-path train_datasets/perfectblend_train.jsonl \
    --output-dir /storage/caden/deepspec/qwen3_5_4b_target_cache \
    --local-batch-size 4 --device cpu
```

### 5.3 阶段二：训练 draft 模型

正式（单卡）：

```bash
bash scripts/train/train_qwen3_5_4b.sh
```

脚本已通过 `--opts` 调小为单卡友好（`global_batch_size=32`、`num_anchors=64`、
`torch_compile=false`）。checkpoint 写到
`~/checkpoints/deepspec/dspark_block8_qwen3_5_4b/step_*`。

仅验证训练通路（CPU，不真训练）：

```bash
PYTHONPATH=$(pwd) python3 scripts/train/train_validate_cpu.py \
    --config config/dspark/dspark_qwen3_5_4b.py \
    --target-cache /storage/caden/deepspec/qwen3_5_4b_target_cache \
    --output-dir ~/checkpoints/deepspec/dspark_block8_qwen3_5_4b/step_latest \
    --max-steps 5
```

该脚本会：构建 draft → 从 target 抽 embedding/LM head 冻结 → 跑 5 个 step（fp32 主参数 + bf16 模型）
→ 按 evaluator 期望的格式存盘（`config` + `pytorch_model.bin` + tokenizer）。

### 5.4 阶段三：评测

```bash
bash scripts/eval/eval_qwen3_5_4b.sh
# 快速验证单任务：
python eval.py --target_name_or_path /storage/caden/models/Qwen3_5-4B \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_block8_qwen3_5_4b/step_latest \
    --task gsm8k:50
```

评测会在 `eval_datasets/` 的基准（gsm8k、math500、aime25、humaneval、mbpp、livecodebench、
mt-bench、alpaca、arena-hard-v2）上跑投机解码，统计接受率与加速比。

### 5.5 最小验证回路（推荐先跑通这个）

无外网/GPU 受限时，按下序打通"小回路"再放大：

```
create_fast_synthetic_data.py   # 造极简数据
  └─▶ prepare_target_cache_cpu.py   # CPU 造 cache
        └─▶ train_validate_cpu.py     # CPU 跑 5 step 存 ckpt
              └─▶ eval.py --task gsm8k:20   # 小样本评测
```

---

## 6. Qwen3.5-4B 关键超参（`config/dspark/dspark_qwen3_5_4b.py`）

| 参数 | 值 | 说明 |
|------|----|------|
| `block_size` | 7 | 每个 anchor 预测的未来 token 数 |
| `num_draft_layers` | 5 | draft backbone 层数（很轻）|
| `target_layer_ids` | `[1,8,16,23,30]` | 取 target 这 5 层隐藏态作输入（Qwen3.5-4B 共 32 层，均匀取非末层）|
| `mask_token_id` | 248063 | `<|fim_pad|>` 作占位 mask token |
| `num_anchors` | 512 | 每条样本采样的锚点数（单卡脚本里调到 64）|
| `markov_rank` / `markov_head_type` | 256 / vanilla | Markov head |
| `confidence_head_alpha` | 1.0 | Confidence 头损失权重 |
| `confidence_head_with_markov` | True | confidence 特征拼接 markov 嵌入 |
| `loss_decay_gamma` | 4.0 | block 内位置损失衰减 |
| `ce_loss_alpha` / `l1_loss_alpha` | 0.1 / 0.9 | CE 与 L1 损失权重 |

训练侧：`lr=6e-4`、`precision=bf16`、`global_batch_size=512`、`num_train_epochs=10`、
`sharding_strategy=no_shard`。

---

## 7. 如何移植到新的 target 家族（本提交即模板）

新增一个 target（如 `Foo`）需要改动以下 6 处，与本次 Qwen3.5 一一对应：

1. `deepspec/modeling/dspark/foo/modeling.py`：从 `qwen3/` 复制，按 target 的结构差异适配
   （RoPE 形式、注意力变体、norm 类型等）；
2. `deepspec/modeling/dspark/foo/config.py`：`build_draft_config`，处理 config 抽取与 `architectures`；
3. `deepspec/trainer/dspark_trainer.py`：新增 `FooDSparkTrainer`，处理 target 权重加载方式；
4. `deepspec/eval/dspark/evaluator.py`：新增 `FooDSparkEvaluator`，设定 `attn_implementation`；
5. `scripts/data/prepare_target_cache.py`：在 `_get_target_backbone` / `_get_target_hidden_size`
   / 加载分支里加入 `foo` 的处理；
6. 一份 `config/dspark/dspark_foo.py` + `scripts/train|eval/*.sh` 入口；
   并在各 `__init__.py`、`eval.py` 的 `EVALUATORS` 字典登记新类。

---

## 8. 排错清单（本提交踩过的坑）

| 现象 | 原因 | 对策（已内置）|
|------|------|------|
| flex_attention 共享内存溢出 | `head_dim=256` 太大 | 训练用 `sdpa`；CPU 用 `eager` |
| target 权重加载失败/key 不匹配 | 多模态 key 在 `model.language_model.*` | 用 `Qwen3_5ForConditionalGeneration` 加载 |
| `last_hidden_state` 为 None | ForConditionalGeneration 不暴露 | hook 末层 + 补 `backbone.norm` |
| 单卡 `dist.get_world_size()` 崩 | 未初始化分布式 | `dist.is_initialized()` 判空 |
| 评测 KV cache 报错 | 混合模型需 config-aware cache | 用 `output.past_key_values` |
| cross_entropy dtype 报错 | target_ids 非 long | `.long()` 转换 |
| sdpa/flash 不支持 GatedDeltaNet | target 含线性注意力层 | 评测/造 cache 用 `eager` |

---

## 9. 推荐学习顺序

1. 读本文第 3 章 + `deepspec/modeling/dspark/common.py`（算法骨架）。
2. 读 `qwen3_5/modeling.py` 的 `forward` 与 `_forward_backbone`（理解 target 隐藏态注入）。
3. 读 `loss.py` 的 `compute_dspark_loss`（三项损失与接受率估计）。
4. 跑第 5.5 节的最小验证回路，打通三阶段。
5. 读 `eval/dspark/draft_ops.py`，理解推理时的投机"提议—验证"与 confidence 截断。
6. 对照第 4 章逐块复看本次 diff，体会"为多模态/混合注意力做适配"的工程取舍。

---

参考：仓库根 `DSpark_paper.pdf`、`README.md`、`scripts/data/README.md`。

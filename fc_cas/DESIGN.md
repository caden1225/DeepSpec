# FC-CAS Design Spec

> Date: 2026-07-16  
> Owner: 郑冬  
> Status: Approved direction (session-derived); implementation starts from this spec

## 1. Problem

座舱 FC 模型输出同时含高熵自然语言与低熵 JSON tool_calls。全局统一的投机 `block_size` / `confidence_threshold` 无法吃满结构化区段加速，又在 NL 区浪费验证。

## 2. Non-goals

- 不改变目标模型任务准确率（默认保持验证无损）  
- 不重新发明 DSpark 训练框架  
- 不把某一私有座舱协议锁进独立权利要求（本数据为 OpenAI 兼容 JSON FC）

## 3. Approaches considered

| 方案 | 做法 | 利 | 弊 | 结论 |
|------|------|----|----|------|
| A. 仅调全局超参 | 领域数据训 draft | 实现快 | 无专利区分度 | 否 |
| B. 仅两区 NL/TOOL 调度 | 标记驱动策略表 | 清晰可证 | 未吃 enum 红利 | 作独权主干 |
| C. B + Schema 槽位三分 | 骨架/枚举/自由 | 与数据集同构、加速预期最大 | 实现稍重 | **采纳** |

## 4. Architecture

```
messages + tools
       │
       ▼
┌──────────────────┐
│  Segmenter       │  增量 JSON + Schema 指针 → Region
└────────┬─────────┘
         ▼
┌──────────────────┐
│  StrategyTable   │  Region → (block_size, θ, mode)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  SchemaGuard     │  mode∈{template, enum_mask, free}
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Draft + Verify  │  复用 DeepSpec DSpark 验证；可选 confidence 二级截断
└──────────────────┘
```

### 4.1 Region enum

```python
class Region(str, Enum):
    NL = "nl"
    TOOL_SKELETON = "tool_skeleton"
    TOOL_ENUM = "tool_enum"
    TOOL_FREE = "tool_free"
```

### 4.2 Default strategy (calibratable)

| Region | block_size | confidence_threshold | mode |
|--------|------------|----------------------|------|
| NL | 4 | 0.6 | free |
| TOOL_SKELETON | 24 | 0.2 | template |
| TOOL_ENUM | 16 | 0.2 | enum_mask |
| TOOL_FREE | 8 | 0.4 | free |

### 4.3 Segmenter contract

输入：已生成 token 文本前缀（或 token ids + tokenizer）、本轮 `tools` schema、可选聊天模板边界。  
输出：当前 `Region`，以及枚举约束时的 `allowed_strings: list[str] | None`。

判定优先级：

1. 不在 tool_calls JSON 结构内 → `NL`  
2. 在结构符号 / 固定键名 / 待写函数名位置 → `TOOL_SKELETON`  
3. JSON 指针落在带 `enum` 的 property 值内 → `TOOL_ENUM`  
4. 否则在 arguments 值内 → `TOOL_FREE`

### 4.4 Integration with DeepSpec

- **Phase 1–3**：纯 CPU，不依赖 GPU；用数据集中的 gold assistant 序列做离线区段标注与模拟接受率上界分析  
- **Phase 4+**：在 `deepspec.eval.dspark.draft_ops.build_dspark_proposal` 外包一层：按 Region 覆盖 `block_size` / `confidence_threshold`，并在采样前调用 SchemaGuard  
- Target 模型：座舱 FC 微调模型（路径运行时配置）；草稿可先用已有 Qwen3.5-4B DSpark ckpt 作冷启动，再以本数据集重训

## 5. Data protocol (`processed_FC_dataset`)

- Splits: train 7105 / val 875 / test 917  
- Fields: `id`, `messages`, `tools`, optional `source`/`turn`  
- Assistant patterns: mostly `tool_calls` only; NL for refusals etc.  
- Eval must report \(\tau\), \(\tau_{NL}\), \(\tau_{TOOL}\), speedup, and task exact-match under verification

## 6. Success criteria

| Gate | Criteria |
|------|----------|
| G1 单测 | segmenter + strategy + schema 单测全绿（CPU） |
| G2 离线分析 | test 集上 TOOL 区可模板/枚举覆盖的 token 占比报告产出 |
| G3 在线对照 | 相对全局基线，test 子集 \(\tau\) 或 wall-clock 有可报告增益 |
| G4 专利 | 附图齐；实施例可引用 G2/G3 表；权利要求与实现一一对应 |

## 7. Risks

| Risk | Mitigation |
|------|------------|
| 聊天模板导致 JSON 边界难检 | 先支持「裸 tool_calls JSON 字符串」路径；模板适配单列 |
| 无座舱 target 权重 | 先离线上界 + mock verify；权重到位再跑 G3 |
| 现有技术接近 | 权利要求必须写清「标准 JSON FC 区段 + TOOL 更激进」 |

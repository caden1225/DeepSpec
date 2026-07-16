# 图 1 系统架构

面向标准 JSON Function Call 的区段感知投机解码系统。输入为 `messages` + `tools`；输出经目标模型验证后回填对话。

```mermaid
flowchart TB
  subgraph input["输入"]
    IN["messages + tools<br/>(OpenAI 兼容 FC 协议)"]
  end

  subgraph cas["FC-CAS 调度层"]
    SEG["FC 区段识别器<br/>(Segmenter)<br/>增量 JSON + Schema 指针 → Region"]
    ST["策略调度器<br/>(StrategyTable)<br/>Region → (B, θ, mode)"]
    SG["Schema 约束器<br/>(SchemaGuard)<br/>template / enum_mask / free"]
  end

  subgraph draft["草稿生成"]
    MP["草稿模块 M_p<br/>(条件于 M_q 隐藏状态)"]
    CONF["置信度头<br/>(可选)<br/>token 级截断 @ θ(r)"]
  end

  subgraph verify["目标验证"]
    MQ["目标模型 M_q<br/>(Target Verify)"]
  end

  IN --> SEG
  SEG -->|"Region, allowed_strings"| ST
  ST -->|"π(r) = (B, θ, mode)"| SG
  SG -->|"约束采样 / 模板注入"| MP
  MP --> CONF
  CONF -->|"候选 token 序列"| MQ
  MQ -->|"接受最长合法前缀"| SEG
  MQ --> OUT["assistant 输出<br/>NL 文本 / tool_calls JSON"]
```

**模块对应关系**

| 附图模块 | 代码路径 | 职责 |
|---------|---------|------|
| FC 区段识别器 | `fc_cas/segmenter/json_fc.py` | 判定 NL / TOOL_SKELETON / TOOL_ENUM / TOOL_FREE |
| 策略调度器 | `fc_cas/strategy/table.py` | 查表返回 block_size、confidence_threshold、mode |
| Schema 约束器 | `fc_cas/schema/guard.py` | 枚举掩码、骨架模板注入 |
| 草稿 + 验证 | DeepSpec DSpark 桥（Phase 4+） | 生成候选并由 M_q 并行验证 |

# 图 3 单轮 FC 投机解码时序

用户 utterance 触发 assistant 生成；若需调用工具则序列化 `tool_calls`，tool 角色回传结果后 assistant 可继续 NL 回复。图中展示一次解码步内的区段感知投机循环。

```mermaid
sequenceDiagram
  autonumber
  participant U as 用户
  participant MQ as 目标模型 M_q
  participant SEG as FC 区段识别器
  participant ST as 策略调度器
  participant SG as Schema 约束器
  participant MP as 草稿模块 M_p

  U->>MQ: user message + tools schema
  activate MQ

  loop 自回归解码直至停止符
    MQ->>SEG: 已生成前缀 (token / text)
    SEG->>SEG: 增量 JSON 解析 + Schema 指针
    SEG-->>ST: Region, allowed_strings
    ST-->>SG: π(r) = (B, θ, mode)
    SG-->>MP: 采样约束 / 模板前缀

    MP->>MP: 生成 ≤ B(r) 候选 token
    Note over MP: 置信度头截断至<br/>首个 σ(c_i) < θ(r)

    MP->>MQ: 候选 token 序列
    MQ->>MQ: 并行验证，接受最长合法前缀
    MQ->>SEG: 更新前缀 → 重判 Region
  end

  MQ-->>U: assistant.tool_calls<br/>(function.arguments JSON 字符串)
  deactivate MQ

  U->>MQ: tool 角色回传执行结果
  MQ-->>U: assistant NL 回复 (可选)
```

**时序要点**

1. 每步解码前先经 Segmenter 判定当前 Region，再查 StrategyTable 得差异化超参。  
2. SchemaGuard 仅在 TOOL_SKELETON / TOOL_ENUM 区段施加约束；NL / TOOL_FREE 回退普通草稿。  
3. 目标模型始终执行验证，保持与 M_q 分布一致的无损加速路径。

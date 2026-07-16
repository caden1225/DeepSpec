# 图 2 标准 JSON Function Call 区段状态机

解码过程中维护输出区段状态；识别信号包括聊天模板边界、增量 JSON 语法状态（键名 / 字符串值 / 结构符号）及 tools Schema 字段指针。

```mermaid
stateDiagram-v2
  direction LR

  [*] --> NL: 开始解码

  NL --> TOOL_SKELETON: 检测到 tool_calls<br/>JSON 结构符号 / 固定键名
  NL --> NL: 自然语言文本<br/>(拒识 / 澄清 / 闲聊)

  TOOL_SKELETON --> TOOL_ENUM: JSON 指针落在<br/>properties.field 且 field.enum 存在
  TOOL_SKELETON --> TOOL_FREE: JSON 指针落在<br/>无 enum 的 string / number 槽位
  TOOL_SKELETON --> NL: 离开 FC payload<br/>(无 tool_calls / 序列结束)

  TOOL_ENUM --> TOOL_SKELETON: 枚举值写完<br/>→ 下一键 / 闭合括号
  TOOL_FREE --> TOOL_SKELETON: 自由值写完<br/>→ 下一键 / 闭合括号

  TOOL_SKELETON --> TOOL_RESULT: (可选) tool 角色<br/>回传结果区段
  TOOL_RESULT --> NL: 回传结束<br/>继续 assistant 回复

  note right of NL
    B 小, θ 高
    mode = free (保守草稿)
  end note

  note right of TOOL_SKELETON
    B 大, θ 低
    mode = template (模板注入)
  end note

  note right of TOOL_ENUM
    B 大, θ 低
    mode = enum_mask
  end note

  note right of TOOL_FREE
    B 中, θ 中
    mode = free
  end note
```

**默认策略（DESIGN §4.2，可标定）**

| Region | block_size B | confidence θ | mode |
|--------|-------------|--------------|------|
| NL | 4 | 0.6 | free |
| TOOL_SKELETON | 24 | 0.2 | template |
| TOOL_ENUM | 16 | 0.2 | enum_mask |
| TOOL_FREE | 8 | 0.4 | free |

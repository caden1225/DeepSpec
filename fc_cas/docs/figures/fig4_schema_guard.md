# 图 4 基于 tools Schema 的参数槽位约束草稿

SchemaGuard 依据当前 Region 与 tools JSON Schema（`properties` / `enum` / `required`）对草稿采样施加约束；目标模型验证不放宽。

```mermaid
flowchart TD
  START(["当前 Region r + tools Schema"])

  START --> R{Region 判定}

  R -->|NL| NL["普通草稿<br/>B=4, θ=0.6<br/>无 Schema 约束"]
  R -->|TOOL_SKELETON| SK["模板注入<br/>B=24, θ=0.2"]
  R -->|TOOL_ENUM| EN["枚举约束采样<br/>B=16, θ=0.2"]
  R -->|TOOL_FREE| FR["普通 / 半自回归草稿<br/>B=8, θ=0.4"]

  SK --> SK1["注入确定性片段:<br/>结构符号 { } , :<br/>Schema 键名 / required 字段"]
  SK1 --> SK2["例: arguments 对象刚打开<br/>→ 注入 \"action\": "]

  EN --> EN1["读取 properties.field.enum"]
  EN1 --> EN2["非法 token logits → −∞"]
  EN2 --> EN3["例: action ∈ {check_update, install}<br/>→ 草稿几乎必然命中短枚举值"]

  FR --> FR1["歌名 / POI / 主题等<br/>高熵自由字符串槽位"]
  FR1 --> FR2["仅区段级 B/θ 差异化<br/>不做 enum 掩码"]

  NL --> DRAFT["草稿模块 M_p 输出候选"]
  SK2 --> DRAFT
  EN3 --> DRAFT
  FR2 --> DRAFT

  DRAFT --> CONF{"置信度头?<br/>σ(c_i) ≥ θ(r)"}
  CONF -->|是| DRAFT
  CONF -->|否, 截断| VERIFY

  VERIFY["目标模型 M_q 并行验证<br/>接受最长合法前缀"]
  VERIFY --> UPDATE["更新 JSON 指针 → 下一 Region"]
  UPDATE --> START
```

**示例：OTA 工具 arguments Schema**

```json
{
  "type": "object",
  "properties": {
    "action": { "type": "string", "enum": ["check_update", "install"] }
  },
  "required": ["action"]
}
```

| 生成位置 | Region | SchemaGuard 行为 |
|---------|--------|-----------------|
| `{"action":` 骨架 | TOOL_SKELETON | 模板注入键名与标点 |
| `"check_update"` 值 | TOOL_ENUM | logits 掩码至 enum 集合 |
| 自由 string 槽位 | TOOL_FREE | 无掩码，中等 B/θ |

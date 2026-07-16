# FC-CAS 说明书附图源文件

本目录存放发明专利「附图说明」对应的 Mermaid 源文件。提交专利局前需导出为 PNG 或 SVG 嵌入说明书。

## 文件清单

| 文件 | 对应附图 | 内容 |
|------|---------|------|
| [`fig1_architecture.md`](fig1_architecture.md) | 图 1 | 系统架构：Segmenter → StrategyTable → SchemaGuard → Draft/Verify |
| [`fig2_state_machine.md`](fig2_state_machine.md) | 图 2 | 区段状态机：NL / TOOL_SKELETON / TOOL_ENUM / TOOL_FREE |
| [`fig3_sequence.md`](fig3_sequence.md) | 图 3 | 单轮 FC 投机解码时序 |
| [`fig4_schema_guard.md`](fig4_schema_guard.md) | 图 4 | Schema 槽位约束草稿示意 |

图 5（评测曲线）待实验数据补齐后单独添加。

## 导出为 PNG

### 方式 A：mermaid-cli（推荐）

```bash
cd fc_cas/docs/figures

# 安装（一次性）
npm install -g @mermaid-js/mermaid-cli

# 批量导出 PNG（300 dpi，白底）
for f in fig1_architecture fig2_state_machine fig3_sequence fig4_schema_guard; do
  mmdc -i "${f}.md" -o "${f}.png" -b white -w 1600 -H 1200
done
```

若 mmdc 无法直接解析 `.md`，可只提取 mermaid 代码块：

```bash
sed -n '/^```mermaid$/,/^```$/p' fig1_architecture.md | sed '1d;$d' > /tmp/fig1.mmd
mmdc -i /tmp/fig1.mmd -o fig1_architecture.png -b white
```

### 方式 B：VS Code / Cursor

1. 安装扩展 [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid)。
2. 打开对应 `.md`，预览 Mermaid 图。
3. 右键导出或使用 [Mermaid Export](https://marketplace.visualstudio.com/items?itemName=GeekMermaid.mermaid-export) 扩展保存 PNG/SVG。

### 方式 C：Mermaid Live Editor

1. 打开 <https://mermaid.live>。
2. 粘贴 `.md` 中 ` ```mermaid ` 代码块内容。
3. 点击 **Actions → PNG/SVG** 下载。

## 导出为 SVG（矢量，专利排版优选）

```bash
mmdc -i fig1_architecture.md -o fig1_architecture.svg -b transparent
```

## 命名规范

导出文件与源文件同名、仅改扩展名：

```
fig1_architecture.png
fig2_state_machine.png
fig3_sequence.png
fig4_schema_guard.png
```

## 专利草稿引用

附图说明见 [`../patent/CN-发明专利-座舱FunctionCall内容感知投机解码-草稿.md`](../patent/CN-发明专利-座舱FunctionCall内容感知投机解码-草稿.md) §附图说明。

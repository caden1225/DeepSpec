# FC-CAS Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox syntax.  
> **REQUIRED when implementing:** Prefer one task per session slice; run listed tests before marking done.

**Goal:** 在 `fc_cas/` 内交付可运行的区段感知投机组件、可复核实验，并回填专利实施例。

**Architecture:** Segmenter → StrategyTable → SchemaGuard →（对接 DeepSpec）Draft/Verify；数据用 `processed_FC_dataset`。

**Tech Stack:** Python 3.12（conda `312`）、pytest、（后期）PyTorch + 本仓库 DeepSpec。

## Global Constraints

- 工作根目录：`/home/caden/workspace/DeepSpec/fc_cas/`  
- 专利不主张 DSpark/DFlash 本身；代码注释与文档须标明「现有技术载体」  
- 先 CPU 可测，再 GPU 实验  
- 申请人：郑冬  
- 数据集只读：经 `data/processed_FC_dataset` 软链访问  

---

## File map（目标态）

| Path | Responsibility |
|------|----------------|
| `src/fc_cas/types.py` | `Region`, `Strategy`, `SegmentInfo` |
| `src/fc_cas/segmenter/json_fc.py` | 增量 JSON + Schema 指针区段识别 |
| `src/fc_cas/strategy/table.py` | 区段→策略查表 |
| `src/fc_cas/schema/guard.py` | 模板注入候选、枚举 mask |
| `src/fc_cas/eval/offline_stats.py` | 离线区段/覆盖率统计 |
| `src/fc_cas/runtime/dspark_bridge.py` | 包装 DeepSpec proposal（后期） |
| `experiments/scripts/*.py` | 对照实验入口 |
| `docs/patent/*.md` | 专利文稿与递交清单 |
| `docs/figures/*.md` | 附图 mermaid 源 |

---

### Task 0: 工作区验收（已完成骨架则勾选验证）

**Files:**
- Verify: `README.md`, `DESIGN.md`, `PLAN.md`, `STATUS.md`, `data/processed_FC_dataset`

- [x] **Step 1:** 确认软链可读（`../../processed_FC_dataset` → total 8897）
- [x] **Step 2:** 更新 `STATUS.md` 将 P0 标为 done

---

### Task 1: 类型与包入口

**Files:**
- Create: `src/fc_cas/__init__.py`
- Create: `src/fc_cas/types.py`
- Create: `pyproject.toml`
- Test: `tests/test_types.py`

**Produces:**
- `Region`, `DraftMode`, `Strategy`, `SegmentInfo`

- [ ] **Step 1: Write failing test**

```python
# tests/test_types.py
from fc_cas.types import Region, DraftMode, Strategy

def test_strategy_fields():
    s = Strategy(block_size=4, confidence_threshold=0.6, mode=DraftMode.FREE)
    assert s.block_size == 4
    assert Region.NL.value == "nl"
```

- [ ] **Step 2: Run test — expect import fail**

```bash
cd /home/caden/workspace/DeepSpec && conda run -n 312 pytest fc_cas/tests/test_types.py -v
```

- [ ] **Step 3: Implement types + pyproject**

`types.py` 须含：

```python
from dataclasses import dataclass
from enum import Enum

class Region(str, Enum):
    NL = "nl"
    TOOL_SKELETON = "tool_skeleton"
    TOOL_ENUM = "tool_enum"
    TOOL_FREE = "tool_free"

class DraftMode(str, Enum):
    FREE = "free"
    TEMPLATE = "template"
    ENUM_MASK = "enum_mask"

@dataclass(frozen=True)
class Strategy:
    block_size: int
    confidence_threshold: float
    mode: DraftMode

@dataclass(frozen=True)
class SegmentInfo:
    region: Region
    allowed_strings: tuple[str, ...] | None = None
    json_pointer: str | None = None
```

`pyproject.toml`：包名 `fc_cas`，`packages` 指向 `src/fc_cas`。

- [ ] **Step 4: Install editable + pass test**

```bash
pip install -e ./fc_cas && pytest fc_cas/tests/test_types.py -v
```

Expected: PASS

---

### Task 2: StrategyTable

**Files:**
- Create: `src/fc_cas/strategy/__init__.py`
- Create: `src/fc_cas/strategy/table.py`
- Test: `tests/test_strategy.py`

**Produces:** `StrategyTable.lookup(region) -> Strategy`；`DEFAULT_TABLE` 符合 DESIGN §4.2

- [ ] **Step 1: Failing test** — TOOL_SKELETON.block_size > NL.block_size；ENUM 的 mode 为 `enum_mask`
- [ ] **Step 2: Implement `StrategyTable` with override dict**
- [ ] **Step 3: pytest PASS**

---

### Task 3: SchemaGuard（纯函数）

**Files:**
- Create: `src/fc_cas/schema/__init__.py`
- Create: `src/fc_cas/schema/guard.py`
- Test: `tests/test_schema_guard.py`

**Produces:**
- `next_template_prefix(prefix, tools, fn_name) -> str | None`
- `enum_allowed(tools, fn_name, arg_key) -> list[str] | None`
- `filter_logits_by_allowed(tokenizer, logits, allowed: list[str])` 可先 stub 为「返回 allowed 集合」，GPU 版后补

- [ ] **Step 1: Test** — 对 `ota_update` + field `action`，`enum_allowed` 含 `check_update`
- [ ] **Step 2: 从 tools schema 解析 properties.enum**
- [ ] **Step 3: pytest PASS**（用 `processed_FC_dataset/test.json` 中真实样本）

---

### Task 4: JSON FC Segmenter（核心）

**Files:**
- Create: `src/fc_cas/segmenter/__init__.py`
- Create: `src/fc_cas/segmenter/json_fc.py`
- Test: `tests/test_segmenter.py`

**Produces:** `classify_prefix(text: str, tools: list[dict], active_fn: str | None) -> SegmentInfo`

**算法（须实现，勿留空）：**

1. 若 `text` 不包含 tool call JSON 特征（无 `{` 作为 FC 载荷，或显式处于纯自然语言）→ `NL`  
2. 维护简易状态：寻找 `"arguments"` 后的 JSON 对象；对当前 key 查 schema  
3. 若在 key 名 / 结构符号 → `TOOL_SKELETON`  
4. 若在 value 且 schema 有 enum → `TOOL_ENUM` + `allowed_strings`  
5. 若在 value 且无 enum → `TOOL_FREE`

- [ ] **Step 1: 固定夹具** — 取 test 集一条 `ota_update` 的 `arguments` 逐步前缀断言 region 序列
- [ ] **Step 2: 实现最小状态机（可用正则+括号深度，不必上完整 JSON parser 库）**
- [ ] **Step 3: pytest PASS**
- [ ] **Step 4: 对 val 随机 50 条跑冒烟，写入 `artifacts/segmenter_smoke.json`**

---

### Task 5: 离线统计（专利图/表原料）

**Files:**
- Create: `src/fc_cas/eval/__init__.py`
- Create: `src/fc_cas/eval/offline_stats.py`
- Create: `experiments/scripts/run_offline_stats.py`
- Output: `experiments/results/offline_stats_test.json`

**Produces:** 对每条 assistant 序列（将 tool_calls 规范序列化为统一 JSON 文本）统计：

- 各 Region token 占比（按字符近似即可，注明近似）  
- 可被 template/enum 覆盖的字符占比  
- NL vs TOOL 样本计数

- [ ] **Step 1: 实现序列化** — `serialize_assistant(message) -> str`（content 与 tool_calls 的确定性拼接）
- [ ] **Step 2: 跑 test 全集**

```bash
python fc_cas/experiments/scripts/run_offline_stats.py --split test
```

- [ ] **Step 3: 将关键表粘贴进 `docs/patent/实施例数据-占位.md`**

---

### Task 6: 说明书附图

**Files:**
- Create: `docs/figures/fig1_architecture.md`
- Create: `docs/figures/fig2_state_machine.md`
- Create: `docs/figures/fig3_sequence.md`
- Create: `docs/figures/fig4_schema_guard.md`
- Update: `docs/patent/CN-发明专利-...草稿.md` 附图说明指向上述文件

- [ ] **Step 1: 用 mermaid 画齐图 1–4**
- [ ] **Step 2: 导出说明：如何用 mermaid-cli/IDE 出 PNG（写入 `docs/figures/README.md`）**

---

### Task 7: DeepSpec Runtime 桥（GPU 可选）

**Files:**
- Create: `src/fc_cas/runtime/dspark_bridge.py`
- Create: `experiments/configs/baseline_global.yaml`
- Create: `experiments/configs/cas_segmented.yaml`
- Create: `experiments/scripts/run_compare_eval.py`

**Produces:** 包装函数：

```python
def cas_proposal_hparams(segment: SegmentInfo, table: StrategyTable) -> dict:
    """return {block_size, confidence_threshold, mode, allowed_strings}"""
```

在线评测依赖：座舱 target 路径、draft ckpt。若缺失，脚本须清晰报错并仍能跑 **mock verify**（用 gold 下一 token 模拟接受，仅用于管线打通）。

- [ ] **Step 1: mock verify 通路在 CPU 跑通 20 条 test**
- [ ] **Step 2: 真实 GPU 对照（权重就绪后）写入 `experiments/results/compare_*.json`**
- [ ] **Step 3: 回填专利「实施例 3」数值表**

---

### Task 8: 领域草稿训练（可选增强，不阻塞专利方法稿）

**Files:**
- Create: `experiments/scripts/prepare_fc_target_cache.md`（步骤说明）
- Create: `experiments/configs/dspark_fc_cas.yaml`（或 py config 说明）

- [ ] **Step 1: 将 train 划分转为 DeepSpec jsonl + loss_mask 约定**
- [ ] **Step 2: 小样本 cache（≤200）+ 短训验证 loss 下降**
- [ ] **Step 3: 全量训练与 Task 7 复评（有算力再做）**

---

### Task 9: 专利定稿包

**Files:**
- Update: `docs/patent/CN-发明专利-座舱FunctionCall内容感知投机解码-草稿.md`
- Create: `docs/patent/递交清单.md`
- Create: `docs/patent/权利要求-实现对照表.md`

- [ ] **Step 1: 权利要求每条映射到模块/函数名**
- [ ] **Step 2: 摘要/说明书与实现一致化（去掉内部备忘或移到 `docs/patent/INTERNAL.md`）**
- [ ] **Step 3: 递交清单勾选：申请人、附图、实施例数据、代理检索**

---

## Phase timeline（切实可行）

| Phase | Tasks | 预估 | 依赖 |
|-------|-------|------|------|
| P0 工作区 | T0 | 0.5h | 无 |
| P1 核心库 CPU | T1–T4 | 1–2d | 无 GPU |
| P2 离线证据 | T5–T6 | 0.5–1d | P1 |
| P3 在线对照 | T7 | 1–3d | 座舱模型权重 / 或 mock |
| P4 增强训练 | T8 | 按算力 | 可选 |
| P5 专利包 | T9 | 0.5d | P2 至少；P3 更佳 |

**最小可申报路径：** P0→P1→P2→T9（方法+离线统计实施例）。  
**完整说服力路径：** 再加上 P3（在线 τ/speedup）。

---

## Execution handoff

Plan 已写入本文件。执行方式二选一：

1. **Subagent-Driven** — 每 Task 一个子代理，Task 间人工过目  
2. **Inline** — 本会话按 Task 1→…连续推进  

默认建议：先 Inline 完成 **Task 1–4**（纯 CPU 核心），再决定 GPU 实验。

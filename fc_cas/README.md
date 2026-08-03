# FC-CAS：座舱 Function Call 区段感知投机解码

> **全称**：Function-Call Content-Aware Speculation  
> **申请人 / 发明人**：郑冬  
> **宿主仓库**：DeepSpec（复用 DSpark 草稿训练/验证通路，**不修改**其作为现有技术的主张边界）  
> **本目录**：专利、实现、评测、附图与实验的唯一工作区

## 目标（一句话）

在标准 JSON Function Call 输出上，用**区段感知调度 + Schema 槽位约束草稿**提升投机接受长度与加速比，并以 `processed_FC_dataset` 完成可复核实验，回填发明专利实施例。

## 保护重心（已锁定）

1. **独立权利要求**：NL vs TOOL 区段差异化草稿长度 / 置信度阈值  
2. **从属强化**：TOOL 内再分骨架 / 枚举 / 自由槽 + Schema 约束与模板注入  
3. **组合从属**：区段策略 × 置信度头两级截断  

不主张 DSpark/DFlash/EAGLE 本身。

## 目录

```
fc_cas/
├── README.md                 # 本文件
├── DESIGN.md                 # 设计规格
├── PLAN.md                   # 可执行实施计划（按 Task 推进）
├── STATUS.md                 # 进度看板
├── pyproject.toml            # 本包最小依赖
├── data/processed_FC_dataset → ../../processed_FC_dataset
├── docs/
│   ├── patent/               # 专利文稿与递交清单
│   └── figures/              # 说明书附图源（mermaid/svg）
├── src/fc_cas/               # 实现包
│   ├── segmenter/            # 区段识别（JSON 状态机 + Schema 指针）
│   ├── strategy/             # 分区策略表
│   ├── schema/               # 枚举约束 / 模板注入
│   ├── runtime/              # 与 DeepSpec 评测对接的 runtime
│   └── eval/                 # 指标与对照实验
├── tests/                    # 单测（优先 CPU、无 GPU）
├── experiments/              # 配置、脚本、结果
└── artifacts/                # 导出表、图、专利补强片段
```

## 环境

```bash
conda activate 312
cd /home/caden/workspace/DeepSpec
pip install -e ./fc_cas   # 待 pyproject 就绪后
pytest fc_cas/tests -q
```

## 阅读顺序

1. `DESIGN.md` — 为何如此切  
2. `PLAN.md` — 按 Task 执行  
3. `docs/patent/` — 法律文稿  
4. `STATUS.md` — 当前进度  

## 状态

见 `STATUS.md`。当前阶段：**P0 工作区已建，进入 P1 区段识别实现**。

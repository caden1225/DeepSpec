# FC-CAS Status

| Phase | Status | Notes |
|-------|--------|-------|
| P0 工作区 | **done** | `fc_cas/` 已建 |
| P1 核心库 CPU (T1–T4) | **done** | segmenter / strategy / schema / runtime |
| P2 离线证据 (T5–T6) | **done** | test 可覆盖字符 **85.75%** |
| P3 在线对照 (T7) | **done** | tools-aware 30 条：\(\tau=1.30\)；分区 \(\tau_{\text{ENUM}}=1.40>\tau_{\text{NL}}=1.06\) |
| P4 领域训练 (T8) | **done** | 200 样本 cache + 60 step，loss≈2.45，ckpt `step_60` |
| P5 专利定稿包 (T9) | **done** | 实施例已回填 tools-aware 数据 |

## 最近更新

- 2026-07-16：v1 尝试（B=16，1k 样本，200 step）→ tools \(\tau\)=2.10，ENUM/NL=2.22×；CAS≈全局
- 理想画像：`fc_cas/docs/patent/理想实施例数据画像.md`
- 结果：`tools_aware_compare_b16.json` / `tools_aware_compare_b16_casgap.json`
- 下一步：Schema 在线约束冲 CAS>全局；注意根盘空间（ckpt 已迁 `/storage`）

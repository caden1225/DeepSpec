# FC-CAS Status

| Phase | Status | Notes |
|-------|--------|-------|
| P0 工作区 | **done** | `fc_cas/` 已建；数据软链已修；专利稿已复制 |
| P1 核心库 CPU (T1–T4) | **done** | 区段分类、策略表、Schema 守卫与运行时桥接已完成 |
| P2 离线证据 (T5–T6) | **done** | test 字符级统计：template/enum 可覆盖 94,004 / 109,625 字符（85.75%） |
| P3 在线对照 (T7) | **partial** | CPU mock 路由对照完成（20 条 test）；真实 \(\tau\)/Speedup 待座舱 target 与草稿权重 |
| P4 领域训练 (T8) | **deferred** | 不阻塞最小专利申报路径 |
| P5 专利定稿包 (T9) | **done** | v3 草稿、内部备忘分离、实施例数据、对照表及递交清单已更新 |

## 最近更新

- 2026-07-16：创建工作区、DESIGN、PLAN；复制专利 v2；链到 `processed_FC_dataset`
- 2026-07-16：完成 CPU 核心、离线统计及 CPU mock 路由对照；test 可覆盖字符占比为 85.75%
- 2026-07-16：完成 T9 专利定稿包；真实在线实验与领域训练保留为后续工作

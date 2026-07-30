# MMA — Stage A 条件对齐（进行中 / 模板）

> **所指 tip：** `feat/cond-mm-stage-a` · entry `eccv-3view-mm-a`  
> **文档源：** `main`/`docs/eccv_stage_reports/`  
> **路线：** [COND_MM_ROADMAP.md](./COND_MM_ROADMAP.md)

## 设定

- parent ckpt: `260723-162838` best
- 冻：AR + FSQ
- 训：MultiModal Encoder + decoder cross-attn + surface-type head
- 数据：`processed/cond_cache_v2`
- 早停：mid official gen_success（兼 summary）

## 状态

- [ ] cache v2 manifest 完成
- [ ] R_smoke（40 step）
- [ ] Stage A train
- [ ] GT test（best + analytic）
- [ ] 对照 R1 / 旧基线后硬门禁判定

## 指标表（跑完填）

| 指标 | 旧 GT 260725-002218 | R1 | MMA best |
|------|---------------------|----|----------|
| gen_success | 0.353 | 0.500 | |
| summary | 0.046 | 0.042 | |

## 判定

（过门禁条件：gen≥0.50 且 Δsummary≥−0.005 vs 0.046；否则停、不进 Stage B。）

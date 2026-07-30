# ECCV 阶段报告

> **文档源：`main`/`docs/eccv_stage_reports/`。**  
> 每个 **R\*** / **P1\*** 节点跑完后填表；**硬停**确认后再 enqueue 下一阶段。

## 门禁顺序

| 门禁 | 内容 | tip / 工具 |
|------|------|------------|
| R_smoke | 通路冒烟（40 step） | tip `eccv-p0` · workflow `eccv-challenge-gated` |
| R0 | 旧 pred analytic 后处理 A/B | 脚本（不改 tip） |
| R1 | 现有 3view ckpt + analytic GT test | tip `eccv-p0` · parent `260723-162838` |
| TechDraw 划分 | XY-Cut regions（锁定） | 归档 `main` · 实现 `eccv-p0` / P1a · [TECHDRAW_VIEW_SPLIT.md](./TECHDRAW_VIEW_SPLIT.md) |
| P1-A | 图元 Encoder soft prefix | tip `feat/p1-prim-encoder-prefix` |
| P1-B | decoder cross-attn | tip `feat/p1-decoder-cross-attn` |

## 对照基线

| 角色 | run | 指标 |
|------|-----|------|
| parent train | `260723-162838` | 3view-geom |
| GT test（旧） | `260725-002218` | gen≈35.3%, summary≈0.046 |

## 报告索引

| 文件 | 所指 tip / 说明 |
|------|-----------------|
| [TECHDRAW_VIEW_SPLIT.md](./TECHDRAW_VIEW_SPLIT.md) | 划分错误原因 + XY-Cut 锁定方案（`main`） |
| [GEN_FAIL_TAXONOMY.md](./GEN_FAIL_TAXONOMY.md) | Gen 低：L1 失败占比 + L2 打标计划 |
| [R0_latest.md](./R0_latest.md) | 后处理 A/B |
| [R1_latest.md](./R1_latest.md) | `eccv-p0` GT test |
| [P1A_latest.md](./P1A_latest.md) | `feat/p1-prim-encoder-prefix` · `260728-212715` |

## 报告填写 checklist

- [ ] summary / gen_success / surface|edge|vertex|topo F1 / CD（若有）
- [ ] vs `260725-002218`
- [ ] 标明所用 tip / commit / ckpt（best vs last）
- [ ] 下一动作：停 / 重训 / 进下一门禁

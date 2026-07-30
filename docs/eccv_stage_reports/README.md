# ECCV 阶段报告

> **文档源：`main`/`docs/eccv_stage_reports/`。**  
> 每个门禁跑完后填表；**硬停**确认后再 enqueue 下一阶段。

## 门禁顺序

| 门禁 | 内容 | tip / 工具 |
|------|------|------------|
| R_smoke | 通路冒烟（40 step） | tip `eccv-p0` |
| R0 | 旧 pred analytic 后处理 A/B | 脚本 |
| R1 | 现有 3view ckpt + analytic GT test | tip `eccv-p0` · parent `260723-162838` |
| TechDraw 划分 | L-layout 硬切（锁定） | [TECHDRAW_VIEW_SPLIT.md](./TECHDRAW_VIEW_SPLIT.md) |
| P1-A | soft prefix（**失败归档**） | `feat/p1-prim-encoder-prefix` |
| **MM Stage A** | 多模态条件对齐 | **`feat/cond-mm-stage-a`** · [COND_MM_ROADMAP.md](./COND_MM_ROADMAP.md) |
| MM Stage B | 局部解冻 | 过 A 门禁后开 tip |
| MM Stage C | 解析参数 + 约束 | 过 B 后 |

## 对照基线

| 角色 | run | 指标 |
|------|-----|------|
| parent train | `260723-162838` | 3view-geom |
| GT test（旧） | `260725-002218` | gen≈35.3%, summary≈0.046 |
| R1 | `260728-201614` | gen≈50%, summary≈0.042 |

## 报告索引

| 文件 | 所指 tip / 说明 |
|------|-----------------|
| [COND_MM_ROADMAP.md](./COND_MM_ROADMAP.md) | 条件多模态主线 + cache 约定 |
| [MMA_latest.md](./MMA_latest.md) | Stage A 实验记录 |
| [TECHDRAW_VIEW_SPLIT.md](./TECHDRAW_VIEW_SPLIT.md) | L-layout 划分 |
| [GEN_FAIL_TAXONOMY.md](./GEN_FAIL_TAXONOMY.md) | Gen 失败 L1/L2 |
| [R0_latest.md](./R0_latest.md) | 后处理 A/B |
| [R1_latest.md](./R1_latest.md) | R1 GT test |
| [P1A_latest.md](./P1A_latest.md) | P1-A 失败归档 |

## 报告填写 checklist

- [ ] summary / gen_success / surface|edge|vertex|topo F1 / CD（若有）
- [ ] vs `260725-002218` 与 R1
- [ ] 标明 tip / commit / ckpt（best vs last）
- [ ] 下一动作：停 / 重训 / 进下一门禁

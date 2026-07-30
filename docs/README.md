# AutoBrep 文档索引（唯一源：`main`）

> **约定（2026-07-30）：** 所有项目文档统一维护在 **`main` 的 `docs/`**。  
> 功能 tip（`eccv-p0`、`feat/p1-*` 等）只承载**代码**；文档变更请提交到 `main`，各 tip 需要时再 cherry-pick/合并文档目录。  
> 每篇文档应标明「所指实现分支」，避免在分支上再分叉一份过期说明。

## 分支速查

| 分支 / tip | 用途 | 入口 / tag |
|------------|------|------------|
| `main` | 文档归档 + TechDraw 库代码归档 | — |
| `eccv-3view-geom` | 3view 条件基线 | entry `eccv-3view-geom` |
| `eccv-p0` | P0：fast metrics + XY-Cut TechDraw + analytic STEP | tag `autobrep-eccv-p0` · entry `eccv-3view-p0` |
| `feat/p1-prim-encoder-prefix` | P1-A：PrimTransformer → soft prefix | entry `eccv-3view-p1a` |
| `feat/p1-decoder-cross-attn` / `eccv-p1` | P1-B：decoder cross-attn | entry `eccv-3view-p1b` |
| `eccv-p2` | P2 冲榜开关位 | entry 见 registry |

## 文档目录

### ECCV 挑战赛

| 文档 | 内容 | 所指分支 |
|------|------|----------|
| [eccv_upgrade_p0_p1_p2.md](./eccv_upgrade_p0_p1_p2.md) | P0→P1→P2 路线、开关、TechDraw 锁定管线 | `eccv-p0` / `feat/p1-*` / `eccv-p2` |
| [eccv_pipeline_completeness.md](./eccv_pipeline_completeness.md) | 管线完备性与优先级 | ECCV tip 族 |
| [eccv_sft.md](./eccv_sft.md) | ECCV SFT 说明 | ECCV tip 族 |
| [eccv_stage_reports/](./eccv_stage_reports/) | 阶段门禁报告（R0/R1/P1A…） | 见各报告内标注 |
| [eccv_stage_reports/TECHDRAW_VIEW_SPLIT.md](./eccv_stage_reports/TECHDRAW_VIEW_SPLIT.md) | **TechDraw XY-Cut 锁定方案**（问题→修复） | `main` 归档；实现同步 `eccv-p0` / P1a |
| [eccv_stage_reports/GEN_FAIL_TAXONOMY.md](./eccv_stage_reports/GEN_FAIL_TAXONOMY.md) | **Gen 失败归因**（L1 硬统计 + L2 计划） | 数据自 `eccv-p0` / P1a runs；文档在 `main` |

### 阶段报告

| 报告 | 所指实验 tip |
|------|----------------|
| [R0_latest.md](./eccv_stage_reports/R0_latest.md) | 后处理 A/B（脚本，不改 tip） |
| [R1_latest.md](./eccv_stage_reports/R1_latest.md) | `eccv-p0` + parent `260723-162838` GT test |
| [P1A_latest.md](./eccv_stage_reports/P1A_latest.md) | `feat/p1-prim-encoder-prefix` · run `260728-212715` |
| [GEN_FAIL_TAXONOMY.md](./eccv_stage_reports/GEN_FAIL_TAXONOMY.md) | R1/`260728-201614` 等 · L1 decode vs rebuild |

### 其它

| 文档 | 内容 | 所指分支 |
|------|------|----------|
| [dataflow_and_operators.md](./dataflow_and_operators.md) | 数据流与算子 | 多 tip，以文内为准 |
| [pc_cond_dev.md](./pc_cond_dev.md) | 点云条件 | `pc-cond` |
| [advisor_2page_brief.md](./advisor_2page_brief.md) | 顾问简报 | — |
| [ppt_*.md](./ppt_storyboard_4slides.md) | PPT 素材 | — |

## TechDraw 一句话

**L-layout 硬切**：先按 gutter 分主/侧栏（或俯+侧行），剩余为第三视图；同宽/同高对齐打分；按切缝象限归属。详见 [TECHDRAW_VIEW_SPLIT.md](./eccv_stage_reports/TECHDRAW_VIEW_SPLIT.md)。

## Gen 失败一句话

R1（348 test，gen=50%）失败里 **rebuild≈76% / decode≈24%**；细到「引用/拟合/缝合」尚未打标。详见 [GEN_FAIL_TAXONOMY.md](./eccv_stage_reports/GEN_FAIL_TAXONOMY.md)。

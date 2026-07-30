# R1 — Existing 3view ckpt + analytic GT test

> **所指 tip：** `eccv-p0` / entry `eccv-3view-p0` @ `9a931b0`  
> **文档源：** `main`/`docs/eccv_stage_reports/`

- 时间: 20260729-151839
- jid: `25616aec`
- status: **finished**
- tip: `eccv-3view-p0` @ `9a931b0`
- checkpoint: `260723-162838` (via R1_ckpt_host)
- postprocess_analytic: 1
- R0: 事后 analytic Δsummary≈+0.0003

## 判定（硬门禁 1）

- gen↑ 明显（35%→50%），但 summary↓：更多合法 STEP，几何/拓扑 F1 略降。
- 事后拟合 alone 不够；继续 P1-A 图元编码已在跑。
- **默认不停训**：P1a 已按要求接上。

## 指标表

| 指标 | baseline 260725-002218 | R1 (analytic) | Δ |
|------|------------------------|---------------|---|
| summary | 0.045996 | 0.042056 | -0.003940 |
| gen_success | 0.353448 | 0.500000 | +0.146552 |
| n_generated | 123 | 174 | +51.000000 |
| valid_ratio | 1.000000 | 1.000000 | +0.000000 |
| surface_f1 | 0.064516 | 0.063843 | -0.000673 |
| edge_f1 | 0.020709 | 0.017340 | -0.003369 |
| vertex_f1 | 0.016469 | 0.015390 | -0.001079 |
| topo_f1 | 0.018589 | 0.016365 | -0.002224 |

**Δsummary = -0.003940**

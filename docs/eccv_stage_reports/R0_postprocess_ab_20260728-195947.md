# R0 — Analytic STEP post-process A/B

- 时间: 20260728-195947
- 基线 pred: `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260725-002218/eccv-3view-geom-resume__test/metrics/official_test/pred`
- analytic out: `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/stage_gates/R0_postprocess_ab/pred_analytic`
- 评测 work: `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/stage_gates/R0_postprocess_ab/official_test_analytic`
- n_pred STEP: 123

## 判定（赛题）

- Δsummary (analytic − raw) = **+0.000287**
- 若 Δ 明显为正：后处理可保留作推理默认；曲面类型问题仍建议后续进模型。
- 若 Δ≈0：事后拟合几乎无用，优先做模型内 surf-type，而不是重训 P1。

## 指标表

| 指标 | raw (260725-002218) | analytic | Δ |
|------|---------------------|----------|---|
| summary | 0.045996 | 0.046283 | +0.000287 |
| gen_success | 0.353448 | 0.353448 | +0.000000 |
| valid_ratio | 1.000000 | 1.000000 | +0.000000 |
| IR≈invalid_ratio | 0.000000 | 0.000000 | +0.000000 |
| surface_f1 | 0.064516 | 0.065395 | +0.000879 |
| edge_f1 | 0.020709 | 0.020709 | -0.000001 |
| vertex_f1 | 0.016469 | 0.017269 | +0.000800 |
| topo_f1 | 0.018589 | 0.018989 | +0.000400 |
| CD_surface (median) | — | 0.151505 | — |
| CD_edge | — | 0.207269 | — |
| CD_vertex | — | 0.323979 | — |

## 下一步（硬门禁）

- 通过 → 继续 **R1**：现有 3view ckpt (`260723-162838`) + `postprocess_analytic=1` 全量 GT test。
- 停在本报告：不自动开 P0 50ep / P1 全量训。

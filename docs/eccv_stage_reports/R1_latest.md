# R1 — Existing 3view ckpt + analytic GT test（进行中）

- 状态: **RUNNING**
- tip: `eccv-p0` @ `39bd634`（`postprocess_analytic=1`）
- run_dir: `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/stage_gates/R1_ckpt_gt_test`
- checkpoint: `260723-162838` `last.ckpt`
- 对照基线: `260725-002218`（summary≈0.046, gen≈35.3%）
- R0 参考: 事后 analytic Δsummary≈+0.0003
- 最新进度: `8/348 (2.3%) ok=3 fail=5 | 277.5/h ETA 1.2h | chunk 8/348 8.9s (AR 7.6s + decode 1.1s) ids=['000756']`
- 说明: 因 launcher 对 parent `run_id` 会落到 `eccv-3view-geom` tip，R1 改为在 p0 tip 上直接跑 `eval_eccv_split.py`。

**硬门禁 1**：完成后写正式指标表并停等确认；不自动开 P0-50ep / P1。

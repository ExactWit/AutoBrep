# R1 — Existing 3view ckpt + analytic GT test（launcher 进行中）

- 状态: **RUNNING on exp_launcher**
- jid: `25616aec`
- workflow: `eccv-r1-then-p1a` (`1cdd1417ce63`)
- 下一步排队: `p1a-prim-token-train`（图元 Encoder soft prefix，50ep）
- tip 应以 `eccv-3view-p0` / `autobrep-eccv-p0` 为准（host 避免继承 geom tip）
- 对照: `260725-002218`；R0 Δsummary≈+0.0003

**硬门禁 1**：本 Job 出 `metrics/test.json` 后写正式表；P1a 已按要求自动接上。

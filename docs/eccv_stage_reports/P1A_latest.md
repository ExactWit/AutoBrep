# P1-A 结论（图元 PrimTransformer → soft prefix）

> **所指 tip：** `feat/p1-prim-encoder-prefix` · entry `eccv-3view-p1a`  
> **文档源：** `main`/`docs/eccv_stage_reports/`

- 时间: 2026-07-29
- jid: `65e84657`
- run: `260728-212715/p1a-prim-token-train__train`
- entry: `eccv-3view-p1a` @ `feat/p1-prim-encoder-prefix` (`1448dbd`)
- 设定: 冻 AR+FSQ；只训 PrimTransformerEncoder + SoftPrefixCompressor；50ep；M=64

## 判定（硬）

**P1-A 以 ep50/last 为准失败，不应直接进 P1-B。**

- full val（694，**last @ step 76400**）: gen **23.3%**，summary **0.025**  
  对比 R1 GT test: gen 50% / summary 0.042；旧 GT: gen 35% / summary 0.046
- mid-24 的 summary↑（0.027→0.086）是**幸存者偏差**（gen 54%→33%），不能当涨分
- `best.ckpt` = step **10696**（val_loss 0.225），之后 val_loss 漂到 ~0.258；**full official 没用 best**
- 失败结构（full）: rebuild 321、decode 211、ok 162 → gen 主战场仍在，且 decode 比例偏高（跨 run 归因见 [GEN_FAIL_TAXONOMY.md](./GEN_FAIL_TAXONOMY.md)）

## 指标表

| 阶段 | gen | summary | surface_f1 | 备注 |
|------|-----|---------|------------|------|
| mid ep12 (24) | 0.542 | 0.027 | 0.031 | |
| mid ep25 (24) | 0.458 | 0.059 | 0.092 | |
| mid ep38 (24) | 0.333 | 0.086 | 0.115 | mid 峰值，样本极少 |
| **full ep50 last** | **0.233** | **0.025** | **0.036** | 正式结论用这个 |
| R1 GT (best+analytic) | 0.500 | 0.042 | 0.064 | 对照 |
| 旧 GT 260725-002218 | 0.353 | 0.046 | 0.065 | 对照 |

fast val（冻 AR 下仍恶化）: ppl 1.26→1.30，token/geom acc 缓慢下降 → prefix 后期损害 CE。

## 根因（与 viz 对齐）

1. **三视图空间切分不可靠** → view_id / 局部归一化几何被污染（P1-A 更依赖 view embed + per-view geom）
2. **训太久过拟合条件侧**：best 很早，last 官方崩
3. mid-24 误导：summary 与 gen 反向

## 下一步（建议顺序）

1. **P1a-best GT test**（不重训）：`best.ckpt` + analytic，对标 R1  
2. **修 TechDraw 视图划分**（数据/规则），再决定是否重训 encoder  
3. 划分修好后短训 **P1a-v2**（早停 / 跟 gen 或 summary，勿盲跟 val_loss）  
4. **暂缓 P1-B**（decoder xattn），除非 best GT ≥ R1  
5. 并行：**rebuild/decode** 鲁棒性（与条件编码正交的 gen 主战场）

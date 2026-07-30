# MMA — Stage A 条件对齐

> **所指 tip：** `feat/cond-mm-stage-a` @ `f6d48d0` · entry `eccv-3view-mm-a`  
> **文档源：** `main`/`docs/eccv_stage_reports/`  
> **路线：** [COND_MM_ROADMAP.md](./COND_MM_ROADMAP.md)

## 设定

- parent ckpt: `260723-162838` best（`eccv-view-step=022920-val_loss=0.2253.ckpt`，已链 `best.ckpt`）
- 冻：AR + FSQ
- 训：ViewConditionEncoder + decoder cross-attn + SurfaceTypeHead
- 数据：`processed/cond_cache_v2`（manifest n=7152, n_fail=0）
- 早停：mid official gen_success（兼 summary）

## 状态

- [x] cache v2 manifest 完成（train/val/test；parquet 子集 7152）
- [x] R_smoke（本地 8 step，`stage_gates/mma_smoke`，train_loss↓，通路 OK）
- [x] Stage A 正式训 **RUNNING** via **exp_launcher**：`jid=ac395002` · `260730-174645/mma-stage-a__train`（bs=1, accum=4, 50ep；cond_cache_v2；xattn+surf）
- [ ] GT test（best + analytic）
- [ ] 对照 R1 / 旧基线后硬门禁判定

> 已废弃并删除的 nohup 跑：`260730-172526` / `260730-173000` / `260730-173100`（未走 launcher）。

## Smoke 备忘

- run dir: `/data/hdd/exps/runs/.../stage_gates/mma_smoke`
- trainable ≈ 309M（encoder+xattn+surf head）；frozen ≈ 1.0B
- 命令要点：`--use-decoder-cross-attn 1 --enable-aux-surf-type 1 --cond-cache-root ... --ar-ckpt .../best.ckpt`

## 指标表（正式训完填）

| 指标 | 旧 GT 260725-002218 | R1 260728-201614 | MMA best |
|------|---------------------|------------------|----------|
| gen_success | 0.353 | 0.500 | |
| summary | 0.046 | 0.042 | |

## 判定

过门禁：gen≥0.50 且 Δsummary≥−0.005 vs 0.046 → 开 `feat/cond-mm-stage-b`；否则停。

## Job 启动（正式训）

**必须经 exp_launcher**（禁止 nohup / 直接 `./run.sh train`）：

```bash
# Web: entry eccv-3view-mm-a → train
# 或 API / CLI；默认已含 bs=1 accum=4、cond_cache_v2、ar_ckpt、xattn、surf-type
conda activate launcher
# 查状态
exp_launcher jobs show ac395002 --json
```

tmux: `exp-AutoBrep-mma-stage-a-260730-174645-train`  
log: `.../260730-174645/mma-stage-a__train/logs/out.log`

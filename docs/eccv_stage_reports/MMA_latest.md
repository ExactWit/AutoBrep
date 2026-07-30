# MMA — Stage A 条件对齐

> **所指 tip：** `feat/cond-mm-stage-a` @ `334f7f5` · entry `eccv-3view-mm-a`  
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
- [ ] Stage A 正式训（launcher entry `eccv-3view-mm-a` / workflow `workflows/eccv_mm_stage_a.yaml`）
- [ ] GT test（best + analytic）
- [ ] 对照 R1 / 旧基线后硬门禁判定

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

```bash
# preprocess cache（若尚未跑）
BUILD_COND_CACHE=1 ./run.sh preprocess --data-dir /data/hdd/datasets/eccv2026ws-cad-data --num-workers 8

# launcher: entry eccv-3view-mm-a，或：
./run.sh train --exp-dir <out> --dataset eccv2026ws-cad-data \
  --use-decoder-cross-attn 1 --enable-aux-surf-type 1 \
  --cond-cache-root /data/hdd/datasets/eccv2026ws-cad-data/processed/cond_cache_v2 \
  --ar-ckpt /data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260723-162838/eccv-3view-geom-resume__train/checkpoints/best.ckpt \
  --max-epochs 50
```

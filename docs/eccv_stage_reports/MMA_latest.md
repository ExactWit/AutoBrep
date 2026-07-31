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
- [x] Stage A 正式训 via **exp_launcher**：`jid=ac395002` · `260730-174645/mma-stage-a__train`（bs=1, accum=4, 50ep；cond_cache_v2；xattn+surf）
  - ⚠️ **ep42 崩溃终止**（非自然完成）：DataLoader worker Segmentation fault，`status=finished` 但 full-694 official eval **从未跑**
  - ckpt：`best.ckpt`（step3056, val_loss=0.3308，实际很早）/ `last.ckpt`（ep42 崩前）
- [x] 归档判定（见下）：**失败，不过门禁，停 Stage B**

> 已废弃并删除的 nohup 跑：`260730-172526` / `260730-173000` / `260730-173100`（未走 launcher）。

## Smoke 备忘

- run dir: `/data/hdd/exps/runs/.../stage_gates/mma_smoke`
- trainable ≈ 309M（encoder+xattn+surf head）；frozen ≈ 1.0B
- 命令要点：`--use-decoder-cross-attn 1 --enable-aux-surf-type 1 --cond-cache-root ... --ar-ckpt .../best.ckpt`

## 结果（归档 2026-07-31）

**训练在 ep42 因 DataLoader worker SegFault 崩溃终止，full-694 official eval 未执行。** 仅有 mid-24 三次快照 + fast_val 全程。

### mid-24 official（固定 24 样本，gen_success_rate / n_generated）

| 阶段 | gen_success | n_generated | 备注 |
|------|------------|-------------|------|
| ep12 (step18336) | 0.375 | 9 | |
| ep25 (step38200) | **0.625** | 15 | mid 峰值 |
| ep38 (step58064) | 0.333 | 8 | 回落 |

失败结构：rebuild_failed 为主，decode_failed 次之；高 n_prims 样本（52/81/128）几乎全崩 → 复杂多图元样本仍是主战场。

### fast_val（冻 AR 下持续恶化 → 条件侧损害 CE）

| epoch | ppl | token_acc | geom_acc |
|-------|-----|-----------|----------|
| 1 | 1.285 | 0.9469 | 0.9438 |
| 11 | 1.391 | 0.9412 | 0.9378 |
| 22 | 1.403 | 0.9400 | 0.9366 |
| 32 | 1.467 | 0.9369 | 0.9333 |
| 42 | 1.437 | 0.9372 | 0.9335 |

ppl 单调↑（1.29→1.47），token/geom acc 缓慢↓ → **cross-attn + surf-type 注入在冻 AR 下损害其 CE 表征**，与 P1-A 结论同向。

## 指标表（正式训完填）

| 指标 | 旧 GT 260725-002218 | R1 260728-201614 | MMA best |
|------|---------------------|------------------|----------|
| gen_success | 0.353 | 0.500 | 仅 mid-24 峰值 0.625（不可比，样本极少） |
| summary | 0.046 | 0.042 | 未测（full eval 未跑） |
| fast_val ppl | — | — | 1.29→1.47（恶化） |

## 判定

**失败，不过门禁，停 `feat/cond-mm-stage-b`。** 理由：

1. 训练本身未完成（ep42 崩溃），full-694 未评，无有效 gen_success/summary 正式值。
2. mid-24 gen_success 不稳（0.375→0.625→0.333），无法支撑门禁。
3. **fast_val ppl 持续恶化**：冻结 AR 下 cross-attn+surf 注入损害 CE，说明**双注入（soft-prefix + decoder cross-attn）+ aux head 这条路在冻 AR 设定下有害**。
4. 结论与 P1-A 一致：条件注入损害 frozen AR。→ 转向**更轻的注入**：`eccv-prim-prefix-direct`（per-prim 直进 prepend，**无 compressor、无 decoder cross-attn、无 aux head**）消融已在跑（jid=8de616d4，见 [P1A_latest.md](./P1A_latest.md) 路线）。

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
**终止原因**: DataLoader worker SegFault @ ep42（`torch/utils/data/_utils/signal_handling.py`）；非自然完成。

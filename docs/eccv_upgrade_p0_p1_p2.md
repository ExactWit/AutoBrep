# ECCV 升级路线（P0 → P1 → P2）

## 分支图

```
eccv-3view-geom (baseline @ 0b2c8b9)
  ├─ feat/p0-fast-metrics
  ├─ feat/p0-techdraw-split
  ├─ feat/p0-step-postprocess
  └─ eccv-p0  (tag: autobrep-eccv-p0)
       └─ feat/p1-prim-encoder-prefix  → entry eccv-3view-p1a
            └─ feat/p1-decoder-cross-attn → entry eccv-3view-p1b / tip eccv-p1
                 └─ eccv-p2
```

## Registry entries

| entry_id | git_ref | 相对基线 |
|----------|---------|----------|
| `eccv-3view-geom` | `eccv-3view-geom` | 3view + KV fix |
| `eccv-3view-p0` | `autobrep-eccv-p0` / `eccv-p0` | fast metrics + hist-split/groups + analytic STEP postprocess |
| `eccv-3view-p1a` | `feat/p1-prim-encoder-prefix` | 图元 Transformer Encoder→M=64 soft prefix（冻 AR）；CLI `--use-prim-seq-encoder 1` |
| `eccv-3view-p1b` | `feat/p1-decoder-cross-attn` | AR 每层 cross-attn（冻 AR 原权重） |

P1-A note 模板：`P1a prim-encoder prefix; parent=260723-162838; compare vs p0 / 260725-002218`

## 开关

| 开关 | 默认 | 说明 |
|------|------|------|
| Level-1 fast metrics | on | `val/fast/*` + `metrics/fast_val_epochXXX.json` |
| `--postprocess-analytic` | 1 | STEP 写出前解析曲面替换 + ShapeFix；`0` 做 A/B |
| hist-split | on | `split_into_views(use_histogram=True)`；失败回退 k-means |
| P1 prefix encoder | entry 决定 | `eccv-3view-p1a` |
| P1 decoder xattn | entry 决定 | `eccv-3view-p1b` |
| P2 aux losses | off | `eccv-p2` 可配开关 |
| FSQ 精度上调 | off | P2 冲榜单独开关 |

## 评测对照 run_id

| 角色 | run_id | 备注 |
|------|--------|------|
| 3view train parent | `260723-162838` | `eccv-3view-geom-resume__train` |
| 3view GT test（旧） | `260725-002218` | gen≈35.3%, summary≈0.046；P0 对照基线 |
| P0 note 模板 | — | `P0 postprocess+split; parent=260723-162838; compare vs 260725-002218 GT test` |

离线后处理（不重训）：

```bash
PYTHONPATH=core/src python scripts/postprocess_step_batch.py \
  --pred-dir <旧 pred STEP 目录> --out-dir <对比输出> --analytic 1
```

## 自由曲面光顺

OCC fairing API 环境差异大：当前 **skip**，仅做解析替换 + `ShapeFix_Shape` + sewing（默认 tol 略放宽至 ≥0.005）。

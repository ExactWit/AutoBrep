# ECCV 升级路线（P0 → P1 → MM Stage A/B/C）

> **文档源：`main`。** 实现 tip 见下表；改文档请只改 `main` 的 `docs/`。  
> TechDraw：[eccv_stage_reports/TECHDRAW_VIEW_SPLIT.md](./eccv_stage_reports/TECHDRAW_VIEW_SPLIT.md)（**L-layout 硬切**）  
> 条件多模态主线：[eccv_stage_reports/COND_MM_ROADMAP.md](./eccv_stage_reports/COND_MM_ROADMAP.md)

## 分支图

```
eccv-3view-geom (baseline)
  └─ eccv-p0  (tag: autobrep-eccv-p0)  ← L-layout TechDraw + fast metrics + analytic STEP
       ├─ feat/p1-prim-encoder-prefix   (P1-A soft prefix · 已失败，勿作主路径)
       ├─ feat/p1-decoder-cross-attn    (P1-B 代码源 · 注入方式复用)
       ├─ eccv-p2                       (aux / FSQ 冲榜开关)
       └─ feat/cond-mm-stage-a          → entry eccv-3view-mm-a  ★ 当前主线
            └─ (过门禁后) feat/cond-mm-stage-b / -c
```

## Registry entries

| entry_id | git_ref | 说明 |
|----------|---------|------|
| `eccv-3view-geom` | `eccv-3view-geom` | 3view + KV fix |
| `eccv-3view-p0` | `autobrep-eccv-p0` / `eccv-p0` | L-layout TechDraw + metrics + analytic post |
| `eccv-3view-p1a` | `feat/p1-prim-encoder-prefix` | soft prefix（失败归档） |
| `eccv-3view-p1b` | `feat/p1-decoder-cross-attn` | decoder cross-attn 代码 |
| `eccv-3view-p2` | `eccv-p2` | aux surf-type / FSQ 开关 |
| **`eccv-3view-mm-a`** | **`feat/cond-mm-stage-a`** | **MM Stage A：冻 AR+FSQ；encoder+xattn+surf head；cond_cache_v2** |

## 开关

| 开关 | 默认 | 说明 |
|------|------|------|
| Level-1 fast metrics | on | `val/fast/*` |
| `--postprocess-analytic` | 1 | STEP 解析替换 + ShapeFix |
| TechDraw 划分 | **L-layout 硬切** | 见 TECHDRAW_VIEW_SPLIT；递归 XY-Cut 仅 fallback |
| `--cond-cache-root` | 空则在线算 | `cond_cache_v2` 加速 |
| `--enable-aux-surf-type` | Stage A on | 曲面类型 CE |
| MM decoder xattn | entry `eccv-3view-mm-a` | 主注入路径（非 soft-prefix-only） |

## TechDraw 管线（锁定）

```
DXF (+ compatible SVG)
  → filter_and_merge
  → merge_dxfir（不相容 SVG 丢弃）
  → split_into_views：L-layout 双路径硬切（xy / yx）
  → (x_cut, y_cut) 象限硬归属
  → assign_loop_groups → tensorize
```

## 评测对照 run_id

| 角色 | run_id | 备注 |
|------|--------|------|
| 3view train parent | `260723-162838` | MM Stage A 加载并冻结 |
| 3view GT test（旧） | `260725-002218` | gen≈35.3%, summary≈0.046 |
| R1 best+analytic | `260728-201614` | gen≈50%, summary≈0.042 |
| MM Stage A | 见 [MMA_latest.md](./eccv_stage_reports/MMA_latest.md) | |

## 自由曲面光顺

OCC fairing API 环境差异大：当前 **skip**，仅做解析替换 + `ShapeFix_Shape` + sewing（默认 tol 略放宽至 ≥0.005）。

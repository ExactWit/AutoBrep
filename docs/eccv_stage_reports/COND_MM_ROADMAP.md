# 条件多模态路线（COND MM · Stage A/B/C）

> **文档源：`main`。** 实现 tip：`feat/cond-mm-stage-a`（entry `eccv-3view-mm-a`）。  
> 一句话：保留 AutoBrep 作 B-Rep 先验；三视图 DXF + 透明/半透明 render → encoder + **decoder cross-attn**；解析曲面显式 type/param 头逐步拆出。

## 与旧 P1 的关系

| 旧实验 | 结论 | 对本路线 |
|--------|------|----------|
| P1-A soft prefix | full gen 23%，失败 | **不作** Stage A 主路径 |
| P1-B cross-attn 代码 | tip 已有，未过门禁 | **移植**为注入方式 |
| TechDraw L-layout | `000008` 等已修 | cache / 训练默认 |

对照锚点：R1 `260728-201614`（gen≈50%，summary≈0.042）；旧 GT `260725-002218`（gen≈35%，summary≈0.046）。

## 系统形态

```text
三视图 DXF (L-layout) + HLG / translucent / transparent renders
            ↓
    MultiModal Encoder  (Stage A 可训)
            ↓
   Decoder Cross-Attn   (Stage A 可训)
            ↓
   AutoBrep Decoder + FSQ  (Stage A 冻结)
            ↓
   Surface-type head (+ 后续 param head)
            ↓
        OCC / STEP (+ analytic post)
```

## Stage 门禁

| Stage | tip / entry | 训什么 | 冻什么 | 硬门禁 |
|-------|-------------|--------|--------|--------|
| **A** | `feat/cond-mm-stage-a` · `eccv-3view-mm-a` | encoder + xattn + surf-type head | AR + FSQ | GT test best+analytic：gen≥50% 且 Δsummary≥−0.005 vs 0.046 |
| **B** | `feat/cond-mm-stage-b`（过 A 后开） | + decoder 后 2–4 层 | FSQ 仍冻 | gen 不降、summary ≥ A |
| **C** | `feat/cond-mm-stage-c` | 拓扑约束 + analytic param + OCC 回退 | — | 冲榜 / P2 叙事 |

早停：**跟 mid official gen_success**（兼看 summary）；禁止只跟 val_loss。

## cond_cache_v2 字段约定

根目录：`/data/hdd/datasets/eccv2026ws-cad-data/processed/cond_cache_v2/{train,val,test}/{id}.pt`

| key | 含义 |
|-----|------|
| `images` | `(3,3,H,W)` float01；顺序锁定：`hlg` / `hlg_translucent` / `transparent_shaded_edges` |
| `prim_types` / `prim_linetypes` / `prim_geom` / `prim_mask` | 3-view TechDraw（L-layout 后 tensorize） |
| `surf_type_ids` | `(max_faces,)` int64；plane/cylinder/cone/sphere/bspline/+pad |
| `n_faces` / `face_mask` | 面数与有效 mask |
| `adjacency` | 可选，拓扑辅助 |
| `meta` | `cache_version=2`，`split_algo=l_layout`，路径指纹 |

构建：`scripts/build_eccv_cond_cache_v2.py`（可 resume；写 `manifest.json`）。  
训练：`--cond-cache-root` 命中则跳过 DXF/SVG/读图。

**不要**覆盖旧 `processed/cache/`。

## Job / workflow

- Launcher entry：`eccv-3view-mm-a`
- 建议 workflow：`workflows/eccv_mm_stage_a.yaml`（smoke → train → GT test）
- 阶段报告：`MMA_latest.md` / `MMB_latest.md` / `MMC_latest.md`

## Surf type id

| id | 名称 |
|----|------|
| 0 | plane |
| 1 | cylinder |
| 2 | cone |
| 3 | sphere |
| 4 | bspline |
| -1 | pad / invalid |

来源：`processed/brepir/{split}/{id}.json` → `faces[].type`。

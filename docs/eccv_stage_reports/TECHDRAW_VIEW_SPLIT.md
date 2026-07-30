# TechDraw 三视图划分（锁定方案 · 归档于 main）

> **状态：L-layout 硬切（2026-07-30 修订）**  
> **实现分支 / tip：** 训练侧同步在 `eccv-p0`（tag `autobrep-eccv-p0`）、`feat/p1-prim-encoder-prefix`（entry `eccv-3view-p1a`）。  
> **文档唯一源：** 本文件在 `main` 的 `docs/`；其它分支只应引用，勿再分叉改文档。

## 问题定义（之前错在哪）

### 旧方案 1：KMeans

```
图元中心点 → 投影直方图谷底粗切 → k-means 细化 → 按质心贴 TL/BL/TR
```

典型失败（如 `train/000001`）：k-means 跨 gutter 吸图元。

### 旧方案 2：递归 XY-Cut（仍不够）

递归「每次切当前最好的缝」会在侧视内部再切一刀，把主+俯粘在一起。  
`train/000008`（DXF-only）：views 变成 `[35,9,9]`，主视顶端横线（y≈108）与俯视同框。

根因：**三视图是带对齐约束的 L 版面，不是任意二叉树分割。**

版面先验（本数据集常见）：

| 约束 | 含义 |
|------|------|
| 主 ↔ 俯 | **同宽**（同一 x 带） |
| 俯 ↔ 侧 | **同高**（同一 y 带） |
| 三块之间 | 大间距 gutter，bbox **不应重叠** |

## 锁定方案（如何修）

```
Primitive Set
  → 布局对象（bbox / center / length，噪声降权）
  → L-layout 双路径硬切（见下）→ 3 个 view regions
  → 按 (x_cut, y_cut) 象限硬归属（禁止软 bbox 偷边界线）
  → assign_loop_groups → tensorize
```

### 双路径（取对齐分更高者）

1. **xy（先主/侧栏，剩余俯视）**  
   全局最佳 **x gutter** → 左栏 vs 侧视；再在左栏上切 **y gutter** → 主 / 俯。  
2. **yx（先俯+侧行，剩余主视）**  
   全局最佳 **y gutter** → 主视 vs 俯+侧行；再在该行上切 **x gutter** → 俯 / 侧。  
   （`000008` 走此路径：俯与侧同高。）

评分：gutter 宽度 + **同宽/同高对齐** + 三块规模平衡。  
递归 XY-Cut 仅作 fallback。

`000008`：DXF-only 由错误 `[35,9,9]` → `[15,20,18]`，主视顶边 y=108 留在主视；merged `[11,20,13]`。

## 实现位置（main）

| 模块 | 路径 |
|------|------|
| 划分 | `core/src/autobrep/data/techdraw_dxf/split_views.py`（`_l_layout_plan`） |
| 训练入口 | `load_techdraw_geometry` → `core/src/autobrep/data/eccv_data.py` |
| 单测 | `tests/test_techdraw_split.py`（含「勿切侧视」回归） |
| 可视化脚本 | `scripts/viz_techdraw_splits.py` |

## 运行时说明

- **不必重做 `processed/`**：parquet 只存 `techdraw_*_path`，划分在 `load_techdraw_geometry` **在线**执行。  
- 换 tip 到含本方案的 commit 后直接训即可。

## 可视化产物

`/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/stage_gates/techdraw_viz/`  
（pipeline：`L-layout cuts→hard assign`）

## 相关分支索引

| 用途 | 分支 / tip |
|------|------------|
| 文档与代码归档 | **`main`（本文件）** |
| P0 训练 tip | `eccv-p0` / `autobrep-eccv-p0` |
| P1-A 图元 encoder | `feat/p1-prim-encoder-prefix` |
| 升级路线总览 | [`docs/eccv_upgrade_p0_p1_p2.md`](../eccv_upgrade_p0_p1_p2.md) |

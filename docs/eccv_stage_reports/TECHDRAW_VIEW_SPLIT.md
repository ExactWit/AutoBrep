# TechDraw 三视图划分（锁定方案 · 归档于 main）

> **状态：已敲定并归档到 `main`（2026-07-30）**  
> **实现分支 / tip：** 训练侧同步在 `eccv-p0`（tag `autobrep-eccv-p0`）、`feat/p1-prim-encoder-prefix`（entry `eccv-3view-p1a`）。  
> **文档唯一源：** 本文件在 `main` 的 `docs/`；其它分支只应引用，勿再分叉改文档。

## 问题定义（之前错在哪）

旧方案本质是：

```
图元中心点 → 投影直方图谷底粗切 → k-means 细化 → 按质心贴 TL/BL/TR
```

典型失败（如 `train/000001`）：

| 现象 | 原因 |
|------|------|
| 俯视下半（孔/弧，y≈98）被并进主视 | **k-means 跨空白带吸图元**：主视簇更密，边界图元被拉走 |
| 切缝落在错误位置（y≈78 而非真正 gutter≈132） | 谷底取「靠近 median」+ **投影一致性打分** 偏好了错误切 |
| SVG 并入后版面炸裂 | DXF 与 SVG **坐标框不相容**仍硬 merge |
| slot 名与观感拧巴 | `(-y,x)` 命名是纸面位置，不是语义「主/俯/侧」；viz 还 `invert_yaxis` |

根因一句话：**三视图分离是版面/区域问题，不是 primitive feature 上的自然簇。**  
对图元做 `KMeans(3)` 会被长线、跨区线、尺寸噪声干扰。

## 锁定方案（如何修）

```
Primitive Set
  → 布局对象（bbox / center / length，噪声降权）
  → Recursive XY-Cut 找出 3 个 view regions（投影空白 gutter，硬切）
  → 按 bbox-overlap（辅 center-in-box）把每个图元归属到 region
  → assign_loop_groups → tensorize
```

相对旧方案的关键差异：

1. **先区域、后归属** — 禁止主路径对图元做 3-means  
2. **硬 gutter** — 空白带两侧不再跨缝 refine  
3. **评分以 gutter 宽度为主** — 不再被假的投影一致性带偏  
4. **不相容 SVG 丢弃** — `merge_dxfir` 用 bbox IoU/尺度过滤像素系污染  
5. **短线/标注层降权** — 避免填平投影直方图  

`000001` 验证：旧 `[17,13,9]` 孔进主视 → 新 `[13,13,13]` 三块干净分离。

## 实现位置（main）

| 模块 | 路径 |
|------|------|
| 划分 | `core/src/autobrep/data/techdraw_dxf/split_views.py` |
| 训练入口 | `load_techdraw_geometry` → `core/src/autobrep/data/eccv_data.py` |
| 单测 | `tests/test_techdraw_split.py` |
| 可视化脚本 | `scripts/viz_techdraw_splits.py` |

## 运行时说明

- **不必重做 `processed/`**：parquet 只存 `techdraw_*_path`，划分在 `load_techdraw_geometry` **在线**执行。  
- 换 tip 到含本方案的 commit 后直接训即可。

## 可视化产物

`/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/stage_gates/techdraw_viz/`  
（pipeline 标记：`XY-Cut regions→bbox-assign`）

## 相关分支索引

| 用途 | 分支 / tip |
|------|------------|
| 文档与代码归档 | **`main`（本文件）** |
| P0 训练 tip | `eccv-p0` / `autobrep-eccv-p0` |
| P1-A 图元 encoder | `feat/p1-prim-encoder-prefix` |
| 升级路线总览 | [`docs/eccv_upgrade_p0_p1_p2.md`](../eccv_upgrade_p0_p1_p2.md) |

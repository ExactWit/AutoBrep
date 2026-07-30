# TechDraw 三视图划分（锁定方案）

> 状态：**已敲定**（2026-07-30）  
> 主路径：`Primitive Set → 3 View Regions (XY-Cut) → Assignment`  
> 禁止主路径：`Primitive Set → KMeans(3)`

## 流程

1. **解析** DXF（+ 坐标相容的 SVG）
2. **预处理** `filter_and_merge`：短线/标注层降权，共线合并
3. **区域检测** Recursive XY-Cut：对布局对象（bbox/center/加权长度）做 x/y 投影，找空白 gutter，递归切到 3 个 region box
4. **归属** 每个图元按与 region 的 bbox-overlap（辅以 center-in-box）分配；跨区取重叠最大
5. **后处理** `assign_loop_groups` → `tensorize_dxf_views`

## 代码

| 模块 | 路径 |
|------|------|
| 划分 | `core/src/autobrep/data/techdraw_dxf/split_views.py` |
| 训练入口 | `load_techdraw_geometry` in `eccv_data.py` |
| 单测 | `tests/test_techdraw_split.py` |
| 可视化 | `scripts/viz_techdraw_splits.py` |

## 可视化产物

`/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/stage_gates/techdraw_viz/`

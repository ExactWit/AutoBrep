# Classical Orthographic → B-rep Baseline (Wesley–Markowsky)

非深度学习对照基线：从三视图 TechDraw（DXF/SVG）自底向上重建 B-rep / STEP。  
代码入口：`core/src/autobrep/classical/orthographic_wm.py`；评测：`scripts/eval_classical_orthographic.py`。  
exp_launcher entry：`eccv-classical-wm`（`classical_baseline=1`：host stub → official test）。

## 1. 问题与形式化框架

工业 CAD 中「三视图线元 → B-rep」的经典路线是 **B-rep oriented bottom-up / wireframe-based**。  
Wesley & Markowsky (1981) *Fleshing Out Projections* 把映射写成五步复合：

\[
O^\* = f_{SL}\bigl(f_{BL}\bigl(f_{FA}\bigl(f_{ED}\bigl(f_{VR}(P_s)\bigr)\bigr)\bigr)\bigr)
\]

| 映射 | 输入 | 输出 | 作用 |
|------|------|------|------|
| \(f_{VR}\) | 三视图 2D 顶点 | 3D 候选顶点 | 「长对正、高平齐、宽相等」反投影配对 |
| \(f_{ED}\) | 3D 顶点 | 3D 候选边 | 投影回三视图验证，删 ghost 边 |
| \(f_{FA}\) | 3D 边 | 候选面 | 极小环（左邻边 / 最大转角） |
| \(f_{BL}\) | 候选面 | 候选块 | 切割边分割 → 虚块组合 |
| \(f_{SL}\) | 候选块 | 有效实体 | 决策求解 + 反投影验证 |

核心思想：投影重合 / 积聚会产生大量「投影看起来合法但三维不存在」的假元（ghost），每一步都要检测并剔除。

本仓库实现是该框架的 **工程化子集**：完整实现 \(f_{VR}/f_{ED}/f_{FA}\)，用 OCC sewing + 反投影打分近似 \(f_{BL}/f_{SL}\)，并附带轻量圆柱扩展。

## 2. 本仓库数据与坐标约定

### 2.1 输入

1. 从 parquet 取 `techdraw_dxf_path` / `techdraw_svg_path`。
2. `extract_dxf_primitives`（+ 可选 SVG）→ `filter_and_merge` → `merge_dxfir`。
3. `split_into_views`：L-layout / XY-Cut 把整张 TechDraw 切成最多 3 个视图 IR。
4. `label_orthographic_views`：按 sheet 质心标注 **Front / Top / Side**  
   （默认第三角画法启发：Top 在 Front 上方，Side 在 Front 右侧）。

### 2.2 视图局部坐标

每个视图减去自身 `bbox_min`，再按工程制图轴映射：

| 视图 | sheet 局部 \((u,v)\) | 世界坐标 |
|------|----------------------|----------|
| Front | \((u,v)\) | \((x,y)=(u,v)\) |
| Top | \((u,v)\) | \((x,z)=(u,v)\) |
| Side（默认） | \((u,v)\) | \((y,z)=(v,u)\)（\(u\)=深度/\(Z\)，\(v\)=高度/\(Y\)） |

Side 另有一变体：\((y,z)=(u,v)\)。重建时对 **Front↔Top 互换** 与 **Side 轴映射** 做笛卡尔积试探，按反投影分数选最优。

容差：`tol = max(1e-3, 1e-3 × span)`，`span` 为三视图 bbox 最大边长。

## 3. 实现细节（按模块）

### 3.1 \(f_{VR}\)：2D 顶点 → 3D 顶点

1. 从每个视图的线段端点（圆取四象限点）量化哈希聚类，得到 2D 顶点。
2. Front\((x,y)\) 与 Top\((x,z)\) 按 \(x\) 分桶配对；要求 Side 上也存在 \((y,z)\)。
3. 另有 Front+Side 回填路径（Top 稀疏时）。
4. 输出去重量化后的 `V3(x,y,z)` 列表。

### 3.2 \(f_{ED}\)：3D 顶点 → 3D 边

1. 顶点两两连边，按长度排序，上限约 8000。
2. 将候选边分别投影到 Front/Top/Side；投影线段须被对应视图的 2D 线元覆盖（共线 + 端点在容差内）。
3. **三视图均通过才保留**（ghost 边的主过滤器）。
4. 投影退化为点（边垂直于该投影面）时，两端点投影重合也视为通过。

### 3.3 \(f_{FA}\)：3D 边 → 候选面

1. 在线框邻接图上，对每条有向种子边 + 第三点确定平面法向。
2. **左邻边行走**（绕法向取最大左转角）闭合成环；平面性复核。
3. 按环长度排序，去掉被更小环顶点集真包含的非极小环；上限约 200 环。

### 3.4 \(f_{BL}/f_{SL}\) 的工程近似

完整虚块 / 切割边 / 组合爆炸求解未实现。当前路径：

1. 每个极小环 → OCC `MakePolygon` → `MakeFace`。
2. `BRepBuilderAPI_Sewing` → 尽量 `MakeSolid`。
3. 候选还包括：
   - **圆柱轻量扩展**：Front 圆 + Top/Side 高度线索 → `MakeCylinder`（可 fuse）；
   - **AABB fallback**：仅用 3D 顶点包围盒（保底出 STEP，分数通常很低）。
4. **反投影验证**：实体边投影回三视图，覆盖输入 2D 线段的比例作 `score`；多假设取最高分。
5. `occwl.io.save_step` / `STEPControl_Writer` 写出 `.step`。

### 3.5 评测与启动

- `scripts/classical_host_train.py`：写 sentinel ckpt（无训练），满足 launcher「test 需 parent」约束。
- `scripts/eval_classical_orthographic.py`：按 datasplit `test` 逐样本重建 → `prepare_gt_pred_dirs` → `min_eval/eval.py` → `metrics/test.json`。
- `run.sh`：`--classical-baseline 1` 时 train/test 走上述脚本（不加载 AR/FSQ）。

## 4. 与「教科书五映射」的差距

| 教科书步骤 | 本实现 | 说明 |
|------------|--------|------|
| \(f_{VR}\) | ✓ | 哈希分桶；依赖视图标注与轴约定 |
| \(f_{ED}\) | ✓ | 直线边为主；弧弦化近似 |
| \(f_{FA}\) | ✓ 部分 | 平面极小环；无完整二次曲面环 |
| \(f_{BL}\) 切割边 / 虚块 | ✗ 近似 | sewing 代替切割边与块枚举 |
| \(f_{SL}\) 决策求解 | ✗ 近似 | 反投影分数选优，非完备搜索 |
| 共轭直径 / 一般二次曲线边 | ✗ 极简 | 仅圆→圆柱启发式 |
| Moebius / 双向边面规则 | ✗ | 未强制每边两面反向 |

## 5. 局限性（实验观察）

1. **视图分割 / 角色标注错误**  
   L-layout 失败或 Front/Top/Side 标错 → \(f_{VR}\) 顶点为 0，整例失败（试跑中约 30–40% 样本）。

2. **轴约定与第一角 / 第三角**  
   Side \((u,v)\mapsto(y,z)\) 与图纸习惯强相关；虽有变体搜索，仍覆盖不全。

3. **假元与漏元并存**  
   容差过松：ghost 边/面进入 sewing；过紧：真边被滤掉。复杂件上环爆炸或空环。

4. **曲面体能力弱**  
   椭圆、一般二次曲线、自由曲面边界基本不支持；圆柱仅「圆+高度」启发式。

5. **sewing ≠ 拓扑完备 B-rep**  
   常得到 shell/compound；官方 Valid 可能为 1，但 Surface/Edge/Vertex/Topo F1 接近 0，Summary 极低（试跑 7 例 Valid=1.0、Summary≈0.002）。即「能成实体 ≠ 几何正确」。

6. **组合爆炸被砍掉换稳健性**  
   不做完整虚块枚举，病理多解 / 开槽 / 通孔等难正确消解。

7. **隐藏线 / 中心线**  
   linetype 已解析但未深度参与可见性推理。

8. **与 DL SOTA 不可比肩**  
   定位是 **可复现的传统方法对照**，不是挑战榜冲分方案。R1+retry4 等 DL 管线在 Summary / F1 上显著更高。

## 6. 文件索引

| 路径 | 角色 |
|------|------|
| `core/src/autobrep/classical/orthographic_wm.py` | 重建主算法 |
| `scripts/eval_classical_orthographic.py` | Official test |
| `scripts/classical_host_train.py` | Launcher host stub |
| `models.tsv` → `eccv-classical-wm` | Entry（feature 分支） |
| `run.sh` → `--classical-baseline` | 模式开关 |

## 7. 参考文献（框架）

- Wesley, M. A., & Markowsky, G. (1981). Fleshing out projections. *IBM Journal of Research and Development*.
- 后续三视图重建（含清华刘世霞等对曲面体 / 共轭直径的扩展）均建立在上述五映射骨架上；本实现仅覆盖其直线体主路径的可运行子集。

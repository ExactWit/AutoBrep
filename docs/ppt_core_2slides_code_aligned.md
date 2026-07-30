# AutoBrep 核心技术 PPT（对照代码重设计）

> 只重做 **两页核心技术**。风格：学术、示意清晰；主色深蓝灰 + 青灰强调。  
> 给做图 AI：按下面「版式 / 图形 / 文案 / 代码锚点」逐页生成；图标风格左右栏统一。

**代码锚点总览（页脚可写极小字，可不印）**

| 概念 | 代码位置 |
|------|----------|
| 面/边 UV 点阵 | parquet 列 `face_points_normalized` / `edge_points_normalized`；`ARDataModule` |
| 面/边 bbox | `face_bbox_world` / `edge_bbox_world`；`quantize_pos` |
| Surface/Edge FSQ-VAE | `models/vaes.py`：`SurfaceFSQVAE` / `EdgeFSQVAE` |
| FSQ 量化 | `models/fsq.py`：`FSQ` |
| Face 邻接矩阵 | `face_edge_incidence`；`nx.Graph` 建面邻接 |
| BFS 序列化 | `abc_data.py`：`cad_tokenization` / `convert2seq` |
| Face ID 引用边 | `convert2seq` 中 `prev_face_id + edge_bbox + edge_z` |
| AR 生成 | `AutoBrepModel.generate` ← `XTransformer` |
| Token→几何 | `decode` + FSQ `decode` + `convert_to_cad_data` |
| → STEP | `AutoBrepBuilder.rebuild_brep`（共享顶点 / `joint_optimize` / BSpline / Sewing） |

---

# 第 1 页｜B-Rep 如何变成可学习的离散表示

## 本页一句话
左边讲 **几何怎么编码**，右边讲 **拓扑怎么编码**；二者最终都会落成 **离散 token**，供下一页的自回归模型书写。

## 版式（严格左右分栏）
- 顶栏：标题 + 一行总起  
- **左 48%**：几何编码流水线（从上到下）  
- 中间细竖分割线，线中写小字 `统一进词表`  
- **右 48%**：拓扑编码流水线（从上到下）  
- 底栏：一行「汇合公式」式示意（一个面在序列里长什么样）

---

## 顶栏文案
- **标题**：`表示层：几何 × 拓扑 → 同一套离散 Token`
- **总起**：`AutoBrep 不直接回归连续 CAD；先把面/边的形状与邻接关系编成可预测的整数序列。`

---

## 左栏｜几何特征如何被编码

### 栏标题
`几何编码 Geometric Encoding`

### 建议主图（竖向 4 步，每步一框 + 向下箭头）

**步骤 G1｜参数域采样点阵（连续）**  
画：一张曲面小片上的 UV 网格（面）+ 一条曲线上的折线点（边）  
标注：
- 面：`face_points_normalized`，形状示意 `(U×V×3)`  
- 边：`edge_points_normalized`，形状示意 `(U×3)`  
小字：`局部坐标系 / 归一化点阵（NCS），不含全局位姿`

**步骤 G2｜全局位姿：包围盒（连续→离散）**  
画：同一曲面外包一个 3D 线框盒子  
标注：`face_bbox_world / edge_bbox_world` = `minxyz ∥ maxxyz`（6 个数）  
箭头：`quantize_pos(bit=10)` → **6 个坐标 bin token**  
小字：`把 [-1,1] 量化到 1024 档；决定面/边在实体里的位置与尺寸`

**步骤 G3｜形状压缩：FSQ-VAE（连续→码本下标）**  
画两条并行小管道：

```
面 UV ── SurfaceFSQVAE.encode ── FSQ ── 4 个 surface code
边折线 ── EdgeFSQVAE.encode ──── FSQ ── 2 个 edge code
```

标注：
- `SurfaceFSQVAE` / `EdgeFSQVAE`（`vaes.py`）  
- 量化器：`FSQ`（有限标量量化，码本大小默认 1000 级）  
小字：`平面/柱面等类型不写文字，而体现在码所解码出的点阵形态`

**步骤 G4｜几何侧产物（给序列用）**  
三个小徽章并排：
1. `bbox tokens ×6`  
2. `surf FSQ ×4`（每个面）  
3. `edge FSQ ×2`（每条边）

### 左栏底部警示条（浅灰）
`训练时：AR 看到的几何槽位会被替换成真实 FSQ 码（copy_fsq_code）；推理时：模型直接生成这些码。`

---

## 右栏｜拓扑关系如何被编码

### 栏标题
`拓扑编码 Topology Encoding`

### 建议主图（竖向 4 步）

**步骤 T1｜面–边关联矩阵**  
画：一个小表格 / 热力示意 `F×E`  
标注：`face_edge_incidence`（布尔）  
约束小字（重要）：`流形边：每列求和 = 2（一条边恰连两个面）`

**步骤 T2｜面邻接图**  
画：若干圆节点（面）+ 连线（若两面共享边）  
标注：`networkx.Graph`：由 incidence 的每一列抽出两个面连边  
隐喻：`房间=面，门=共享边`

**步骤 T3｜遍历顺序：Ordered-BFS**  
画：从「bbox 字典序最小的起始面」出发，一层层向外扩展的同心圈  
标注：`cad_tokenization`：`xyz_order` 选起点 → BFS 得 `faces_sorted` + `levels`  
小字：`序列顺序 = 拓扑图上的广度优先，不是随意排列`

**步骤 T4｜用 Face ID「声明 / 引用」写邻接（本栏重点）**  
左右两个小面板：

**声明（新面出场）**
```
BOF → FaceID=k →（几何 tokens）→ …
```
图示：给面 k 挂名牌

**引用（表达共边）**
```
… → FaceID=j（旧面）→ 边 bbox×6 → 边 FSQ×2 → …
```
图示：从当前面画箭头指回旧面 j，中间画出共享边  
代码对应：`convert2seq` 里对 `faces_sorted[:index]` 找 `adj sum==2` 的边，再按边 bbox 排序写出

小字金句：  
`Face ID embedding 学的是可绑定的身份指针，不是外观；邻接靠「引用旧面 + 吐出边」表达，没有单独的邻接矩阵 token。`

---

## 底栏｜左右汇合：一个面在序列里的「完整短语」

画成一条横向胶带 / 乐高条（务必具体）：

```
[BOF] [FaceID] [bbox×6] [surfFSQ×4]   [旧FaceID][edgeBBox×6][edgeFSQ×2] × N   [EOF]
 └─ 拓扑：我是谁 ─┘ └────── 几何：我在哪、长啥样 ──────┘ └──── 拓扑：我和谁共哪些边 ────┘
```

两侧小注：
- 左：`几何来自左栏 G2+G3`  
- 右：`引用结构来自右栏 T4`  
- 层间还有：`BOL / EOL`（一层 BFS）、整块包在 `BOC … EOC`

### 页脚过渡
`下一页：模型如何按此语法自回归写完整句，以及如何组装回 STEP。`

---

# 第 2 页｜序列化生成 → 组装为 STEP

## 本页一句话
上半：**怎么生成这条序列**；下半：**序列如何变回可缝合的实体并导出 STEP**（对照推理代码路径）。

## 版式
- 顶栏标题  
- **上半 45%**：生成（从 prompt 到 token 流）——横向时间线  
- **中部分隔条**：`decode` 解析出三张表  
- **下半 45%**：组装流水线到 STEP——横向 5 步  
- 右下角小窗：失败点（重建失败 ≠ 没采样到 token）

---

## 顶栏文案
- **标题**：`生成层：自回归写序列 → 解码几何/拓扑 → OCC 重建 STEP`
- **总起**：`表示定语法；生成是按语法抽样；STEP 是把抽样结果「装配」回 B-Rep。`

---

## 上半｜序列化生成（Serialization & Generation）

### 区块标题
`A. 自回归生成（AutoBrepModel.generate）`

### 主图：时间线（左→右）

**① Prompt（固定前缀）**  
胶囊：`BOS | BOM | GEN_* | EOM | BOC`  
小字：`无条件几乎只给复杂度档位；对应 sample_tokens`

**② Transformer 逐步写**  
画一个解码器方块：`XTransformer / AutoregressiveWrapper`  
循环箭头标注：  
`logits → top-p 截断 → ÷temperature → multinomial 抽 1 个 token → 接到序列末尾`  
直到：`EOC / EOS` 或达到 `max_seq`

**③ 写出的内容形态**  
画一层层：
```
BOL
  面0（仅声明+几何，无边）
EOL
BOL
  面1：声明+几何 + 引用面0的边…
  面2：…
EOL
…
EOC
```
旁注：`顺序遵循训练时的 BFS 语法；推理时模型「学着」吐出同构结构`

### 关键对比小条（上半右上角，浅底）
| 训练 | 推理 |
|------|------|
| 真值序列 + 冻结 VAE 填 FSQ 码 | 无真值，抽样生成 FSQ 码 |
| `common_step` + CE loss | `generate` + top-p |

---

## 中部分隔｜Token 解析成三张「表」

标题小条：`B. decode_tokens / decode / convert_to_cad_data`

画三个并排表卡（从序列解析出来）：

1. **面表**：每个面 → `bbox(6)` + `surf code(4)` → FSQ decode → `face UV (NCS)`  
2. **边表**：每条边 → `bbox(6)` + `edge code(2)` → FSQ decode → `edge polyline (NCS)`  
3. **邻接**：每次「引用旧面」→ 在 `face_edge_adj[F,E]` 里点亮「当前面与旧面共享该边」

再画一个小箭头：`NCS × bbox中心/尺寸 → WCS 点阵`（`compute_bbox_center_and_size` / `convert_to_cad_data`）  
产出结构名：`BrepGenCAD`

---

## 下半｜组装为 STEP（Assembly）

### 区块标题
`C. AutoBrepBuilder.rebuild_brep → save_step`

### 主图：5 步装配线（必须画具体，不要抽象云朵）

**步 1｜共享顶点**  
图标：多条边端点聚成少数红点  
文案：`detect_shared_vertex`（距离阈值合并该重合的角点）  
代码：`AutoBrepPostProcess.compute_shared_vertex`

**步 2｜联合优化**  
图标：面网格与边折线被弹簧拉齐  
文案：`joint_optimize`：让面边几何贴合共享顶点，减少缝隙  
（可旁注 `eval_mode` 可跳过）

**步 3｜曲线/曲面拟合**  
图标：点列 → 光滑 B 样条  
文案：  
- 面：`GeomAPI_PointsToBSplineSurface`（`rebuild_surfaces`）  
- 边：`GeomAPI_PointsToBSpline`（`rebuild_curves`，精度失败会回退放宽容差）

**步 4｜Wire 裁剪出面 + 缝合**  
图标：曲面上画外环/内环（通孔场景）→ 多个面缝成壳/实体  
文案：  
- `_cut_surface_by_wire`：边连成 Wire，裁剪曲面（`linker` 定环顺序）  
- `ShapeFix_*` 修线修面、补 pcurve  
- `BRepBuilderAPI_Sewing` → `MakeSolid`（`rebuild_solid`）

**步 5｜导出**  
图标：文件徽章 `.step`  
文案：`occwl.io.save_step` / 我们的 `infer_pipeline` 写入产品目录

### 右下警告窗（橙边）
标题：`失败发生在哪？`  
正文：  
- Token 可能已生成（debug 里仍有点阵图）  
- `rebuild_brep` 返回 `None` 或异常 → **没有 STEP**  
- 常见：顶点合并失败、BSpline/Sewing 失败、非水密  

一句话：`生成成功 ≠ 重建成功`

---

## 本页底部金句（居中一条）
`表示规定「句子语法」；生成负责「按语法抽样」；重建负责「把句子焊回 CAD」。条件生成要改的是抽样前的条件输入，不是重写 OCC 焊枪。`

---

# 给做图 AI 的执行清单

## 第 1 页必须出现的视觉元素
- [ ] 左：UV 网格 → bbox 盒子 → FSQ 双管道 → 三类几何 token 徽章  
- [ ] 右：incidence 矩阵 → 面邻接图 → BFS 层圈 → 声明/引用对照  
- [ ] 底：一条完整的「面短语」胶带，标出几何段 vs 拓扑段  
- [ ] 不要在第 1 页画 Transformer 大模型（留给第 2 页）

## 第 2 页必须出现的视觉元素
- [ ] Prompt 五元组  
- [ ] top-p / temperature / multinomial 三步抽样标注  
- [ ] decode 后的三表：面 / 边 / 邻接  
- [ ] 重建五步：共享顶点 → 优化 → BSpline → Wire+Sewing → STEP  
- [ ] 失败窗：debug 有、STEP 无

## 禁止
- 把 Face ID 画成「圆柱/平面分类标签」  
- 把无条件 prompt 画成「输入一张零件图」  
- 第 1 页堆词表 3245 细节（可备注「坐标 bin + 两套码本 + Face ID + 控制符」一笔带过）

---

# 演讲者备注（口述 90 秒/页）

**第 1 页**  
先指左：「形状是 UV 点阵，经 FSQ 变成几个整数码；位置是 bbox。」  
再指右：「谁挨着谁不靠矩阵进网络，而靠 Face ID 指回旧面并吐出边。」  
最后指底胶带：「一个面 = 名字 + 几何 + 若干条引用边。」

**第 2 页**  
先指上：「从 5 个控制 token 开始往下抽字，抽的是符合这种语法的长句。」  
再指中：「解码器读懂句子，恢复面边点阵和邻接。」  
再指下：「OCC 把点阵拟合成面边并缝合；这里才出 STEP，也最容易失败。」  
收束：「要做点云条件，是在『抽字之前』加条件，表示与重建可复用。」

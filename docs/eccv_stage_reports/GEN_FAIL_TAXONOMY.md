# Gen 失败归因（生成率低：为什么？失败在哪？）

> **文档源：`main`/`docs/eccv_stage_reports/`**  
> **所指 tip / 产物：** `eccv-p0` · R1 run `260728-201614`；对照 `260725-002218`、P1-A `260729-192232` / full val `260728-212715`  
> **状态（2026-07-30）：** L1（decode vs rebuild）已有硬统计并锁定叙事；L2（拓扑引用 / 拟合 / OCC 数值…）**尚未打标**，见文末行动项。

## 评委式问题（我们要回答的）

若只说「gen≈30–50%，因为 AutoBrep 不好」——说服力不够。

应回答成：

> 我们在 N 个失败样本上统计了失败环节；主因是 X%（因此提 Y），次因是 Z%（因此提 W）。

本文归档的是：**目前能量化到哪一层、数字是什么、还缺什么。**

## 失败发生在哪条链上

```text
AR 抽样 token
  → decode / convert_to_cad   ← decode_failed（序列无法解析为面边表）
  → joint_optimize + OCC 重建  ← rebuild_failed（有 CAD 数据结构，但缝不成 STEP）
  → 写出 .step                 ← 计入 gen_success
  → 官方 F1 / summary          ← 仅对「已生成 STEP」评几何质量
```

要点：**官方 summary 只看成功写出的 STEP**；生成率低首先是「根本没有 STEP」，不是 F1 算得差。  
R1 上 `official.valid_ratio=1.0`（174/174 写出的 STEP 都能进官方评测）→ **当前瓶颈在 gen 通路，不在「写出后非法」**。

## L1 硬统计（已有，可汇报）

来源：各 run 的 `metrics.json` → `gen_log[].error`（仅两档：`decode_failed` / `rebuild_failed`）。

| 实验 | n | ok | gen | fail 中 rebuild | fail 中 decode |
|------|--:|---:|----:|----------------:|---------------:|
| 旧 3view GT test（`260725-002218`，last） | 348 | 123 | **35.3%** | 158 (**70.2%**) | 67 (**29.8%**) |
| **R1** best+analytic（`260728-201614`） | 348 | 174 | **50.0%** | 133 (**76.4%**) | 41 (**23.6%**) |
| P1-A best GT test（`260729-192232`） | 348 | 134 | **38.5%** | 127 (**59.3%**) | 87 (**40.7%**) |
| P1-A full val last（`260728-212715` ep50） | 694 | 162 | **23.3%** | 321 (**60.3%**) | 211 (**39.7%**) |

### R1 细表（会议可用）

- 源：`…/260728-201614/R1-ckpt-host__test/metrics/official_test/metrics.json`
- n=348，ok=174，fail=174，gen_success=**0.50**

| error | count | share_of_fail |
|-------|------:|-------------:|
| `rebuild_failed` | 133 | **76.4%** |
| `decode_failed` | 41 | **23.6%** |

### L1 结论（可对评委说的）

1. **主战场是重建，不是「完全不会写序列」。** 失败里约 **60–76%** 已 decode 成面/边几何，却在 OCC / 后处理缝合阶段丢掉。  
2. **decode 仍占 24–40%。** 非法 / 不可 reshape 的 token 序列是第二战场（引用错乱、缺 EOC、面边表不一致等都会落在这一桶——**当前未再细分**）。  
3. **抬 gen 优先于抬 summary。** 没有 STEP 就没有官方分；R0 已表明事后 analytic  alone 几乎不动 summary。  
4. P1-A 把 decode 占比抬高、gen 下降 → 条件编码改动可能伤了序列合法性；与「只优化几何条件」的假设不完全一致，需 L2 才能说清。

## L2 目标 taxonomy（研究价值所在，**尚未打标**）

评委期待的细表（示例结构，**非实测百分比**）：

| L2 桶（建议名） | 直觉含义 | 大致落在代码哪段 | 与 L1 关系 |
|-----------------|----------|------------------|------------|
| **Sequence Syntax** | 控制符/长度/reshape 失败，构不成面边表 | `decode_tokens` / `transformer.decode` | ⊆ `decode_failed` |
| **Reference / Topology** | Face ID 引用越界、邻接不一致、共享边对不上 | decode 解析 `prev_face`；或 rebuild 前邻接 | 多在 decode；部分「看起来 decode 成功」的拓扑烂可能拖到 rebuild |
| **Geometry Fit** | BSpline 曲面/曲线拟合失败或精度回退后仍坏 | `rebuild_surfaces` / `rebuild_curves` | ⊆ `rebuild_failed` |
| **Wire / Face Fix** | Wire 不闭合、pcurve、`ShapeFix_*` 失败 | `form_wires` / `fix_face` / `fix_face_edge` | ⊆ `rebuild_failed` |
| **OCC Sewing / Solid** | Sewing 不成 Shell、`MakeSolid` 失败、数值缝宽 | `rebuild_solid` / `BRepBuilderAPI_Sewing` | ⊆ `rebuild_failed` |
| **Postprocess** | analytic 替换等后处理异常（当前多回退保留原 shape） | `postprocess_shape` | 一般**不**计入 gen fail |

### 为什么现在还不能报「52% Reference / 31% Spline…」

当前 `_tokens_to_step`（tip `eccv-p0`：`eccv_val_eval.py`）在失败时只写：

- `cad_list` 空 → `"decode_failed"`
- `reconstruct_compound` 为 `None` → `"rebuild_failed"`

**没有**把异常类型、OCC 阶段名、引用校验结果写入 `gen_log`。  
因此任何细百分比若现在报出，都属于猜测，不能进正式结论。

### 旁证（非 L2，但可支撑叙事）

P1-A 训练 Level-1 fast（teacher-forced val，非自由生成）：

- `topo_acc` ≈ **0.99**，`geom_acc` ≈ **0.94**，`topo_compliance` ≈ **0.90**  
→ 在「看过真前缀」时拓扑 token 很好学；**自由生成**仍大量 decode/rebuild 失败，说明瓶颈更像是 **开环抽样累积误差 + 重建鲁棒性**，而不是「拓扑类别完全学不会」。

## 与质量指标的关系（避免叙事混淆）

| 现象 | 含义 |
|------|------|
| gen↑、summary↓（如 R1 vs 旧基线） | 多出来的 STEP 多为「勉强缝上」的差几何；先扩覆盖再提质量 |
| mid-24 summary↑ 伴随 gen↓（P1-A） | 幸存者偏差：活下来的少而「好看」 |
| R0 Δsummary≈0 | 事后解析拟合几乎不解 gen；曲面类型问题要进模型或 L2 拟合阶段 |

## 下一步（把 L2 做成可汇报数字）

1. **打标**：在 `_tokens_to_step` / `rebuild_brep` 各阶段 `try/except`，把 `error` 写成 `stage:exc_type`（或枚举上表 L2 桶），重跑 R1 同 ckpt 的 348 test（或先 100 子集）。  
2. **引用专项**：对 `decode_failed` 样本解析 Face ID 是否越界 / 重复声明 / 引用未生成面 → 得到 Reference 占比。  
3. **Oracle 对照（可选）**：GT token → 只跑 rebuild，测「表示+OCC」天花板，与自由生成对比。  
4. **再写一版本文「L2 实测表」**；未测满前勿用假百分比对外。

## 一句话对外版

> 在 R1（348 test，gen=50%）上，失败样本中 **76% 是 OCC/后处理重建失败，24% 是 token 解码失败**；官方对已生成 STEP 的 valid_ratio=1。  
> 因此优先攻 **重建鲁棒性 + 序列合法性**，而不是空泛说「模型不好」。细到「引用 / 拟合 / 缝合」的占比需要下一轮带阶段标签的失败日志。

## 相关索引

| 资源 | 路径 |
|------|------|
| 本文件（文档源） | `docs/eccv_stage_reports/GEN_FAIL_TAXONOMY.md` |
| 早期一页摘要 | `/data/hdd/.../stage_gates/R1_gen_fail_taxonomy.md` |
| 汇总脚本（tip） | `eccv-p0`:`scripts/eccv_gen_fail_taxonomy.py` |
| 打标位点 | `eccv_val_eval._tokens_to_step` · `AutoBrepBuilder.rebuild_*` |
| TechDraw（条件输入，正交） | [TECHDRAW_VIEW_SPLIT.md](./TECHDRAW_VIEW_SPLIT.md) |
| 阶段总索引 | [README.md](./README.md) |

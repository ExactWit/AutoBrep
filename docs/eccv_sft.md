# ECCV × AutoBrep 监督微调（3 渲染图 + 结构化 TechDraw）

面向 ECCV 2026 CAD 挑战：输入 **3D 渲染图 + TechDraw**，输出 **normalized STEP**。冻结 AutoBrep AR/FSQ，只训条件编码器。

**exp_launcher**

| 项 | 值 |
|----|-----|
| 模型条目 | `models.tsv` → `eccv`（固定 medium）/ `eccv-cx-cond`（条件推断复杂度，`git_ref=eccv-cx-from-cond`） |
| 数据集 ID | `eccv2026ws-cad-data`（与 `DATA_DIR/registry/datasets.tsv` 一致） |
| task | `gen`（亦声明 `cad`） |
| processed | `{DATA_DIR}/eccv2026ws-cad-data/processed/autobrep` |

## 复杂度 token

| 阶段 | 行为 |
|------|------|
| **训练** | parquet **不存**离散序列；dataloader 按 GT 面数写 BOM…EOM（`<25` easy / `<50` mid / else hard；`meta_ratio` 下约 20% uncond） |
| **推理 `eccv`** | 固定 `--complexity medium`（或 easy/hard/random） |
| **推理 `eccv-cx-cond`** | `--complexity from_condition`：view+DXF → prepend 后，在 `BOS,BOM` 上做一步 AR，在 `{easy,mid,hard}` 上 argmax，再拼完整 prompt 生成 |

训练时 prepend 与复杂度 token 同属一条 CE 序列，因此 `P(C\|cond)` 被联合学习；`from_condition` 即在推理时取该条件分布。

## 验证节奏（`eccv-base` / `eccv-3view-geom`）

| 频率 | 内容 |
|------|------|
| **每 epoch** | CE `val_loss`（无 STEP） |
| **25% / 50% / 75%** | 固定小子集 STEP（默认 24）+ `min_eval`；AR `gen_batch` 默认 4 吃显存 |
| **100%（末 epoch）** | 全量 official val STEP（~694） |

CLI：`--official-val-samples-mid 24`、`--official-val-gen-batch 4`、`--official-val-samples -1`。

## 条件模态（重要）

| 输入 | 处理方式 |
|------|----------|
| 3 张渲染 PNG（transparent / hlg / hlg_translucent） | **图像**：ImageNet-pretrained **ResNet-18**（三种着色风格，非三投影） |
| TechDraw DXF + SVG | **几何**：解析图元（LINE/ARC/… 与 SVG path），**按纸面空间拆成 3 个投影视图**，各视图独立 Set-Encoder；**不光栅化** |

### 方案对照

| 条目 | 分支 / tag | TechDraw |
|------|------------|----------|
| **`eccv-base`**（autobrep_on_eccv base） | tag `autobrep-eccv-base` / `eccv-cx-from-cond@d8023b5` | 整张 DXF 扁平底池 → 1 token |
| **`eccv-3view-geom`** | `eccv-3view-geom` | DXF+SVG 几何图元 → 3 视图分别编码 → 3 tokens |

## STEP → parquet（对齐 AutoBrep 采样）

与 AutoBrep / ABC 约定一致：

1. 面：参数域均匀 **32×32**（`BRepAdaptor_Surface(face, True)` 面内 UV）
2. 边：参数域均匀 **32** 点
3. 实体 AABB 归一到约 `[-1,1]`，再按面/边 bbox 得到 NCS
4. 写盘前做 `sort_uv_grids` / `sort_u_grids`（与训练 `uv_invariant` 一致）
5. 流形检查：每边恰 2 面；跳过非流形 / 超 `max_face`

输出：`{data-dir}/processed/autobrep/{train,val,test}/*.parquet`

## 模型与损失

- `ViewConditionEncoder`：ResNet-18（3 视图）+ `TechDrawSetEncoder`（DXF）→ cross-attn latents → `prepend_embeds`
- 损失：AR token **CE**；prepend 不计入 CE
- 冻结：`cad_gpt` + FSQ
- 训练启动日志：`metrics/model_info.json`（模块 / 参数量 / FLOPs / 连接关系）

依赖：`torchvision`、`ezdxf`、`thop`（可 `pip install`）。

## TensorBoard 指标

| 指标 | 含义 |
|------|------|
| `train_loss` / `val_loss` | AR token CE（parquet train / **官方 datasplit val**） |
| `lr` | 学习率 |
| `val/fast/ppl` | Level-1：序列困惑度（无需 OCC） |
| `val/fast/token_acc` | Level-1：token Top-1 准确率 |
| `val/fast/geom_acc` | Level-1：几何/FSQ 码段准确率 |
| `val/fast/topo_acc` | Level-1：拓扑标记 token 准确率 |
| `val/fast/complexity_acc` | Level-1：复杂度 token 准确率 |
| `val/fast/topo_compliance` | Level-1：轻量拓扑合规率（规则校验，无 OCC） |
| `val/official_gen_success` | 官方 val 上 STEP 生成成功率 |
| `val/official_summary` | 挑战 `min_eval/eval.py` 综合分（含几何/拓扑 F1；非独立 CD 字段时以 summary 对齐赛题质量） |
| `val/official_valid_ratio` | 合法 B-Rep 比例 |
| `val/official_surface_f1` / `edge_f1` / `vertex_f1` / `topo_f1` | 几何 / 拓扑 F1 |

Level-1 每个 val epoch 还会写入 `metrics/fast_val_epochXXX.json`。  
Level-2/3（OCC STEP）仍由 `EccvOfficialValCallback`：mid 子集（默认 24，可配到 100–200）+ 末 epoch full。

每次官方评测会在 `metrics/official_val_stepXXXXXX/` 写下 `gt/`、`pred/`、`metrics.json`。  
CLI：`--official-val-samples 4`、`--official-val-every 1`、`--no-official-val`、`--eval-py ...`。

## Launcher / CLI

```bash
./run.sh preprocess --data-dir /data/hdd/datasets/eccv2026ws-cad-data --num-workers 4
# 小数据集默认按 epoch 重复遍历（非点云；条件=3 渲染图+DXF）
./run.sh train --dataset eccv2026ws-cad-data --data-dir /data/hdd/datasets/eccv2026ws-cad-data \
  --weight-folder /data/hdd/outputs/AutoBrep --exp-dir <exp> \
  --max-epochs 50 --official-val-samples 4 --official-val-every 1
./run.sh infer --view-conditioned 1 --checkpoint <exp>/checkpoints --sample-id 000029 \
  --data-dir /data/hdd/datasets/eccv2026ws-cad-data --exp-dir <exp> --output-dir <out>
```

`run.sh capabilities` 声明 `preprocess/train/infer` 与数据集 `eccv2026ws-cad-data`。  
exp_launcher 按 `git_ref` 探测 capabilities，**需提交 `eccv` 分支上的 `run.sh` / `models.tsv`** 后 UI 才能看到新接口。

详见 `configs/autobrep_eccv.yaml`。

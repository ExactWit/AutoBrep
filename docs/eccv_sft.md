# ECCV × AutoBrep 监督微调（3 渲染图 + 结构化 TechDraw）

面向 ECCV 2026 CAD 挑战：输入 **3D 渲染图 + TechDraw**，输出 **normalized STEP**。冻结 AutoBrep AR/FSQ，只训条件编码器。

**exp_launcher**

| 项 | 值 |
|----|-----|
| 模型条目 | `models.tsv` → `eccv`（`git_ref=eccv`） |
| 数据集 ID | `eccv2026ws-cad-data`（与 `DATA_DIR/registry/datasets.tsv` 一致） |
| task | `gen`（亦声明 `cad`） |
| processed | `{DATA_DIR}/eccv2026ws-cad-data/processed/autobrep` |

## 条件模态（重要）

| 输入 | 处理方式 |
|------|----------|
| 3 张渲染 PNG（transparent / hlg / hlg_translucent） | **图像**：ImageNet-pretrained **ResNet-18** |
| TechDraw（三视图工程图） | **结构化**：解析 **DXF** 图元（LINE/ARC/CIRCLE/…），Set-Transformer 编码；**不做光栅化** |
| SVG | 仅元数据路径；与 DXF 同源，网络侧用 DXF |

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

## Launcher / CLI

```bash
./run.sh preprocess --data-dir /data/hdd/datasets/eccv2026ws-cad-data --num-workers 4
./run.sh train --dataset eccv2026ws-cad-data --data-dir /data/hdd/datasets/eccv2026ws-cad-data \
  --weight-folder /data/hdd/outputs/AutoBrep --exp-dir <exp> --max-steps 10000
./run.sh infer --view-conditioned 1 --checkpoint <exp>/checkpoints --sample-id 000029 \
  --data-dir /data/hdd/datasets/eccv2026ws-cad-data --exp-dir <exp> --output-dir <out>
```

`run.sh capabilities` 声明 `preprocess/train/infer` 与数据集 `eccv2026ws-cad-data`。  
exp_launcher 按 `git_ref` 探测 capabilities，**需提交 `eccv` 分支上的 `run.sh` / `models.tsv`** 后 UI 才能看到新接口。

详见 `configs/autobrep_eccv.yaml`。

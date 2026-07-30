# AutoBrep 点云条件生成（pc-cond）

## Git 底线

| 锚点 | 含义 |
|------|------|
| tag `base` = `0e88b1b` | 上游无条件原版（未改） |
| branch `pc-cond` | 点云条件 + exp_launcher 接入 |

## 做法（冻结骨干）

1. 数据：ABC-1M parquet；从 `face_bbox + face_ncs` 现采点云 `(2048,3)`，归一化到 `[-1,1]`
2. 模型：`PointCloudConditionEncoder` → `(B, 66, 2048)` soft tokens（含 BOPC/EOPC 软边界）
3. 注入：`XTransformer(..., prepend_embeds=...)`（x_transformers 原生）
4. 冻结：`cad_gpt` + FSQ VAE；只训 `pc_encoder`（约数 M 参数）
5. 损失：仍是 AR CE；prepend 段不计入 CE（wrapper 已裁掉）

## 权重规模（实测 `ar.ckpt`）

| 项 | 数值 |
|----|------|
| state_dict 参数量 | **~1018M**（cad_gpt ~987M + surf_vae ~23M + edge_vae ~8M） |
| state_dict 体积 | ~4.1 GB（fp32 张量） |
| 文件 12GB | 另含 Adam 优化器状态 ~7.9 GB |
| 配置 | dim=2048, depth=16, heads=32, kv_groups=8, max_seq=3000, codebook=1024 |

## 4090D（24GB）能否训

实测 `scripts/probe_pc_vram.py`（冻骨干 + prepend CE backward，**未含** FSQ encode）：

| seq_len | peak allocated |
|---------|----------------|
| 512 | **~7.0 GB** |
| 1536 | **~8.6 GB** |

线性外推 seq=3000 大约 **~11 GB** 量级；加上真实 `common_step` 里 FSQ encode（no_grad）与 DataLoader 开销，预期 **14–18 GB**，单卡 4090D **能跑**。

建议：`batch_size=1`，`accumulate_grad_batches=4`，`bf16-mixed`。若 OOM 再降 `max_seq` 或开 checkpointing。

可训参数：PC encoder **~5.7M**（总参数 ~1023M，冻结 ~1018M）。

## launcher

流式训练主旋钮是 **`max_steps`（优化器更新次数）**，不是 epoch。

```bash
# capabilities 含 train + infer；datasets: abc / abc-1m
./run.sh train --exp-dir ... --data-dir /data/hdd/datasets/ABC-1M \
  --weight-folder /data/hdd/outputs/AutoBrep \
  --max-steps 10000 --accumulate-grad-batches 4 \
  --val-check-interval 500 --limit-val-batches 50
./run.sh infer --pc-conditioned 1 --point-cloud /path/to.npy --checkpoint .../last.ckpt ...
```

含义：
- `max_steps`：停训条件（Lightning optimizer steps）
- `accumulate_grad_batches`：每步对应若干 microbatch；大约吃掉 `max_steps × accum` 条样本（batch_size=1 时）
- `val_check_interval` / `limit_val_batches`：周期性抽一段 val 流，避免扫完整 val

`models.tsv`：`pretrained`（无条件 infer）与 `pc-cond`（点云训/推）均指向本分支 tip。

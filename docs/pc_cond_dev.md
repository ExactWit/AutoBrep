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

| 设置 | 预期 |
|------|------|
| 冻骨干 + 只训 PC 头 | **可以**，建议 `batch_size=1`，`accumulate_grad_batches=4`，`bf16-mixed` |
| 激活 | 仍要过完整 16 层算对 prefix 的梯度，**激活显存是大头**（seq=3000 时约数 GB～十余 GB） |
| 建议 | 先 `scripts/probe_pc_vram.py`；若 OOM 把 `max_seq` 降到 2048 或加 gradient checkpointing |

可训参数量：PC encoder（hidden=256, latents=64）约 **~2–5M**，优化器状态可忽略。

## launcher

```bash
# capabilities 含 train + infer；datasets: abc / abc-1m
./run.sh train --exp-dir ... --data-dir /data/hdd/datasets/ABC-1M --weight-folder /data/hdd/outputs/AutoBrep
./run.sh infer --pc-conditioned 1 --point-cloud /path/to.npy --checkpoint .../last.ckpt ...
```

`models.tsv`：`pretrained`（无条件 infer）与 `pc-cond`（点云训/推）均指向本分支 tip。

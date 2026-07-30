# AutoBrep Autocomplete

条件补全管线：给定局部 B-Rep（面 bbox + UV + 边），拼成 `BOGEOM…EOGEOM` token 前缀，再用官方 `ar.ckpt` 自回归续写到 STEP。

**自定义输入不是自然语言，也不是任意 STEP。** 是与训练 `geom_tokenization` 同构的离散几何条件。

## 条件序列

```text
BOS BOM GEN_* EOM
BOGEOM BOL
  [每个条件面] BOF  bbox×6  FSQ面码×4
               [边] prevFace|DUMMYID  bbox×6  FSQ边码×2 …
  EOF
EOL EOGEOM
BOC
→ AR 续写到 EOC/EOS
```

- 位置：10-bit bbox
- 形状：推理时用 `surf-fsq` / `edge-fsq` **编码器**得到真实 FSQ 码（不能用训练时的 z 占位符）
- 悬空边：`DUMMYID`；两条件面共边：引用对方局部 Face ID

解码端若序列含 `BOGEOM…EOGEOM`，会把条件几何并进 CAD 再 Sewing。

## 两档输入

### 1. ABC-1M 样本子集

```bash
./run.sh infer \
  --exp-dir /tmp/ab_ac_exp --output-dir /tmp/ab_ac_out \
  --weight-folder /data/hdd/outputs/AutoBrep \
  --autocomplete 1 \
  --data-dir /data/hdd/datasets/ABC-1M \
  --abc-stem 00007410_d681bb73885e41b39c681d22_step_047_0074 \
  --face-ids 0,3 \
  --batch-size 1 --num-batches 1
```

也可用 `--num-condition-faces 4` 或 `--condition-mode constraint`（读行内 `constraint_faces`）。

### 2. JSON 局部 BRep

```json
{
  "faces": [
    {"id": 0, "bbox": [6 floats], "uv": [[[32,32,3]]]},
    {"id": 1, "bbox": [...], "uv": [...]}
  ],
  "edges": [
    {"face_a": 0, "face_b": null, "bbox": [6], "uv": [[32,3]]},
    {"face_a": 0, "face_b": 1, "bbox": [...], "uv": [...]}
  ]
}
```

- `face_b: null` → 悬空边 → `DUMMYID`
- `uv` 建议始终提供；缺 UV 时效果会很差

```bash
./run.sh infer ... --autocomplete 1 --condition-json /path/to/cond.json
```

## Smoke 测试

```bash
python scripts/autocomplete_smoke.py --case all \
  --data-dir /data/hdd/datasets/ABC-1M \
  --weight-folder /data/hdd/outputs/AutoBrep \
  --output-dir /data/hdd/outputs/autobrep_autocomplete_smoke
```

| 案例 | 做法 |
|------|------|
| A | ABC 取 2 个相邻面 |
| B | `constraint_faces` |
| C | A 导出的 JSON 再读入 |

输出：`{output}/infer/*.step` + sidecar JSON（`face_ids`、`prompt_len`、`stem`）。

## 说明与风险

官方 `ar.ckpt` 训练时 `geom_tokens` 曾被注释，BOGEOM 为零样本。续写 CAD 可能无法 Sewing；infer 会：

1. 尝试解码完整生成序列
2. 失败则回退为「仅条件 BOGEOM」重建
3. 仍失败则写 sidecar（`error=rebuild_failed` / `decode_failed`），进程不崩

成功标准以「可喂条件、管线通、可出 STEP 或明确 soft-fail」为准。

## 分支说明

- 本功能在分支 `autocomplete`（从 tag `base` / `0e88b1b` 分出）
- 与 `pc-cond` 点云条件训练隔离；`models.tsv` 中 `autocomplete` 与 `pretrained` 并存

## 非目标

- 任意 STEP → 自动拆面条件
- 重新训练带 `load_geom` 的 AR
- 与点云条件合并

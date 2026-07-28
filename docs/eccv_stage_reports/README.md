# ECCV 阶段报告模板

> 每个 **R\*** 节点跑完后填一张表，**硬停**等确认后再 enqueue 下一阶段。  
> 最新报告：同目录 `R*_latest.md`。

## 门禁顺序

| 门禁 | 内容 | 自动连跑？ |
|------|------|------------|
| R_smoke | 通路冒烟（40 step） | workflow `eccv-challenge-gated` |
| R0 | 旧 pred analytic 后处理 A/B | **脚本** `scripts/eccv_stage_r0_postprocess_ab.py` |
| R1 | 现有 3view ckpt + analytic GT test | 单独 enqueue（见下） |
| R2 | P0 hist-split 全量训+test | **仅**门禁1确认后 |
| P1a/P1b | 条件编码升级 | **仅**门禁2后按需 |

## 对照基线

| 角色 | run | 指标 |
|------|-----|------|
| parent train | `260723-162838` | 3view-geom |
| GT test（旧） | `260725-002218` | gen≈35.3%, summary≈0.046 |

## R1 启动示例（门禁通过后）

```bash
# 用 HTTP / workflow 单步；run_id 指向 parent train
curl -s -X POST http://127.0.0.1:8765/api/runs/start \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_name":"AutoBrep",
    "entry_id":"eccv-3view-p0",
    "dataset_id":"eccv2026ws-cad-data",
    "task":"gen",
    "mode":"test",
    "gpu":"0",
    "tag":"R1-ckpt-gt-test",
    "run_id":"260723-162838",
    "enqueue":true,
    "note":"【R1】现有 3view ckpt + postprocess_analytic=1 GT test；对照 260725-002218；硬门禁1",
    "train_config":{"gen_batch":1,"complexity":"from_condition","postprocess_analytic":1}
  }'
exp_launcher queue tick --json
```

## 报告填写 checklist

- [ ] summary / gen_success / surface|edge|vertex|topo F1 / CD（若有）
- [ ] vs `260725-002218`
- [ ] 下一动作：停 / R2 重训 / 进 surf-type / 开 P1

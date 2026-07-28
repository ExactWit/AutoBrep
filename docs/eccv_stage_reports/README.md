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

## R1 启动（门禁通过后）

> 注意：launcher 对 `run_id=260723-162838`（geom tip）会继承 parent entry，**checkout 到无 postprocess 的 tip**。  
> R1 请在 **`eccv-p0` tip** 上直接跑（当前正式路径）：

```bash
cd /home/divisor/workspace/repo/AutoBrep && git checkout eccv-p0
conda activate autobrep
EXP=/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/stage_gates/R1_ckpt_gt_test
CKPT=/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260723-162838/eccv-3view-geom-resume__train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=core/src python -u scripts/eval_eccv_split.py \
  --exp-dir "$EXP" --output-dir "$EXP" \
  --data-dir /data/hdd/datasets/eccv2026ws-cad-data \
  --weight-folder /data/hdd/outputs/AutoBrep \
  --checkpoint "$CKPT" --gpu 0 --split test \
  --complexity from_condition --gen-batch 1 --postprocess-analytic 1
# 完成后写报告：
python scripts/eccv_stage_r1_write_report.py --test-json "$EXP/metrics/test.json" --run-dir "$EXP"
```

## 报告填写 checklist

- [ ] summary / gen_success / surface|edge|vertex|topo F1 / CD（若有）
- [ ] vs `260725-002218`
- [ ] 下一动作：停 / R2 重训 / 进 surf-type / 开 P1

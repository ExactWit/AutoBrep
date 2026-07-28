# ECCV 管线完整性对照（base / 3view / pc-cond）

更新时间：2026-07-28

图例：✅ 可用 · ⚠️ 有产物但不可信/不完整 · ❌ 缺失 · 🔄 进行中 · 🆕 P0+

数据与实验根目录：

| 域 | 路径 |
|----|------|
| 数据集 | `/data/hdd/datasets/eccv2026ws-cad-data` |
| 实验 runs | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep` |
| 产出 outputs | `/data/hdd/outputs/eccv2026ws-cad-data/gen/AutoBrep` |
| 预训练权重 | `/data/hdd/outputs/AutoBrep` |
| 代码仓库 | `/home/divisor/workspace/repo/AutoBrep` |

---

## 五阶段对照表

| 阶段 | 含义 | **base**（flat TechDraw） | **3view-geom** | **pc-cond** |
|------|------|---------------------------|----------------|-------------|
| **训练** | 50 epoch CE，有 ckpt | ✅ | ✅ | ❌ 无 ECCV run（仅有 ABC-1M 旧训） |
| **val** | 训练中 CE `val_loss` | ✅ 有 best/last | ✅ 有 best/last | — |
| **official val** | 训中/训后 STEP + 官方分 | ⚠️ mid/full 多为 0（旧 KV bug）；仅 mid-24 repaired 可用 | ❌ mid×3 + full 全 gen=0（同 KV bug，无 repaired） | — |
| **official test** | datasplit `test`，有 GT | ✅ | ✅ | — |
| **public infer** | `public_test`，无 GT | ✅ 有 submission.zip | 🔄 全量重跑中 | — |

---

## 各模型关键路径与数字

### base（entry `eccv-base`，tag `eccv-base-resume`）

| 阶段 | 状态 | Run 目录 | 关键指标 |
|------|------|----------|----------|
| 训练 | ✅ | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260721-194515/eccv-base-resume__train` | best val_loss ≈ **0.2253**（step 18336）；`last.ckpt` + best ckpt |
| val（CE） | ✅ | 同上 `metrics/train_summary.json`、`tensorboard/` | 与 best 监控一致 |
| official val | ⚠️ | 同上 `metrics/official_val_*` | mid/full 定时评多为 **gen=0**；可用：`official_val_mid_best_repaired` → gen **13/24 (54.2%)**，summary **≈0.077** |
| official test | ✅ | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260722-132547/eccv-base-resume__test` | 348：gen **107 (30.7%)**，summary **0.0506**，surf F1 0.0676；见 `metrics/test.json` |
| public infer | ✅ | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260724-191452/eccv-base-resume__public_infer` | 927：gen **285 (30.7%)**；zip：`/data/hdd/outputs/eccv2026ws-cad-data/gen/AutoBrep/260724-191452/eccv-base-resume__public_infer/submission.zip` |

### 3view-geom（entry `eccv-3view-geom`，tag `eccv-3view-geom-resume`）

| 阶段 | 状态 | Run 目录 | 关键指标 |
|------|------|----------|----------|
| 训练 | ✅ | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260723-162838/eccv-3view-geom-resume__train` | best val_loss ≈ **0.2253**（step 22920）；`last.ckpt` + best ckpt |
| val（CE） | ✅ | 同上 `metrics/train_summary.json`、`tensorboard/` | 与 base 同量级 |
| official val | ❌ | 同上 `metrics/official_val_mid_*`、`official_val_full_*` | mid×3 + full：**gen=0**（训中 KV bug）；**无** mid-24 repaired |
| official test | ✅ | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260725-002218/eccv-3view-geom-resume__test` | 348：gen **123 (35.3%)**，summary **0.0460**，surf F1 0.0645；见 `metrics/test.json` |
| public infer | 🔄 | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260727-172702/eccv-3view-geom-resume__public_infer` | 全量重跑中（PATH 修复后）；历史：`260725-014123` 曾到 ~418/927 被主机挂死；尚无完整 submission |

### pc-cond

| 说明 | 路径 |
|------|------|
| 不在本 ECCV 数据闭环 | 无 `/data/hdd/exps/runs/eccv2026ws-cad-data/...` 下的 train/test/public |
| 旧 ABC-1M 训练（参考） | `/data/hdd/exps/runs/abc-1m/gen/AutoBrep/260719-183113/pc-cond__train` |

若要对齐五阶段，需在 `eccv2026ws-cad-data` 上单独开训评。

---

## GT test 对比（可写报告）

| 模型 | Run | gen 成功率 | summary | surf F1 |
|------|-----|-----------|---------|---------|
| base | `.../260722-132547/eccv-base-resume__test` | 30.7% (107/348) | **0.0506** | 0.0676 |
| 3view | `.../260725-002218/eccv-3view-geom-resume__test` | **35.3%** (123/348) | 0.0460 | 0.0645 |

3view 合法率略高，官方 summary 略低；同量级。

---

## 缺口与补齐优先级（挑战赛门禁）

详见 [`docs/eccv_stage_reports/README.md`](eccv_stage_reports/README.md)；workflow：`workflows/eccv_challenge_gated.yaml`。

1. **R_smoke**：通路冒烟（40 step，非正式训）— `eccv-challenge-gated`。
2. **R0**：旧 GT test pred（`260725-002218`）analytic 后处理 A/B — `scripts/eccv_stage_r0_postprocess_ab.py`（零训练）。
3. **R1**：现有 3view ckpt（`260723-162838`）+ `postprocess_analytic=1` 官方 GT test；**硬停**写报告。
4. **R2（按需）**：仅当门禁确认 hist-split 值得重训时，再开 P0 50ep + GT test。
5. **曲面类型进模型**：相对事后拟合更对症；优先于盲跑 P1 全量。
6. **P1-A / P1-B**：门禁通过且条件编码仍是瓶颈时再 enqueue（不自动连跑）。
7. **3view public**：交 submission 单独盯 `260727-172702`，与抬分门禁解耦。
8. **pc-cond**：仅当需要第三条件分支时再在 ECCV 数据上立项。

代码合入状态：P0 tip（`eccv-3view-p0`）fast metrics ✅ · hist-split+groups ✅ · analytic postprocess ✅ — 见 `docs/eccv_upgrade_p0_p1_p2.md`。

---

## 工程修复备忘（影响评测可信度）

| 问题 | 影响 | 状态 |
|------|------|------|
| `gen_batch=2` OOM thrash | public/test 极慢或挂 | 已改默认 `gen_batch=1` |
| KV cache rotary / prepend | official val / public STEP 全 0 | 已修（`input_not_include_cache`） |
| `conda activate` 弄丢 `exp_launcher` | 空日志、~6s 假完成 | 已修（绝对路径 + filter 在 activate 外） |
| 主机硬挂 / CUDA 804 | 3view public 多次中断 | 已重提；需盯当前 run |

---

## 一句话

对「能交报告的主结果」：base 齐；3view 差完整 public，且两边都缺可信 official val STEP；CE 训练与 official test 两边都齐。
